#!/usr/bin/env python3
"""
Generate synthetic research questions for SFT distillation and RL training.

Uses Claude Sonnet 5 to generate diverse research questions that exercise
the agent's tool-use capabilities across different domains and tool combinations.

The generated questions replace the HLE-derived rl_train_data.jsonl — they're
designed to teach research workflow and tool usage, not test parametric knowledge.

Output format matches what both SFT and RL pipelines expect:
    {"prompt": [{"role": "user", "content": "..."}], "metadata": {"prompt": "...", "domain": "...", "tools": [...]}}

Usage:
    # Generate 500 training questions + 100 eval questions
    uv run test-scripts/sft_generate_questions.py --count 500 --eval-count 100

    # Generate for specific domains only
    uv run test-scripts/sft_generate_questions.py --domains finance,science --count 200

    # Custom output paths
    uv run test-scripts/sft_generate_questions.py --output my_train.jsonl --eval-output my_eval.jsonl
"""

import argparse
import json
import logging
import random
import sys
from pathlib import Path

import boto3

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Domain definitions — each maps to tools the question should exercise
# ---------------------------------------------------------------------------

DOMAINS = {
    "general": {
        "tools": ["nova"],
        "description": "General web research questions requiring current information",
        "examples": [
            "What are the latest developments in quantum computing error correction?",
            "Compare the regulatory approaches to AI governance in the EU vs US",
            "What is the current state of nuclear fusion energy research?",
        ],
    },
    "finance": {
        "tools": ["nova", "edgar"],
        "description": "Finance and economics questions requiring market data and filings",
        "examples": [
            "Analyze the competitive position of NVIDIA in the AI chip market based on recent filings",
            "What are the macroeconomic implications of recent central bank policy changes?",
            "Compare revenue growth trends across major cloud providers",
        ],
    },
    "science": {
        "tools": ["nova", "arxiv"],
        "description": "Science and technology questions requiring academic literature",
        "examples": [
            "What are the leading approaches to protein structure prediction beyond AlphaFold?",
            "Summarize recent advances in room-temperature superconductor research",
            "What are the current challenges in scaling large language models?",
        ],
    },
    "medical": {
        "tools": ["nova", "pubmed", "openfda", "clinicaltrials"],
        "description": "Medical and life science questions requiring clinical literature",
        "examples": [
            "What are the latest FDA-approved treatments for treatment-resistant depression?",
            "Compare outcomes of different immunotherapy approaches for lung cancer",
            "What does recent research say about long COVID mechanisms and treatments?",
        ],
    },
    "policy": {
        "tools": ["nova"],
        "description": "Policy and governance questions requiring multi-source analysis",
        "examples": [
            "How are different countries approaching regulation of autonomous vehicles?",
            "What are the arguments for and against universal basic income based on recent pilot studies?",
            "Analyze the global response to semiconductor supply chain vulnerabilities",
        ],
    },
    "technology": {
        "tools": ["nova", "arxiv"],
        "description": "Technology industry analysis requiring current and academic sources",
        "examples": [
            "What are the leading approaches to AI safety and alignment?",
            "Compare the architectures and tradeoffs of different LLM serving frameworks",
            "What is the current state of brain-computer interface technology?",
        ],
    },
}

# ---------------------------------------------------------------------------
# Generation prompt
# ---------------------------------------------------------------------------

GENERATION_PROMPT = """You are generating diverse research questions for training an AI deep research agent.

Domain: {domain}
Description: {description}
Tools available: {tools}

Example questions in this domain:
{examples}

Generate {batch_size} NEW and DIVERSE research questions for this domain. Requirements:
1. Questions should require searching multiple sources and synthesizing information
2. Questions should be answerable with web search and the listed tools (not parametric knowledge)
3. Questions should require a multi-paragraph research report, not a one-word answer
4. Questions should be specific enough to have a clear research direction
5. Questions should cover different subtopics within the domain
6. Questions should be phrased as a user would naturally ask them
7. Avoid questions that can be answered from memory alone — they must need current data
8. Mix complexity: some straightforward, some requiring multi-hop reasoning across sources

Return ONLY a JSON array of strings, one question per element. No other text.
Example: ["Question 1?", "Question 2?", "Question 3?"]"""


# ---------------------------------------------------------------------------
# Generation logic
# ---------------------------------------------------------------------------


def generate_batch(
    bedrock_client,
    domain: str,
    domain_config: dict,
    batch_size: int = 20,
    model_id: str = "global.anthropic.claude-sonnet-5",
) -> list[dict]:
    """Generate a batch of questions for a given domain."""
    examples = "\n".join(f"- {e}" for e in domain_config["examples"])
    tools = ", ".join(domain_config["tools"])

    prompt = GENERATION_PROMPT.format(
        domain=domain,
        description=domain_config["description"],
        tools=tools,
        examples=examples,
        batch_size=batch_size,
    )

    response = bedrock_client.converse(
        modelId=model_id,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={"maxTokens": 4096, "temperature": 1.0},
    )

    # Extract text from response — handle thinking models that return reasoningContent
    content_blocks = response["output"]["message"]["content"]
    text = ""
    for block in content_blocks:
        if "text" in block:
            text = block["text"].strip()
            break
    if not text:
        logger.warning(f"No text block in response for domain={domain}, skipping batch")
        return []

    # Parse JSON array from response
    # Handle markdown code blocks if present
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    try:
        questions = json.loads(text)
    except json.JSONDecodeError:
        logger.warning(f"Failed to parse response for domain={domain}, skipping batch")
        return []

    if not isinstance(questions, list):
        logger.warning(f"Response not a list for domain={domain}, skipping batch")
        return []

    # Format into training data format
    results = []
    for q in questions:
        if not isinstance(q, str) or len(q) < 20:
            continue
        results.append(
            {
                "prompt": [{"role": "user", "content": q}],
                "enabled_sources": domain_config["tools"],
                "metadata": {
                    "prompt": q,
                    "domain": domain,
                    "tools": domain_config["tools"],
                },
            }
        )

    return results


def generate_questions(
    count: int,
    domains: list[str],
    model_id: str = "global.anthropic.claude-sonnet-5",
    region: str | None = None,
) -> list[dict]:
    """Generate research questions across specified domains."""
    region = region or "us-east-1"
    bedrock = boto3.client("bedrock-runtime", region_name=region)

    # Distribute questions across domains evenly
    per_domain = count // len(domains)
    remainder = count % len(domains)

    all_questions = []

    for i, domain in enumerate(domains):
        domain_config = DOMAINS[domain]
        target = per_domain + (1 if i < remainder else 0)
        generated = 0

        logger.info(f"Generating {target} questions for domain: {domain}")

        while generated < target:
            batch_size = min(25, target - generated)  # Max 25 per API call
            batch = generate_batch(
                bedrock, domain, domain_config, batch_size, model_id
            )
            all_questions.extend(batch)
            generated += len(batch)
            logger.info(f"  {domain}: {generated}/{target} generated")

            if not batch:
                logger.warning(f"  Empty batch for {domain}, retrying...")

    # Shuffle to mix domains
    random.shuffle(all_questions)
    return all_questions[:count]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic research questions for SFT and RL training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--count", type=int, default=500, help="Number of training questions (default: 500)"
    )
    parser.add_argument(
        "--eval-count",
        type=int,
        default=100,
        help="Number of held-out eval questions (default: 100)",
    )
    parser.add_argument(
        "--domains",
        type=str,
        default=None,
        help=f"Comma-separated domains (default: all). Available: {','.join(DOMAINS.keys())}",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="test-scripts/results/rl_train_data.jsonl",
        help="Output path for training questions (default: test-scripts/results/rl_train_data.jsonl)",
    )
    parser.add_argument(
        "--eval-output",
        type=str,
        default="test-scripts/results/sft_eval_questions.jsonl",
        help="Output path for eval questions (default: test-scripts/results/sft_eval_questions.jsonl)",
    )
    parser.add_argument(
        "--force-eval-overwrite",
        action="store_true",
        help=(
            "Overwrite an existing eval question set. This invalidates every "
            "score already measured against it, since models would then be "
            "compared on different questions."
        ),
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="global.anthropic.claude-sonnet-5",
        help="Bedrock model ID for question generation (default: Sonnet 5)",
    )
    parser.add_argument(
        "--region",
        type=str,
        default=None,
        help="AWS region (default: AWS_DEFAULT_REGION or us-east-1)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for shuffling (default: 42)"
    )

    args = parser.parse_args()
    random.seed(args.seed)

    # Parse domains
    if args.domains:
        domains = [d.strip() for d in args.domains.split(",")]
        invalid = [d for d in domains if d not in DOMAINS]
        if invalid:
            logger.error(f"Invalid domains: {invalid}. Available: {list(DOMAINS.keys())}")
            sys.exit(1)
    else:
        domains = list(DOMAINS.keys())

    region = args.region or "us-east-1"
    total = args.count + args.eval_count

    logger.info("=" * 60)
    logger.info("Generating Synthetic Research Questions")
    logger.info("=" * 60)
    logger.info(f"Training questions: {args.count}")
    logger.info(f"Eval questions:     {args.eval_count}")
    logger.info(f"Domains:            {domains}")
    logger.info(f"Model:              {args.model_id}")
    logger.info(f"Region:             {region}")
    logger.info("")

    # Generate all questions at once, then split
    all_questions = generate_questions(
        count=total, domains=domains, model_id=args.model_id, region=region
    )

    if len(all_questions) < total:
        logger.warning(
            f"Generated {len(all_questions)} questions (requested {total}). "
            f"Adjusting split proportionally."
        )

    # Split into train and eval
    eval_count = min(args.eval_count, len(all_questions) // 5)  # Max 20% for eval
    train_questions = all_questions[eval_count:]
    eval_questions = all_questions[:eval_count]

    # Write training questions
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for q in train_questions:
            f.write(json.dumps(q) + "\n")
    logger.info(f"✓ Wrote {len(train_questions)} training questions to {output_path}")

    # Write eval questions
    # Only touch the eval file when eval questions were actually requested, and
    # never overwrite an existing one without --force: it is a benchmark
    # reference set, and silently replacing it invalidates every score already
    # measured against it (models end up compared on different questions).
    eval_path = Path(args.eval_output)
    if eval_count == 0:
        logger.info("eval_count=0: leaving %s untouched", eval_path)
    elif eval_path.exists() and not args.force_eval_overwrite:
        logger.warning(
            "%s already exists; refusing to overwrite a benchmark reference set. "
            "Pass --force-eval-overwrite to replace it (this invalidates "
            "comparisons with previously measured scores).",
            eval_path,
        )
    else:
        eval_path.parent.mkdir(parents=True, exist_ok=True)
        with open(eval_path, "w") as f:
            for q in eval_questions:
                f.write(json.dumps(q) + "\n")
        logger.info(f"✓ Wrote {len(eval_questions)} eval questions to {eval_path}")

    # Print domain distribution
    logger.info("\nDomain distribution (training):")
    domain_counts = {}
    for q in train_questions:
        d = q["metadata"]["domain"]
        domain_counts[d] = domain_counts.get(d, 0) + 1
    for d, c in sorted(domain_counts.items()):
        logger.info(f"  {d}: {c}")


if __name__ == "__main__":
    main()
