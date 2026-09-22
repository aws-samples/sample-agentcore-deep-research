"""
Shared rubric for scoring deep research reports.

SINGLE SOURCE OF TRUTH. Imported by both:
  - test-scripts/eval-agent.py   (offline eval metric)
  - patterns/strands-deep-research/rl_app.py  (RL training reward)

These must be the same measurement applied at different points in the pipeline.
They were previously duplicated by copy-paste with a "keep in sync" comment,
which is exactly how an eval metric and a training reward silently diverge.

Design notes — why this is not just "count the citations":

An earlier version scored citations by counting occurrences of the literal
string "[Source:" (3 hits => full marks) and format by testing for five
heading strings. A calibration probe showed a report consisting of the five
headings plus three fabricated `[Source: https://example.com/...]` links, with
deliberately vacuous body text, scored 0.412 — above a real frontier-model
report's own rubric sub-score, and 2.3x a genuine competitor baseline. The LLM
judge correctly rated that probe 2/10 on every criterion; all of its score came
from the two string-matching heuristics.

That is a reward-hacking vector: emitting section headings and citation-shaped
strings is the easiest pattern for a student model to copy from teacher traces,
and under RL it is the first thing a policy would exploit. So here:

  * Citations must be well-formed, de-duplicated URLs, and placeholder domains
    are rejected. When the set of URLs actually returned by tool calls is
    known, citations must appear in it — fabricated citations score zero.
  * Section credit requires substantive content beneath each heading, not just
    the heading text.
  * The heuristics are weighted down (10% each, from 15%) so the LLM-judged
    criteria dominate, since those were shown to resist the gaming probe.
  * A grounding/faithfulness criterion is scored explicitly.

When changing the weights or criteria, re-check that a deliberately vacuous
report (correct headings, fabricated citations, no substance) still scores far
below a real one. An earlier version scored such a report 0.412 against a real
report's 0.762 — a 1.85x gap, too narrow to survive being optimised against. The
current weighting puts that gap above 5x. The gamed probe must
score far below a real report.
"""

import json
import re
from urllib.parse import urlparse

# --- LLM-judged criteria -----------------------------------------------------

RUBRICS = [
    {
        "criterion": (
            "Coverage: How thoroughly does the report address all aspects of the "
            "research question? (1=superficial/misses key aspects, 5=partial "
            "coverage with gaps, 10=comprehensive, addresses all facets)"
        ),
        "category": "coverage",
    },
    {
        "criterion": (
            "Source Quality: How well does the report cite specific, credible "
            "sources with inline references? (1=no citations, 5=some citations but "
            "vague/few, 10=abundant specific citations from authoritative sources "
            "throughout)"
        ),
        "category": "sources",
    },
    {
        "criterion": (
            "Synthesis: Does the report synthesize information across sources into "
            "a coherent narrative, or just list facts? (1=disconnected bullet "
            "points, 5=some connections made, 10=deeply integrated analysis "
            "drawing novel connections across sources)"
        ),
        "category": "synthesis",
    },
    {
        "criterion": (
            "Depth: Does the report provide substantive analysis with specific "
            "data, examples, and nuance? (1=vague generalities only, 5=some "
            "specifics but shallow, 10=rich detail with data points, expert "
            "perspectives, counterarguments, and nuance)"
        ),
        "category": "depth",
    },
    {
        "criterion": (
            "Actionability: Are conclusions well-supported and useful? Does the "
            "report distinguish established facts from emerging trends? (1=no "
            "clear takeaways, 5=some conclusions but unsupported, 10=precise "
            "conclusions with caveats, clearly distinguishes certainty levels)"
        ),
        "category": "actionability",
    },
    {
        "criterion": (
            "Grounding: Are the report's specific claims (numbers, dates, named "
            "entities, quotes) actually attributable to the cited sources, and "
            "does it avoid confident but unsupported assertions? (1=specifics "
            "appear invented or wholly uncited, 5=a mix of supported and "
            "unsupported claims, 10=every substantive claim is traceable to a "
            "named source and uncertainty is stated explicitly)"
        ),
        "category": "grounding",
    },
]

# Prefix inserted when tool observations are available. Without it the judge cannot
# verify grounding at all -- it can only see whether claims look attributed. Showing
# the retrieved context is the fix DR Tulu (arXiv 2511.19399) recommends.
SEARCH_CONTEXT_BLOCK = (
    "The agent retrieved the following source material. Use it to verify the "
    "report's specific claims. A claim that contradicts this material, or that "
    "cannot be found in it, is NOT grounded however confidently it is stated.\n"
    "-- retrieved source material --\n{context}\n-- end source material --\n\n"
)

RUBRIC_JUDGE_PROMPT = (
    "You are a strict evaluator of deep research reports. Score each criterion on "
    "a 1-10 scale.\n"
    "Be discriminating: a score of 7-8 means genuinely good, 9-10 means "
    "exceptional.\n"
    "Most adequate reports should score 5-7. Only outstanding reports reach 8+.\n"
    "Correct section headings and citation-shaped text are NOT evidence of "
    "quality: judge the substance beneath them. A well-formatted report whose "
    "claims are vague, generic, or unsupported must score low.\n\n"
    "Question: {question}\n\n{context_block}Report:\n{report}\n\n"
    "Criteria:\n{criteria}\n\n"
    "Return ONLY a JSON object mapping criterion index to integer score 1-10.\n"
    'Example: {{"0": 7, "1": 5, "2": 8, "3": 6, "4": 7, "5": 6}}'
)

# Component weights. The LLM-judged rubric dominates because the two
# string-matching heuristics below are inherently gameable.
WEIGHT_RUBRIC = 0.80
WEIGHT_CITATION = 0.10
WEIGHT_FORMAT = 0.10

# Distinct valid citations needed for full citation credit.
CITATION_TARGET = 5

# Minimum characters of body text under a heading for that section to count.
SECTION_MIN_CHARS = 200

# Reports shorter than this cannot earn structural credit at all.
MIN_REPORT_CHARS = 1500

PLACEHOLDER_HOSTS = {
    "example.com",
    "www.example.com",
    "example.org",
    "example.net",
    "localhost",
    "test.com",
    "foo.com",
    "url.com",
}

CITATION_RE = re.compile(r"\[Source:\s*([^\]]+?)\s*\]", re.IGNORECASE)


def extract_citations(report: str) -> set[str]:
    """Return the set of well-formed, non-placeholder URLs cited in the report."""
    urls: set[str] = set()
    for raw in CITATION_RE.findall(report or ""):
        match = re.search(r"https?://[^\s\)\]]+", raw)
        if not match:
            continue
        url = match.group(0).rstrip(".,;")
        host = (urlparse(url).netloc or "").lower()
        if not host or host in PLACEHOLDER_HOSTS:
            continue
        if "." not in host:
            continue
        urls.add(url)
    return urls


def score_citations(report: str, retrieved_urls: set[str] | None = None) -> float:
    """
    Validated citation score in [0, 1].

    Counts distinct well-formed URLs rather than occurrences of a marker string.
    When `retrieved_urls` is supplied (the URLs actually returned by tool calls
    in this session), citations absent from it are treated as fabricated and
    ignored, which closes the "invent plausible links" exploit.

    Full credit additionally requires at least two distinct domains, so a report
    cannot max the component by citing one page five times.
    """
    cited = extract_citations(report)
    if retrieved_urls:
        normalized = {u.rstrip("/") for u in retrieved_urls}
        cited = {u for u in cited if u.rstrip("/") in normalized}
    if not cited:
        return 0.0

    count_score = min(len(cited) / CITATION_TARGET, 1.0)
    domains = {(urlparse(u).netloc or "").lower() for u in cited}
    diversity = min(len(domains) / 2.0, 1.0)
    return count_score * diversity


def score_format(report: str) -> float:
    """
    Structural compliance in [0, 1], requiring real content per section.

    Headings alone earn nothing: each expected section must be followed by at
    least SECTION_MIN_CHARS of body text, and very short documents are excluded
    entirely. This is what stops a heading-only stub from scoring 1.0.
    """
    text = (report or "").strip()
    if len(text) < MIN_REPORT_CHARS:
        return 0.0

    # Map each heading to the text that follows it, up to the next heading.
    sections: dict[str, str] = {}
    parts = re.split(r"^(#{1,3})\s*(.+?)\s*$", text, flags=re.MULTILINE)
    for i in range(1, len(parts) - 2, 3):
        title = parts[i + 1].strip().lower()
        body = parts[i + 2]
        sections[title] = sections.get(title, "") + body

    def has_section(*keywords: str) -> bool:
        for title, body in sections.items():
            if (
                any(k in title for k in keywords)
                and len(body.strip()) >= SECTION_MIN_CHARS
            ):
                return True
        return False

    checks = [
        text.startswith("#"),
        has_section("executive summary", "summary"),
        has_section("finding", "key findings"),
        has_section("analysis"),
        has_section("conclusion"),
    ]
    return sum(checks) / len(checks)


class RubricJudgeError(RuntimeError):
    """
    The judge could not be scored — as distinct from scoring zero.

    Raised rather than returning 0.0 because the two are not interchangeable: a
    throttled or unparsable judge is missing data, while 0.0 is a claim about
    report quality. Conflating them corrupts eval means and, in RL, becomes a
    false training label.
    """


def score_rubric_with_judge(
    question: str,
    report: str,
    bedrock_client,
    judge_model: str,
    observations: list[str] | None = None,
    context_chars: int | None = 24000,
) -> tuple[float, dict]:
    """
    Run the LLM judge. Returns (normalized_score_0_1, per_criterion_dict).

    Criteria are equally weighted; each is scored 1-10 and normalized by 10.
    """
    criteria_text = "\n".join(f"{i}. {r['criterion']}" for i, r in enumerate(RUBRICS))
    context_block = ""
    if observations:
        joined = "\n\n".join(observations)
        # context_chars=None sends everything the agent retrieved. The cap exists only to
        # bound judge input cost; a capped judge can mark an early-sourced claim ungrounded
        # simply because the supporting snippet is no longer in view, which biases grounding
        # down for agents that search more. Keep it constant across models being compared.
        if context_chars and len(joined) > context_chars:
            # Drop whole observations, oldest first, rather than slicing mid-token.
            kept: list[str] = []
            total = 0
            for obs in reversed(observations):
                if total + len(obs) > context_chars:
                    break
                kept.append(obs)
                total += len(obs)
            joined = "[earlier results omitted]\n" + "\n\n".join(reversed(kept))
        context_block = SEARCH_CONTEXT_BLOCK.format(context=joined)
    prompt = RUBRIC_JUDGE_PROMPT.format(
        question=question,
        report=report,
        criteria=criteria_text,
        context_block=context_block,
    )
    # Newer reasoning models reject `temperature` outright rather than ignoring it,
    # so it is only sent to models that accept it. Judge determinism is unaffected:
    # those models do not expose the knob at all.
    # Reasoning models spend tokens thinking before emitting any text, so a 300-token
    # cap returns an empty completion. They also reject `temperature` rather than
    # ignoring it.
    reasoning_judge = "opus-5" in judge_model or "sonnet-5" in judge_model
    inference_config: dict = {"maxTokens": 4000 if reasoning_judge else 300}
    if not reasoning_judge:
        inference_config["temperature"] = 0.0
    resp = bedrock_client.converse(
        modelId=judge_model,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig=inference_config,
    )
    text = "".join(
        block.get("text", "") for block in resp["output"]["message"]["content"]
    )
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        # Returning 0.0 here would be a lie: it is indistinguishable from a
        # genuinely worthless report. In eval that silently depresses a model's
        # score; in RL it actively teaches the policy that a good report is bad.
        # Raise so the caller can treat it as missing data.
        raise RubricJudgeError(
            f"Judge returned no JSON object (first 200 chars: {text[:200]!r})"
        )
    parsed = json.loads(match.group(0))
    scores = [int(parsed.get(str(i), 0)) for i in range(len(RUBRICS))]
    normalized = sum(s / 10.0 for s in scores) / len(RUBRICS)
    per_criterion = {RUBRICS[i]["category"]: scores[i] for i in range(len(RUBRICS))}
    return normalized, per_criterion


def combine(rubric: float, citation: float, fmt: float) -> float:
    """Weighted total in [0, 1]."""
    return WEIGHT_RUBRIC * rubric + WEIGHT_CITATION * citation + WEIGHT_FORMAT * fmt
