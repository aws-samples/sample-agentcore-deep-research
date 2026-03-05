"""
Copyright Amazon.com and Affiliates
This code is being licensed under the terms of the Amazon Software License available at https://aws.amazon.com/asl/
"""

import json
import os
import traceback
from pathlib import Path

# Bypass tool confirmation prompts for headless operation in AgentCore Runtime
os.environ["BYPASS_TOOL_CONSENT"] = "true"

from bedrock_agentcore.memory.integrations.strands.config import AgentCoreMemoryConfig
from bedrock_agentcore.memory.integrations.strands.session_manager import (
    AgentCoreMemorySessionManager,
)
from bedrock_agentcore.runtime import BedrockAgentCoreApp, RequestContext
from mcp.client.streamable_http import streamablehttp_client
from report_upload_hook import ReportS3UploadHook
from safe_bedrock_model import SafeBedrockModel
from strands import Agent
from strands.tools.mcp import MCPClient
from strands_tools import editor, file_read, file_write
from utils.auth import extract_user_id_from_context, get_gateway_access_token
from utils.inference import get_bedrock_config, get_inference_configs
from utils.ssm import get_ssm_parameter

# load inference configurations
INFERENCE_CONFIG, REASONING_CONFIG = get_inference_configs()
BEDROCK_CONFIG = get_bedrock_config()

app = BedrockAgentCoreApp()

SYSTEM_PROMPT_PATH = Path(__file__).parent / "system_prompt.txt"

# Available data sources with their tool names
DATA_SOURCES = {
    "tavily": {
        "name": "Tavily Web Search",
        "tool": "tavily_web_search",
        "description": "Search the web for current information",
    },
    "nova": {
        "name": "Nova Web Search",
        "tool": "nova_web_search",
        "description": "Web search via Amazon Nova with citations",
    },
    "arxiv": {
        "name": "ArXiv Search",
        "tool": "arxiv_search",
        "description": "Search academic papers on arXiv",
    },
    "openfda": {
        "name": "OpenFDA Drug Search",
        "tool": "openfda_drug_search",
        "description": "Search FDA drug label database for pharmaceutical information",
    },
    "s3": {
        "name": "S3 Text Reader",
        "tool": "s3_text_reader",
        "description": "Read text files from S3 (txt, md, csv, json, etc.)",
    },
    "kb": {
        "name": "Knowledge Base Search",
        "tool": "knowledge_base_search",
        "description": "Query Amazon Bedrock Knowledge Bases",
        "requires_params": True,
    },
}

# Default enabled sources (KB and S3 excluded by default)
DEFAULT_ENABLED_SOURCES = ["tavily", "nova", "arxiv", "openfda"]


def load_system_prompt(
    enabled_sources: list[str] | None = None,
    s3_file_uris: list[str] | None = None,
) -> str:
    """
    Load the system prompt and customize based on enabled data sources.

    Parameters
    ----------
    enabled_sources : list[str] | None
        List of enabled source keys (tavily, nova, arxiv). If None, all are enabled.
    s3_file_uris : list[str] | None
        List of S3 URIs provided by the user for the S3 data source.
    """
    with open(SYSTEM_PROMPT_PATH) as f:
        base_prompt = f.read()

    if enabled_sources is None:
        enabled_sources = DEFAULT_ENABLED_SOURCES

    # Build the data retrieval tools section
    tools_section = "### Data Retrieval (via Gateway)\n"
    for source_key in enabled_sources:
        if source_key in DATA_SOURCES:
            source = DATA_SOURCES[source_key]
            tools_section += f"- `{source['tool']}`: {source['description']}\n"

    # If no sources enabled, add a note
    if not any(s in DATA_SOURCES for s in enabled_sources):
        tools_section += "- No external data sources enabled\n"

    # Add S3 file list when S3 source is enabled
    if "s3" in enabled_sources and s3_file_uris:
        tools_section += "\n### Available S3 Files\n"
        tools_section += (
            "The user has provided these S3 files as data sources. "
            "Use `s3_text_reader` to read them during your research. "
            "Start with end_line=100 to explore, then read more "
            "lines if the content is relevant. "
            "Cite S3 files as [Source: s3://bucket/path/file.ext].\n"
        )
        for uri in s3_file_uris:
            tools_section += f"- `{uri}`\n"

    # Replace the data retrieval section in the prompt
    import re

    pattern = r"### Data Retrieval \(via Gateway\)\n(?:- .*\n)*"
    base_prompt = re.sub(pattern, tools_section, base_prompt)

    return base_prompt


def create_gateway_mcp_client(access_token: str) -> MCPClient:
    """
    Create MCP client for AgentCore Gateway with OAuth2 authentication.

    MCP (Model Context Protocol) is how agents communicate with tool providers.
    This creates a client that can talk to the AgentCore Gateway using the provided
    access token for authentication.
    """
    stack_name = os.environ.get("STACK_NAME")
    if not stack_name:
        raise ValueError("STACK_NAME environment variable is required")

    # Validate stack name format to prevent injection
    if not stack_name.replace("-", "").replace("_", "").isalnum():
        raise ValueError("Invalid STACK_NAME format")

    print(f"[AGENT] Creating Gateway MCP client for stack: {stack_name}")

    # Fetch Gateway URL from SSM
    gateway_url = get_ssm_parameter(f"/{stack_name}/gateway_url")
    print(f"[AGENT] Gateway URL from SSM: {gateway_url}")

    # Create MCP client with Bearer token authentication
    gateway_client = MCPClient(
        lambda: streamablehttp_client(
            url=gateway_url, headers={"Authorization": f"Bearer {access_token}"}
        ),
        prefix="gateway",
    )

    print("[AGENT] Gateway MCP client created successfully")
    return gateway_client


def create_deep_research_agent(
    user_id: str,
    session_id: str,
    enabled_sources: list[str] | None = None,
    s3_file_uris: list[str] | None = None,
) -> Agent:
    """
    Create a deep research agent with Gateway MCP tools and memory.

    Parameters
    ----------
    user_id : str
        User identifier from JWT token
    session_id : str
        Session identifier for conversation continuity
    enabled_sources : list[str] | None
        List of enabled data sources. Default: all enabled.
    s3_file_uris : list[str] | None
        List of S3 file URIs provided by the user.
    """
    system_prompt = load_system_prompt(enabled_sources, s3_file_uris)
    print(f"[AGENT] Enabled data sources: {enabled_sources or DEFAULT_ENABLED_SOURCES}")

    model_id = os.environ.get(
        "MODEL_ID", "global.anthropic.claude-sonnet-4-5-20250929-v1:0"
    )
    bedrock_model = SafeBedrockModel(
        model_id=model_id,
        temperature=INFERENCE_CONFIG["temperature"],
        max_tokens=INFERENCE_CONFIG["maxTokens"],
        streaming=True,
        boto_client_config=BEDROCK_CONFIG,
        cache_prompt="default",
        cache_tools="default",
    )

    memory_id = os.environ.get("MEMORY_ID")
    if not memory_id:
        raise ValueError("MEMORY_ID environment variable is required")

    # Configure AgentCore Memory
    agentcore_memory_config = AgentCoreMemoryConfig(
        memory_id=memory_id, session_id=session_id, actor_id=user_id
    )

    session_manager = AgentCoreMemorySessionManager(
        agentcore_memory_config=agentcore_memory_config,
        region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
    )

    # Collect all tools - file tools for local report management
    tools: list = [
        file_read,
        file_write,
        editor,
    ]

    try:
        print("[AGENT] Starting agent creation with Gateway tools...")

        # Get OAuth2 access token and create Gateway MCP client
        print("[AGENT] Step 1: Getting OAuth2 access token...")
        access_token = get_gateway_access_token()
        print(f"[AGENT] Got access token: {access_token[:20]}...")

        # Create Gateway MCP client with authentication
        print("[AGENT] Step 2: Creating Gateway MCP client...")
        gateway_client = create_gateway_mcp_client(access_token)
        print("[AGENT] Gateway MCP client created successfully")

        # Add Gateway client to tools
        tools.append(gateway_client)

        print("[AGENT] Step 3: Creating Deep Research Agent...")

        # Create hook for S3 report upload
        report_upload_hook = ReportS3UploadHook()

        agent = Agent(
            name="DeepResearchAgent",
            system_prompt=system_prompt,
            tools=tools,
            model=bedrock_model,
            session_manager=session_manager,
            hooks=[report_upload_hook],
            trace_attributes={
                "user.id": user_id,
                "session.id": session_id,
            },
        )
        print(
            "[AGENT] Agent created successfully with Gateway tools and S3 upload hook"
        )
        return agent

    except Exception as e:
        print(f"[AGENT ERROR] Error creating Gateway client: {e}")
        print(f"[AGENT ERROR] Exception type: {type(e).__name__}")
        print("[AGENT ERROR] Traceback:")
        traceback.print_exc()
        print("[AGENT] Gateway connection failed - raising exception")
        raise


@app.entrypoint
async def agent_stream(payload, context: RequestContext):
    """
    Main entrypoint for the deep research agent with streaming and Gateway integration.

    This is the function that AgentCore Runtime calls when the agent receives a request.
    It extracts the user's query from the payload, securely obtains the user ID from
    the validated JWT token in the request context, creates a deep research agent with
    file tools and Gateway tools, and streams the response back.

    Payload fields:
    - prompt: User's research query (required)
    - runtimeSessionId: Session ID for continuity (required)
    - enabledSources: List of enabled data sources (optional, default: all)
      Valid values: "tavily", "nova", "arxiv", "openfda", "s3"
    - s3FileUris: List of S3 file URIs (optional, used when "s3" is enabled)
    """
    user_query = payload.get("prompt")
    session_id = payload.get("runtimeSessionId")
    enabled_sources = payload.get("enabledSources")
    s3_file_uris = payload.get("s3FileUris")  # Optional: ["s3://bucket/file.txt"]

    if not all([user_query, session_id]):
        yield {
            "status": "error",
            "error": "Missing required fields: prompt or runtimeSessionId",
        }
        return

    try:
        # Extract user ID securely from the validated JWT token
        user_id = extract_user_id_from_context(context)

        print(f"[STREAM] Starting deep research for user: {user_id}")
        print(f"[STREAM] Query: {user_query}")
        print(f"[STREAM] Enabled sources: {enabled_sources or 'all (default)'}")
        if s3_file_uris:
            print(f"[STREAM] S3 files: {s3_file_uris}")

        agent = create_deep_research_agent(
            user_id, session_id, enabled_sources, s3_file_uris
        )

        # Use the agent's stream_async method for true token-level streaming
        # Pass session_id in invocation state for the S3 upload hook
        async for event in agent.stream_async(user_query, session_id=session_id):
            yield json.loads(json.dumps(dict(event), default=str))

    except Exception as e:
        print(f"[STREAM ERROR] Error in agent_stream: {e}")
        traceback.print_exc()
        yield {"status": "error", "error": str(e)}


if __name__ == "__main__":
    app.run()
