# Correlate 2.0 Deep Search Implementation Plan

This document outlines the implementation plan for building a Deep Search application ("Correlate 2.0") on top of the FAST template.

## Overview

Correlate 2.0 is a deep research agent that can search across multiple data sources, synthesize information, and generate comprehensive reports. It leverages the FAST architecture with AgentCore Runtime, Memory, and Gateway.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              React Frontend                                  │
│                    (Chat UI + Report Display + Source Views)                │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          AgentCore Runtime                                   │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                    Deep Research Agent (Strands)                       │  │
│  │  • Multi-step research planning                                        │  │
│  │  • Source orchestration                                                │  │
│  │  • Report generation & revision                                        │  │
│  │  • Code Interpreter (data analysis)                                    │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
           │                              │
           │ Direct Tools                 │ Gateway Tools (Lambda)
           ▼                              ▼
┌─────────────────────┐    ┌─────────────────────────────────────────────────┐
│ Strands Built-in:   │    │              AgentCore Gateway                   │
│ • file_read         │    │  ┌─────────────────────────────────────────────┐│
│ • file_write        │    │  │  Data Retrieval Tools (per-user auth)       ││
│ • editor            │    │  │                                             ││
│ • Code Interpreter  │    │  │                                             ││
└─────────────────────┘    │  │  • tavily_web_search                        ││
                           │  │  • s3_document_reader (BDA)                 ││
                           │  │  • knowledge_base_search                    ││
                           │  │  • arxiv_search                             ││
                           │  │  • nova_web_search                          ││
                           │  └─────────────────────────────────────────────┘│
                           └─────────────────────────────────────────────────┘
                                              │
                                              ▼
                           ┌─────────────────────────────────────────────────┐
                           │             External Data Sources               │
                           │  • Tavily API                                   │
                           │  • S3 Documents (via Bedrock Data Automation)   │
                           │  • Bedrock Knowledge Base                       │
                           │  • ArXiv API                                    │
                           │  • Nova Web Search API                          │
                           └─────────────────────────────────────────────────┘
```

## Implementation Phases

### Phase 1: Deep Research Agent Core

**Goal**: Create the main Strands agent with research capabilities and report generation.

#### 1.1 Create New Agent Pattern

Create `patterns/strands-deep-research/` with the deep research agent:

- [ ] **`deep_research_agent.py`** - Main agent implementation
  - System prompt optimized for deep research tasks (in a separate .txt file)
  - Iterative research workflow (3 rounds)
  - Source citation tracking
  - Uses Strands built-in file tools for report management

- [ ] **`system_prompt.txt`** - Research agent system prompt
  - Instructions for iterative refinement workflow
  - Report structure guidelines
  - Citation format requirements

- [ ] **`requirements.txt`** - Dependencies
  - strands-agents
  - strands-tools (file_read, file_write, editor)
  - bedrock-agentcore-runtime

- [ ] **`Dockerfile`** - Container configuration

#### 1.2 Agent System Prompt & Iterative Workflow

Design a system prompt that implements a **3-round iterative refinement** workflow:

**Round 1 - Initial Draft:**
- Analyze the research question
- Generate initial report outline/structure
- Write draft to file using `file_write`

**Round 2 - Research & Enrich:**
- Run searches across data sources (Gateway tools)
- Read current report using `file_read`
- Update report with findings using `editor`
- Add citations and sources

**Round 3 - Refine & Finalize:**
- Run additional targeted searches to fill gaps
- Polish report structure and flow
- Verify all claims have citations
- Write final version

**File Tools Usage:**
```python
from strands_tools import file_read, file_write, editor

# Round 1: Create initial draft
file_write(path="/tmp/report.md", content="# Research Report\n...")

# Round 2-3: Read and update
current = file_read(path="/tmp/report.md")
editor(command="str_replace", path="/tmp/report.md",
       old_str="[placeholder]", new_str="actual findings...")
```

---

### Phase 2: Gateway Data Retrieval Tools

**Goal**: Implement Lambda-based data retrieval tools behind AgentCore Gateway for per-user access control.

#### 2.1 Tavily Web Search Tool

Create `gateway/tools/tavily_search/`:

- [ ] **`tavily_search_lambda.py`** - Lambda handler
  - Open-ended web search via Tavily API
  - Result ranking and filtering
  - Snippet extraction

- [ ] **`tool_spec.json`** - Tool schema
  ```json
  {
    "name": "tavily_web_search",
    "description": "Search the web for current information using Tavily",
    "inputSchema": {
      "type": "object",
      "properties": {
        "query": {"type": "string", "description": "Search query"},
        "max_results": {"type": "integer", "description": "Max results (default 10)"},
        "search_depth": {"type": "string", "enum": ["basic", "advanced"]}
      },
      "required": ["query"]
    }
  }
  ```

- [ ] **Environment**: Store Tavily API key in Secrets Manager

#### 2.2 S3 Document Reader Tool (BDA)

Create `gateway/tools/s3_bda_reader/`:

- [ ] **`s3_bda_reader_lambda.py`** - Lambda handler
  - Takes S3 URI as input
  - Processes document through Bedrock Data Automation
  - Returns parsed/extracted content
  - Support multiple formats (PDF, DOCX, TXT, images, etc.)

- [ ] **`tool_spec.json`** - Tool schema
  ```json
  {
    "name": "s3_document_reader",
    "description": "Read and parse a document from S3 using Bedrock Data Automation. Takes an S3 URI and returns the extracted content.",
    "inputSchema": {
      "type": "object",
      "properties": {
        "s3_uri": {"type": "string", "description": "S3 URI of the document (e.g., s3://bucket/path/document.pdf)"},
        "extract_tables": {"type": "boolean", "description": "Extract tables as structured data (default: true)"},
        "extract_images": {"type": "boolean", "description": "Extract and describe images (default: false)"}
      },
      "required": ["s3_uri"]
    }
  }
  ```

- [ ] **IAM**: Lambda role needs S3 read + Bedrock BDA permissions

#### 2.3 Bedrock Knowledge Base Search Tool

Create `gateway/tools/kb_search/`:

- [ ] **`kb_search_lambda.py`** - Lambda handler
  - Query Bedrock Knowledge Base by URI
  - Return relevant chunks with metadata
  - Support filtering and ranking

- [ ] **`tool_spec.json`** - Tool schema
  ```json
  {
    "name": "knowledge_base_search",
    "description": "Search a Bedrock Knowledge Base",
    "inputSchema": {
      "type": "object",
      "properties": {
        "query": {"type": "string", "description": "Search query"},
        "knowledge_base_id": {"type": "string", "description": "KB ID"},
        "max_results": {"type": "integer", "description": "Max results"},
        "filters": {"type": "object", "description": "Metadata filters"}
      },
      "required": ["query", "knowledge_base_id"]
    }
  }
  ```

#### 2.4 ArXiv Search Tool

Create `gateway/tools/arxiv_search/`:

- [ ] **`arxiv_search_lambda.py`** - Lambda handler
  - Search ArXiv papers via API
  - Return paper metadata, abstracts
  - Support category filtering

- [ ] **`tool_spec.json`** - Tool schema
  ```json
  {
    "name": "arxiv_search",
    "description": "Search academic papers on ArXiv",
    "inputSchema": {
      "type": "object",
      "properties": {
        "query": {"type": "string", "description": "Search query"},
        "categories": {"type": "array", "items": {"type": "string"}},
        "max_results": {"type": "integer", "description": "Max results"},
        "sort_by": {"type": "string", "enum": ["relevance", "date"]}
      },
      "required": ["query"]
    }
  }
  ```

#### 2.5 Nova Web Search Tool

Create `gateway/tools/nova_search/`:

- [ ] **`nova_search_lambda.py`** - Lambda handler
  - Web search using Amazon Nova capabilities
  - Grounded search with citations

- [ ] **`tool_spec.json`** - Tool schema
  ```json
  {
    "name": "nova_web_search",
    "description": "Search the web using Amazon Nova",
    "inputSchema": {
      "type": "object",
      "properties": {
        "query": {"type": "string", "description": "Search query"},
        "max_results": {"type": "integer"}
      },
      "required": ["query"]
    }
  }
  ```

---

### Phase 3: CDK Infrastructure Updates

**Goal**: Update CDK to deploy new Lambda tools and configure Gateway targets.

#### 3.1 Lambda Definitions

In `infra-cdk/lib/backend-stack.ts`, add:

- [ ] Lambda functions for each data retrieval tool
- [ ] IAM roles with appropriate permissions per tool
- [ ] Environment variables and Secrets Manager references
- [ ] VPC configuration if needed for external APIs

#### 3.2 Gateway Target Configuration

- [ ] Add new `CfnGatewayTarget` for each tool Lambda
- [ ] Define tool schemas from `tool_spec.json` files
- [ ] Configure appropriate timeout values

#### 3.3 Secrets Management

- [ ] Tavily API key in Secrets Manager
- [ ] Any other external API credentials

#### 3.4 Optional: Per-User Data Source Access

For fine-grained access control:
- [ ] Pass Cognito user claims to Lambda via Gateway
- [ ] Lambda validates user permissions for specific data sources
- [ ] Example: User A can access KB-1 but not KB-2

---

### Phase 4: Frontend Enhancements

**Goal**: Enhance the React frontend for deep research use cases.

#### 4.1 Research-Specific UI Components

- [ ] **Research Query Input** - Multi-line input for complex queries
- [ ] **Source Selection Panel** - Toggle which data sources to use
- [ ] **Progress Indicator** - Show research phases (searching, analyzing, writing)
- [ ] **Report Display** - Structured view with sections, citations
- [ ] **Source Viewer** - View original sources inline

#### 4.2 Report Features

- [ ] Export to Markdown/PDF
- [ ] Section-by-section editing
- [ ] Citation management
- [ ] Revision history

---

### Phase 5: Testing & Validation

#### 5.1 Unit Tests

- [ ] Test each Lambda tool independently
- [ ] Test agent logic with mocked tools
- [ ] Test report generation

#### 5.2 Integration Tests

- [ ] End-to-end research flow
- [ ] Multi-source query handling
- [ ] Gateway authentication flow

#### 5.3 Test Scripts

Update `test-scripts/` with:
- [ ] `test-tavily-tool.py`
- [ ] `test-arxiv-tool.py`
- [ ] `test-deep-research-agent.py`

---

## File Structure (New/Modified)

```
fullstack-agentcore-solution-template/
├── patterns/
│   └── strands-deep-research/           # NEW: Deep research agent
│       ├── deep_research_agent.py
│       ├── system_prompt.txt
│       ├── requirements.txt
│       └── Dockerfile
├── gateway/
│   └── tools/
│       ├── tavily_search/               # NEW
│       │   ├── tavily_search_lambda.py
│       │   └── tool_spec.json
│       ├── s3_bda_reader/               # NEW
│       │   ├── s3_bda_reader_lambda.py
│       │   └── tool_spec.json
│       ├── kb_search/                   # NEW
│       │   ├── kb_search_lambda.py
│       │   └── tool_spec.json
│       ├── arxiv_search/                # NEW
│       │   ├── arxiv_search_lambda.py
│       │   └── tool_spec.json
│       └── nova_search/                 # NEW
│           ├── nova_search_lambda.py
│           └── tool_spec.json
├── infra-cdk/
│   ├── lib/
│   │   └── backend-stack.ts             # MODIFIED: Add new Lambdas + Gateway targets
│   └── config.yaml                      # MODIFIED: Add pattern option
├── frontend/
│   └── src/
│       └── components/                  # MODIFIED: Research UI components
└── test-scripts/                        # MODIFIED: Add tool test scripts
```

---

## Configuration Changes

### `infra-cdk/config.yaml`

```yaml
stack_name_base: Correlate-stack

backend:
  pattern: strands-deep-research  # NEW pattern
  deployment_type: docker

# NEW: Data source configuration
data_sources:
  tavily:
    enabled: true
    secret_name: tavily-api-key
  s3_bda_reader:
    enabled: true
    allowed_buckets:  # buckets the tool can read from
      - my-research-bucket
  knowledge_base:
    enabled: true
    allowed_kb_ids:
      - KB-123456
  arxiv:
    enabled: true
  nova_search:
    enabled: true
```

---

## Implementation Order (Recommended)

1. **Phase 1.1**: Create basic deep research agent structure
2. **Phase 2.4**: ArXiv search tool (simplest - no API key needed)
3. **Phase 3.1-3.2**: CDK updates for ArXiv tool
4. **Test**: Verify ArXiv tool works end-to-end
5. **Phase 2.1**: Tavily web search tool
6. **Phase 2.3**: Knowledge Base search tool
7. **Phase 2.2**: S3 BDA document reader tool
8. **Phase 2.5**: Nova web search tool
9. **Phase 4**: Frontend enhancements
10. **Phase 5**: Comprehensive testing

---

## Dependencies & Prerequisites

### External Services
- [ ] Tavily API account and key
- [ ] Amazon Nova access (preview/GA)
- [ ] Existing Bedrock Knowledge Base (optional)
- [ ] S3 bucket with documents to parse (optional)

### AWS Services
- AgentCore Runtime (included in FAST)
- AgentCore Gateway (included in FAST)
- AgentCore Memory (included in FAST)
- Bedrock Data Automation (for S3 document parsing)
- Secrets Manager (for API keys)

---

## Risk Considerations

| Risk | Mitigation |
|------|------------|
| Nova Search API availability | Implement as optional, fallback to Tavily |
| BDA parsing latency for large documents | Add caching layer, limit document size |
| Rate limits on external APIs | Implement retry logic, request throttling |
| Per-user access control complexity | Start simple, add claims-based auth later |

---

## Next Steps

1. Review and approve this plan
2. Create feature branch: `feature/correlate-deep-search`
3. Start with Phase 1 (agent core)
4. Iterate through phases with testing at each step
