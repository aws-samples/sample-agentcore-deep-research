# AgentCore Deep Research Accelerator

AgentCore Deep Research template is a sample agentic AI solution for deep research built on Amazon Bedrock AgentCore. It conducts thorough research across multiple data sources and generates comprehensive reports with proper citations. The application features a modern React frontend with a split-pane interface showing real-time report generation alongside the chat. This sample is built using the [Fullstack Solution Template for AgentCore](https://github.com/awslabs/fullstack-solution-template-for-agentcore).

![Workflow](docs/figures/workflow.png)

**Key Features**

- **Multi-Source Retrieval**: Search across the enterprise data (S3 or Knowledge Base) and public sources (Internet, academic papers, OpenFDA, AlphaVantage and more)
- **Iterative Research Workflow**: AI agent scaffolds report, researches across sources, writes all sections with citations, and verifies completeness
- **Real-Time Report Display**: Split-pane UI shows the research report being built in real-time and allows follow-up questions
- **Fact-Checking and Citations**: Every factual claim includes inline source citations with the references section

## 🚀 Deployment

**Prerequisites**: Node.js 20+, AWS CLI, [AWS CDK CLI](https://docs.aws.amazon.com/cdk/v2/guide/getting-started.html), Python 3.10+, [uv](https://docs.astral.sh/uv/), and Docker. See the [deployment guide](docs/DEPLOYMENT.md) for details.

Deploying AgentCore Deep Research stack requires a few commands:

```bash
cd infra-cdk
cp .config_example.yaml config.yaml  # Create your config (edit as needed)
npm install
cdk bootstrap  # Once per account/region
npm run deploy
```

Available deploy commands (run from `infra-cdk/`):

```bash
npm run deploy            # Backend + frontend (if auto_deploy_frontend: true in config.yaml)
npm run deploy:frontend   # Frontend only
cdk deploy                # Backend only
```

See the [deployment guide](docs/DEPLOYMENT.md) for detailed instructions.

## ▶️ Usage

1. Open the application URL (from CDK outputs)
2. Log in with Cognito credentials
3. Toggle data sources (AlphaVantage, Tavily, Nova, ArXiv, etc.) as needed
4. Enter a research question
5. Watch as the agent:
   - Scaffolds report structure with key themes
   - Researches across enabled data sources
   - Writes all sections with citations
   - Verifies completeness and fills gaps
6. Ask follow-up questions and download the report


## ℹ️ Architecture

![Architecture Diagram](docs/figures/adr-architecture.png)

The architecture uses Amazon Cognito in four places:
1. User-based login to the frontend web application on CloudFront
2. Token-based authentication for the frontend to access AgentCore Runtime
3. Token-based authentication for the agents in AgentCore Runtime to access AgentCore Gateway
4. Token-based authentication when making API requests to API Gateway.

### Gateway Tools

The application includes multiple Lambda-based tools behind AgentCore Gateway with OAuth authentication:

1. **Tavily Web Search** - Search the web for current information with relevance scoring
2. **Nova Web Grounding** - AWS-powered web search via Amazon Nova with citations
3. **ArXiv Search** - Search academic papers on arXiv by topic, author, or keywords
4. **OpenFDA Drug Search** - Search FDA drug label database for pharmaceutical information
5. **AlphaVantage Research** - Commodity prices (gold, oil, silver, etc.), US economic indicators (CPI, inflation, Fed rate, GDP, unemployment), and market news with sentiment analysis
6. **S3 File Reader** - Read text files and PDFs from S3 (PDFs auto-converted to markdown via pymupdf4llm)
7. **Knowledge Base Search** - Query Amazon Bedrock Knowledge Bases (requires configuration)

The modular architecture makes it easy to integrate additional data sources for developers.

> **Note:** Several tools connect to external (non-AWS) APIs: Tavily, ArXiv, OpenFDA, and AlphaVantage. Of these, Tavily and AlphaVantage require API keys obtained through external registration. All external APIs, whether free or paid, are subject to the terms and conditions of their respective providers. We are not responsible for the availability, accuracy, or usage policies of third-party APIs. Please review each provider's terms before use. See the [deployment guide](docs/DEPLOYMENT.md) for stack setup instructions and which tools require API keys.

### Tech Stack

- **Frontend**: React with TypeScript, Vite, Tailwind CSS, and shadcn components
- **Agent**: Strands Agents SDK with BedrockModel
- **Authentication**: AWS Cognito User Pool with OAuth support
- **Infrastructure**: CDK deployment with Amplify Hosting for frontend and AgentCore backend


## 💻 Local Development

Local development requires a deployed stack because the agent depends on AWS services that cannot run locally:
- **AgentCore Memory** - stores conversation history
- **AgentCore Gateway** - provides tool access via MCP
- **SSM Parameters** - stores configuration (Gateway URL, client IDs)
- **Secrets Manager** - stores Gateway authentication credentials

You must first deploy the stack with `npm run deploy` (from `infra-cdk/`), then you can run the frontend and agent locally using Docker Compose while connecting to these deployed AWS resources:

```bash
# Set required environment variables (see below for how to find these)
export MEMORY_ID=your-memory-id
export STACK_NAME=your-stack-name
export AWS_DEFAULT_REGION=us-east-1

# Start the full stack locally
cd docker
docker-compose up --build
```

**Finding the environment variable values:**
- `STACK_NAME`: Use the `stack_name_base` value from your `infra-cdk/config.yaml`
- `MEMORY_ID`: Extract from the `MemoryArn` CloudFormation output (the ID is the last segment after `/`)
  ```bash
  aws cloudformation describe-stacks --stack-name <your-stack-name> \
    --query 'Stacks[0].Outputs[?OutputKey==`MemoryArn`].OutputValue' --output text
  # Returns: arn:aws:bedrock-agentcore:region:account:memory/MEMORY_ID
  ```
- `AWS_DEFAULT_REGION`: The region where you deployed the stack (e.g., `us-east-1`)

See the [local development guide](docs/LOCAL_DEVELOPMENT.md) for detailed setup instructions.


## 📂 Project Structure

```
agentcore-deep-research/
├── frontend/                 # React frontend application
│   ├── src/
│   │   ├── components/     # React components (shadcn/ui)
│   │   ├── hooks/          # Custom React hooks
│   │   ├── lib/            # Utility libraries
│   │   ├── services/       # API service layers
│   │   └── types/          # TypeScript type definitions
│   ├── public/             # Static assets and aws-exports.json
│   └── package.json
├── infra-cdk/               # CDK infrastructure code
│   ├── lib/                # CDK stack definitions
│   ├── bin/                # CDK app entry point
│   ├── lambdas/            # Lambda function code
│   ├── .config_example.yaml # Example deployment configuration (copy to config.yaml)
│   └── config.yaml         # Your deployment configuration (gitignored)
├── patterns/               # Agent pattern implementations
│   └── strands-deep-research/ # Deep Research agent
│       ├── deep_research_agent.py  # Main agent with Gateway tools
│       ├── report_upload_hook.py   # S3 upload for real-time display
│       ├── system_prompt.txt       # 4-round research workflow
│       ├── requirements.txt        # Agent dependencies
│       └── Dockerfile              # Container configuration
├── gateway/                # Gateway utilities and tools
│   └── tools/              # Gateway tool implementations
├── scripts/                # Deployment and test scripts
│   ├── deploy-frontend.py  # Cross-platform frontend deployment
│   └── test-*.py           # Various test utilities
├── docs/                   # Documentation source files
│   └── architecture-diagram/ # Architecture diagrams
├── tests/                  # Test suite
├── docker-compose.yml      # Local development stack
└── README.md
```

## 🔒 Security

Note: this asset represents a proof-of-value for the services included and is not intended as a production-ready solution. You must determine how the AWS Shared Responsibility applies to your specific use case and implement the needed controls to achieve your desired security outcomes. AWS offers a broad set of security tools and configurations to enable our customers.

Ultimately it is your responsibility as the developer to ensure all aspects of the application are secure. We provide security best practices in repository documentation and provide a secure baseline but Amazon holds no responsibility for the security of applications built from this tool.

## 👤 Team

| ![image](docs/figures/team/nikita.jpeg) | ![image](docs/figures/team/aiham.jpeg) | ![image](docs/figures/team/jack.jpeg) | ![image](docs/figures/team/elizaveta.jpeg) |
|---|---|---|---|
| [Nikita Kozodoi](https://www.linkedin.com/in/kozodoi/) | [Aiham Taleb](https://www.linkedin.com/in/aihamtaleb/) | [Jack Butler](https://www.linkedin.com/in/jackbutler-a/) | [Elizaveta Zinovyeva](https://www.linkedin.com/in/zinov-liza/) |
