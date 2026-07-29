// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0
import * as path from "path"
import * as cdk from "aws-cdk-lib"
import * as s3 from "aws-cdk-lib/aws-s3"
import * as ecr from "aws-cdk-lib/aws-ecr"
import * as iam from "aws-cdk-lib/aws-iam"
import * as ec2 from "aws-cdk-lib/aws-ec2"
import * as ssm from "aws-cdk-lib/aws-ssm"
import * as ecr_assets from "aws-cdk-lib/aws-ecr-assets"
import * as agentcore from "@aws-cdk/aws-bedrock-agentcore-alpha"
import { Construct } from "constructs"

/**
 * RL Training Stack — infrastructure for fine-tuning with AgentCore RL Toolkit.
 *
 * Creates:
 * - S3 bucket for rollout data exchange (prompts ↔ rewards ↔ checkpoints)
 * - RL-adapted agent deployed to AgentCore Runtime
 * - IAM role for SageMaker training jobs
 *
 * Usage: npm run deploy:rl (or cdk deploy deep-research-rl)
 */
export class RLTrainingStack extends cdk.Stack {
  public readonly rolloutBucket: s3.Bucket
  public readonly trainingRole: iam.Role

  constructor(scope: Construct, id: string, props?: cdk.StackProps) {
    super(scope, id, {
      ...props,
      description: "AgentCore Deep Research - RL Training Infrastructure",
    })

    // S3 bucket for rollout data exchange between training engine and agents
    this.rolloutBucket = new s3.Bucket(this, "RLRolloutBucket", {
      removalPolicy: cdk.RemovalPolicy.DESTROY,
      autoDeleteObjects: true,
      lifecycleRules: [
        {
          // Auto-cleanup old rollout data after 30 days
          expiration: cdk.Duration.days(30),
          prefix: "rollouts/",
        },
        {
          // Keep checkpoints longer (90 days)
          expiration: cdk.Duration.days(90),
          prefix: "checkpoints/",
        },
      ],
    })

    // IAM role for the RL agent on AgentCore Runtime
    const agentRole = new iam.Role(this, "RLAgentRole", {
      assumedBy: new iam.ServicePrincipal("bedrock-agentcore.amazonaws.com"),
      description: "Execution role for RL-adapted deep research agent",
    })

    // Agent needs: Bedrock (reward judge), S3 (rollout results), SSM (gateway URL), Secrets Manager (gateway auth)
    agentRole.addToPolicy(
      new iam.PolicyStatement({
        effect: iam.Effect.ALLOW,
        actions: ["bedrock:InvokeModel", "bedrock:Converse"],
        resources: ["*"],
      })
    )
    this.rolloutBucket.grantReadWrite(agentRole)
    agentRole.addToPolicy(
      new iam.PolicyStatement({
        effect: iam.Effect.ALLOW,
        actions: ["ssm:GetParameter", "ssm:GetParameters"],
        resources: [`arn:aws:ssm:${this.region}:${this.account}:parameter/*`],
      })
    )
    agentRole.addToPolicy(
      new iam.PolicyStatement({
        effect: iam.Effect.ALLOW,
        actions: ["secretsmanager:GetSecretValue"],
        resources: [`arn:aws:secretsmanager:${this.region}:${this.account}:secret:*`],
      })
    )
    agentRole.addToPolicy(
      new iam.PolicyStatement({
        effect: iam.Effect.ALLOW,
        actions: ["cognito-idp:InitiateAuth"],
        resources: ["*"],
      })
    )
    agentRole.addToPolicy(
      new iam.PolicyStatement({
        effect: iam.Effect.ALLOW,
        actions: ["sagemaker:InvokeEndpoint"],
        resources: ["*"],
      })
    )

    // VPC for RL training — shared between AgentCore agent and SageMaker training
    const vpc = new ec2.Vpc(this, "RLVpc", {
      maxAzs: 2,
      natGateways: 1, // Agent needs outbound internet for tools (Tavily, Nova)
      subnetConfiguration: [
        { name: "Public", subnetType: ec2.SubnetType.PUBLIC, cidrMask: 24 },
        { name: "Private", subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS, cidrMask: 24 },
      ],
    })

    const securityGroup = new ec2.SecurityGroup(this, "RLSecurityGroup", {
      vpc,
      description: "Security group for RL training (agent + SageMaker)",
      allowAllOutbound: true,
    })
    // Allow all traffic within the security group (agent <-> training gateway)
    securityGroup.addIngressRule(securityGroup, ec2.Port.allTraffic(), "Allow intra-SG traffic")

    // Deploy RL agent to AgentCore Runtime using Docker
    const rlAgentArtifact = agentcore.AgentRuntimeArtifact.fromAsset(
      path.resolve(__dirname, "..", ".."),
      {
        platform: ecr_assets.Platform.LINUX_ARM64,
        file: "patterns/strands-deep-research/Dockerfile.rl",
      }
    )

    const rlRuntime = new agentcore.Runtime(this, "RLRuntime", {
      runtimeName: `${id.replace(/-/g, "_")}_rl_agent`,
      agentRuntimeArtifact: rlAgentArtifact,
      executionRole: agentRole,
      networkConfiguration: agentcore.RuntimeNetworkConfiguration.usingVpc(this, {
        vpc,
        vpcSubnets: { subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS },
        securityGroups: [securityGroup],
      }),
      protocolConfiguration: agentcore.ProtocolType.HTTP,
      environmentVariables: {
        AWS_DEFAULT_REGION: this.region,
        STACK_NAME: "deep-research", // For SSM parameter lookup (gateway URL)
      },
      description: "RL-adapted deep research agent for GRPO training",
      lifecycleConfiguration: {
        idleRuntimeSessionTimeout: cdk.Duration.minutes(15),
      },
    })

    // Fine-tuned inference agent — same as production but uses SageMaker endpoint
    // JWT auth using same Cognito as production (read from SSM parameters written by main stack)
    const cognitoUserPoolId = ssm.StringParameter.valueForStringParameter(this, "/deep-research/cognito-user-pool-id")
    const cognitoClientId = ssm.StringParameter.valueForStringParameter(this, "/deep-research/cognito-user-pool-client-id")

    const finetunedAuthConfig = agentcore.RuntimeAuthorizerConfiguration.usingJWT(
      `https://cognito-idp.${this.region}.amazonaws.com/${cognitoUserPoolId}/.well-known/openid-configuration`,
      [cognitoClientId]
    )

    const finetunedAgentArtifact = agentcore.AgentRuntimeArtifact.fromAsset(
      path.resolve(__dirname, "..", ".."),
      {
        platform: ecr_assets.Platform.LINUX_ARM64,
        file: "patterns/strands-deep-research/Dockerfile",
      }
    )

    const finetunedRuntime = new agentcore.Runtime(this, "FinetunedRuntime", {
      runtimeName: `${id.replace(/-/g, "_")}_finetuned_agent`,
      agentRuntimeArtifact: finetunedAgentArtifact,
      executionRole: agentRole,
      networkConfiguration: agentcore.RuntimeNetworkConfiguration.usingPublicNetwork(),
      protocolConfiguration: agentcore.ProtocolType.HTTP,
      authorizerConfiguration: finetunedAuthConfig,
      requestHeaderConfiguration: {
        allowlistedHeaders: ["Authorization"],
      },
      environmentVariables: {
        AWS_DEFAULT_REGION: this.region,
        STACK_NAME: "deep-research",
        USE_SAGEMAKER_MODEL: "true",
        SAGEMAKER_ENDPOINT_NAME: "dr-finetuned", // Updated by deploy_finetuned_agent.py
      },
      description: "Fine-tuned deep research agent (SageMaker endpoint)",
      lifecycleConfiguration: {
        idleRuntimeSessionTimeout: cdk.Duration.minutes(15),
      },
    })

    // IAM role for SageMaker training jobs + Bedrock Custom Model Import
    this.trainingRole = new iam.Role(this, "RLTrainingRole", {
      assumedBy: new iam.CompositePrincipal(
        new iam.ServicePrincipal("sagemaker.amazonaws.com"),
        new iam.ServicePrincipal("bedrock.amazonaws.com"),
      ),
      description: "Role for RL training jobs (GRPO with AgentCore RL Toolkit) and Bedrock model import",
      managedPolicies: [
        iam.ManagedPolicy.fromAwsManagedPolicyName("AmazonSageMakerFullAccess"),
      ],
    })

    this.rolloutBucket.grantReadWrite(this.trainingRole)
    this.trainingRole.addToPolicy(
      new iam.PolicyStatement({
        effect: iam.Effect.ALLOW,
        actions: [
          "bedrock-agentcore:InvokeAgentRuntime",
          "bedrock-agentcore:InvokeAgentRuntimeStream",
          "bedrock-agentcore:StopRuntimeSession",
        ],
        resources: [
          rlRuntime.agentRuntimeArn,
          `${rlRuntime.agentRuntimeArn}/*`,
        ],
      })
    )
    this.trainingRole.addToPolicy(
      new iam.PolicyStatement({
        effect: iam.Effect.ALLOW,
        actions: ["bedrock:InvokeModel", "bedrock:Converse"],
        resources: ["*"],
      })
    )

    // --- Outputs ---
    new cdk.CfnOutput(this, "RLBucketName", {
      value: this.rolloutBucket.bucketName,
      description: "S3 bucket for RL rollout data exchange",
    })

    new cdk.CfnOutput(this, "RLAgentRuntimeArn", {
      value: rlRuntime.agentRuntimeArn,
      description: "AgentCore Runtime ARN for the RL agent",
    })

    new cdk.CfnOutput(this, "FinetunedAgentRuntimeArn", {
      value: finetunedRuntime.agentRuntimeArn,
      description: "AgentCore Runtime ARN for the fine-tuned inference agent",
    })

    new cdk.CfnOutput(this, "RLTrainingRoleArn", {
      value: this.trainingRole.roleArn,
      description: "IAM role ARN for SageMaker training jobs",
    })

    new cdk.CfnOutput(this, "RLVpcSubnets", {
      value: vpc.selectSubnets({ subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS }).subnetIds.join(","),
      description: "Private subnet IDs for SageMaker training",
    })

    new cdk.CfnOutput(this, "RLSecurityGroupId", {
      value: securityGroup.securityGroupId,
      description: "Security group ID for SageMaker training",
    })
  }
}
