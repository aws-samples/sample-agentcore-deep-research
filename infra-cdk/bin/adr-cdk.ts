#!/usr/bin/env node
// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0
import * as cdk from "aws-cdk-lib"
import { ADRMainStack } from "../lib/adr-main-stack"
import { RLTrainingStack } from "../lib/rl-training-stack"
import { ConfigManager } from "../lib/utils/config-manager"

// Load configuration using ConfigManager
const configManager = new ConfigManager("config.yaml")

// Initial props consist of configuration parameters
const props = configManager.getProps()

const app = new cdk.App()

// Deploy the ADR stack
const amplifyStack = new ADRMainStack(app, props.stack_name_base, {
  config: props,
  env: {
    account: process.env.CDK_DEFAULT_ACCOUNT,
    region: props.region || process.env.CDK_DEFAULT_REGION,
  },
})

// Deploy RL training infrastructure (via `npm run deploy:rl`)
const rlStack = new RLTrainingStack(app, `${props.stack_name_base}-rl`, {
  finetunedEndpointName: props.training?.finetuned_endpoint_name,
  mainStackName: props.stack_name_base,
  stagingBucketName: props.training?.staging_bucket_name,
  toolsConfig: JSON.stringify(props.tools ?? {}),
  sagemakerMaxTokens: props.training?.sagemaker_max_tokens,
  sagemakerEnableThinking: props.training?.sagemaker_enable_thinking,
  sagemakerTemperature: props.training?.sagemaker_temperature,
  env: {
    account: process.env.CDK_DEFAULT_ACCOUNT,
    region: props.region || process.env.CDK_DEFAULT_REGION,
  },
})

// The RL stack reads the Cognito user pool and client ids the main stack writes
// to SSM, so it cannot stand alone: deployed into a fresh account or region on
// its own, those lookups resolve to nothing and the RL agent cannot authenticate
// to Gateway -- it runs, gets no tools, and produces empty reports. Declaring the
// dependency makes `cdk deploy deep-research-rl` bring the main stack with it, in
// the right order, on every deploy path.
rlStack.addDependency(amplifyStack)

app.synth()
