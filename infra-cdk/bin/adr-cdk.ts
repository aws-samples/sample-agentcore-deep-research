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

// Deploy RL training infrastructure (optional, via `npm run deploy:rl`)
new RLTrainingStack(app, `${props.stack_name_base}-rl`, {
  env: {
    account: process.env.CDK_DEFAULT_ACCOUNT,
    region: props.region || process.env.CDK_DEFAULT_REGION,
  },
})

app.synth()
