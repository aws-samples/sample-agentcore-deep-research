#!/usr/bin/env node
import * as cdk from "aws-cdk-lib"
import { ADRMainStack } from "../lib/adr-main-stack"
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
    region: process.env.CDK_DEFAULT_REGION,
  },
})

app.synth()
