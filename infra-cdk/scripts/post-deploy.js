#!/usr/bin/env node
// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0
//
// Post-CDK-deploy hook: conditionally deploys the frontend
// based on the auto_deploy_frontend flag in config.yaml.

const { execSync } = require("child_process")
const fs = require("fs")
const yaml = require("yaml")

const config = yaml.parse(fs.readFileSync("config.yaml", "utf8"))

if (config.auto_deploy_frontend) {
  console.log("\n>> auto_deploy_frontend is enabled — deploying frontend...\n")
  execSync("python3 ../scripts/deploy-frontend.py", { stdio: "inherit" })
} else {
  console.log(
    "\n>> Skipping frontend deploy (auto_deploy_frontend is not enabled in config.yaml)."
  )
  console.log("   To deploy frontend manually, run: npm run deploy:frontend\n")
}
