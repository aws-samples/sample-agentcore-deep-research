// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0
import * as fs from "fs"
import * as path from "path"
import * as yaml from "yaml"

const MAX_STACK_NAME_BASE_LENGTH = 35

export type DeploymentType = "docker" | "zip"

export interface ToolConfig {
  enabled: boolean
  default_on: boolean
  knowledge_base_id?: string | null
}

export interface AppConfig {
  stack_name_base: string
  admin_user_email?: string | null
  auto_deploy_frontend?: boolean
  backend: {
    pattern: string
    deployment_type: DeploymentType
    model_id?: string
  }
  tools?: Record<string, ToolConfig>
  api_keys?: {
    tavily?: string | null
    alphavantage?: string | null
  }
}

export class ConfigManager {
  private config: AppConfig

  constructor(configFile: string) {
    this.config = this._loadConfig(configFile)
  }

  private _loadConfig(configFile: string): AppConfig {
    const configDir = path.join(__dirname, "..", "..")
    let configPath = path.join(configDir, configFile)

    if (!fs.existsSync(configPath)) {
      const examplePath = path.join(configDir, ".config_example.yaml")
      if (fs.existsSync(examplePath)) {
        console.log(
          `⚠ ${configFile} not found, using .config_example.yaml defaults. ` +
            `To customize, run: cp infra-cdk/.config_example.yaml infra-cdk/config.yaml`
        )
        configPath = examplePath
      } else {
        throw new Error(
          `Configuration file not found. Please create config.yaml by copying the example:\n` +
            `  cp infra-cdk/.config_example.yaml infra-cdk/config.yaml`
        )
      }
    }

    try {
      const fileContent = fs.readFileSync(configPath, "utf8")
      const parsedConfig = yaml.parse(fileContent) as AppConfig

      const deploymentType = parsedConfig.backend?.deployment_type || "docker"
      if (deploymentType !== "docker" && deploymentType !== "zip") {
        throw new Error(`Invalid deployment_type '${deploymentType}'. Must be 'docker' or 'zip'.`)
      }

      const stackNameBase = parsedConfig.stack_name_base
      if (!stackNameBase) {
        throw new Error("stack_name_base is required in config.yaml")
      }
      if (stackNameBase.length > MAX_STACK_NAME_BASE_LENGTH) {
        throw new Error(
          `stack_name_base '${stackNameBase}' is too long (${stackNameBase.length} chars). ` +
            `Maximum length is ${MAX_STACK_NAME_BASE_LENGTH} characters due to AWS AgentCore runtime naming constraints.`
        )
      }

      // Parse tools config with defaults
      const defaultTools: Record<string, ToolConfig> = {
        tavily: { enabled: true, default_on: true },
        nova: { enabled: true, default_on: true },
        arxiv: { enabled: true, default_on: false },
        openfda: { enabled: true, default_on: false },
        s3: { enabled: true, default_on: false },
        alphavantage: { enabled: true, default_on: true },
        bedrock_kb: { enabled: false, default_on: false },
      }
      const tools: Record<string, ToolConfig> = {}
      for (const [key, defaults] of Object.entries(defaultTools)) {
        const raw = (parsedConfig as any).tools?.[key]
        tools[key] = {
          enabled: raw?.enabled ?? defaults.enabled,
          default_on: raw?.default_on ?? defaults.default_on,
          ...(raw?.knowledge_base_id ? { knowledge_base_id: raw.knowledge_base_id } : {}),
        }
      }

      return {
        stack_name_base: stackNameBase,
        admin_user_email: parsedConfig.admin_user_email || null,
        auto_deploy_frontend: parsedConfig.auto_deploy_frontend ?? false,
        backend: {
          pattern: parsedConfig.backend?.pattern || "strands-deep-research",
          deployment_type: deploymentType,
          model_id: parsedConfig.backend?.model_id,
        },
        tools,
        api_keys: {
          tavily: parsedConfig.api_keys?.tavily || null,
          alphavantage: parsedConfig.api_keys?.alphavantage || null,
        },
      }
    } catch (error) {
      throw new Error(`Failed to parse configuration file ${configPath}: ${error}`)
    }
  }

  public getProps(): AppConfig {
    return this.config
  }

  public get(key: string, defaultValue?: any): any {
    const keys = key.split(".")
    let value: any = this.config

    for (const k of keys) {
      if (typeof value === "object" && value !== null && k in value) {
        value = value[k]
      } else {
        return defaultValue
      }
    }

    return value
  }
}
