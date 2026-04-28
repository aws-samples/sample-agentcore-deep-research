// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0
import * as fs from "fs"
import * as path from "path"
import * as yaml from "yaml"

const MAX_STACK_NAME_BASE_LENGTH = 35

export type DeploymentType = "docker" | "zip"
export type ServiceTier = "default" | "priority" | "flex"

const VALID_SERVICE_TIERS: ServiceTier[] = ["default", "priority", "flex"]

export interface ToolConfig {
  enabled: boolean
  default_on: boolean
  required?: Record<string, string | null>
}

export interface AppConfig {
  stack_name_base: string
  region?: string | null
  admin_user_email?: string | null
  auto_deploy_frontend?: boolean
  backend: {
    pattern: string
    deployment_type: DeploymentType
    model_id?: string
    service_tier: ServiceTier
  }
  tools?: Record<string, ToolConfig>
}

export class ConfigManager {
  private config: AppConfig

  constructor(configFile: string) {
    this.config = this._loadConfig(configFile)
  }

  private _loadConfig(configFile: string): AppConfig {
    // nosemgrep: path-join-resolve-traversal -- build-time config loading from known directory
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

      const serviceTier = (parsedConfig.backend?.service_tier || "default") as ServiceTier
      if (!VALID_SERVICE_TIERS.includes(serviceTier)) {
        throw new Error(
          `Invalid service_tier '${serviceTier}'. Must be one of: ${VALID_SERVICE_TIERS.join(
            ", "
          )}.`
        )
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

      // Parse tools config directly from config.yaml
      const tools: Record<string, ToolConfig> = {}
      for (const [key, raw] of Object.entries((parsedConfig as any).tools ?? {})) {
        const t = raw as any
        const required = t?.required as Record<string, string | null> | undefined
        tools[key] = {
          enabled: t?.enabled ?? true,
          default_on: t?.default_on ?? false,
          ...(required ? { required } : {}),
        }
      }

      return {
        stack_name_base: stackNameBase,
        region: parsedConfig.region || null,
        admin_user_email: parsedConfig.admin_user_email || null,
        auto_deploy_frontend: parsedConfig.auto_deploy_frontend ?? false,
        backend: {
          pattern: parsedConfig.backend?.pattern || "strands-deep-research",
          deployment_type: deploymentType,
          model_id: parsedConfig.backend?.model_id,
          service_tier: serviceTier,
        },
        tools,
      }
    } catch (error) {
      throw new Error(`Failed to parse configuration file ${configPath}: ${error}`)
    }
  }

  public getProps(): AppConfig {
    return this.config
  }

  public get(key: string, defaultValue?: any): any {
    const UNSAFE_KEYS = new Set(["__proto__", "constructor", "prototype"])
    const keys = key.split(".")
    let value: any = this.config

    for (const k of keys) {
      if (UNSAFE_KEYS.has(k)) {
        return defaultValue
      }
      if (typeof value === "object" && value !== null && k in value) {
        value = value[k]
      } else {
        return defaultValue
      }
    }

    return value
  }
}
