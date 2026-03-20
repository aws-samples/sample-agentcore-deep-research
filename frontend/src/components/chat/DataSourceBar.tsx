// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0
"use client";

import { type ReactNode, useState } from "react";
import { ChevronUp, Plus, Lock, RotateCcw } from "lucide-react";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";

export type ToolTag =
  | "Web Search"
  | "Science"
  | "Finance"
  | "Healthcare"
  | "Internal";

export interface ToolMeta {
  name: string;
  icon: string;
  tag: ToolTag;
  description: string;
}

export const TOOL_METADATA: Record<string, ToolMeta> = {
  alphavantage: {
    name: "AlphaVantage",
    icon: "📈",
    tag: "Finance",
    description: "Commodity prices, economic indicators, and market news",
  },
  arxiv: {
    name: "ArXiv Papers",
    icon: "📚",
    tag: "Science",
    description: "Search academic papers on arXiv by topic or keywords",
  },
  bedrock_kb: {
    name: "Knowledge Base",
    icon: "🧠",
    tag: "Internal",
    description: "Query Amazon Bedrock Knowledge Bases",
  },
  clinicaltrials: {
    name: "ClinicalTrials",
    icon: "🔬",
    tag: "Healthcare",
    description: "Search clinical studies by condition, intervention, or phase",
  },
  edgar: {
    name: "SEC EDGAR",
    icon: "🏛️",
    tag: "Finance",
    description: "Search SEC company filings (10-K, 10-Q, 8-K)",
  },
  fred: {
    name: "FRED Economic",
    icon: "🏦",
    tag: "Finance",
    description: "800K+ economic time series from the Federal Reserve",
  },
  nova: {
    name: "Nova Web Grounding",
    icon: "🔍",
    tag: "Web Search",
    description: "AWS-powered web search via Amazon Nova with citations",
  },
  openfda: {
    name: "OpenFDA Drugs",
    icon: "💊",
    tag: "Healthcare",
    description: "Search FDA drug label database for pharmaceutical info",
  },
  pubmed: {
    name: "PubMed",
    icon: "🏥",
    tag: "Science",
    description: "Search peer-reviewed biomedical and life sciences literature",
  },
  s3: {
    name: "S3 Files",
    icon: "📁",
    tag: "Internal",
    description: "Read text files and PDFs from S3 buckets",
  },
  tavily: {
    name: "Tavily Web",
    icon: "🌐",
    tag: "Web Search",
    description:
      "Search the web for current information with relevance scoring",
  },
};

const TAG_ORDER: ToolTag[] = [
  "Web Search",
  "Science",
  "Finance",
  "Healthcare",
  "Internal",
];

const TAG_COLORS: Record<
  ToolTag,
  { bg: string; text: string; border: string }
> = {
  "Web Search": {
    bg: "bg-sky-50 dark:bg-sky-950/30",
    text: "text-sky-700 dark:text-sky-300",
    border: "border-sky-200 dark:border-sky-800",
  },
  Science: {
    bg: "bg-violet-50 dark:bg-violet-950/30",
    text: "text-violet-700 dark:text-violet-300",
    border: "border-violet-200 dark:border-violet-800",
  },
  Finance: {
    bg: "bg-emerald-50 dark:bg-emerald-950/30",
    text: "text-emerald-700 dark:text-emerald-300",
    border: "border-emerald-200 dark:border-emerald-800",
  },
  Healthcare: {
    bg: "bg-rose-50 dark:bg-rose-950/30",
    text: "text-rose-700 dark:text-rose-300",
    border: "border-rose-200 dark:border-rose-800",
  },
  Internal: {
    bg: "bg-amber-50 dark:bg-amber-950/30",
    text: "text-amber-700 dark:text-amber-300",
    border: "border-amber-200 dark:border-amber-800",
  },
};

function Tip({ text, children }: { text: string; children: ReactNode }) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>{children}</TooltipTrigger>
      <TooltipContent side="top" sideOffset={4}>
        {text}
      </TooltipContent>
    </Tooltip>
  );
}

// Sort tools by tag order first, then alphabetically by name within each tag
function sortTools<T extends { tag: ToolTag; name: string }>(tools: T[]): T[] {
  return [...tools].sort((a, b) => {
    const tagDiff = TAG_ORDER.indexOf(a.tag) - TAG_ORDER.indexOf(b.tag);
    if (tagDiff !== 0) return tagDiff;
    return a.name.localeCompare(b.name);
  });
}

interface DataSourceBarProps {
  toolsConfig: Record<string, { enabled: boolean; default_on: boolean }>;
  enabledSources: Record<string, boolean>;
  onToggle: (id: string) => void;
  onReset: () => void;
  s3FileInput?: string;
  onS3FileInputChange?: (value: string) => void;
}

export function DataSourceBar({
  toolsConfig,
  enabledSources,
  onToggle,
  onReset,
  s3FileInput,
  onS3FileInputChange,
}: DataSourceBarProps) {
  const [expanded, setExpanded] = useState(false);

  const allTools = Object.entries(TOOL_METADATA).map(([id, meta]) => {
    const cfg = toolsConfig[id];
    const deployed = cfg?.enabled ?? false;
    const active = deployed && (enabledSources[id] ?? false);
    return { id, ...meta, deployed, active };
  });

  const activeCount = allTools.filter((t) => t.active).length;
  const activeTools = sortTools(allTools.filter((t) => t.active));
  const inactiveDeployed = allTools.filter((t) => t.deployed && !t.active);
  const notDeployed = allTools.filter((t) => !t.deployed);

  const groupedTools = TAG_ORDER.map((tag) => ({
    tag,
    tools: sortTools(allTools.filter((t) => t.tag === tag)),
  })).filter((g) => g.tools.length > 0);

  const hasNotDeployed = notDeployed.length > 0;

  return (
    <div className="w-full">
      {/* Collapsed bar */}
      <div className="flex items-center gap-2 flex-wrap">
        <span className="text-xs font-medium text-muted-foreground whitespace-nowrap">
          {activeCount} source{activeCount !== 1 ? "s" : ""} enabled
        </span>
        <div className="flex items-center gap-1.5 flex-wrap">
          {activeTools.map((tool) => (
            <Tip key={tool.id} text={tool.description}>
              <button
                onClick={() => onToggle(tool.id)}
                className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium border transition-all hover:opacity-80 ${
                  TAG_COLORS[tool.tag].bg
                } ${TAG_COLORS[tool.tag].text} ${TAG_COLORS[tool.tag].border}`}
              >
                <span>{tool.icon}</span>
                {tool.name}
              </button>
            </Tip>
          ))}
        </div>
        <button
          onClick={() => setExpanded(!expanded)}
          className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium text-muted-foreground bg-muted hover:bg-accent border border-border transition-all"
        >
          {expanded ? (
            <>
              <ChevronUp className="h-3 w-3" />
              Less
            </>
          ) : (
            <>
              <Plus className="h-3 w-3" />
              {inactiveDeployed.length + notDeployed.length} more
            </>
          )}
        </button>
        <Tip text="Reset to defaults">
          <button
            onClick={onReset}
            className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium text-muted-foreground hover:bg-accent border border-transparent hover:border-border transition-all"
          >
            <RotateCcw className="h-3 w-3" />
          </button>
        </Tip>
      </div>

      {/* Expanded panel */}
      {expanded && (
        <div className="mt-3 p-3 rounded-lg border border-border bg-card">
          <div className="space-y-3">
            {groupedTools.map(({ tag, tools }) => (
              <div key={tag}>
                <div
                  className={`text-[10px] font-semibold uppercase tracking-wider mb-1.5 ${TAG_COLORS[tag].text}`}
                >
                  {tag}
                </div>
                <div className="flex flex-wrap gap-1.5">
                  {tools.map((tool) => {
                    if (!tool.deployed) {
                      return (
                        <Tip
                          key={tool.id}
                          text={`${tool.description} — disabled in deployment configuration`}
                        >
                          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium border border-border bg-muted/50 text-muted-foreground/50 cursor-not-allowed select-none">
                            <Lock className="h-2.5 w-2.5" />
                            <span className="opacity-60">{tool.icon}</span>
                            <span className="line-through opacity-60">
                              {tool.name}
                            </span>
                          </span>
                        </Tip>
                      );
                    }
                    return (
                      <Tip key={tool.id} text={tool.description}>
                        <button
                          onClick={() => onToggle(tool.id)}
                          className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium border transition-all hover:opacity-80 ${
                            tool.active
                              ? `${TAG_COLORS[tool.tag].bg} ${
                                  TAG_COLORS[tool.tag].text
                                } ${TAG_COLORS[tool.tag].border}`
                              : "bg-muted text-muted-foreground border-border hover:bg-accent"
                          }`}
                        >
                          <span>{tool.icon}</span>
                          {tool.name}
                          {tool.active && (
                            <span className="text-[10px] ml-0.5">✓</span>
                          )}
                        </button>
                      </Tip>
                    );
                  })}
                </div>
              </div>
            ))}
          </div>

          {/* Notes */}
          <div className="mt-3 pt-2 border-t border-border space-y-1">
            {hasNotDeployed && (
              <p className="text-[10px] text-muted-foreground">
                <Lock className="h-2.5 w-2.5 inline mr-1 -mt-0.5" />
                Grayed-out tools are disabled in the deployment configuration.
              </p>
            )}
            <p className="text-[10px] text-muted-foreground">
              <span className="inline-block mr-1 -mt-0.5">🔑</span>
              Some tools require API keys or resource IDs to work properly.
              Check the deployment configuration for details.
            </p>
          </div>

          {/* S3 file URIs input */}
          {enabledSources["s3"] && onS3FileInputChange && (
            <div className="mt-3 pt-3 border-t border-border">
              <textarea
                placeholder="s3://bucket/path/to/file.txt&#10;s3://bucket/another/file.csv"
                value={s3FileInput}
                onChange={(e) => onS3FileInputChange(e.target.value)}
                rows={2}
                className="w-full px-3 py-2 border border-border rounded-lg text-xs font-mono text-left placeholder:text-muted-foreground bg-background text-foreground focus:outline-none focus:ring-2 focus:ring-blue-300 dark:focus:ring-blue-600"
              />
              <p className="text-[10px] text-muted-foreground mt-1">
                One S3 URI per line (supports txt, md, csv, json, pdf, etc.)
              </p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
