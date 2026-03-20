// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0
"use client";

import { useEffect, useRef, useState } from "react";
import { ChatHeader } from "./ChatHeader";
import { ChatInput } from "./ChatInput";
import { ChatMessages } from "./ChatMessages";
import { Message, MessageSegment, ToolCall } from "./types";
import { ResearchReportPanel } from "./ResearchReportPanel";
import { ResizableSplitPane } from "./ResizableSplitPane";

import { useGlobal } from "@/app/context/GlobalContext";
import { AgentCoreClient } from "@/lib/agentcore-client";
import type { AgentPattern } from "@/lib/agentcore-client";
import { submitFeedback } from "@/services/feedbackService";
import { useAuth } from "react-oidc-context";
import { useDefaultTool } from "@/hooks/useToolRenderer";
import { ToolCallDisplay } from "./ToolCallDisplay";

// Extract report URL from tool result
function extractReportUrl(result: string): string | null {
  const match = result.match(/\[REPORT_URL:(https?:\/\/[^\]]+)\]/);
  return match ? match[1] : null;
}

// Fetch report content from pre-signed S3 URL
async function fetchReportContent(url: string): Promise<string | null> {
  try {
    const response = await fetch(url);
    if (!response.ok) return null;
    return await response.text();
  } catch {
    return null;
  }
}

// Tool metadata (display names and icons)
const TOOL_METADATA: Record<string, { name: string; icon: string }> = {
  alphavantage: { name: "AlphaVantage", icon: "📈" },
  tavily: { name: "Tavily Web", icon: "🌐" },
  nova: { name: "Nova Web Grounding", icon: "🔍" },
  arxiv: { name: "ArXiv Papers", icon: "📚" },
  openfda: { name: "OpenFDA Drugs", icon: "💊" },
  s3: { name: "S3 Files", icon: "📁" },
  bedrock_kb: { name: "Bedrock KB", icon: "🧠" },
  pubmed: { name: "PubMed", icon: "🏥" },
  clinicaltrials: { name: "ClinicalTrials", icon: "🔬" },
  fred: { name: "FRED Economic", icon: "🏦" },
  edgar: { name: "SEC EDGAR", icon: "🏛️" },
};

// Tool config from aws-exports.json (populated on mount)
interface ToolConfig {
  enabled: boolean;
  default_on: boolean;
}

// Fallback defaults if tools config is not in aws-exports.json
const FALLBACK_TOOLS: Record<string, ToolConfig> = {
  alphavantage: { enabled: false, default_on: false },
  tavily: { enabled: false, default_on: false },
  nova: { enabled: true, default_on: true },
  arxiv: { enabled: true, default_on: false },
  openfda: { enabled: true, default_on: false },
  s3: { enabled: true, default_on: false },
  bedrock_kb: { enabled: false, default_on: false },
  pubmed: { enabled: true, default_on: false },
  clinicaltrials: { enabled: true, default_on: false },
  fred: { enabled: true, default_on: false },
  edgar: { enabled: true, default_on: false },
};

export default function ChatInterface() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [client, setClient] = useState<AgentCoreClient | null>(null);
  const [sessionId, setSessionId] = useState(() => crypto.randomUUID());

  // Tools config loaded from aws-exports.json
  const [toolsConfig, setToolsConfig] =
    useState<Record<string, ToolConfig>>(FALLBACK_TOOLS);

  // Derived: visible data sources (enabled tools only)
  const dataSources = Object.entries(toolsConfig)
    .filter(([, cfg]) => cfg.enabled)
    .filter(([id]) => TOOL_METADATA[id])
    .map(([id, cfg]) => ({
      id,
      name: TOOL_METADATA[id].name,
      icon: TOOL_METADATA[id].icon,
      defaultEnabled: cfg.default_on,
    }));

  // Data source toggles - initialize from defaults
  const [enabledSources, setEnabledSources] = useState<Record<string, boolean>>(
    {},
  );

  // S3 file URIs (one per line)
  const [s3FileInput, setS3FileInput] = useState<string>("");

  // Research report state
  const [reportContent, setReportContent] = useState<string>("");
  const [researchRound, setResearchRound] = useState<number>(0);
  const [showReportPanel, setShowReportPanel] = useState<boolean>(false);
  const fileWriteCountRef = useRef<number>(0);
  const lastReportUrlRef = useRef<string | null>(null);

  // Get array of enabled source IDs for API
  const getEnabledSourceIds = () =>
    Object.entries(enabledSources)
      .filter(([, enabled]) => enabled)
      .map(([id]) => id);

  // Toggle a data source
  const toggleSource = (sourceId: string) => {
    setEnabledSources((prev) => ({
      ...prev,
      [sourceId]: !prev[sourceId],
    }));
  };

  const { isLoading, setIsLoading } = useGlobal();
  const auth = useAuth();

  // Ref for message container to enable auto-scrolling
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const messagesContainerRef = useRef<HTMLDivElement>(null);
  const userScrolledUpRef = useRef(false);

  // Track user scroll position to pause/resume auto-scroll
  useEffect(() => {
    const container = messagesContainerRef.current;
    if (!container) return;
    const handleScroll = () => {
      const { scrollTop, scrollHeight, clientHeight } = container;
      userScrolledUpRef.current = scrollHeight - scrollTop - clientHeight > 100;
    };
    container.addEventListener("scroll", handleScroll);
    return () => container.removeEventListener("scroll", handleScroll);
  }, []);

  // Register default tool renderer (wildcard "*")
  useDefaultTool(({ name, args, status, result }) => (
    <ToolCallDisplay name={name} args={args} status={status} result={result} />
  ));

  // Load agent configuration and create client on mount
  useEffect(() => {
    async function loadConfig() {
      try {
        const response = await fetch("/aws-exports.json");
        if (!response.ok) {
          throw new Error("Failed to load configuration");
        }
        const config = await response.json();

        if (!config.agentRuntimeArn) {
          throw new Error("Agent Runtime ARN not found in configuration");
        }

        const agentClient = new AgentCoreClient({
          runtimeArn: config.agentRuntimeArn,
          region: config.awsRegion || "us-east-1",
          pattern: (config.agentPattern ||
            "strands-deep-research") as AgentPattern,
        });

        setClient(agentClient);

        // Load tools config from aws-exports.json
        const tools: Record<string, ToolConfig> =
          config.tools || FALLBACK_TOOLS;
        setToolsConfig(tools);

        // Initialize enabled sources from tools config
        const defaults: Record<string, boolean> = {};
        for (const [id, cfg] of Object.entries(tools)) {
          if (cfg.enabled) {
            defaults[id] = cfg.default_on;
          }
        }
        setEnabledSources(defaults);
      } catch (err) {
        const errorMessage =
          err instanceof Error ? err.message : "Unknown error";
        setError(`Configuration error: ${errorMessage}`);
        console.error("Failed to load agent configuration:", err);
      }
    }

    loadConfig();
  }, []);

  useEffect(() => {
    if (!userScrolledUpRef.current) {
      messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages]);

  const sendMessage = async (userMessage: string) => {
    if (!userMessage.trim() || !client) return;

    // Clear any previous errors
    setError(null);

    // Add user message to chat
    const newUserMessage: Message = {
      role: "user",
      content: userMessage,
      timestamp: new Date().toISOString(),
    };

    setMessages((prev) => [...prev, newUserMessage]);
    setInput("");
    setIsLoading(true);

    // Create placeholder for assistant response
    const assistantResponse: Message = {
      role: "assistant",
      content: "",
      timestamp: new Date().toISOString(),
    };

    setMessages((prev) => [...prev, assistantResponse]);

    try {
      // Get auth token from react-oidc-context
      const accessToken = auth.user?.access_token;

      if (!accessToken) {
        throw new Error("Authentication required. Please log in again.");
      }

      const segments: MessageSegment[] = [];
      const toolCallMap = new Map<string, ToolCall>();

      const updateMessage = () => {
        // Build content from text segments for backward compat
        const content = segments
          .filter(
            (s): s is Extract<MessageSegment, { type: "text" }> =>
              s.type === "text",
          )
          .map((s) => s.content)
          .join("");

        setMessages((prev) => {
          const updated = [...prev];
          updated[updated.length - 1] = {
            ...updated[updated.length - 1],
            content,
            segments: [...segments],
          };
          return updated;
        });
      };

      // User identity is extracted server-side from the validated JWT token,
      // not passed as a parameter — prevents impersonation via prompt injection.
      const enabledSourceIds = getEnabledSourceIds();
      const s3Uris = enabledSources["s3"]
        ? s3FileInput
            .split("\n")
            .map((u) => u.trim())
            .filter((u) => u.startsWith("s3://"))
        : undefined;
      await client.invoke(
        userMessage,
        sessionId,
        accessToken,
        (event) => {
          switch (event.type) {
            case "text": {
              // If text arrives after a tool segment, mark all pending tools as complete
              const prev = segments[segments.length - 1];
              if (prev && prev.type === "tool") {
                for (const tc of toolCallMap.values()) {
                  if (tc.status === "streaming" || tc.status === "executing") {
                    tc.status = "complete";
                  }
                }
              }
              // Append to last text segment, or create new one
              const last = segments[segments.length - 1];
              if (last && last.type === "text") {
                last.content += event.content;
              } else {
                segments.push({ type: "text", content: event.content });
              }
              updateMessage();
              break;
            }
            case "tool_use_start": {
              const tc: ToolCall = {
                toolUseId: event.toolUseId,
                name: event.name,
                input: "",
                status: "streaming",
              };
              toolCallMap.set(event.toolUseId, tc);
              segments.push({ type: "tool", toolCall: tc });

              // update research round and show panel when new file_write/editor starts
              if (event.name === "file_write" || event.name === "editor") {
                fileWriteCountRef.current += 1;
                setResearchRound(fileWriteCountRef.current);
                setShowReportPanel(true); // Show panel immediately when tool starts
              }

              updateMessage();
              break;
            }
            case "tool_use_delta": {
              const tc = toolCallMap.get(event.toolUseId);
              if (tc) {
                tc.input += event.input;
              }
              updateMessage();
              break;
            }
            case "tool_result": {
              const tc = toolCallMap.get(event.toolUseId);
              if (tc) {
                tc.result = event.result;
                tc.status = "complete";

                // Re-fetch report from S3 after any file_write/editor completes
                if (tc.name === "file_write" || tc.name === "editor") {
                  const newUrl = extractReportUrl(tc.result || "");
                  if (newUrl) {
                    lastReportUrlRef.current = newUrl;
                  }
                  const url = lastReportUrlRef.current;
                  if (url) {
                    fetchReportContent(url).then((content) => {
                      if (content) {
                        setReportContent(content);
                      }
                    });
                  }
                }
              }
              updateMessage();
              break;
            }
            case "message": {
              if (event.role === "assistant") {
                for (const tc of toolCallMap.values()) {
                  if (tc.status === "streaming") tc.status = "executing";
                }
                updateMessage();
              }
              break;
            }
          }
        },
        enabledSourceIds,
        s3Uris,
      );
    } catch (err) {
      // Silently ignore aborted requests (user clicked New Research)
      if (err instanceof DOMException && err.name === "AbortError") return;

      const errorMessage = err instanceof Error ? err.message : "Unknown error";
      setError(`Failed to get response: ${errorMessage}`);
      console.error("Error invoking AgentCore:", err);

      // Update the assistant message with error
      setMessages((prev) => {
        const updated = [...prev];
        updated[updated.length - 1] = {
          ...updated[updated.length - 1],
          content:
            "I apologize, but I encountered an error processing your request. Please try again.",
        };
        return updated;
      });
    } finally {
      setIsLoading(false);
    }
  };

  // Handle form submission
  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();

    sendMessage(input);
  };

  // Handle feedback submission
  const handleFeedbackSubmit = async (
    messageContent: string,
    feedbackType: "positive" | "negative",
    comment: string,
  ) => {
    try {
      // Use ID token for API Gateway Cognito authorizer (not access token)
      const idToken = auth.user?.id_token;

      if (!idToken) {
        throw new Error("Authentication required. Please log in again.");
      }

      await submitFeedback(
        {
          sessionId,
          message: messageContent,
          feedbackType,
          comment: comment || undefined,
        },
        idToken,
      );

      console.log("Feedback submitted successfully");
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : "Unknown error";
      console.error("Error submitting feedback:", err);
      setError(`Failed to submit feedback: ${errorMessage}`);
    }
  };

  // Start a new chat (generates new session ID)
  const startNewChat = () => {
    client?.abort();
    setSessionId(crypto.randomUUID());
    setMessages([]);
    setInput("");
    setError(null);
    setIsLoading(false);
    setReportContent("");
    setResearchRound(0);
    setShowReportPanel(false);
    fileWriteCountRef.current = 0;
    const defaults: Record<string, boolean> = {};
    for (const [id, cfg] of Object.entries(toolsConfig)) {
      if (cfg.enabled) {
        defaults[id] = cfg.default_on;
      }
    }
    setEnabledSources(defaults);
    setS3FileInput("");
  };

  // Check if this is the initial state (no messages)
  const isInitialState = messages.length === 0;

  // Check if there are any assistant messages
  const hasAssistantMessages = messages.some(
    (message) => message.role === "assistant",
  );

  // Show split view when report panel should be visible (triggered on first file_write/editor start)
  const showSplitView = showReportPanel;

  return (
    <div className="flex flex-col h-screen w-full">
      {/* Fixed header */}
      <div className="flex-none">
        <ChatHeader
          onNewChat={startNewChat}
          canStartNewChat={hasAssistantMessages}
        />
        {error && (
          <div className="bg-red-50 dark:bg-red-950/50 border-l-4 border-red-500 p-4 mx-4 mt-2">
            <p className="text-sm text-red-700 dark:text-red-400">{error}</p>
          </div>
        )}
      </div>

      {/* Conditional layout based on whether there are messages */}
      {isInitialState ? (
        // Initial state - input in the middle
        <>
          {/* Empty space above */}
          <div className="grow" />

          {/* Centered welcome message */}
          <div className="text-center mb-6">
            <h2 className="text-2xl font-bold text-foreground">
              AgentCore Deep Research
            </h2>
            <p className="text-muted-foreground mt-2">
              Ask a question and I will search across multiple sources to create
              a comprehensive report
            </p>

            {/* Data source toggles */}
            <div className="flex flex-wrap justify-center gap-3 mt-6">
              {dataSources.map((source) => (
                <button
                  key={source.id}
                  onClick={() => toggleSource(source.id)}
                  className={`px-3 py-2 rounded-lg text-sm font-medium transition-all ${
                    enabledSources[source.id]
                      ? "bg-blue-100 dark:bg-blue-900/40 text-blue-800 dark:text-blue-300 border-2 border-blue-300 dark:border-blue-700"
                      : "bg-muted text-muted-foreground border-2 border-transparent"
                  }`}
                >
                  <span className="mr-1">{source.icon}</span>
                  {source.name}
                  <span className="ml-2">
                    {enabledSources[source.id] ? "✓" : "○"}
                  </span>
                </button>
              ))}
            </div>
            <p className="text-xs text-muted-foreground mt-2">
              Click to toggle data sources
            </p>

            {/* S3 file URIs input */}
            {enabledSources["s3"] && (
              <div className="mt-4 max-w-lg mx-auto">
                <textarea
                  placeholder={
                    "s3://bucket/path/to/file.txt\ns3://bucket/another/file.csv"
                  }
                  value={s3FileInput}
                  onChange={(e) => setS3FileInput(e.target.value)}
                  rows={3}
                  className="w-full px-3 py-2 border border-border rounded-lg text-sm font-mono text-left placeholder:text-muted-foreground bg-background text-foreground focus:outline-none focus:ring-2 focus:ring-blue-300 dark:focus:ring-blue-600"
                />
                <p className="text-xs text-muted-foreground mt-1">
                  One S3 URI per line (supports txt, md, csv, json, pdf, etc.)
                </p>
              </div>
            )}
          </div>

          {/* Centered input */}
          <div className="px-4 mb-16 max-w-4xl mx-auto w-full">
            <ChatInput
              input={input}
              setInput={setInput}
              handleSubmit={handleSubmit}
              isLoading={isLoading}
            />
          </div>

          {/* Empty space below */}
          <div className="grow" />
        </>
      ) : showSplitView ? (
        // Split view - chat on left, report on right
        <div className="grow overflow-hidden">
          <ResizableSplitPane
            defaultLeftWidth={33}
            minLeftWidth={25}
            maxLeftWidth={50}
            left={
              <div className="flex flex-col h-full border-r">
                {/* Chat messages */}
                <div className="grow overflow-hidden">
                  <ChatMessages
                    messages={messages}
                    messagesEndRef={messagesEndRef}
                    containerRef={messagesContainerRef}
                    sessionId={sessionId}
                    isLoading={isLoading}
                    onFeedbackSubmit={handleFeedbackSubmit}
                  />
                </div>
                {/* Chat input */}
                <div className="flex-none p-2 border-t border-border bg-background">
                  <ChatInput
                    input={input}
                    setInput={setInput}
                    handleSubmit={handleSubmit}
                    isLoading={isLoading}
                  />
                </div>
              </div>
            }
            right={
              <ResearchReportPanel
                content={reportContent}
                isLoading={isLoading}
                currentRound={researchRound}
              />
            }
          />
        </div>
      ) : (
        // Chat in progress without report - normal layout
        <>
          {/* Scrollable message area */}
          <div className="grow overflow-hidden">
            <div className="max-w-4xl mx-auto w-full h-full">
              <ChatMessages
                messages={messages}
                messagesEndRef={messagesEndRef}
                sessionId={sessionId}
                isLoading={isLoading}
                onFeedbackSubmit={handleFeedbackSubmit}
              />
            </div>
          </div>

          {/* Fixed input area at bottom */}
          <div className="flex-none">
            <div className="max-w-4xl mx-auto w-full">
              <ChatInput
                input={input}
                setInput={setInput}
                handleSubmit={handleSubmit}
                isLoading={isLoading}
              />
            </div>
          </div>
        </>
      )}
    </div>
  );
}
