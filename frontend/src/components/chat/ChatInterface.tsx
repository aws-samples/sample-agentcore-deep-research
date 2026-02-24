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

// Available data sources configuration
const DATA_SOURCES = [
  { id: "tavily", name: "Tavily Web", icon: "🌐", defaultEnabled: true },
  { id: "nova", name: "Nova Search", icon: "🔍", defaultEnabled: true },
  { id: "arxiv", name: "ArXiv Papers", icon: "📚", defaultEnabled: true },
] as const;

export default function ChatInterface() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [client, setClient] = useState<AgentCoreClient | null>(null);
  const [sessionId] = useState(() => crypto.randomUUID());

  // Data source toggles - initialize from defaults
  const [enabledSources, setEnabledSources] = useState<Record<string, boolean>>(
    () => Object.fromEntries(DATA_SOURCES.map((s) => [s.id, s.defaultEnabled])),
  );

  // Research report state
  const [reportContent, setReportContent] = useState<string>("");
  const [researchRound, setResearchRound] = useState<number>(0);
  const [showReportPanel, setShowReportPanel] = useState<boolean>(false);
  const fileWriteCountRef = useRef<number>(0);

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
            "strands-single-agent") as AgentPattern,
        });

        setClient(agentClient);
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
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
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

                // Extract report URL from file_write/editor results and fetch from S3
                if (tc.name === "file_write" || tc.name === "editor") {
                  const reportUrl = extractReportUrl(tc.result || "");
                  if (reportUrl) {
                    fetchReportContent(reportUrl).then((content) => {
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
      );
    } catch (err) {
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
    setMessages([]);
    setInput("");
    setError(null);
    setReportContent("");
    setResearchRound(0);
    setShowReportPanel(false);
    fileWriteCountRef.current = 0;
    // Note: sessionId stays the same for the component lifecycle
    // If you want a new session ID, you'd need to remount the component
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
          <div className="bg-red-50 border-l-4 border-red-500 p-4 mx-4 mt-2">
            <p className="text-sm text-red-700">{error}</p>
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
            <h2 className="text-2xl font-bold text-gray-800">
              Correlate: Deep Research
            </h2>
            <p className="text-gray-600 mt-2">
              Ask a research question and I'll search across multiple sources to
              create a comprehensive report
            </p>

            {/* Data source toggles */}
            <div className="flex flex-wrap justify-center gap-3 mt-6">
              {DATA_SOURCES.map((source) => (
                <button
                  key={source.id}
                  onClick={() => toggleSource(source.id)}
                  className={`px-3 py-2 rounded-lg text-sm font-medium transition-all ${
                    enabledSources[source.id]
                      ? "bg-blue-100 text-blue-800 border-2 border-blue-300"
                      : "bg-gray-100 text-gray-400 border-2 border-transparent"
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
            <p className="text-xs text-gray-400 mt-2">
              Click to toggle data sources
            </p>
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
                    sessionId={sessionId}
                    isLoading={isLoading}
                    onFeedbackSubmit={handleFeedbackSubmit}
                  />
                </div>
                {/* Chat input */}
                <div className="flex-none p-2 border-t bg-white">
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
