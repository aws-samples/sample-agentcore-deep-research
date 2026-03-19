// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0
import { RefObject } from "react";
import { Message } from "./types";
import { ChatMessage } from "./ChatMessage";
import { Loader2 } from "lucide-react";

interface ChatMessagesProps {
  messages: Message[];
  messagesEndRef: RefObject<HTMLDivElement | null>;
  containerRef?: RefObject<HTMLDivElement | null>;
  sessionId: string;
  isLoading?: boolean;
  onFeedbackSubmit: (
    messageContent: string,
    feedbackType: "positive" | "negative",
    comment: string,
  ) => Promise<void>;
}

function ThinkingIndicator() {
  return (
    <div className="flex items-start gap-3 animate-fade-in">
      <div className="flex items-center gap-2 px-4 py-3 bg-muted rounded-2xl rounded-bl-none">
        <Loader2 className="w-4 h-4 text-blue-600 dark:text-blue-400 animate-spin" />
        <span className="text-sm text-muted-foreground">Searching...</span>
      </div>
    </div>
  );
}

export function ChatMessages({
  messages,
  messagesEndRef,
  containerRef,
  sessionId,
  isLoading = false,
  onFeedbackSubmit,
}: ChatMessagesProps) {
  // Check if we should show thinking indicator
  // Show when loading and the last message is empty assistant message (waiting for first response)
  const lastMessage = messages[messages.length - 1];
  const showThinking =
    isLoading &&
    lastMessage?.role === "assistant" &&
    !lastMessage.content &&
    (!lastMessage.segments || lastMessage.segments.length === 0);

  return (
    <div
      ref={containerRef}
      className={`h-full p-4 space-y-4 w-full ${
        messages.length > 0 ? "overflow-y-auto" : "overflow-hidden"
      }`}
    >
      {messages.length === 0 ? (
        <div className="flex items-center justify-center h-full text-muted-foreground">
          Start a new conversation
        </div>
      ) : (
        <>
          {messages.map((message, index) => {
            // Skip rendering empty assistant message if we're showing thinking indicator
            if (
              showThinking &&
              index === messages.length - 1 &&
              message.role === "assistant"
            ) {
              return null;
            }
            return (
              <ChatMessage
                key={index}
                message={message}
                sessionId={sessionId}
                onFeedbackSubmit={async (feedbackType, comment) => {
                  await onFeedbackSubmit(
                    message.content,
                    feedbackType,
                    comment,
                  );
                }}
              />
            );
          })}
          {showThinking && <ThinkingIndicator />}
        </>
      )}
      <div ref={messagesEndRef} />
    </div>
  );
}
