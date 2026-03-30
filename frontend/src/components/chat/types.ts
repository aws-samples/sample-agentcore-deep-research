// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0
export type ToolCallStatus = "streaming" | "executing" | "complete";

export interface ToolCall {
  toolUseId: string;
  name: string;
  input: string;
  result?: string;
  status: ToolCallStatus;
}

export type MessageSegment =
  | { type: "text"; content: string }
  | { type: "tool"; toolCall: ToolCall };

export interface Message {
  role: "user" | "assistant";
  content: string;
  timestamp: string;
  segments?: MessageSegment[];
}
