// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0
/**
 * Session History Service
 * Handles CRUD operations for user research sessions
 */

import type { Message } from "@/components/chat/types";

let API_BASE_URL = "";

async function loadApiUrl(): Promise<string> {
  if (API_BASE_URL) {
    return API_BASE_URL;
  }

  try {
    const response = await fetch("/aws-exports.json");
    const config = await response.json();
    API_BASE_URL = config.feedbackApiUrl || "";
    return API_BASE_URL;
  } catch (error) {
    console.error("Failed to load API URL from aws-exports.json:", error);
    throw new Error("API URL not configured");
  }
}

export interface SessionSummary {
  sessionId: string;
  title: string;
  createdAt: number;
  updatedAt: number;
  hasReport: boolean;
}

export interface SessionData {
  sessionId: string;
  title: string;
  messages: Message[];
  enabledSources: Record<string, boolean>;
  s3FileInput: string;
  reportContent: string | null;
  reportPdfUrl: string | null;
  createdAt: number;
  updatedAt: number;
}

export interface SaveSessionPayload {
  title: string;
  messages: Message[];
  enabledSources: Record<string, boolean>;
  s3FileInput: string;
  hasReport: boolean;
}

async function apiRequest<T>(
  path: string,
  idToken: string,
  options: RequestInit = {},
): Promise<T> {
  const baseUrl = await loadApiUrl();
  // cache-bust GET requests to bypass API Gateway response cache
  const cacheBuster = !options.method || options.method === "GET"
    ? `${path.includes("?") ? "&" : "?"}_t=${Date.now()}`
    : "";
  const url = `${baseUrl}${path}${cacheBuster}`;

  const response = await fetch(url, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${idToken}`,
      ...options.headers,
    },
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(
      errorData.error || `HTTP error! status: ${response.status}`,
    );
  }

  return response.json();
}

export async function listSessions(
  idToken: string,
): Promise<SessionSummary[]> {
  const data = await apiRequest<{ sessions: SessionSummary[] }>(
    "sessions",
    idToken,
  );
  return data.sessions;
}

export async function loadSession(
  sessionId: string,
  idToken: string,
): Promise<SessionData> {
  return apiRequest<SessionData>(`sessions/${sessionId}`, idToken);
}

export async function saveSession(
  sessionId: string,
  payload: SaveSessionPayload,
  idToken: string,
): Promise<{ success: boolean; sessionId: string; title: string; createdAt: number; updatedAt: number }> {
  return apiRequest(`sessions/${sessionId}`, idToken, {
    method: "PUT",
    body: JSON.stringify(payload),
  });
}

export async function deleteSession(
  sessionId: string,
  idToken: string,
): Promise<void> {
  await apiRequest(`sessions/${sessionId}`, idToken, {
    method: "DELETE",
  });
}
