"use client";
// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

import { useCallback, useEffect, useState } from "react";
import ChatInterface from "@/components/chat/ChatInterface";
import { SessionSidebar } from "@/components/sidebar/SessionSidebar";
import { Button } from "@/components/ui/button";
import { useAuth } from "@/hooks/useAuth";
import { GlobalContextProvider } from "@/app/context/GlobalContext";
import { listSessions, loadSession } from "@/services/sessionService";
import type { SessionSummary, SessionData } from "@/services/sessionService";
import { useAuth as useOidcAuth } from "react-oidc-context";

export default function ChatPage() {
  const { isAuthenticated, signIn } = useAuth();
  const oidcAuth = useOidcAuth();

  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [sessionsLoading, setSessionsLoading] = useState(false);
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  const [sessionToLoad, setSessionToLoad] = useState<SessionData | null>(null);
  const [loadCounter, setLoadCounter] = useState(0);

  const toggleSidebar = useCallback(() => {
    setSidebarOpen((prev) => !prev);
  }, []);

  const closeSidebar = useCallback(() => {
    setSidebarOpen(false);
  }, []);

  const refreshSessions = useCallback(async () => {
    const idToken = oidcAuth.user?.id_token;
    if (!idToken) return;

    setSessionsLoading(true);
    try {
      const result = await listSessions(idToken);
      setSessions(result);
      if (result.length > 0) {
        setSidebarOpen(true);
      }
    } catch (err) {
      console.error("Failed to load sessions:", err);
    } finally {
      setSessionsLoading(false);
    }
  }, [oidcAuth.user?.id_token]);

  useEffect(() => {
    if (isAuthenticated && oidcAuth.user?.id_token) {
      refreshSessions();
    }
  }, [isAuthenticated, oidcAuth.user?.id_token, refreshSessions]);

  const handleSelectSession = useCallback(
    async (sessionId: string) => {
      const idToken = oidcAuth.user?.id_token;
      if (!idToken) return;

      try {
        const data = await loadSession(sessionId, idToken);
        setActiveSessionId(sessionId);
        setSessionToLoad(data);
        setLoadCounter((c) => c + 1);
        setSidebarOpen(false);
      } catch (err) {
        console.error("Failed to load session:", err);
      }
    },
    [oidcAuth.user?.id_token],
  );

  // Called by ChatInterface when the header's "New Research" is clicked
  const handleNewChat = useCallback(() => {
    setActiveSessionId(null);
    setSessionToLoad(null);
  }, []);

  const handleSessionSaved = useCallback((summary: SessionSummary) => {
    setActiveSessionId(summary.sessionId);
    setSessions((prev) => {
      const existing = prev.findIndex((s) => s.sessionId === summary.sessionId);
      if (existing >= 0) {
        const updated = [...prev];
        updated[existing] = summary;
        updated.sort((a, b) => b.updatedAt - a.updatedAt);
        return updated;
      }
      return [summary, ...prev];
    });
  }, []);

  if (!isAuthenticated) {
    return (
      <div className="flex flex-col items-center justify-center min-h-screen gap-4">
        <p className="text-4xl">Please sign in</p>
        <Button onClick={() => signIn()}>Sign In</Button>
      </div>
    );
  }

  return (
    <GlobalContextProvider>
      <div className="flex h-screen relative">
        <SessionSidebar
          sessions={sessions}
          activeSessionId={activeSessionId}
          isOpen={sidebarOpen}
          isLoading={sessionsLoading}
          onClose={toggleSidebar}
          onSelectSession={handleSelectSession}
        />
        <div className="flex-1 min-w-0">
          <ChatInterface
            onToggleSidebar={toggleSidebar}
            sessionToLoad={sessionToLoad}
            loadTrigger={loadCounter}
            onSessionSaved={handleSessionSaved}
            onResearchStart={closeSidebar}
            onNewChat={handleNewChat}
          />
        </div>
      </div>
    </GlobalContextProvider>
  );
}
