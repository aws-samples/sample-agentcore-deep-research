// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0
import { useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { Plus, Moon, Sun, PanelLeft } from "lucide-react";
import { useAuth } from "@/hooks/useAuth";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from "@/components/ui/alert-dialog";

function useDarkMode() {
  const [dark, setDark] = useState(() => {
    if (typeof window === "undefined") return false;
    const stored = localStorage.getItem("theme");
    if (stored) return stored === "dark";
    return window.matchMedia("(prefers-color-scheme: dark)").matches;
  });

  useEffect(() => {
    document.documentElement.classList.toggle("dark", dark);
    localStorage.setItem("theme", dark ? "dark" : "light");
  }, [dark]);

  return [dark, () => setDark((d) => !d)] as const;
}

type ChatHeaderProps = {
  title?: string | undefined;
  onNewChat: () => void;
  canStartNewChat: boolean;
  onToggleSidebar?: () => void;
};

export function ChatHeader({
  title,
  onNewChat,
  canStartNewChat,
  onToggleSidebar,
}: ChatHeaderProps) {
  const { isAuthenticated, signOut } = useAuth();
  const [dark, toggleDark] = useDarkMode();

  return (
    <header className="flex items-center justify-between p-4 border-b w-full">
      <div className="flex items-center gap-2">
        {onToggleSidebar && (
          <Button
            onClick={onToggleSidebar}
            variant="ghost"
            size="icon"
            aria-label="Toggle sidebar"
          >
            <PanelLeft className="h-5 w-5" />
          </Button>
        )}
        <h1 className="text-xl font-bold">
          {title || "AgentCore Deep Research"}
        </h1>
      </div>
      <div className="flex items-center gap-2">
        <Button
          onClick={toggleDark}
          variant="ghost"
          size="icon"
          aria-label="Toggle dark mode"
        >
          {dark ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
        </Button>
        <Button
          onClick={onNewChat}
          variant="outline"
          className="gap-2"
          disabled={!canStartNewChat}
        >
          <Plus className="h-4 w-4" />
          New Research
        </Button>
        {isAuthenticated && (
          <AlertDialog>
            <AlertDialogTrigger asChild>
              <Button variant="outline">Logout</Button>
            </AlertDialogTrigger>
            <AlertDialogContent>
              <AlertDialogHeader>
                <AlertDialogTitle>Confirm Logout</AlertDialogTitle>
                <AlertDialogDescription>
                  Are you sure you want to log out? You will need to sign in
                  again to access your account.
                </AlertDialogDescription>
              </AlertDialogHeader>
              <AlertDialogFooter>
                <AlertDialogCancel>Cancel</AlertDialogCancel>
                <AlertDialogAction onClick={() => signOut()}>
                  Confirm
                </AlertDialogAction>
              </AlertDialogFooter>
            </AlertDialogContent>
          </AlertDialog>
        )}
      </div>
    </header>
  );
}
