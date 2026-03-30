// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0
"use client";

import { useMemo } from "react";
import {
  Download,
  FileText,
  Loader2,
  BookOpen,
  Link,
  BarChart3,
} from "lucide-react";
import {
  ReportMarkdownRenderer,
  extractReportTitle,
} from "./ReportMarkdownRenderer";

interface ResearchReportPanelProps {
  content: string;
  isLoading: boolean;
  currentRound: number;
  onAnalyze?: () => void;
}

export function ResearchReportPanel({
  content,
  isLoading,
  currentRound,
  onAnalyze,
}: ResearchReportPanelProps) {
  // Extract H1 title from markdown content
  const reportTitle = useMemo(
    () => extractReportTitle(content) || "Research Report",
    [content],
  );

  // Extract unique [Source: XXX] references from report
  const references = useMemo(() => {
    if (!content) return [];
    const matches = content.matchAll(/\[Source:\s*([^\]]+)\]/g);
    const seen = new Set<string>();
    const refs: { label: string; url: string | null }[] = [];
    for (const m of matches) {
      const raw = m[1].trim();
      if (seen.has(raw)) continue;
      seen.add(raw);
      // detect if the source is a URL
      const isUrl = /^https?:\/\//.test(raw);
      refs.push({ label: raw, url: isUrl ? raw : null });
    }
    return refs;
  }, [content]);

  const handleDownload = async () => {
    const imgRegex = /!\[([^\]]*)\]\((https?:\/\/[^)]+)\)/g;
    let localContent = content;
    const images: { filename: string; url: string }[] = [];
    let match;
    let i = 1;

    while ((match = imgRegex.exec(content)) !== null) {
      const pathMatch = match[2].match(/\/([^/?]+\.png)/);
      const filename = pathMatch ? pathMatch[1] : `chart_${i}.png`;
      images.push({ filename, url: match[2] });
      localContent = localContent.replace(match[2], filename);
      i++;
    }

    // Download markdown (with local paths if images exist)
    const blob = new Blob([images.length ? localContent : content], { type: "text/markdown" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "research_report.md";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    // Download each image with a small delay to avoid browser blocking
    for (const img of images) {
      try {
        const resp = await fetch(img.url);
        if (!resp.ok) continue;
        const imgBlob = await resp.blob();
        const imgUrl = URL.createObjectURL(imgBlob);
        const imgA = document.createElement("a");
        imgA.href = imgUrl;
        imgA.download = img.filename;
        document.body.appendChild(imgA);
        imgA.click();
        document.body.removeChild(imgA);
        URL.revokeObjectURL(imgUrl);
        await new Promise((r) => setTimeout(r, 300));
      } catch { /* skip */ }
    }
  };

  return (
    <div className="flex flex-col h-full bg-gradient-to-b from-gray-50 to-gray-100 dark:from-gray-900 dark:to-gray-950">
      {/* Header */}
      <div className="flex-none flex items-center justify-between px-4 py-3 border-b border-border bg-background shadow-sm">
        <div className="flex items-center gap-2 min-w-0 flex-1">
          <div className="p-1.5 bg-blue-100 dark:bg-blue-900/40 rounded-lg shrink-0">
            <FileText className="w-5 h-5 text-blue-600" />
          </div>
          <h2
            className="font-semibold text-foreground truncate"
            title={reportTitle}
          >
            {reportTitle}
          </h2>
          {isLoading && (
            <div className="flex items-center gap-2 ml-3 px-3 py-1 bg-gradient-to-r from-blue-100 to-blue-50 dark:from-blue-900/40 dark:to-blue-950/30 rounded-full border border-blue-200 dark:border-blue-800 shrink-0">
              <Loader2 className="w-4 h-4 text-blue-600 dark:text-blue-400 animate-spin" />
              <span className="text-sm text-blue-700 dark:text-blue-300 font-medium">
                Version {currentRound}
              </span>
            </div>
          )}
        </div>
        {content && (
          <div className="flex items-center gap-2 shrink-0 ml-2">
            {onAnalyze && !isLoading && (
              <button
                onClick={onAnalyze}
                className="flex items-center gap-2 px-3 py-1.5 text-sm font-medium text-white bg-blue-600 border border-blue-700 rounded-lg hover:bg-blue-700 transition-all shadow-sm"
              >
                <BarChart3 className="w-4 h-4" />
                Analyze
              </button>
            )}
            <button
              onClick={handleDownload}
              className="flex items-center gap-2 px-3 py-1.5 text-sm font-medium text-foreground bg-background border border-border rounded-lg hover:bg-muted hover:border-muted-foreground/30 transition-all shadow-sm"
            >
              <Download className="w-4 h-4" />
              Download
            </button>
          </div>
        )}
      </div>

      {/* Content */}
      <div className="grow overflow-auto">
        {content ? (
          <div className="p-6">
            <div className="bg-background rounded-2xl shadow-sm border border-border p-6">
              <ReportMarkdownRenderer
                content={content}
                collapsibleSections={true}
                isLoading={isLoading}
              />
            </div>

            {/* References section */}
            {references.length > 0 && (
              <div className="mt-4 bg-background rounded-2xl shadow-sm border border-border p-6">
                <div className="flex items-center gap-2 mb-3">
                  <Link className="w-4 h-4 text-muted-foreground" />
                  <h2 className="font-semibold text-foreground">References</h2>
                </div>
                <ol className="list-decimal list-inside space-y-1.5 text-sm text-muted-foreground">
                  {references.map((ref, i) => (
                    <li key={i} className="break-all">
                      {ref.url ? (
                        <a
                          href={ref.url}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-blue-600 hover:text-blue-800 hover:underline"
                        >
                          {ref.label}
                        </a>
                      ) : (
                        <span>{ref.label}</span>
                      )}
                    </li>
                  ))}
                </ol>
              </div>
            )}
          </div>
        ) : (
          <div className="flex flex-col items-center justify-center h-full text-muted-foreground">
            <div className="p-6 bg-background rounded-2xl shadow-sm border border-border mb-4">
              <BookOpen className="w-16 h-16 text-muted-foreground/50" />
            </div>
            <p className="text-lg font-medium text-muted-foreground">
              Generating report...
            </p>
            <p className="text-sm mt-1 text-muted-foreground/70 max-w-xs text-center">
              The research report will appear here as it's being generated
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
