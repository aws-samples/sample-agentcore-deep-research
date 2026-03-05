"use client";

import { useMemo } from "react";
import { Download, FileText, Loader2, BookOpen, Link } from "lucide-react";
import {
  ReportMarkdownRenderer,
  extractReportTitle,
} from "./ReportMarkdownRenderer";

interface ResearchReportPanelProps {
  content: string;
  isLoading: boolean;
  currentRound: number;
}

export function ResearchReportPanel({
  content,
  isLoading,
  currentRound,
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

  const handleDownload = () => {
    const blob = new Blob([content], { type: "text/markdown" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "research_report.md";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return (
    <div className="flex flex-col h-full bg-gradient-to-b from-gray-50 to-gray-100">
      {/* Header */}
      <div className="flex-none flex items-center justify-between px-4 py-3 border-b bg-white shadow-sm">
        <div className="flex items-center gap-2 min-w-0 flex-1">
          <div className="p-1.5 bg-blue-100 rounded-lg shrink-0">
            <FileText className="w-5 h-5 text-blue-600" />
          </div>
          <h2
            className="font-semibold text-gray-800 truncate"
            title={reportTitle}
          >
            {reportTitle}
          </h2>
          {isLoading && (
            <div className="flex items-center gap-2 ml-3 px-3 py-1 bg-gradient-to-r from-blue-100 to-blue-50 rounded-full border border-blue-200 shrink-0">
              <Loader2 className="w-4 h-4 text-blue-600 animate-spin" />
              <span className="text-sm text-blue-700 font-medium">
                Version {currentRound}
              </span>
            </div>
          )}
        </div>
        {content && (
          <button
            onClick={handleDownload}
            className="flex items-center gap-2 px-3 py-1.5 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-lg hover:bg-gray-50 hover:border-gray-400 transition-all shadow-sm shrink-0 ml-2"
          >
            <Download className="w-4 h-4" />
            Download
          </button>
        )}
      </div>

      {/* Content */}
      <div className="grow overflow-auto">
        {content ? (
          <div className="p-6">
            <div className="bg-white rounded-2xl shadow-sm border border-gray-200 p-6">
              <ReportMarkdownRenderer
                content={content}
                collapsibleSections={true}
                isLoading={isLoading}
              />
            </div>

            {/* References section */}
            {references.length > 0 && (
              <div className="mt-4 bg-white rounded-2xl shadow-sm border border-gray-200 p-6">
                <div className="flex items-center gap-2 mb-3">
                  <Link className="w-4 h-4 text-gray-500" />
                  <h3 className="font-semibold text-gray-700">
                    References ({references.length})
                  </h3>
                </div>
                <ol className="list-decimal list-inside space-y-1.5 text-sm text-gray-600">
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
          <div className="flex flex-col items-center justify-center h-full text-gray-400">
            <div className="p-6 bg-white rounded-2xl shadow-sm border border-gray-200 mb-4">
              <BookOpen className="w-16 h-16 text-gray-300" />
            </div>
            <p className="text-lg font-medium text-gray-500">
              Generating report...
            </p>
            <p className="text-sm mt-1 text-gray-400 max-w-xs text-center">
              The research report will appear here as it's being generated
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
