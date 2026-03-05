"use client";

import { useMemo } from "react";
import { Download, FileText, Loader2, BookOpen } from "lucide-react";
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
