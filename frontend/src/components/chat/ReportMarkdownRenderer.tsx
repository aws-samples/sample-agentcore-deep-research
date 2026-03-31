// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0
"use client";

import { useState, useMemo, useEffect, useRef } from "react";
import ReactMarkdown from "react-markdown";
import rehypeRaw from "rehype-raw";
import remarkGfm from "remark-gfm";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { oneLight } from "react-syntax-highlighter/dist/esm/styles/prism";
import {
  Copy,
  Check,
  ChevronDown,
  ChevronRight,
  ExternalLink,
} from "lucide-react";

function completePartialMarkdown(text: string): string {
  const fenceCount = (text.match(/^```/gm) || []).length;
  if (fenceCount % 2 !== 0) return text + "\n```";
  return text;
}

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);
  const handleCopy = () => {
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };
  return (
    <button
      onClick={handleCopy}
      className="p-1 text-muted-foreground hover:text-foreground transition-colors"
      aria-label="Copy code"
    >
      {copied ? <Check size={14} /> : <Copy size={14} />}
    </button>
  );
}

interface Section {
  id: string;
  level: number;
  title: string;
  content: string;
}

// Extract H1 title from markdown content
export function extractReportTitle(content: string): string | null {
  const match = content.match(/^#\s+(.+)$/m);
  return match ? match[1] : null;
}

// Parse markdown into sections based on headings
function parseIntoSections(content: string): Section[] {
  const lines = content.split("\n");
  const sections: Section[] = [];
  let currentSection: Section | null = null;
  let contentLines: string[] = [];

  for (const line of lines) {
    // Check H2 first (more specific) before H1
    const h2Match = line.match(/^##\s+(.+)$/);
    const h1Match = !h2Match ? line.match(/^#\s+(.+)$/) : null;

    if (h1Match || h2Match) {
      // Save previous section
      if (currentSection) {
        currentSection.content = contentLines.join("\n").trim();
        sections.push(currentSection);
      }

      const level = h1Match ? 1 : 2;
      const title = h1Match ? h1Match[1] : h2Match![1];
      const id = title.toLowerCase().replace(/[^a-z0-9]+/g, "-");

      currentSection = { id, level, title, content: "" };
      contentLines = [];
    } else if (currentSection) {
      contentLines.push(line);
    } else {
      // Content before first heading - create intro section
      if (!currentSection) {
        currentSection = { id: "_intro", level: 0, title: "", content: "" };
      }
      contentLines.push(line);
    }
  }

  // Save last section
  if (currentSection) {
    currentSection.content = contentLines.join("\n").trim();
    sections.push(currentSection);
  }

  return sections;
}

// Markdown components for rendering section content
// eslint-disable-next-line @typescript-eslint/no-explicit-any
const markdownComponents: Record<string, any> = {
  // Skip h1 and h2 as they're rendered separately
  h1: () => null,
  h2: () => null,
  h3({ children }: { children?: React.ReactNode }) {
    return (
      <h3 className="text-base font-semibold text-foreground mt-4 mb-2 pl-2 border-l-3 border-blue-300 dark:border-blue-600">
        {children}
      </h3>
    );
  },
  p({ children }: { children?: React.ReactNode }) {
    return <p className="my-2.5 text-foreground leading-relaxed">{children}</p>;
  },
  a({ href, children }: { href?: string; children?: React.ReactNode }) {
    const text = String(children);
    const isAnchor = href?.startsWith("#");
    const isSource = text.startsWith("http") || text.includes("Source");
    // Citation superscript links like [1], [2] — open source URL or scroll to ref
    const isCitation = /^\[\d+\]$/.test(text);
    if (isCitation) {
      return (
        <a
          href={href}
          {...(isAnchor
            ? {}
            : { target: "_blank", rel: "noopener noreferrer" })}
          className="text-blue-600 hover:text-blue-800 no-underline text-[0.75em]"
        >
          {children}
        </a>
      );
    }
    return (
      <a
        href={href}
        target="_blank"
        rel="noopener noreferrer"
        className={`inline-flex items-center gap-1 ${
          isSource
            ? "text-xs px-2 py-0.5 bg-amber-50 dark:bg-amber-900/30 text-amber-700 dark:text-amber-400 rounded-full border border-amber-200 dark:border-amber-800 hover:bg-amber-100 dark:hover:bg-amber-900/50 no-underline"
            : "text-blue-600 hover:text-blue-800 underline decoration-blue-300 hover:decoration-blue-500"
        } transition-colors`}
      >
        {isSource ? (
          <>
            <ExternalLink className="w-3 h-3" />
            <span className="max-w-[200px] truncate">
              {text.replace(/^https?:\/\//, "").split("/")[0]}
            </span>
          </>
        ) : (
          children
        )}
      </a>
    );
  },
  ul({ children }: { children?: React.ReactNode }) {
    return (
      <ul className="my-2.5 pl-5 list-disc space-y-1.5 text-foreground">
        {children}
      </ul>
    );
  },
  ol({ children }: { children?: React.ReactNode }) {
    return (
      <ol className="my-2.5 pl-5 list-decimal space-y-1.5 text-foreground">
        {children}
      </ol>
    );
  },
  li({ children }: { children?: React.ReactNode }) {
    return <li className="leading-relaxed">{children}</li>;
  },
  blockquote({ children }: { children?: React.ReactNode }) {
    return (
      <blockquote className="my-3 pl-4 border-l-4 border-blue-300 dark:border-blue-600 bg-blue-50/50 dark:bg-blue-950/30 py-2 pr-3 rounded-r-lg text-muted-foreground italic">
        {children}
      </blockquote>
    );
  },
  table({ children }: { children?: React.ReactNode }) {
    return (
      <div className="my-4 overflow-x-auto rounded-lg border border-border shadow-sm">
        <table className="min-w-full divide-y divide-border">{children}</table>
      </div>
    );
  },
  thead({ children }: { children?: React.ReactNode }) {
    return <thead className="bg-muted/50">{children}</thead>;
  },
  th({ children }: { children?: React.ReactNode }) {
    return (
      <th className="px-4 py-2.5 text-left text-xs font-semibold text-muted-foreground uppercase tracking-wider">
        {children}
      </th>
    );
  },
  td({ children }: { children?: React.ReactNode }) {
    return (
      <td className="px-4 py-2.5 text-sm text-foreground border-t border-border/50">
        {children}
      </td>
    );
  },
  code({
    className,
    children,
  }: {
    className?: string;
    children?: React.ReactNode;
  }) {
    const match = /language-(\w+)/.exec(className || "");
    const codeString = String(children).replace(/\n$/, "");
    if (match) {
      return (
        <div className="my-3 rounded-lg overflow-hidden border border-border shadow-sm">
          <div className="flex items-center justify-between px-4 py-2 bg-muted border-b border-border">
            <span className="text-xs font-medium text-muted-foreground uppercase tracking-wide">
              {match[1]}
            </span>
            <CopyButton text={codeString} />
          </div>
          <SyntaxHighlighter
            style={oneLight}
            language={match[1]}
            PreTag="div"
            customStyle={{
              margin: 0,
              padding: "1rem",
              fontSize: "0.8rem",
              background: "#fafafa",
            }}
          >
            {codeString}
          </SyntaxHighlighter>
        </div>
      );
    }
    return (
      <code className="px-1.5 py-0.5 bg-muted text-foreground rounded text-[0.85em] font-mono border border-border">
        {children}
      </code>
    );
  },
  pre({ children }: { children?: React.ReactNode }) {
    return <>{children}</>;
  },
  hr() {
    return <hr className="my-6 border-t-2 border-border" />;
  },
};

interface CollapsibleSectionProps {
  section: Section;
  isCollapsed: boolean;
  onToggle: () => void;
  isNew?: boolean;
}

function CollapsibleSection({
  section,
  isCollapsed,
  onToggle,
  isNew = false,
}: CollapsibleSectionProps) {
  const styles =
    section.level === 1
      ? {
          wrapper: "mt-6 mb-4 scroll-mt-4",
          button:
            "w-full flex items-center gap-2 px-3 py-3 rounded-xl bg-gradient-to-r from-blue-100 via-blue-50 to-transparent dark:from-blue-900/40 dark:via-blue-950/20 dark:to-transparent hover:from-blue-200 dark:hover:from-blue-900/60 transition-all text-left group border-b-2 border-blue-200 dark:border-blue-800",
          text: "text-xl font-bold text-foreground group-hover:text-blue-800 dark:group-hover:text-blue-300 transition-colors",
          icon: "w-5 h-5 text-blue-600",
        }
      : {
          wrapper: "mt-5 mb-3 scroll-mt-4",
          button:
            "w-full flex items-center gap-2 px-3 py-2.5 rounded-lg bg-gradient-to-r from-blue-50 via-blue-50/50 to-transparent dark:from-blue-950/30 dark:via-blue-950/15 dark:to-transparent hover:from-blue-100 dark:hover:from-blue-950/50 transition-all text-left group",
          text: "text-lg font-semibold text-foreground group-hover:text-blue-700 dark:group-hover:text-blue-300 transition-colors",
          icon: "w-5 h-5 text-blue-500",
        };

  // Add green highlight ring for new sections
  const newHighlight = isNew
    ? "ring-2 ring-green-400 ring-offset-2 bg-green-50/30"
    : "";

  return (
    <div
      id={section.id}
      className={`${styles.wrapper} ${newHighlight} rounded-xl transition-all duration-500`}
    >
      <button onClick={onToggle} className={styles.button}>
        {isCollapsed ? (
          <ChevronRight className={`${styles.icon} shrink-0`} />
        ) : (
          <ChevronDown className={`${styles.icon} shrink-0`} />
        )}
        <span className={styles.text}>{section.title}</span>
        {isNew && (
          <span className="ml-2 px-2 py-0.5 text-xs font-medium bg-green-100 text-green-700 rounded-full">
            Updated
          </span>
        )}
      </button>
      {!isCollapsed && section.content && (
        <div className="pl-6 pt-2">
          <ReactMarkdown
            remarkPlugins={[remarkGfm]}
            rehypePlugins={[rehypeRaw]}
            components={markdownComponents}
          >
            {completePartialMarkdown(section.content)}
          </ReactMarkdown>
        </div>
      )}
    </div>
  );
}

interface ReportMarkdownRendererProps {
  content: string;
  collapsibleSections?: boolean;
  isLoading?: boolean;
}

export function ReportMarkdownRenderer({
  content,
  collapsibleSections = true,
  isLoading = false,
}: ReportMarkdownRendererProps) {
  const [collapsedIds, setCollapsedIds] = useState<Set<string>>(new Set());
  const [changedSectionIds, setChangedSectionIds] = useState<Set<string>>(
    new Set(),
  );
  const prevContentRef = useRef<string>("");
  const prevSectionsRef = useRef<Map<string, string>>(new Map());

  const sections = useMemo(() => parseIntoSections(content), [content]);

  // Track changes between versions
  useEffect(() => {
    if (content === prevContentRef.current) return;

    const currentSectionsMap = new Map<string, string>();
    const newChangedIds = new Set<string>();

    // Only mark sections as changed if we had previous content (not V1)
    const hasPreviousContent = prevSectionsRef.current.size > 0;

    for (const section of sections) {
      const sectionKey = `${section.level}-${section.id}`;
      currentSectionsMap.set(sectionKey, section.content);

      if (hasPreviousContent) {
        const prevContent = prevSectionsRef.current.get(sectionKey);
        // Mark as changed only if content actually changed (not new sections on V1)
        if (prevContent !== undefined && prevContent !== section.content) {
          if (section.level > 0) {
            // Don't highlight intro section or H1 title
            newChangedIds.add(section.id);
          }
        }
      }
    }

    setChangedSectionIds(newChangedIds);
    prevContentRef.current = content;
    prevSectionsRef.current = currentSectionsMap;
  }, [content, sections]);

  // Clear highlights when loading completes
  useEffect(() => {
    if (!isLoading && changedSectionIds.size > 0) {
      // Clear highlights after a delay when generation completes
      const timer = setTimeout(() => {
        setChangedSectionIds(new Set());
      }, 3000);
      return () => clearTimeout(timer);
    }
  }, [isLoading, changedSectionIds.size]);

  // Extract table of contents from H2 sections only (H1 is used as title in panel header)
  const toc = useMemo(() => sections.filter((s) => s.level === 2), [sections]);

  const toggleSection = (id: string) => {
    setCollapsedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
      }
      return next;
    });
  };

  const scrollToSection = (id: string) => {
    const element = document.getElementById(id);
    if (element) {
      element.scrollIntoView({ behavior: "smooth", block: "start" });
      // Expand section if collapsed
      setCollapsedIds((prev) => {
        const next = new Set(prev);
        next.delete(id);
        return next;
      });
    }
  };

  if (!content) return null;

  return (
    <div className="report-markdown">
      {/* Table of Contents */}
      {toc.length > 3 && (
        <nav className="mb-6 p-4 bg-muted/50 rounded-xl border border-border">
          <h4 className="text-sm font-semibold text-muted-foreground uppercase tracking-wide mb-3">
            Contents
          </h4>
          <ul className="space-y-1.5">
            {toc.map((section) => (
              <li
                key={section.id}
                style={{ paddingLeft: `${(section.level - 1) * 12}px` }}
              >
                <button
                  onClick={() => scrollToSection(section.id)}
                  className="text-sm text-muted-foreground hover:text-blue-600 dark:hover:text-blue-400 transition-colors text-left"
                >
                  {section.title}
                  {changedSectionIds.has(section.id) && (
                    <span className="ml-2 inline-block w-2 h-2 bg-green-500 rounded-full" />
                  )}
                </button>
              </li>
            ))}
          </ul>
        </nav>
      )}

      {/* Render sections */}
      <div className="prose-container">
        {sections.map((section) => {
          // Skip H1 - it's displayed in the panel header
          if (section.level === 1) {
            return null;
          }

          // Intro section (content before first heading)
          if (section.level === 0) {
            return section.content ? (
              <div key={section.id} className="mb-4">
                <ReactMarkdown
                  remarkPlugins={[remarkGfm]}
                  rehypePlugins={[rehypeRaw]}
                  components={markdownComponents}
                >
                  {completePartialMarkdown(section.content)}
                </ReactMarkdown>
              </div>
            ) : null;
          }

          // H2 sections are collapsible
          if (collapsibleSections && section.level === 2) {
            return (
              <CollapsibleSection
                key={section.id}
                section={section}
                isCollapsed={collapsedIds.has(section.id)}
                onToggle={() => toggleSection(section.id)}
                isNew={changedSectionIds.has(section.id)}
              />
            );
          }

          // Non-collapsible fallback for H2
          return (
            <div key={section.id} id={section.id} className="mb-4 scroll-mt-4">
              <h2 className="text-lg font-semibold text-foreground mt-5 mb-3 px-3 py-2 rounded-lg bg-gradient-to-r from-blue-50 to-transparent dark:from-blue-950/30 dark:to-transparent">
                {section.title}
              </h2>
              {section.content && (
                <ReactMarkdown
                  remarkPlugins={[remarkGfm]}
                  rehypePlugins={[rehypeRaw]}
                  components={markdownComponents}
                >
                  {completePartialMarkdown(section.content)}
                </ReactMarkdown>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
