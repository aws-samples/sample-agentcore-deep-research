// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

import { describe, it, expect } from "vitest";
import { parseIntoSections } from "../components/chat/ReportMarkdownRenderer";
import { detectTextDirection } from "../lib/utils";

const ARABIC_SKELETON = `# مستقبل الذكاء الاصطناعي

## الملخص التنفيذي
[placeholder]

## الخلفية
[placeholder]

## أهم النتائج

### النتيجة 1: المسار التقني
[placeholder]

## التحليل
[placeholder]`;

describe("Report section parsing", () => {
  it("gives Arabic headings distinct ids", () => {
    const ids = parseIntoSections(ARABIC_SKELETON).map((s) => s.id);
    expect(new Set(ids).size).toBe(ids.length);
    expect(ids).toContain("الملخص-التنفيذي");
  });

  it("keeps section ids stable when a placeholder is filled in", () => {
    const before = parseIntoSections(ARABIC_SKELETON).map((s) => s.id);
    const filled = ARABIC_SKELETON.replace(
      "## الخلفية\n[placeholder]",
      "## الخلفية\nنص جديد للقسم.",
    );
    const after = parseIntoSections(filled);
    expect(after.map((s) => s.id)).toEqual(before);
    expect(after.find((s) => s.title === "الخلفية")?.content).toBe(
      "نص جديد للقسم.",
    );
  });

  it("disambiguates repeated headings", () => {
    const ids = parseIntoSections("## Notes\na\n## Notes\nb").map((s) => s.id);
    expect(ids).toEqual(["notes", "notes-2"]);
  });

  it("keeps the existing slug for English headings", () => {
    const [section] = parseIntoSections("## Executive Summary\ntext");
    expect(section.id).toBe("executive-summary");
  });
});

describe("detectTextDirection", () => {
  it("detects Arabic text with embedded English terms as rtl", () => {
    expect(
      detectTextDirection(
        "تتوقع شركة McKinsey أن يضيف الذكاء الاصطناعي (AGI) قيمة كبيرة [Source: https://www.mckinsey.com/featured-insights]",
      ),
    ).toBe("rtl");
  });

  it("detects English text as ltr", () => {
    expect(detectTextDirection("The market grew by 15% in 2025.")).toBe("ltr");
    expect(detectTextDirection("")).toBe("ltr");
  });
});
