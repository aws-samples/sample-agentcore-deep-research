// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0
import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

const RTL_CHARS = /[\u0590-\u08FF\uFB1D-\uFDFF\uFE70-\uFEFF]/g;
const LTR_CHARS = /[A-Za-z\u00C0-\u024F]/g;

// Base direction for a block of (markdown) text: RTL when Hebrew/Arabic letters
// outnumber Latin ones. URLs are ignored so citations don't skew the count.
export function detectTextDirection(text: string): "rtl" | "ltr" {
  const stripped = text.replace(/https?:\/\/\S+/g, "");
  const rtl = stripped.match(RTL_CHARS)?.length ?? 0;
  const ltr = stripped.match(LTR_CHARS)?.length ?? 0;
  return rtl > ltr ? "rtl" : "ltr";
}
