# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import re
from html import escape

import markdown
import weasyprint

_CSS = """
@page {
    size: A4;
    margin: 2cm 2.5cm;
    @bottom-center {
        content: counter(page);
        font-size: 8pt;
        color: #888;
    }
}
body {
    font-family: 'DejaVu Sans', sans-serif;
    font-size: 10pt;
    line-height: 1.55;
    color: #222;
    text-align: justify;
    hyphens: auto;
}
h1 {
    font-size: 22pt;
    margin-top: 0;
    margin-bottom: 0.4em;
    border-bottom: 2px solid #333;
    padding-bottom: 0.2em;
    text-align: left;
}
h2 {
    font-size: 16pt;
    margin-top: 1.2em;
    margin-bottom: 0.4em;
    border-bottom: 1px solid #ccc;
    padding-bottom: 0.15em;
    text-align: left;
}
h3 { font-size: 13pt; margin-top: 1em; margin-bottom: 0.3em; text-align: left; }
h4 { font-size: 11pt; margin-top: 0.8em; margin-bottom: 0.3em; text-align: left; }
p { margin: 0.4em 0; }
ul, ol { margin: 0.4em 0; padding-left: 1.5em; }
li { margin: 0.15em 0; }
table {
    border-collapse: collapse;
    width: 100%;
    margin: 0.8em 0;
    page-break-inside: avoid;
}
th, td {
    border: 1px solid #ccc;
    padding: 5px 8px;
    text-align: left;
    font-size: 9pt;
}
th { background: #f0f0f0; font-weight: bold; }
tr:nth-child(even) { background: #fafafa; }
code {
    font-family: monospace;
    font-size: 9pt;
    background: #f4f4f4;
    padding: 1px 3px;
    border-radius: 3px;
}
pre {
    background: #f4f4f4;
    padding: 8px;
    border-radius: 4px;
    overflow-x: auto;
    font-size: 8pt;
    line-height: 1.4;
    page-break-inside: avoid;
}
pre code { background: none; padding: 0; }
img {
    max-width: 100%;
    height: auto;
    display: block;
    margin: 0.8em auto;
    page-break-inside: avoid;
}
blockquote {
    border-left: 3px solid #ccc;
    margin: 0.4em 0;
    padding: 0.2em 0.8em;
    color: #555;
}
a { color: #1a5ca8; text-decoration: none; }
a:hover { text-decoration: underline; }
sup.citation a {
    font-size: 8pt;
    color: #1a5ca8;
    text-decoration: none;
}
.references {
    margin-top: 1.5em;
    padding-top: 0.5em;
    border-top: 1px solid #ccc;
}
.references h2 { border-bottom: none; font-size: 14pt; }
.references ol { font-size: 9pt; padding-left: 1.5em; }
.references li { margin: 0.3em 0; word-break: break-all; }
"""

_MD_EXTENSIONS = ["tables", "fenced_code", "toc", "nl2br"]

# matches [Source: ...] patterns (URL or plain text)
_SOURCE_RE = re.compile(r"\[Source:\s*([^\]]+)\]")
_URL_RE = re.compile(r"^https?://\S+$")


def _process_citations(html_body: str) -> str:
    """
    Replace inline [Source: ...] with superscript numbered citations
    and append a References section at the end.
    """
    sources: list[str] = []
    source_index: dict[str, int] = {}

    def _replace(match: re.Match) -> str:
        raw = match.group(1).strip()
        if raw not in source_index:
            source_index[raw] = len(sources) + 1
            sources.append(raw)
        idx = source_index[raw]
        return f'<sup class="citation"><a href="#ref-{idx}">[{idx}]</a></sup>'

    html_body = _SOURCE_RE.sub(_replace, html_body)

    if sources:
        refs_html = '<div class="references"><h2>References</h2><ol>'
        for i, src in enumerate(sources, 1):
            safe = escape(src, quote=True)
            if _URL_RE.match(src):
                refs_html += f'<li id="ref-{i}"><a href="{safe}">{safe}</a></li>'
            else:
                refs_html += f'<li id="ref-{i}">{safe}</li>'
        refs_html += "</ol></div>"
        html_body += refs_html

    return html_body


def generate_pdf(markdown_content: str) -> bytes:
    """
    Convert markdown content to a styled PDF document.

    Parameters
    ----------
    markdown_content : str
        Markdown text, may include image URLs and tables.

    Returns
    -------
    bytes
        PDF file content.
    """
    html_body = markdown.markdown(markdown_content, extensions=_MD_EXTENSIONS)
    html_body = _process_citations(html_body)
    html_doc = (
        "<!DOCTYPE html>"
        '<html><head><meta charset="utf-8">'
        f"<style>{_CSS}</style></head>"
        f"<body>{html_body}</body></html>"
    )
    return weasyprint.HTML(string=html_doc).write_pdf()
