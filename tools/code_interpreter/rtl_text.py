# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Right-to-left (Arabic/Hebrew) text support for matplotlib.

matplotlib < 3.11 draws glyphs one code point at a time, left to right, with
no bidi reordering (and, before 3.10, no shaping). Arabic labels therefore come out as disconnected letters in reverse
order. This module converts a logical string into its visual form (contextual
Arabic presentation forms, lam-alef ligatures, simplified Unicode bidi
reordering) and patches ``matplotlib.text.Text.set_text`` to apply it.

The Code Interpreter sandbox has no ``arabic_reshaper``/``python-bidi`` and no
network access, so this file must stay self-contained (stdlib only). Its source
is sent to the sandbox and executed ahead of the agent's code; see
``execute_python_tool.py``.
"""

import re

# isolated, final, initial, medial. Right-joining letters have no initial/medial.
_FORMS = {
    "ء": ("\ufe80", None, None, None),
    "آ": ("\ufe81", "\ufe82", None, None),
    "أ": ("\ufe83", "\ufe84", None, None),
    "ؤ": ("\ufe85", "\ufe86", None, None),
    "إ": ("\ufe87", "\ufe88", None, None),
    "ئ": ("\ufe89", "\ufe8a", "\ufe8b", "\ufe8c"),
    "ا": ("\ufe8d", "\ufe8e", None, None),
    "ب": ("\ufe8f", "\ufe90", "\ufe91", "\ufe92"),
    "ة": ("\ufe93", "\ufe94", None, None),
    "ت": ("\ufe95", "\ufe96", "\ufe97", "\ufe98"),
    "ث": ("\ufe99", "\ufe9a", "\ufe9b", "\ufe9c"),
    "ج": ("\ufe9d", "\ufe9e", "\ufe9f", "\ufea0"),
    "ح": ("\ufea1", "\ufea2", "\ufea3", "\ufea4"),
    "خ": ("\ufea5", "\ufea6", "\ufea7", "\ufea8"),
    "د": ("\ufea9", "\ufeaa", None, None),
    "ذ": ("\ufeab", "\ufeac", None, None),
    "ر": ("\ufead", "\ufeae", None, None),
    "ز": ("\ufeaf", "\ufeb0", None, None),
    "س": ("\ufeb1", "\ufeb2", "\ufeb3", "\ufeb4"),
    "ش": ("\ufeb5", "\ufeb6", "\ufeb7", "\ufeb8"),
    "ص": ("\ufeb9", "\ufeba", "\ufebb", "\ufebc"),
    "ض": ("\ufebd", "\ufebe", "\ufebf", "\ufec0"),
    "ط": ("\ufec1", "\ufec2", "\ufec3", "\ufec4"),
    "ظ": ("\ufec5", "\ufec6", "\ufec7", "\ufec8"),
    "ع": ("\ufec9", "\ufeca", "\ufecb", "\ufecc"),
    "غ": ("\ufecd", "\ufece", "\ufecf", "\ufed0"),
    "ف": ("\ufed1", "\ufed2", "\ufed3", "\ufed4"),
    "ق": ("\ufed5", "\ufed6", "\ufed7", "\ufed8"),
    "ك": ("\ufed9", "\ufeda", "\ufedb", "\ufedc"),
    "ل": ("\ufedd", "\ufede", "\ufedf", "\ufee0"),
    "م": ("\ufee1", "\ufee2", "\ufee3", "\ufee4"),
    "ن": ("\ufee5", "\ufee6", "\ufee7", "\ufee8"),
    "ه": ("\ufee9", "\ufeea", "\ufeeb", "\ufeec"),
    "و": ("\ufeed", "\ufeee", None, None),
    "ى": ("\ufeef", "\ufef0", None, None),
    "ي": ("\ufef1", "\ufef2", "\ufef3", "\ufef4"),
    # Persian/Urdu letters
    "پ": ("\ufb56", "\ufb57", "\ufb58", "\ufb59"),
    "چ": ("\ufb7a", "\ufb7b", "\ufb7c", "\ufb7d"),
    "ژ": ("\ufb8a", "\ufb8b", None, None),
    "ک": ("\ufb8e", "\ufb8f", "\ufb90", "\ufb91"),
    "گ": ("\ufb92", "\ufb93", "\ufb94", "\ufb95"),
    "ی": ("\ufbfc", "\ufbfd", "\ufbfe", "\ufbff"),
}
# lam + alef variant -> (isolated, final) ligature
_LAM_ALEF = {
    "آ": ("\ufef5", "\ufef6"),
    "أ": ("\ufef7", "\ufef8"),
    "إ": ("\ufef9", "\ufefa"),
    "ا": ("\ufefb", "\ufefc"),
}
_LAM = "\u0644"
_TATWEEL = "\u0640"
_TRANSPARENT = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")

_RTL = re.compile(
    r"[\u0590-\u065F\u066B-\u06EF\u06FA-\u08FF\uFB1D-\uFDFF\uFE70-\uFEFF]"
)
# Arabic letters (bidi class AL) turn following digits into Arabic numbers
_ARABIC_LETTER = re.compile(
    r"[\u0600-\u065F\u066B-\u06EF\u06FA-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]"
)
_DIGIT = re.compile(r"[0-9\u0660-\u0669\u06F0-\u06F9]")
_COMMON_SEPARATORS = set(".,:/")
_EURO_SEPARATORS = set("+-")
_NUM_TERMINATORS = set("%#$\u00b0\u066a\u20ac\u00a3")
_OPEN_BRACKETS = {"(": ")", "[": "]", "{": "}", "\u00ab": "\u00bb"}
_MIRROR = dict(zip("()[]{}<>\u00ab\u00bb", ")(][}{><\u00bb\u00ab"))


def _joins_forward(ch):
    """True when ``ch`` can connect to the letter that follows it."""
    return ch == _TATWEEL or (ch in _FORMS and _FORMS[ch][2] is not None)


def _joins_backward(ch):
    return ch == _TATWEEL or (ch in _FORMS and _FORMS[ch][1] is not None)


def shape_arabic(text):
    """Replace Arabic letters with their contextual presentation forms."""
    chars = list(text)
    out = []
    prev = None  # last non-transparent character emitted
    i = 0
    while i < len(chars):
        ch = chars[i]
        if _TRANSPARENT.match(ch):
            out.append(ch)
            i += 1
            continue
        joins_prev = prev is not None and _joins_forward(prev)
        if ch == _LAM and i + 1 < len(chars) and chars[i + 1] in _LAM_ALEF:
            iso, fin = _LAM_ALEF[chars[i + 1]]
            out.append(fin if joins_prev else iso)
            prev = chars[i + 1]  # alef never joins forward
            i += 2
            continue
        if ch not in _FORMS:
            out.append(ch)
            prev = ch
            i += 1
            continue
        j = i + 1
        while j < len(chars) and _TRANSPARENT.match(chars[j]):
            j += 1
        nxt = chars[j] if j < len(chars) else None
        joins_next = _joins_forward(ch) and nxt is not None and _joins_backward(nxt)
        iso, fin, ini, med = _FORMS[ch]
        if joins_prev and joins_next:
            out.append(med)
        elif joins_prev:
            out.append(fin or iso)
        elif joins_next:
            out.append(ini)
        else:
            out.append(iso)
        prev = ch
        i += 1
    return "".join(out)


def _resolve_types(line, base):
    """Bidi classes after the weak- and neutral-type rules (W1-W7, N1-N2).

    Returns one of "R", "L", "EN" (European number) or "AN" (Arabic number)
    per character, for a paragraph whose direction is ``base`` ("R" or "L").
    """
    types = []
    last_strong = base  # start of line takes the paragraph direction
    for ch in line:
        if _RTL.match(ch):
            last_strong = "AL" if _ARABIC_LETTER.match(ch) else "R"
            types.append("R")
        elif _DIGIT.match(ch):
            # W2: digits after an Arabic letter are Arabic numbers
            types.append("AN" if last_strong == "AL" else "EN")
        elif ch.isalpha():
            last_strong = "L"
            types.append("L")
        else:
            types.append("N")
    n = len(types)
    # W4: a single separator between two numbers of the same kind joins them
    for i in range(1, n - 1):
        if types[i] != "N" or types[i - 1] != types[i + 1]:
            continue
        ch, num = line[i], types[i - 1]
        if (num == "EN" and ch in _COMMON_SEPARATORS | _EURO_SEPARATORS) or (
            num == "AN" and ch in _COMMON_SEPARATORS
        ):
            types[i] = num
    # W5: terminators (%, $, ...) next to European numbers belong to them
    for i in range(n):
        if types[i] == "EN":
            for step in (-1, 1):
                j = i + step
                while 0 <= j < n and types[j] == "N" and line[j] in _NUM_TERMINATORS:
                    types[j] = "EN"
                    j += step
    # W7: European numbers inside Latin text behave as Latin
    last_strong = base
    for i in range(n):
        if types[i] in ("R", "L"):
            last_strong = types[i]
        elif types[i] == "EN" and last_strong == "L":
            types[i] = "L"
    _resolve_brackets(line, types, base)
    # N1/N2: neutrals between same-direction text take it (numbers count as R),
    # otherwise the paragraph direction
    i = 0
    while i < n:
        if types[i] != "N":
            i += 1
            continue
        j = i
        while j < n and types[j] == "N":
            j += 1
        before = types[i - 1] if i > 0 else base
        after = types[j] if j < n else base
        before, after = (("R" if t in ("EN", "AN") else t) for t in (before, after))
        resolved = before if before == after else base
        for k in range(i, j):
            types[k] = resolved
        i = j
    return types


def _strong(t):
    return "R" if t in ("R", "EN", "AN") else t


def _resolve_brackets(line, types, base):
    """N0: a bracket pair takes one direction so both halves mirror together."""
    pairs, stack = [], []
    for i, ch in enumerate(line):
        if ch in _OPEN_BRACKETS:
            stack.append((i, _OPEN_BRACKETS[ch]))
        elif stack and ch == stack[-1][1]:
            pairs.append((stack.pop()[0], i))
    for start, end in sorted(pairs):
        inside = {_strong(t) for t in types[start + 1 : end]} - {"N"}
        if base in inside:
            direction = base
        elif inside:
            # only the opposite direction inside: follow the preceding context
            direction = base
            for t in reversed(types[:start]):
                if t != "N":
                    direction = _strong(t)
                    break
            if direction not in inside:
                direction = base
        else:
            continue
        types[start] = types[end] = direction


def visual_order(line):
    """Reorder one logical line for left-to-right drawing.

    Implements the subset of the Unicode bidi algorithm that chart labels need:
    Latin words and numbers stay left-to-right inside right-to-left text,
    neutrals take the direction of their surroundings, brackets are mirrored.
    The paragraph direction follows the first strong letter (bidi rule P2).
    """
    if not _RTL.search(line):
        return line
    first = next((c for c in line if c.isalpha()), "")
    base = "R" if _RTL.match(first) else "L"
    if base == "R":
        level_of = {"R": 1, "L": 2, "EN": 2, "AN": 2}
    else:
        level_of = {"R": 1, "L": 0, "EN": 2, "AN": 2}
    levels = [level_of[t] for t in _resolve_types(line, base)]
    n = len(line)
    chars = [_MIRROR.get(c, c) if lvl % 2 else c for c, lvl in zip(line, levels)]
    for lvl in (2, 1):
        i = 0
        while i < n:
            if levels[i] < lvl:
                i += 1
                continue
            j = i
            while j < n and levels[j] >= lvl:
                j += 1
            chars[i:j] = chars[i:j][::-1]
            i = j
    return "".join(chars)


def to_display(text):
    """Shape and reorder each line of ``text`` for matplotlib rendering."""
    return "\n".join(visual_order(shape_arabic(line)) for line in text.split("\n"))


def install_matplotlib_patch():
    """Make every matplotlib Text (titles, labels, ticks, legends) RTL-aware.

    No-op on matplotlib >= 3.11, which shapes and reorders RTL text itself;
    converting the text there as well would scramble it again.
    """
    import matplotlib
    import matplotlib.text as mtext

    version = tuple(int(x) for x in re.findall(r"\d+", matplotlib.__version__)[:2])
    if version >= (3, 11) or getattr(mtext.Text.set_text, "_rtl_patched", False):
        return
    original = mtext.Text.set_text

    def set_text(self, s):
        # skip text already converted (e.g. set_text(get_text())) and mathtext
        if (
            isinstance(s, str)
            and s != getattr(self, "_rtl_display", None)
            and _RTL.search(s)
            and s.count("$") < 2
        ):
            s = to_display(s)
            self._rtl_display = s
        return original(self, s)

    set_text._rtl_patched = True
    mtext.Text.set_text = set_text


if __name__ == "_rtl_text":
    install_matplotlib_patch()
