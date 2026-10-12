"""Build an .ass subtitle file with bold word-by-word captions."""

from __future__ import annotations


def _ass_color(hex_color: str) -> str:
    """'#RRGGBB' -> ASS '&H00BBGGRR'."""
    h = hex_color.lstrip("#")
    r, g, b = h[0:2], h[2:4], h[4:6]
    return f"&H00{b}{g}{r}".upper()


def _ts(seconds: float) -> str:
    seconds = max(0.0, seconds)
    cs = int(round(seconds * 100))
    h, cs = divmod(cs, 360000)
    m, cs = divmod(cs, 6000)
    s, cs = divmod(cs, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _escape(text: str) -> str:
    return text.replace("\\", "").replace("{", "(").replace("}", ")").replace("\n", " ")


def clip_words(words: list[dict], start: float, end: float) -> list[dict]:
    """Words inside [start, end], re-timed so 0 = clip start."""
    out = []
    for w in words:
        if w["end"] <= start or w["start"] >= end:
            continue
        out.append({"word": w["word"], "start": max(0.0, w["start"] - start), "end": min(end, w["end"]) - start})
    return out


def build_ass(words: list[dict], style: dict, width: int, height: int) -> str:
    """Words are shown `words_per_line` at a time; the word being spoken is highlighted."""
    primary = _ass_color(style["primary_color"])
    highlight = _ass_color(style["highlight_color"])
    outline_c = _ass_color(style["outline_color"])
    bold = -1 if style.get("bold", True) else 0
    margin_v = int(height * (1 - float(style["position"])))
    margin_h = int(width * 0.06)

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Caption,{style['font']},{style['font_size']},{primary},{primary},{outline_c},&H80000000,{bold},0,0,0,100,100,0,0,1,{style['outline']},{style['shadow']},2,{margin_h},{margin_h},{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    per_line = max(1, int(style["words_per_line"]))
    upper = style.get("uppercase", True)
    lines = []
    for i in range(0, len(words), per_line):
        group = words[i : i + per_line]
        texts = [_escape(w["word"].upper() if upper else w["word"]) for w in group]
        for j, w in enumerate(group):
            # Keep each word on screen until the next one starts, so captions don't flicker.
            if j + 1 < len(group):
                w_end = group[j + 1]["start"]
            elif i + per_line < len(words):
                w_end = min(words[i + per_line]["start"], w["end"] + 0.6)
            else:
                w_end = w["end"] + 0.3
            if w_end <= w["start"]:
                w_end = w["start"] + 0.05
            parts = [
                (f"{{\\c{highlight}\\fscx108\\fscy108}}{t}{{\\c{primary}\\fscx100\\fscy100}}" if k == j else t)
                for k, t in enumerate(texts)
            ]
            lines.append(f"Dialogue: 0,{_ts(w['start'])},{_ts(w_end)},Caption,,0,0,0,,{' '.join(parts)}")
    return header + "\n".join(lines) + "\n"
