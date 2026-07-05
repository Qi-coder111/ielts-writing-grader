from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
import ssl
from dataclasses import dataclass
from typing import Any

from flask import Flask, jsonify, render_template, request
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

app = Flask(__name__)


def clamp(n: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, n))


def round_to_half(n: float) -> float:
    return round(n * 2) / 2


def safe_strip(s: str) -> str:
    return (s or "").strip()


def tokenize_words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text.lower())


def split_sentences(text: str) -> list[str]:
    t = safe_strip(text)
    if not t:
        return []
    t = re.sub(r"\s+", " ", t)
    sentences: list[str] = []
    start = 0
    for m in re.finditer(r"[.!?]+", t):
        end = m.end()
        chunk = t[start:end].strip()
        if chunk:
            sentences.append(chunk)
        start = end
    tail = t[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences


def detect_question_type(title: str) -> str:
    t = title.lower()
    if any(x in t for x in ["advantages", "disadvantages", "benefits", "drawbacks", "pros and cons"]):
        return "advantages/disadvantages"
    if any(x in t for x in ["discuss both views", "both views", "discuss"]):
        return "discussion"
    if any(x in t for x in ["agree", "disagree", "to what extent", "extent do you agree"]):
        return "agree/disagree"
    return "agree/disagree"


def _http_post_json(url: str, headers: dict[str, str], payload: dict[str, Any], timeout_s: int = 180) -> dict[str, Any]:  # 超时从60→180秒
    # 新增：创建不验证SSL的上下文（解决Mac SSL证书问题）
    context = ssl._create_unverified_context()
    
    # 原有逻辑：序列化payload并构造请求
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    for k, v in headers.items():
        req.add_header(k, v)
    
    try:
        # 关键修改：添加context=context + 增大超时时间
        with urllib.request.urlopen(req, timeout=timeout_s, context=context) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            return json.loads(raw)
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else ""
        raise RuntimeError(f"Upstream HTTP {getattr(e, 'code', 'error')}: {raw[:4000]}")
    except Exception as e:
        raise RuntimeError(f"Upstream request failed: {e}")


def _extract_json_object(content: str) -> dict | None:
    # 提取JSON格式的内容（适配AI可能返回的多余文本和HTML特殊字符）
    pattern = r"\{[\s\S]*\}"
    match = re.search(pattern, content)
    if not match:
        return None
    try:
        cleaned_content = match.group().replace(r"\n", "").replace(r"\t", "")
        return json.loads(cleaned_content)
    except json.JSONDecodeError:
        return None


def call_openai(messages: list[dict[str, str]], model: str | None = None, temperature: float = 0.3, max_tokens: int = 1800) -> dict[str, Any]:
    api_key = os.getenv("SILICONFLOW_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
    if not api_key:
        raise RuntimeError("Missing SILICONFLOW_API_KEY (or OPENAI_API_KEY)")

    base_url = os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1").rstrip("/")
    chosen_model = model or os.getenv("SILICONFLOW_MODEL", "Qwen/Qwen2.5-72B-Instruct")

    url = f"{base_url}/chat/completions"
    payload = {
        "model": chosen_model,
        "messages": messages,
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
        "stream": False,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    return _http_post_json(url, headers, payload, timeout_s=90)


def _llm_text(messages: list[dict[str, str]], model: str | None = None, temperature: float = 0.3, max_tokens: int = 1800) -> str:
    data = call_openai(messages=messages, model=model, temperature=temperature, max_tokens=max_tokens)
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError("Upstream returned no choices")
    msg = (choices[0] or {}).get("message") or {}
    content = msg.get("content")
    if not isinstance(content, str):
        raise RuntimeError("Upstream returned no content")
    return content


def llm_grade_only(title: str, text: str, stats: dict[str, Any]) -> dict[str, Any] | None:
    system = (
        "You are a strict IELTS Writing Task 2 examiner and professional English editor. "
        "Return ONLY a JSON object with keys: overall, criteria, revised_essay, revision_notes. "
        "overall must be 0-9 in 0.5 steps. "
        "criteria must include exactly these keys: "
        "Task Response, Coherence & Cohesion, Lexical Resource, Grammatical Range & Accuracy. "
        "Each criterion value must be an object: {band: number, reason: string}. "
        "Reasons must be concrete, IELTS-style, and avoid generic praise. "
        "revised_essay must be an HTML string. Replace the current annotation behavior with these exact rules: "
        "LAYER PRIORITY (high -> low): "
        "1) Spelling correction -> red; "
        "2) Grammar completion -> purple (append-only); "
        "3) Redundancy deletion -> gray strikethrough; "
        "4) Collocation fix -> orange. "
        "One word = one layer only. Higher priority wins. "
        "LAYER 1 — Spelling (red): "
        "Trigger: misspelled word. "
        "HTML: <span style=\"color:#E53935;\">{misspelled}</span> ({correct}). "
        "Keep original word intact. Add exactly 1 half-width space before the parenthesis. "
        "Correct form must be inside parentheses in black. No explanation in parentheses. "
        "LAYER 2 — Grammar completion (purple, append only): "
        "Trigger: missing 3rd-person -s / missing article a|an|the / any fix solvable by adding characters. "
        "HTML: {original}<span style=\"color:#7C4DFF;margin-left:0.2em;\">{missing}</span>. "
        "Append missing content AFTER original word, never modify original. "
        "This covers ALL add-characters fixes; do NOT use orange for these. "
        "LAYER 3 — Redundancy (gray strikethrough): "
        "Trigger: extra article / extra -s / duplicate word / redundant expression. "
        "HTML: <span style=\"color:#9E9E9E;text-decoration:line-through;\">{word}</span>. "
        "Word remains in place with strikethrough only. "
        "LAYER 4 — Collocation (orange): "
        "Trigger: unnatural phrasing that CANNOT be fixed by adding characters (example: make a photo -> take a photo). "
        "HTML: <span style=\"background:#FFF3E0;color:#E65100;\">{original}</span><span style=\"color:#E65100;font-size:0.85em;\"> ({suggestion})</span>. "
        "Use ONLY when grammar is correct but phrasing is non-native. "
        "If the fix only needs added characters, use Layer 2 instead. "
        "FORMATTING RULES: "
        "All annotations must stay on the same line as original word. No line breaks inside annotations. "
        "Exactly 1 half-width space before parentheses (Layers 1 and 4). "
        "ASCII parentheses only: (). "
        "No extra spaces and no explanatory text anywhere in output. "
        "CONFLICT RESOLUTION: "
        "If a word has multiple issues, apply only the highest-priority layer. "
        "Example: misspelled and missing -s -> Layer 1 only. "
        "revision_notes MUST be a JSON array of objects each with: "
        "position, type (spelling/redundant/missing/collocation), original, action, timestamp (ISO string). "
    )
    import datetime
    user = json.dumps(
        {
            "title": title,
            "essay": text,
            "stats_hint": stats,
            "current_time": datetime.datetime.now().isoformat()
        },
        ensure_ascii=False,
    )
    content = _llm_text(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
        max_tokens=2500,  # 增大token上限，容纳修改内容
    )
    obj = _extract_json_object(content)
    if not obj:
        return None
    required_keys = ["overall", "criteria", "revised_essay", "revision_notes"]
    if not all(key in obj for key in required_keys):
        return None
    return obj


# 示例接口调用逻辑（补充到你的代码中）
def api_grade():
    # 获取前端传入的参数
    title = request.json.get("title", "")
    text = request.json.get("text", "")
    stats = request.json.get("stats", {})
    
    # 调用评分函数
    grading_result = llm_grade_only(title, text, stats)
    if not grading_result:
        return jsonify({"error": "Grading failed"}), 500
    
    # 返回包含修改内容的结果
    return jsonify({
        "overall": grading_result["overall"],
        "criteria": grading_result["criteria"],
        "revised_essay": grading_result["revised_essay"],  # 带格式的修改原文
        "revision_notes": grading_result["revision_notes"]  # 修改说明
    })

def llm_recommendations(title: str) -> dict[str, Any] | None:
    system = (
        "You generate IELTS Task 2 high-band language suggestions based on the topic. "
        "Return ONLY JSON with keys: advanced_vocabulary (array 10-15), collocations (array 10-15), high_band_structures (array 6-10). "
        "CRITICAL RULES FOR 'high_band_structures':\n"
        "1. MUST be TEMPLATIZED skeleton sentences (e.g., 'While it is undeniable that ..., ... is a complex issue.').\n"
        "2. REMOVE all specific topic vocabulary, nouns, data, or concrete examples.\n"
        "3. Use placeholders like '...', 'X', 'Y' for the missing parts.\n"
        "4. Preserve the grammatical structure, conjunctions, and transition words (e.g., concession, cause-effect, exemplification).\n"
        "5. The templates MUST be applicable to ANY IELTS Task 2 essay topic."
    )
    user = json.dumps({"title": title}, ensure_ascii=False)
    content = _llm_text(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.4,
        max_tokens=900,
    )
    obj = _extract_json_object(content)
    if not obj:
        return None
    if not isinstance(obj.get("advanced_vocabulary"), list):
        return None
    if not isinstance(obj.get("collocations"), list):
        return None
    if not isinstance(obj.get("high_band_structures"), list):
        return None
    return obj


def llm_model_essay(title: str) -> str | None:
    system = (
        "Write an IELTS Writing Task 2 Band 7+ essay. "
        "Structure must be: introduction, body paragraph 1, body paragraph 2, conclusion. "
        "Write in clear academic English. No bullet points."
    )
    user = json.dumps({"title": title}, ensure_ascii=False)
    content = _llm_text(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.5,
        max_tokens=1400,
    )
    out = (content or "").strip()
    return out or None


def llm_writing_ideas(title: str) -> dict[str, Any] | None:
    system = (
        "You are an IELTS Task 2 writing coach. "
        "Detect question_type from: agree/disagree, advantages/disadvantages, discussion. "
        "Return ONLY JSON with keys: question_type, idea_1, idea_2. "
        "idea_1 label must be 思路1（部分同意） and idea_2 label must be 思路2（完全同意）. "
        "Each idea must include position (string) and body (array of at least 2 paragraphs). "
        "Each paragraph is {topic_sentence: string, points: array of exactly 3 concrete points}. "
        "Points must be specific, not vague."
    )
    user = json.dumps({"title": title}, ensure_ascii=False)
    content = _llm_text(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.35,
        max_tokens=1400,
    )
    obj = _extract_json_object(content)
    if not obj:
        return None
    if "question_type" not in obj or "idea_1" not in obj or "idea_2" not in obj:
        return None
    return obj


@dataclass
class ErrorItem:
    type: str
    position: tuple[int, int]
    suggestion: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "position": [int(self.position[0]), int(self.position[1])],
            "suggestion": self.suggestion,
        }


COMMON_MISSPELLINGS: dict[str, str] = {
    "becuase": "because",
    "definately": "definitely",
    "enviroment": "environment",
    "goverment": "government",
    "technologie": "technology",
    "teh": "the",
    "recieve": "receive",
    "wich": "which",
    "alot": "a lot",
}


def _apply_spans_replacements(text: str, replacements: list[tuple[int, int, str]]) -> str:
    out = text
    for s, e, rep in sorted(replacements, key=lambda x: (x[0], x[1]), reverse=True):
        out = out[:s] + rep + out[e:]
    return out


def llm_analyze_sentence(sentence: str) -> dict[str, Any]:
    system = (
        "You are an expert IELTS English teacher. Analyze the given sentence for ANY grammar, spelling, "
        "punctuation, collocation, or awkward phrasing errors.\n"
        "Return ONLY a JSON object with this exact structure:\n"
        "{\n"
        "  \"original_sentence\": \"The exact original sentence with HTML tags wrapping errors.\",\n"
        "  \"corrected_sentence\": \"The fully corrected sentence with HTML tags for corrections.\",\n"
        "  \"errors\": []\n"
        "}\n"
        "CRITICAL RULES for 'original_sentence':\n"
        "1. Identify EVERY error (e.g. 3rd person singular '-s', subject-verb agreement, tense).\n"
        "2. Classification priority (IMPORTANT):\n"
        "   - If the correction is achievable by APPENDING characters (additions) to existing adjacent text (examples: missing 3rd-person '-s'/'-es', missing article a/an/the inserted between words, missing end punctuation '.', '!' or '?'), then USE ONLY <span class=\"err-missing\" data-missing=\"...\">... </span>.\n"
        "   - Use <span class=\"err-collocation\">... </span> ONLY when the error is an UNNATURAL collocation / inappropriate phrase that requires REPLACING one or more words/structure (e.g. 'make a photo' -> 'take a photo').\n"
        "   - Do NOT use err-collocation for append-only fixes.\n"
        "3. Wrap the incorrect word/phrase in the original sentence with exactly one of these HTML tags based on the final decision:\n"
        "   - Spelling/Redundant article: <span class=\"err-spelling\">wrong_word</span>\n"
        "   - Missing/appendable characters (e.g. missing 's' or article inserted between words): <span class=\"err-missing\" data-missing=\"...\">word</span>. data-missing must contain ONLY the missing characters to be appended.\n"
        "   - Other grammar error: <span class=\"err-grammar\">wrong_grammar</span>\n"
        "   - Unnatural collocation: <span class=\"err-collocation\">wrong_collocation</span>\n"
        "3. Do NOT change any other words in the original sentence.\n"
        "CRITICAL RULES for 'corrected_sentence':\n"
        "1. Provide the fully fixed sentence.\n"
        "2. Use <span class=\"err-delete\">deleted_word</span> for words that should be removed.\n"
        "3. Use <span class=\"corr-text\">corrected_text</span> for added pieces (insertions) or changed pieces.\n"
        "4. For append-only fixes (err-missing), do not delete/replace the original token; only INSERT the missing characters and wrap ONLY the inserted characters in corr-text.\n"
        "5. Example (appendable -s): original: 'He sustain his work.'; original_sentence: 'He <span class=\"err-missing\" data-missing=\"s\">sustain</span> his work.'; corrected_sentence should contain only an appended <span class=\"corr-text\">s</span>.\n"
        "6. Example (collocation replacement): 'make a photo' -> 'take a photo'. corrected_sentence should delete/replace as needed.\n"
        "7. Fix all the errors you highlighted in the original sentence."
    )
    user = json.dumps({"sentence": sentence}, ensure_ascii=False)
    content = _llm_text(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.1,
        max_tokens=800,
    )
    obj = _extract_json_object(content)
    if not obj or "original_sentence" not in obj or "corrected_sentence" not in obj:
        # Fallback to rule-based if LLM fails
        return analyze_sentence_rules(sentence)
    
    # Ensure errors is a list
    if "errors" not in obj or not isinstance(obj["errors"], list):
        obj["errors"] = []
        
    return obj


def _strip_html_tags(s: str) -> str:
    return re.sub(r"<[^>]+>", "", s or "")


def fix_appendable_errors_tagging(original_sentence_html: str, corrected_sentence_html: str) -> str:
    """
    Post-process LLM output to prevent orange/collocation misuse for append-only fixes.

    In particular, for third-person singular omissions in the form:
      he/she/it + <base verb>
    if the corrected sentence contains <base verb> + ('s' or 'es'), we re-tag the LLM's
    err-collocation span as err-missing.
    """

    plain_original = _strip_html_tags(original_sentence_html).strip()
    plain_corrected = _strip_html_tags(corrected_sentence_html).strip()

    def replacement_for_match(m: re.Match[str]) -> str:
        inner = (m.group(1) or "").strip()
        # Only handle single-word verbs; collocations often contain spaces.
        if not re.fullmatch(r"[A-Za-z]+(?:'[A-Za-z]+)?", inner):
            return m.group(0)

        # Determine whether corrected added 's' or 'es' to this verb.
        missing = None
        if re.search(rf"\b{re.escape(inner)}es\b", plain_corrected, flags=re.IGNORECASE):
            missing = "es"
        elif re.search(rf"\b{re.escape(inner)}s\b", plain_corrected, flags=re.IGNORECASE):
            missing = "s"
        else:
            return m.group(0)

        # Ensure this is in a third-person context in the original.
        if not re.search(
            rf"\b(he|she|it)\s+{re.escape(inner)}\b",
            plain_original,
            flags=re.IGNORECASE,
        ):
            return m.group(0)

        # Rewrite only the tag/class; keep the original inner token text unchanged.
        return f'<span class="err-missing" data-missing="{missing}">{inner}</span>'

    # Replace mis-tagged collocations when they actually behave like appendable suffix fixes.
    return re.sub(
        r'<span class="err-collocation">([^<]+)</span>',
        replacement_for_match,
        original_sentence_html,
    )


def fix_corrected_append_only_spans(original_sentence_html: str, corrected_sentence_html: str) -> str:
    """
    If appendable fixes were re-tagged as err-missing in original_sentence_html,
    try to ensure corrected_sentence_html highlights only the inserted suffix
    (e.g. corr-text should wrap only 's'/'es', not the whole corrected verb).
    """

    missing_spans = re.findall(
        r'<span class="err-missing"\s+data-missing="([^"]+)">([^<]+)</span>',
        original_sentence_html or "",
    )
    if not missing_spans:
        return corrected_sentence_html

    fixed = corrected_sentence_html or ""

    for missing, inner in missing_spans:
        missing = missing or ""
        inner = inner or ""
        if not missing or not inner:
            continue

        # If LLM wrapped the whole corrected token in corr-text, split it so only the
        # inserted suffix remains purple.
        # Example: <span class="corr-text">sustain+s</span> -> sustain<span class="corr-text">s</span>
        pattern = re.compile(rf'<span class="corr-text">([^<]+)</span>')

        def repl(m: re.Match[str]) -> str:
            content = m.group(1) or ""
            if content.lower() != (inner + missing).lower():
                return m.group(0)
            base = content[: -len(missing)] if len(missing) else content
            return f"{base}<span class=\"corr-text\">{missing}</span>"

        fixed = pattern.sub(repl, fixed)

    return fixed

def analyze_sentence_rules(sentence: str) -> dict[str, Any]:
    s = sentence
    errors: list[ErrorItem] = []
    replacements: list[tuple[int, int, str]] = []
    lower = s.lower()

    for wrong, right in COMMON_MISSPELLINGS.items():
        for m in re.finditer(rf"\b{re.escape(wrong)}\b", lower):
            errors.append(ErrorItem("spelling", (m.start(), m.end()), right))
            replacements.append((m.start(), m.end(), right))

    for m in re.finditer(r"\bdo a decision\b", lower):
        errors.append(ErrorItem("collocation", (m.start(), m.end()), "make a decision"))
        replacements.append((m.start(), m.end(), "make a decision"))

    for m in re.finditer(r"\ban university\b", lower):
        errors.append(ErrorItem("grammar", (m.start(), m.end()), "a university"))
        replacements.append((m.start(), m.end(), "a university"))

    for m in re.finditer(r"\b(a|an)\s+([a-z]+)\b", lower):
        art = m.group(1)
        word = m.group(2)
        if word.startswith(("a", "e", "i", "o")) and art == "a":
            errors.append(ErrorItem("grammar", (m.start(), m.start() + 1), "an"))
            replacements.append((m.start(), m.start() + 1, "an"))
        if word.startswith(("u",)) and art == "an":
            errors.append(ErrorItem("grammar", (m.start(), m.start() + 2), "a"))
            replacements.append((m.start(), m.start() + 2, "a"))
        if word.startswith(("b", "c", "d", "f", "g", "h", "j", "k", "l", "m", "n", "p", "q", "r", "s", "t", "v", "w", "x", "y", "z")) and art == "an":
            errors.append(ErrorItem("grammar", (m.start(), m.start() + 2), "a"))
            replacements.append((m.start(), m.start() + 2, "a"))

    for m in re.finditer(r"\b(many|several|various|numerous)\s+([a-z]+)\b", lower):
        noun = m.group(2)
        if not noun.endswith("s") and noun not in {"people", "children", "police"}:
            insert_pos = m.start(2) + len(noun)
            errors.append(ErrorItem("omission", (insert_pos, insert_pos), "s"))
            replacements.append((insert_pos, insert_pos, "s"))

    for m in re.finditer(r"\b(he|she|it)\s+(go|do|have|work|think|make|take|play|want|need|say)\b", lower):
        verb = m.group(2)
        if verb == "have":
            fixed = "has"
        elif verb in {"go", "do"}:
            fixed = verb + "es"
        else:
            fixed = verb + "s"
        errors.append(ErrorItem("grammar", (m.start(2), m.end(2)), fixed))
        replacements.append((m.start(2), m.end(2), fixed))

    if s and s[-1] not in ".!?":
        errors.append(ErrorItem("grammar", (len(s), len(s)), "."))
        replacements.append((len(s), len(s), "."))

    corrected = _apply_spans_replacements(s, replacements)

    return {
        "original_sentence": s,
        "corrected_sentence": corrected,
        "errors": [e.to_dict() for e in errors],
    }


def analyze_sentence(sentence: str) -> dict[str, Any]:
    # Check if LLM is enabled
    use_llm = bool(os.getenv("SILICONFLOW_API_KEY") or os.getenv("OPENAI_API_KEY"))
    if use_llm:
        obj = llm_analyze_sentence(sentence)
        if obj and isinstance(obj.get("original_sentence"), str) and isinstance(obj.get("corrected_sentence"), str):
            fixed_original = fix_appendable_errors_tagging(
                obj["original_sentence"], obj["corrected_sentence"]
            )
            obj["original_sentence"] = fixed_original
            obj["corrected_sentence"] = fix_corrected_append_only_spans(
                fixed_original, obj["corrected_sentence"]
            )
        return obj
    return analyze_sentence_rules(sentence)

def _count_markers(text: str, markers: list[str]) -> int:
    t = text.lower()
    return sum(1 for m in markers if m in t)


def grade_essay(title: str, text: str) -> dict[str, Any]:
    words = tokenize_words(text)
    sentences = split_sentences(text)
    wc = len(words)
    sc = max(1, len(sentences))
    unique_ratio = (len(set(words)) / max(1, wc)) if wc else 0.0
    avg_sent_len = wc / sc
    cohesion_markers = [
        "however",
        "therefore",
        "moreover",
        "in addition",
        "for example",
        "for instance",
        "as a result",
        "on the other hand",
        "in conclusion",
        "overall",
    ]
    marker_hits = _count_markers(text, cohesion_markers)

    structure_hits = _count_markers(
        text,
        ["in conclusion", "overall", "firstly", "secondly", "finally", "to begin with", "in summary"],
    )

    grammar_proxy = clamp(1.0 - abs(avg_sent_len - 18) / 30, 0.0, 1.0)
    lexical_proxy = clamp((unique_ratio - 0.38) / 0.22, 0.0, 1.0)
    cohesion_proxy = clamp(marker_hits / 6, 0.0, 1.0)
    task_proxy = clamp((wc - 170) / 150, 0.0, 1.0)

    tr = 4.5 + 3.5 * (0.65 * task_proxy + 0.35 * (structure_hits > 0))
    cc = 4.5 + 3.5 * (0.55 * cohesion_proxy + 0.45 * (structure_hits > 0))
    lr = 4.5 + 3.5 * lexical_proxy
    gra = 4.5 + 3.5 * grammar_proxy

    tr = round_to_half(clamp(tr, 0, 9))
    cc = round_to_half(clamp(cc, 0, 9))
    lr = round_to_half(clamp(lr, 0, 9))
    gra = round_to_half(clamp(gra, 0, 9))
    overall = round_to_half(clamp((tr + cc + lr + gra) / 4, 0, 9))

    def reason_tr() -> str:
        parts = []
        if wc < 180:
            parts.append("Ideas are relevant but under-developed due to limited extension and examples.")
        else:
            parts.append("Addresses the task with a clear position and mostly relevant support.")
        if structure_hits == 0:
            parts.append("Paragraphing and progression are weak, which reduces clarity of argument.")
        else:
            parts.append("Uses paragraphing to organise main points and support.")
        if "because" not in text.lower() and marker_hits < 2:
            parts.append("Support tends to be general; add specific examples and explanations of why they matter.")
        return " ".join(parts)

    def reason_cc() -> str:
        parts = []
        if marker_hits <= 1:
            parts.append("Linking is limited; transitions between ideas can feel abrupt.")
        else:
            parts.append("A range of cohesive devices is used with generally appropriate referencing.")
        if sc <= 6:
            parts.append("Sentence control is okay but paragraph-level flow could be expanded with clearer topic sentences.")
        else:
            parts.append("Information is sequenced with a generally logical progression.")
        return " ".join(parts)

    def reason_lr() -> str:
        parts = []
        if unique_ratio < 0.45:
            parts.append("Vocabulary is adequate but shows repetition and relies on safe, common wording.")
        else:
            parts.append("Shows a flexible range with some less common vocabulary used appropriately.")
        parts.append("Improve precision by using topic-specific nouns and stronger verb-noun collocations.")
        return " ".join(parts)

    def reason_gra() -> str:
        parts = []
        if grammar_proxy < 0.5:
            parts.append("Frequent errors in agreement, articles and punctuation reduce accuracy and clarity.")
        else:
            parts.append("Mixes simple and some complex structures with generally controlled punctuation.")
        parts.append("To push higher, vary clause types and reduce slips in verb forms and articles.")
        return " ".join(parts)

    return {
        "overall": overall,
        "criteria": {
            "Task Response": {"band": tr, "reason": reason_tr()},
            "Coherence & Cohesion": {"band": cc, "reason": reason_cc()},
            "Lexical Resource": {"band": lr, "reason": reason_lr()},
            "Grammatical Range & Accuracy": {"band": gra, "reason": reason_gra()},
        },
        "stats": {
            "word_count": wc,
            "sentence_count": len(sentences),
            "unique_word_ratio": round(unique_ratio, 3),
        },
    }


def recommend_expressions(title: str) -> dict[str, Any]:
    t = title.lower()
    topic = "the topic"
    if any(k in t for k in ["education", "school", "student", "university"]):
        topic = "education"
    if any(k in t for k in ["technology", "internet", "ai", "social media"]):
        topic = "technology"
    if any(k in t for k in ["environment", "climate", "pollution"]):
        topic = "the environment"
    if any(k in t for k in ["health", "exercise", "diet"]):
        topic = "public health"

    vocab = [
        "a growing body of evidence",
        "a double-edged sword",
        "a far-reaching impact",
        "a pressing concern",
        "a viable alternative",
        "to exacerbate the problem",
        "to bridge the gap",
        "to foster long-term benefits",
        "to strike a balance",
        "to make informed choices",
        "to allocate resources efficiently",
        "to tackle the root cause",
        "to raise public awareness",
        "to ensure equitable access",
        "to deliver measurable outcomes",
    ]
    collocations = [
        "pose a challenge",
        "play a pivotal role",
        "take a proactive approach",
        "a significant proportion of",
        "the likelihood of",
        "in the long run",
        "from a broader perspective",
        "a key driver of change",
        "a matter of priority",
        "at the expense of",
        "bring about improvements",
        "draw a clear distinction",
    ]
    sentence_patterns = [
        "While it is true that ..., I believe ...",
        "One compelling reason is that ..., which means ...",
        "This is largely because ..., and it also leads to ...",
        "From a societal standpoint, ...; from an individual standpoint, ...",
        "Admittedly, ...; nevertheless, ...",
        "Overall, the most sustainable solution is to ..., rather than ...",
    ]

    return {
        "topic_hint": topic,
        "advanced_vocabulary": vocab[:12],
        "collocations": collocations[:12],
        "high_band_structures": sentence_patterns[:6],
    }


def generate_model_essay(title: str) -> str:
    t = safe_strip(title)
    if not t:
        t = "an IELTS Task 2 topic"

    return (
        f"Introduction\\n"
        f"In recent years, {t.lower()} has become a widely discussed issue. "
        f"Although the details vary across countries, the core debate is whether the benefits outweigh the drawbacks. "
        f"This essay argues that a balanced policy can maximise gains while limiting negative side effects.\\n\\n"
        f"Body Paragraph 1\\n"
        f"To begin with, there are clear advantages. "
        f"First, it can improve efficiency and accessibility for ordinary people, particularly those with limited time or resources. "
        f"Second, when institutions invest in quality standards and accountability, the overall outcomes tend to be more consistent and measurable. "
        f"For example, well-designed systems can reduce waste and provide targeted support to those who need it most.\\n\\n"
        f"Body Paragraph 2\\n"
        f"That said, potential risks should not be ignored. "
        f"One concern is that rapid change may widen inequalities if vulnerable groups are left behind. "
        f"Another issue is the temptation to prioritise short-term results over long-term development. "
        f"These problems can be mitigated through clear regulation, transparent evaluation, and public education so that people can make informed choices.\\n\\n"
        f"Conclusion\\n"
        f"In conclusion, {t.lower()} offers substantial benefits, but only when it is implemented with fairness and long-term planning. "
        f"Governments and communities should adopt pragmatic measures that protect disadvantaged groups while encouraging innovation."
    )


def generate_writing_ideas(title: str) -> dict[str, Any]:
    qtype = detect_question_type(title)
    core = safe_strip(title) or "this issue"

    def pack(paragraph_topics: list[str], points: list[list[str]]) -> list[dict[str, Any]]:
        out = []
        for i, topic in enumerate(paragraph_topics):
            out.append({"topic_sentence": topic, "points": points[i]})
        return out

    if qtype == "advantages/disadvantages":
        idea1 = {
            "label": "思路1（部分同意）",
            "position": "Advantages are significant, but they come with manageable disadvantages.",
            "body": pack(
                ["主体段1：主要优势", "主体段2：主要劣势与对策"],
                [
                    [
                        "提升效率与覆盖面，让更多人受益",
                        "促进创新与竞争，带来更高质量选择",
                        "降低长期成本，释放更多社会资源",
                    ],
                    [
                        "可能扩大不平等，弱势群体被边缘化",
                        "质量参差导致信任下降与资源浪费",
                        "通过监管、补贴与信息透明来降低风险",
                    ],
                ],
            ),
        }
        idea2 = {
            "label": "思路2（完全同意）",
            "position": "Overall, the advantages clearly outweigh the disadvantages.",
            "body": pack(
                ["主体段1：结构性优势", "主体段2：劣势可控且可逆"],
                [
                    [
                        "带来制度层面的效率提升与持续改进机制",
                        "提高可及性，缩小地域与时间限制",
                        "形成正向激励，推动长期社会进步",
                    ],
                    [
                        "主要问题来自执行而非理念本身",
                        "通过标准化与评估体系可快速纠偏",
                        "长期收益更稳定，短期阵痛可被政策缓冲",
                    ],
                ],
            ),
        }
        return {"question_type": qtype, "idea_1": idea1, "idea_2": idea2}

    if qtype == "discussion":
        idea1 = {
            "label": "思路1（部分同意）",
            "position": "Both views have merits; a hybrid approach is the most practical.",
            "body": pack(
                ["主体段1：观点A为何合理", "主体段2：观点B为何也重要"],
                [
                    [
                        "满足现实需求，短期可见效果更强",
                        "对资源有限的人群更友好，降低门槛",
                        "在高压力场景中更具可操作性",
                    ],
                    [
                        "长期可持续性更高，避免副作用累积",
                        "能促进个人能力与社会韧性提升",
                        "通过制度设计把短期与长期目标对齐",
                    ],
                ],
            ),
        }
        idea2 = {
            "label": "思路2（完全同意）",
            "position": "View A is more convincing overall.",
            "body": pack(
                ["主体段1：核心理由1", "主体段2：核心理由2"],
                [
                    [
                        "能够覆盖更大规模人群，公平性更强",
                        "成本更可控，投入产出比更高",
                        "对社会整体利益的边际收益更大",
                    ],
                    [
                        "执行路径更清晰，可评估、可迭代",
                        "更容易形成共识，减少对立与摩擦",
                        "长期依然可以通过配套措施补齐短板",
                    ],
                ],
            ),
        }
        return {"question_type": qtype, "idea_1": idea1, "idea_2": idea2}

    idea1 = {
        "label": "思路1（部分同意）",
        "position": f"I partly agree with the statement in '{core}'.",
        "body": pack(
            ["主体段1：同意的理由", "主体段2：保留意见/反例"],
            [
                [
                    "在多数情境下可带来效率与结果提升",
                    "对个人与社会的机会获取更有帮助",
                    "与长期发展目标一致，能产生外溢效应",
                ],
                [
                    "在资源不足或监管薄弱时可能适得其反",
                    "对弱势群体的负面影响需要被提前评估",
                    "更优方案是设定边界与配套措施，而非一刀切",
                ],
            ],
        ),
    }
    idea2 = {
        "label": "思路2（完全同意）",
        "position": f"I fully agree with the statement in '{core}'.",
        "body": pack(
            ["主体段1：宏观层面理由", "主体段2：微观层面理由"],
            [
                [
                    "推动制度与资源配置优化，带来整体收益",
                    "减少结构性问题，长期社会成本更低",
                    "形成正向激励，促进持续改进",
                ],
                [
                    "提升个人能力与机会，改善生活质量",
                    "降低信息不对称，提高决策质量",
                    "增强社会流动性与公平性",
                ],
            ],
        ),
    }
    return {"question_type": qtype, "idea_1": idea1, "idea_2": idea2}


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/api/grade")
def api_grade():
    payload = request.get_json(silent=True) or {}
    title = safe_strip(payload.get("title", ""))
    text = safe_strip(payload.get("text", ""))

    if not title or not text:
        return jsonify({"error": "title and text are required"}), 400

    sentences = split_sentences(text)
    sentence_reports = [analyze_sentence(s) for s in sentences]

    use_llm = bool(os.getenv("SILICONFLOW_API_KEY") or os.getenv("OPENAI_API_KEY"))
    llm_grading = None
    llm_error = ""

    if use_llm:
        fallback_grading = grade_essay(title, text)
        grading = fallback_grading
        try:
            llm_grading = llm_grade_only(title, text, fallback_grading.get("stats", {}))
            if llm_grading:
                grading = {
                    "overall": llm_grading.get("overall", fallback_grading.get("overall")),
                    "criteria": llm_grading.get("criteria", fallback_grading.get("criteria")),
                    "stats": fallback_grading.get("stats", {}),
                }
        except Exception as e:
            llm_error = str(e)
            llm_grading = None

        try:
            rec = llm_recommendations(title) or recommend_expressions(title)
        except Exception as e:
            if not llm_error:
                llm_error = str(e)
            rec = recommend_expressions(title)

        try:
            model = llm_model_essay(title) or generate_model_essay(title)
        except Exception as e:
            if not llm_error:
                llm_error = str(e)
            model = generate_model_essay(title)

        try:
            ideas = llm_writing_ideas(title) or generate_writing_ideas(title)
        except Exception as e:
            if not llm_error:
                llm_error = str(e)
            ideas = generate_writing_ideas(title)
    else:
        grading = grade_essay(title, text)
        rec = recommend_expressions(title)
        model = generate_model_essay(title)
        ideas = generate_writing_ideas(title)

    response_payload = {
        "grading": grading,
        "sentence_reports": sentence_reports,
        "recommendations": rec,
        "model_essay": model,
        "writing_ideas": ideas,
        "revised_essay": llm_grading.get("revised_essay", "") if llm_grading else "",
        "revision_notes": llm_grading.get("revision_notes", []) if llm_grading else [],
        "llm_error": llm_error,
    }
    return jsonify(response_payload)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=True)
