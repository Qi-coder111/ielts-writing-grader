const $ = (sel) => document.querySelector(sel);

const state = {
  loaded: false,
  sentenceStates: [],
  lastResponse: null,
};

function escapeHtml(str) {
  return String(str)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function setLoading(isLoading) {
  $("#loading").classList.toggle("hidden", !isLoading);
  $("#submitBtn").disabled = isLoading;
  $("#fillExampleBtn").disabled = isLoading;
}

function setError(msg) {
  const box = $("#errorBox");
  if (!msg) {
    box.classList.add("hidden");
    box.textContent = "";
    return;
  }
  box.classList.remove("hidden");
  box.textContent = msg;
}

function fetchGrade(title, text) {
  return fetch("/api/grade", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title, text }),
  }).then(async (res) => {
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      const msg = data && data.error ? data.error : "请求失败，请稍后重试";
      throw new Error(msg);
    }
    return data;
  });
}

function classForErrorType(type) {
  if (type === "spelling" || type === "collocation") return "err-word";
  if (type === "grammar") return "err-grammar";
  if (type === "omission") return "err-omission";
  return "err-word";
}

function renderHighlighted(original, errors) {
  const text = String(original || "");
  const insertionsByPos = new Map();
  const replacements = [];

  for (const e of errors || []) {
    const [s, t] = e.position || [0, 0];
    if (s === t) {
      const arr = insertionsByPos.get(s) || [];
      arr.push({ type: e.type, suggestion: e.suggestion });
      insertionsByPos.set(s, arr);
    } else {
      replacements.push({ start: s, end: t, type: e.type });
    }
  }

  replacements.sort((a, b) => (a.start - b.start) || (b.end - a.end));

  const insertionPositions = Array.from(insertionsByPos.keys()).sort((a, b) => a - b);

  function renderInsertionsAt(pos) {
    const arr = insertionsByPos.get(pos) || [];
    if (!arr.length) return "";
    return arr
      .map((x) => `<span class="${classForErrorType(x.type)}">${escapeHtml(x.suggestion)}</span>`)
      .join("");
  }

  function appendPlain(out, from, to) {
    if (from >= to) return out;
    let cur = from;
    for (const ip of insertionPositions) {
      if (ip < from || ip > to) continue;
      if (ip > cur) out.push(escapeHtml(text.slice(cur, ip)));
      out.push(renderInsertionsAt(ip));
      cur = ip;
    }
    if (cur < to) out.push(escapeHtml(text.slice(cur, to)));
    return out;
  }

  const out = [];
  let cursor = 0;
  for (const r of replacements) {
    const s = Math.max(0, Math.min(text.length, r.start));
    const e = Math.max(0, Math.min(text.length, r.end));
    if (e <= cursor || s < cursor) continue;
    appendPlain(out, cursor, s);
    const cls = classForErrorType(r.type);
    out.push(`<span class="${cls}">${escapeHtml(text.slice(s, e))}</span>`);
    cursor = e;
  }
  appendPlain(out, cursor, text.length);
  if (insertionsByPos.has(text.length)) out.push(renderInsertionsAt(text.length));
  return out.join("");
}

function renderScores(grading) {
  const grid = $("#scoreGrid");
  grid.innerHTML = "";

  const overall = grading.overall;
  const overallCard = document.createElement("div");
  overallCard.className = "score";
  overallCard.innerHTML = `
    <div class="score-top">
      <div class="score-name">Overall Band</div>
      <div class="score-band">${escapeHtml(overall)}</div>
    </div>
    <p class="score-reason">四项平均并四舍五入到 0.5 分。建议优先修复语法与衔接，再提高词汇精确度。</p>
  `;
  grid.appendChild(overallCard);

  const criteria = grading.criteria || {};
  for (const [name, v] of Object.entries(criteria)) {
    const card = document.createElement("div");
    card.className = "score";
    card.innerHTML = `
      <div class="score-top">
        <div class="score-name">${escapeHtml(name)}</div>
        <div class="score-band">${escapeHtml(v.band)}</div>
      </div>
      <p class="score-reason">${escapeHtml(v.reason)}</p>
    `;
    grid.appendChild(card);
  }

  const stats = grading.stats || {};
  $("#statsBox").innerHTML = `
    <div class="chip">词数：${escapeHtml(stats.word_count ?? "-")}</div>
    <div class="chip">句子：${escapeHtml(stats.sentence_count ?? "-")}</div>
    <div class="chip">词汇多样性：${escapeHtml(stats.unique_word_ratio ?? "-")}</div>
  `;
}

function modeLabel(mode) {
  if (mode === "replace") return "已替换";
  if (mode === "manual") return "手动修改";
  return "保持不变";
}

function updateExportBox() {
  const lines = state.sentenceStates.map((x) => (x.finalText || "").trim()).filter(Boolean);
  $("#exportBox").value = lines.join(" ");
}

function renderSentences(reports) {
  const list = $("#sentenceList");
  list.innerHTML = "";

  state.sentenceStates = reports.map((r) => ({
    mode: "keep",
    original: r.original_sentence,
    corrected: r.corrected_sentence,
    finalText: r.original_sentence,
  }));

  reports.forEach((r, idx) => {
    const item = document.createElement("div");
    item.className = "sent-item";

    const errCount = (r.errors || []).length;
    const highlighted = r.original_sentence || "";
    const hasSuggestion =
      typeof r.corrected_sentence === "string" &&
      r.corrected_sentence.trim() &&
      r.corrected_sentence.replace(/<[^>]*>?/gm, '').trim() !== String(r.original_sentence || "").replace(/<[^>]*>?/gm, '').trim();
    const suggestionText = hasSuggestion ? r.corrected_sentence : "（未检测到可自动改写，或建议与原句一致）";

    item.innerHTML = `
      <div class="sent-head">
        <div class="sent-index">Sentence ${idx + 1} · 当前：<b>${escapeHtml(
          modeLabel("keep")
        )}</b></div>
        <div class="sent-actions">
          <button class="btn" data-action="keep">保持不变</button>
          <button class="btn primary" data-action="replace">一键替换</button>
          <button class="btn" data-action="manual">手动修改</button>
        </div>
      </div>
      <div class="sent-body">
        <div class="sent-block">
          <div class="sent-label">原句（错误高亮）</div>
          <div class="sent-text">${highlighted}</div>
        </div>
        <div class="sent-block">
          <div class="sent-label">建议改写（紫字部分为修正）</div>
          <div class="sent-text">${suggestionText}</div>
        </div>
      </div>
      <div class="sent-block hidden" data-edit-wrap>
        <div class="sent-label">手动修改（可直接编辑）</div>
        <textarea class="sent-edit" rows="3" data-edit></textarea>
        <div class="diff">
          <div>提示：修改后会自动用于导出。</div>
        </div>
      </div>
    `;

    function refreshHeader() {
      const h = item.querySelector(".sent-index");
      h.innerHTML = `Sentence ${idx + 1} · 检测到 ${errCount} 处问题 · 当前：<b>${escapeHtml(
        modeLabel(state.sentenceStates[idx].mode)
      )}</b>`;
    }

    const editWrap = item.querySelector("[data-edit-wrap]");
    const edit = item.querySelector("[data-edit]");

    item.querySelectorAll("button[data-action]").forEach((btn) => {
      btn.addEventListener("click", () => {
        const action = btn.getAttribute("data-action");
        const ss = state.sentenceStates[idx];
        if (action === "keep") {
          ss.mode = "keep";
          ss.finalText = ss.original;
          editWrap.classList.add("hidden");
        } else if (action === "replace") {
          ss.mode = "replace";
          ss.finalText = ss.corrected;
          editWrap.classList.add("hidden");
        } else {
          ss.mode = "manual";
          editWrap.classList.remove("hidden");
          edit.value = ss.finalText || ss.corrected || ss.original;
          edit.focus();
        }
        refreshHeader();
        updateExportBox();
      });
    });

    edit.addEventListener("input", () => {
      const ss = state.sentenceStates[idx];
      if (ss.mode !== "manual") return;
      ss.finalText = edit.value;
      updateExportBox();
    });

    list.appendChild(item);
  });

  updateExportBox();
}

function renderRecommendations(rec) {
  const box = $("#recBox");
  const vocab = rec.advanced_vocabulary || [];
  const coll = rec.collocations || [];
  const patterns = rec.high_band_structures || [];

  box.innerHTML = `
    <div class="section">
      <h3 class="section-title">高级词汇（10–15）</h3>
      <ul class="list">${vocab.map((x) => `<li>${escapeHtml(x)}</li>`).join("")}</ul>
    </div>
    <div class="section">
      <h3 class="section-title">固定搭配</h3>
      <ul class="list">${coll.map((x) => `<li>${escapeHtml(x)}</li>`).join("")}</ul>
    </div>
    <div class="section">
      <h3 class="section-title">高分句式</h3>
      <ul class="list">${patterns.map((x) => `<li>${escapeHtml(x)}</li>`).join("")}</ul>
    </div>
  `;
}

function renderIdeas(ideas) {
  const box = $("#ideaBox");
  const type = ideas.question_type || "-";

  function ideaCard(idea) {
    if (!idea) return "";
    const body = idea.body || [];
    const bodyHtml = body
      .map(
        (p) => `
        <div class="section">
          <h3 class="section-title">${escapeHtml(p.topic_sentence)}</h3>
          <ul class="list">${(p.points || []).map((x) => `<li>${escapeHtml(x)}</li>`).join("")}</ul>
        </div>
      `
      )
      .join("");
    return `
      <div class="section">
        <h3 class="section-title">${escapeHtml(idea.label)}</h3>
        <div class="diff">题型：${escapeHtml(type)} · 立场：${escapeHtml(idea.position)}</div>
      </div>
      ${bodyHtml}
    `;
  }

  box.innerHTML = ideaCard(ideas.idea_1) + ideaCard(ideas.idea_2);
}

function downloadText(filename, content) {
  const blob = new Blob([content], { type: "text/plain;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

function fillExample() {
  $("#titleInput").value =
    "Some people think governments should invest more in public transportation rather than building new roads. To what extent do you agree or disagree?";
  $("#textInput").value =
    "In many countries, traffic congestion has become a serious issue. Some people argue that governments should focus on improving public transport instead of constructing more roads. I partly agree with this view.\n\nFirstly, investing in public transportation can reduce the number of cars on the road. Many student take buses or subways if the service is affordable and reliable. As a result, cities can lower pollution and save energy. Moreover, public transport is more inclusive for elderly people and low-income groups.\n\nHowever, building roads is still necessary in some cases. For instance, rural areas may not have enough population to support frequent buses, so better roads can help residents travel to work or hospital. If the government ignore this, regional inequality may increase.\n\nIn conclusion, I believe public transportation should be a priority, but road construction should continue when it is clearly needed.";
}

async function onSubmit() {
  setError("");
  const title = $("#titleInput").value.trim();
  const text = $("#textInput").value.trim();
  if (!title || !text) {
    setError("请填写作文标题与正文。");
    return;
  }

  setLoading(true);
  try {
    const data = await fetchGrade(title, text);
    state.lastResponse = data;
    state.loaded = true;
    renderScores(data.grading);
    renderSentences(data.sentence_reports || []);
    renderRecommendations(data.recommendations || {});
    renderIdeas(data.writing_ideas || {});
    $("#modelBox").textContent = data.model_essay || "";

    if (data.llm_error) {
      setError(`AI 调用失败：${data.llm_error}（已回退为规则引擎结果）`);
    } else {
      setError("");
    }

    const annoTitle = document.getElementById("essayAnnoTitle");
    if (annoTitle) {
      annoTitle.textContent = "原文纠错";
    }

    const revisedBox = document.getElementById("revisedEssayBox");
    if (revisedBox) {
      revisedBox.innerHTML = data.revised_essay || "（未返回修改内容）";
    }

    const notesBox = document.getElementById("revisionNotesBox");
    if (notesBox && Array.isArray(data.revision_notes)) {
      console.log("=== JSON 校对日志 ===");
      console.log(JSON.stringify(data.revision_notes, null, 2));
      console.log("=====================");
      notesBox.innerHTML = data.revision_notes.map((note) => {
        if (typeof note === 'object') {
          return `<li>[${note.type}] ${escapeHtml(note.original)} → ${escapeHtml(note.action)}</li>`;
        }
        return `<li>${escapeHtml(String(note))}</li>`;
      }).join("");
    } else if (notesBox) {
      notesBox.innerHTML = "";
    }
  } catch (e) {
    setError(e.message || "请求失败");
  } finally {
    setLoading(false);
  }
}

function init() {
  $("#submitBtn").addEventListener("click", onSubmit);
  $("#fillExampleBtn").addEventListener("click", () => {
    fillExample();
    setError("");
  });
  $("#rebuildBtn").addEventListener("click", () => {
    updateExportBox();
    setError("");
  });
  $("#downloadBtn").addEventListener("click", () => {
    const content = $("#exportBox").value || "";
    if (!content.trim()) {
      setError("请先生成导出内容。");
      return;
    }
    downloadText("ielts_essay_revised.txt", content);
  });
}

init();
