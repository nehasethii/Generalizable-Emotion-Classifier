/* analyzer-script.js — updated
   - sorts emotions highest→lowest
   - maps numeric labels -> readable emotion names using ALL_EMOTIONS
   - nicer Check reasoning UI & explanation rendering (LIME / SHAP -> sentences)
   - keeps debug prediction pre but hidden by default (Option A)
   - safe DOM checks and graceful fallbacks
   - small fixes: unwrap json.explanation, ensure explanation wrapper is placed below prediction-area,
     make Check reasoning button adopt same visual class as analyze button for parity
*/

/* ---------------------------
   EMOTION LABELS (GoEmotions 28)
   Numeric indices map to readable labels if backend doesn't send them.
   --------------------------- */
const ALL_EMOTIONS = [
  "admiration",
  "amusement",
  "anger",
  "annoyance",
  "approval",
  "caring",
  "confusion",
  "curiosity",
  "desire",
  "disappointment",
  "disapproval",
  "disgust",
  "embarrassment",
  "excitement",
  "fear",
  "gratitude",
  "grief",
  "joy",
  "love",
  "nervousness",
  "optimism",
  "pride",
  "realization",
  "relief",
  "remorse",
  "sadness",
  "surprise",
  "neutral",
];

const EMOJI_MAP = {
  admiration: "🤩",
  amusement: "😄",
  anger: "😠",
  annoyance: "😒",
  approval: "👍",
  caring: "🤗",
  confusion: "😕",
  curiosity: "🤔",
  desire: "🤤",
  disappointment: "😞",
  disapproval: "👎",
  disgust: "🤢",
  embarrassment: "😳",
  excitement: "🎉",
  fear: "😨",
  gratitude: "🙏",
  grief: "😭",
  joy: "😊",
  love: "❤️",
  nervousness: "😰",
  optimism: "🌟",
  pride: "😌",
  realization: "💡",
  relief: "😌",
  remorse: "😔",
  sadness: "😢",
  surprise: "😲",
  neutral: "😐",
};

/* ---------------------------
   UI Globals
   --------------------------- */
let selectedModelValue = "distilbert";
let uploadedFile = null;
let currentInputType = "text";

/* ---------------------------
   tiny helper to toggle prediction debug pre from console:
   window.togglePredictionDebug() will show/hide the hidden JSON box
   --------------------------- */
window.togglePredictionDebug = function () {
  const pre = document.getElementById("prediction-text");
  if (!pre) return;
  pre.classList.toggle("debug-visible");
  return pre.classList.contains("debug-visible");
};

/* ---------------------------
   Dropdown & input handlers
   --------------------------- */
function toggleDropdown() {
  const m = document.getElementById("dropdownMenu");
  if (m) m.classList.toggle("active");
}

document.addEventListener("click", (ev) => {
  const dropdown = document.getElementById("dropdownMenu"),
    btn = document.querySelector(".dropdown-btn");
  if (!dropdown || !btn) return;
  if (!dropdown.contains(ev.target) && !btn.contains(ev.target))
    dropdown.classList.remove("active");
});

function selectModel(name, value) {
  selectedModelValue = value;
  const sel = document.getElementById("selectedModel");
  if (sel) sel.textContent = name;
  document.querySelectorAll(".dropdown-item").forEach((i) => i.classList.remove("active"));
  try {
    const el = event.target.closest(".dropdown-item");
    if (el) el.classList.add("active");
  } catch (e) {}
  const m = document.getElementById("dropdownMenu");
  if (m) m.classList.remove("active");
}

function handleInputTypeChange(type) {
  currentInputType = type;
  const textSection = document.getElementById("textInputSection"),
    csvSection = document.getElementById("csvInputSection");
  document.querySelectorAll(".radio-card").forEach((card) => card.classList.remove("active"));
  try {
    const card = event.target.closest(".radio-card");
    if (card) card.classList.add("active");
  } catch (e) {}
  if (type === "text") {
    if (textSection) textSection.classList.remove("hidden");
    if (csvSection) csvSection.classList.add("hidden");
  } else {
    if (textSection) textSection.classList.add("hidden");
    if (csvSection) csvSection.classList.remove("hidden");
  }
}

/* ---------------------------
   Text / CSV helpers
   --------------------------- */
const textInputElem = document.getElementById("textInput");
if (textInputElem) {
  textInputElem.addEventListener("input", () => {
    const n = textInputElem.value.length;
    const cc = document.getElementById("charCount");
    if (cc) cc.textContent = `${n} characters`;
  });
}
function clearTextInput() {
  if (textInputElem) textInputElem.value = "";
  const cc = document.getElementById("charCount");
  if (cc) cc.textContent = "0 characters";
}
function useSample(i) {
  const samples = [
    "I'm so excited about this amazing opportunity! Can't wait!",
    "This is horrible. I'm extremely disappointed.",
    "I feel confused but also relieved at the same time.",
  ];
  const txt = samples[i] || "";
  if (textInputElem) textInputElem.value = txt;
  const cc = document.getElementById("charCount");
  if (cc) cc.textContent = `${txt.length} characters`;
}

function handleFileSelect(e) {
  const f = e.target.files && e.target.files[0];
  if (!f) return;
  if (!f.name.endsWith(".csv")) {
    alert("Please upload a CSV file.");
    return;
  }
  uploadedFile = f;
  const fn = document.getElementById("fileName"),
    fs = document.getElementById("fileSize"),
    ua = document.getElementById("uploadArea"),
    fi = document.getElementById("fileInfo");
  if (fn) fn.textContent = f.name;
  if (fs) fs.textContent = (f.size / 1024).toFixed(1) + " KB";
  if (ua) ua.style.display = "none";
  if (fi) fi.classList.remove("hidden");
}
function removeFile() {
  uploadedFile = null;
  const inp = document.getElementById("csvFileInput");
  if (inp) inp.value = "";
  const ua = document.getElementById("uploadArea");
  const fi = document.getElementById("fileInfo");
  if (ua) ua.style.display = "block";
  if (fi) fi.classList.add("hidden");
}

/* ---------------------------
   Network: /api/predict and CSV upload
   --------------------------- */
async function analyzeEmotion() {
  try {
    const resultsSec = document.getElementById("resultsSection"),
      loadingSec = document.getElementById("loadingSection");
    if (resultsSec) resultsSec.classList.add("hidden");
    if (loadingSec) loadingSec.classList.remove("hidden");

    if (currentInputType === "text") {
      const inputText = (document.getElementById("textInput") || {}).value || "";
      const trimmed = inputText.trim();
      if (!trimmed) {
        alert("Enter text to analyze!");
        return;
      }

      const resp = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text: trimmed,
          model: selectedModelValue,
          apply_rule_override: true,
        }),
      });
      if (!resp.ok) {
        const t = await resp.text();
        throw new Error(t || `HTTP ${resp.status}`);
      }
      const data = await resp.json();
      showPredictionBlock(data, selectedModelValue, trimmed);
      displayResults(data);
    } else {
      if (!uploadedFile) {
        alert("Upload a CSV file first.");
        return;
      }
      const form = new FormData();
      form.append("file", uploadedFile);
      form.append("model", selectedModelValue);
      form.append("apply_rule_override", "true");
      const resp = await fetch("/api/predict/csv", { method: "POST", body: form });
      if (!resp.ok) {
        const t = await resp.text();
        throw new Error(t || `HTTP ${resp.status}`);
      }
      const blob = await resp.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "predictions.csv";
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
    }
  } catch (err) {
    console.error(err);
    alert("Error analyzing text: " + (err.message || err));
  } finally {
    const loadingSec = document.getElementById("loadingSection");
    if (loadingSec) loadingSec.classList.add("hidden");
  }
}

/* ---------------------------
   Prediction preview & Interpret integration
   --------------------------- */
function showPredictionBlock(data, model, textValue) {
  // Build a compact summary for debug pre (kept hidden by default)
  const predictionPre = document.getElementById("prediction-text");
  let topLabel = null,
    topProb = null;

  if (Array.isArray(data.top) && data.top.length > 0) {
    topLabel = data.top[0].label;
    topProb = Number(data.top[0].prob);
  } else if (Array.isArray(data.probs) && data.probs.length > 0) {
    const probs = data.probs.map(Number);
    const maxIdx = probs.indexOf(Math.max(...probs));
    topLabel =
      Array.isArray(data.labels) && data.labels[maxIdx]
        ? data.labels[maxIdx]
        : ALL_EMOTIONS[maxIdx] || String(maxIdx);
    topProb = probs[maxIdx];
  } else {
    topLabel = data.prediction || "N/A";
    topProb = data.confidence ? Number(data.confidence) : null;
  }

  const summary = {
    model: data.model || model,
    prediction: String(topLabel),
    confidence: topProb,
    raw: data,
  };

  if (predictionPre) {
    // keep writing debug JSON so developer can toggle visibility
    try {
      predictionPre.innerText = JSON.stringify(summary, null, 2);
    } catch (e) {
      predictionPre.innerText = String(summary);
    }
  }

  window._lastReasonContext = { model, text: textValue, raw: data };

  // Show check reasoning button and style it consistently
  const btn = document.getElementById("check-reasoning-btn");
  if (btn) {
    // ensure visual parity with predict button
    btn.style.display = "inline-flex";
    // add predict-btn class so CSS matches exactly (id-specific styles still apply)
    btn.classList.add("predict-btn");
    btn.classList.remove("export-btn");
    btn.innerText = "🔎 Check reasoning";
    btn.onclick = async () => {
      await runInterpretation();
    };
  }
}

// ---- Run interpretation (unchanged network call but improved UI placement) ----
async function runInterpretation() {
  const ctx = window._lastReasonContext;
  if (!ctx) return alert("Predict first!");

  const btn = document.getElementById("check-reasoning-btn");
  const wrapper = document.getElementById("explanation-wrapper");
  const container = document.getElementById("explanation-text");

  if (btn) {
    btn.disabled = true;
    btn.innerText = "Computing...";
  }
  if (container) container.innerHTML = "";
  if (wrapper) {
    // ensure wrapper shown and placed AFTER the prediction area so it doesn't push content sideways
    wrapper.style.display = "block";
    // move wrapper below prediction-area in DOM to guarantee vertical stacking
    const predArea = document.getElementById("prediction-area");
    if (predArea && predArea.parentNode) {
      predArea.parentNode.insertBefore(wrapper, predArea.nextSibling);
    }
    wrapper.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  try {
    const resp = await fetch(
      `/api/interpret/${encodeURIComponent(ctx.model)}`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: ctx.text }),
      }
    );
    if (!resp.ok) {
      const t = await resp.text();
      throw new Error(t || `HTTP ${resp.status}`);
    }
    const json = await resp.json();
    renderInterpretation(json, container, ctx);
  } catch (err) {
    if (container) container.innerText = "Interpretation error: " + (err.message || err);
    if (wrapper) wrapper.style.display = "block";
  } finally {
    if (btn) {
      btn.disabled = false;
      btn.innerText = "🔎 Check reasoning";
    }
  }
}

// ---- small helper: convert a raw LIME feature -> friendly sentence ----
function featureToSentence(feature, totalAbs = null) {
  // feature may be ["word", weight] or {feature: "word", weight: 0.12} etc.
  let word = null;
  let weight = null;

  if (Array.isArray(feature) && feature.length >= 2) {
    word = String(feature[0]);
    weight = Number(feature[1]);
  } else if (feature && typeof feature === "object") {
    // common shapes: {feature: 'word', weight: 0.12} or {name:'word', value:0.12}
    word = feature.feature || feature.name || feature.label || feature.token || JSON.stringify(feature);
    weight = feature.weight ?? feature.value ?? feature.score ?? null;
    weight = weight !== null ? Number(weight) : null;
  } else {
    // fallback, put entire item in text
    word = String(feature);
  }

  // If no numeric weight, fallback to a neutral sentence
  if (weight === null || Number.isNaN(weight)) {
    return `Feature "<strong>${escapeHtml(word)}</strong>" had unknown numeric contribution.`;
  }

  // If the caller provided totalAbs (sum of absolute weights), use it for percentage.
  // Otherwise if weight looks like a fraction between -1 and 1, use that * 100.
  // Otherwise fallback to scaling by sum-of-abs = 1 (i.e., treat weight as fraction).
  let pct = null;
  const absW = Math.abs(Number(weight));
  if (totalAbs && totalAbs > 0) {
    pct = (absW / totalAbs) * 100.0;
  } else if (absW <= 1.0) {
    pct = absW * 100.0;
  } else {
    // weight > 1 but no total given — try to present relative magnitude by normalizing by (absW)
    pct = Math.min(100, absW); // best-effort
  }

  const pctStr = pct.toFixed(1) + "%";
  const sign = weight > 0 ? "+" : "";
  const abs = Math.abs(Number(weight));

  // descriptive bucket
  let impact = "slightly";
  if (abs >= 0.20 || pct >= 20.0) impact = "strongly";
  else if (abs >= 0.05 || pct >= 5.0) impact = "moderately";

  const dir = weight > 0 ? "increases" : "decreases";

  return `The word "<strong>${escapeHtml(word)}</strong>" ${impact} ${dir} the prediction (${sign}${Number(weight).toFixed(3)} → ${pctStr}).`;
}

// ---- Render interpretation (prefers sentences, then HTML, then features) ----
function renderInterpretation(json, containerElem, ctx) {
  if (!containerElem) return;
  containerElem.innerHTML = "";

  // unwrap if backend nested explanation under .explanation
  if (json && json.explanation) {
    json = json.explanation;
  }

  // 1) If backend gave explicit sentences array (already human readable), use it
  if (json && Array.isArray(json.sentences) && json.sentences.length > 0) {
    const ul = document.createElement("ul");
    ul.style.margin = "6px 0 0 1rem";
    for (const s of json.sentences) {
      const li = document.createElement("li");
      li.className = "explanation-sentence";
      // allow small HTML snippet (strong tags) from backend mapping, sanitize minimally
      li.innerHTML = sanitizeInlineHtml(String(s));
      ul.appendChild(li);
    }
    // optional note
    if (json.note) {
      const meta = document.createElement("div");
      meta.style.marginTop = "8px";
      meta.style.color = "var(--text-secondary)";
      meta.innerText = json.note;
      containerElem.appendChild(meta);
    }
    containerElem.appendChild(ul);
    return;
  }

  // 2) If server returned ready-made HTML, insert it (we assume trusted server-side HTML)
  if (json && json.explanation_html) {
    // Small sanitizer: only allow certain tags (a minimal approach). If you have a DOM sanitizer lib, use that instead.
    // Here we trust backend for now and insert. This keeps highlighting inline and matches cards.
    containerElem.innerHTML = json.explanation_html;
    // append meta info
    const meta = document.createElement("div");
    meta.style.marginTop = "8px";
    meta.style.color = "var(--text-secondary)";
    meta.innerText = json.note ? String(json.note) : "Explanation (SHAP)";
    containerElem.appendChild(meta);
    return;
  }

  // 3) If backend provided LIME-style features list, convert feature -> sentence (word-level)
  // Accept multiple shapes:
  // - json.features = [ ["word", weight], ... ]
  // - json.features = [ {feature: 'word', weight: 0.12}, ... ]
  // - json.features may be nested under json.explanation.features (we already unwrapped above)
  if (json && Array.isArray(json.features) && json.features.length > 0) {
    // compute total absolute weight if possible
    let totalAbs = 0;
    try {
      for (const f of json.features) {
        let w = null;
        if (Array.isArray(f) && f.length >= 2) w = Number(f[1]);
        else if (f && typeof f === "object") w = f.weight ?? f.value ?? f.score ?? null;
        if (w !== null && !Number.isNaN(Number(w))) totalAbs += Math.abs(Number(w));
      }
    } catch (e) {
      totalAbs = 0;
    }
    // if totalAbs is 0, we will let featureToSentence fall back to fraction interpretation
    const ul = document.createElement("ul");
    ul.style.margin = "6px 0 0 1rem";
    for (const f of json.features) {
      const li = document.createElement("li");
      li.className = "explanation-sentence";
      li.innerHTML = featureToSentence(f, totalAbs || null);
      ul.appendChild(li);
    }
    containerElem.appendChild(ul);
    return;
  }

  // 4) If response included top/probs/labels (an explanation of confidences), render a compact list
  if (json && (Array.isArray(json.top) || Array.isArray(json.probs) || Array.isArray(json.labels))) {
    const list = buildSortedListFromResponse(json);
    if (list && list.length > 0) {
      const frag = document.createDocumentFragment();
      for (const it of list.slice(0, 10)) {
        const div = document.createElement("div");
        div.style.display = "flex";
        div.style.justifyContent = "space-between";
        div.style.padding = "6px 0";
        div.style.borderBottom = "1px solid rgba(255,255,255,0.03)";
        const left = document.createElement("div");
        left.style.fontWeight = "600";
        left.innerText = capitalizeFirst(String(it.label));
        const right = document.createElement("div");
        right.style.opacity = "0.9";
        right.innerText = (it.prob * 100).toFixed(1) + "%";
        div.appendChild(left);
        div.appendChild(right);
        frag.appendChild(div);
      }
      containerElem.appendChild(frag);
      return;
    }
  }

  // 5) fallback: pretty JSON
  const pre = document.createElement("pre");
  pre.style.whiteSpace = "pre-wrap";
  pre.textContent = JSON.stringify(json, null, 2);
  containerElem.appendChild(pre);
}

/* ---------------------------
   Results rendering & sorting (highest-first)
   --------------------------- */
function displayResults(data) {
  const resultsSec = document.getElementById("resultsSection");
  if (resultsSec) resultsSec.classList.remove("hidden");
  const sorted = buildSortedListFromResponse(data);
  displayEmotionCardsFromSorted(sorted);
  displayStatsFromSorted(sorted, data);
  displayChartFromSorted(sorted);
}

/* build sorted list of {label, prob, index} sorted desc */
function buildSortedListFromResponse(data) {
  if (!data) return [];
  if (Array.isArray(data.top) && data.top.length > 0) {
    return data.top
      .map((t, idx) => ({
        label: String(t.label),
        prob: Number(t.prob) || 0,
        index: Number(t.index ?? idx),
      }))
      .sort((a, b) => b.prob - a.prob);
  }
  const labels = Array.isArray(data.labels) ? data.labels.slice() : null;
  const probs = Array.isArray(data.probs) ? data.probs.slice() : null;
  if (!probs) return [];
  const list = [];
  for (let i = 0; i < probs.length; i++) {
    const prob = Number(probs[i]) || 0;
    let label = labels && labels[i] !== undefined ? String(labels[i]) : (ALL_EMOTIONS[i] !== undefined ? ALL_EMOTIONS[i] : String(i));
    list.push({ label: label, prob: prob, index: i });
  }
  list.sort((a, b) => b.prob - a.prob);
  return list;
}

function displayEmotionCardsFromSorted(sortedList) {
  const grid = document.getElementById("emotionGrid");
  if (!grid) return;
  grid.innerHTML = "";
  for (let i = 0; i < sortedList.length; i++) {
    const it = sortedList[i];
    const percent = it.prob * 100;
    const emoji = EMOJI_MAP[it.label] || "😐";
    const div = document.createElement("div");
    div.className = "emotion-card";
    div.innerHTML = `
      <div class="emotion-header">
        <div style="display:flex;align-items:center;gap:10px;">
          <div class="emotion-emoji" style="font-size:1.6rem">${emoji}</div>
          <div class="emotion-name">${escapeHtml(capitalizeFirst(String(it.label)))}</div>
        </div>
        <div class="emotion-confidence">${percent.toFixed(1)}%</div>
      </div>
      <div class="emotion-bar"><div class="emotion-fill" style="width:${Math.max(0, Math.min(100, percent))}%"></div></div>
    `;
    grid.appendChild(div);
  }
}

function displayStatsFromSorted(sortedList, rawData) {
  const grid = document.getElementById("statsGrid");
  if (!grid) return;
  if (!sortedList || sortedList.length === 0) {
    grid.innerHTML = "<div>No stats available.</div>";
    return;
  }
  const top = sortedList[0];
  let avg = 0;
  if (Array.isArray(rawData && rawData.probs) && rawData.probs.length > 0) {
    avg = rawData.probs.reduce((a, b) => a + Number(b), 0) / rawData.probs.length;
  } else {
    avg = sortedList.reduce((s, it) => s + it.prob, 0) / sortedList.length;
  }
  grid.innerHTML = `
    <div class="stat-item">
      <div class="stat-number"><span class="stat-val">${escapeHtml(capitalizeFirst(String(top.label)))}</span></div>
      <div class="stat-label">Primary Emotion</div>
    </div>
    <div class="stat-item">
      <div class="stat-number"><span class="stat-val">${(top.prob * 100).toFixed(1)}</span><span class="stat-suffix">%</span></div>
      <div class="stat-label">Highest Confidence</div>
    </div>
    <div class="stat-item">
      <div class="stat-number"><span class="stat-val">${(avg * 100).toFixed(1)}</span><span class="stat-suffix">%</span></div>
      <div class="stat-label">Average Confidence</div>
    </div>
  `;
}

function displayChartFromSorted(sortedList) {
  const container = document.getElementById("chartContainer");
  if (!container) return;
  container.innerHTML = "";
  const top = sortedList.slice(0, 5);
  top.forEach((it) => {
    const bar = document.createElement("div");
    bar.className = "chart-bar";
    const height = Math.max(6, Math.round(it.prob * 200));
    bar.style.height = `${height}px`;
    bar.title = `${capitalizeFirst(it.label)}: ${(it.prob * 100).toFixed(1)}%`;
    bar.innerHTML = `<div class="chart-value">${(it.prob * 100).toFixed(0)}%</div><div class="chart-label">${escapeHtml(capitalizeFirst(it.label))}</div>`;
    container.appendChild(bar);
  });
}

/* Utilities */
function capitalizeFirst(s) {
  return s ? s.charAt(0).toUpperCase() + s.slice(1) : "";
}
function escapeHtml(u) {
  if (u === undefined || u === null) return "";
  return String(u)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

/* minimal inline HTML sanitizer for produced sentences (allow <strong> only) */
function sanitizeInlineHtml(html) {
  // remove any tags except <strong> and closing tag
  return String(html).replace(/<(?!\/?strong\b)[^>]*>/gi, (m) => {
    return "";
  });
}

/* Keyboard shortcut: Ctrl/Cmd + Enter */
document.addEventListener("keydown", (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
    analyzeEmotion();
  }
});
