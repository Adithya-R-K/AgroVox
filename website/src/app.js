(function(){
"use strict";

function safe(label, fn){
  try { fn(); } catch(e){ console.error("[AgroVox] " + label + " failed:", e); }
}

/* ============================== GLOBAL STATE ============================== */
let currentTheme = "light";
let activeTab = "evaluation";
let activeSubtab = "classification";
let evalFilterMode = "all"; // all, good, exact, review

const EVAL_DATA = [
  { id: 0, ref: "my rice crop has yellow leaves", hyp: "my rice crop has yellow leaf", wer: 0.1687, conf: 92, res: "Good" },
  { id: 1, ref: "what fertilizer is good for tomato", hyp: "what fertilizer is good for tomatoes", wer: 0.1667, conf: 88, res: "Good" },
  { id: 2, ref: "how to control leaf blight in paddy", hyp: "how to control leaf blight in paddy", wer: 0.0000, conf: 96, res: "Exact" },
  { id: 3, ref: "best time to irrigate sugarcane", hyp: "when is the best time to irrigate sugarcane", wer: 0.1429, conf: 85, res: "Good" },
  { id: 4, ref: "weather forecast for next week", hyp: "weather forecast for next weak", wer: 0.2000, conf: 78, res: "Review" },
  { id: 5, ref: "suitable crops for red soil", hyp: "best crops for red soil", wer: 0.3333, conf: 70, res: "Review" },
  { id: 6, ref: "how to increase milk yield in cows", hyp: "how to increase milk yield in cow", wer: 0.1429, conf: 80, res: "Good" },
  { id: 7, ref: "symptoms of panama wilt in banana", hyp: "symptoms panama wilt banana", wer: 0.1667, conf: 94, res: "Good" },
  { id: 8, ref: "organic pest management for cotton", hyp: "organic pest management for cotton", wer: 0.0000, conf: 98, res: "Exact" },
  { id: 9, ref: "how much dap fertilizer for groundnut", hyp: "how much dap fertilizer groundnut", wer: 0.1429, conf: 86, res: "Good" },
  { id: 10, ref: "harvest time for blackgram in salem", hyp: "harvest timing for blackgram in salem", wer: 0.1667, conf: 82, res: "Good" },
];

const SAMPLE_QUERIES = [
  { text: "My rice crop has yellow leaves, what fertilizer should I use?" },
  { text: "How to control leaf blight disease in paddy crop?" },
  { text: "When is the best time to irrigate sugarcane in summer?" },
  { text: "What are suitable crops for black cotton soil?" },
  { text: "என் நெற்பயிரில் இலைகள் மஞ்சளாகின்றன, என்ன மருந்து தெளிக்கலாம்?" },
  { text: "தக்காளி செடியில் புழு தாக்குதல் கட்டுப்படுத்த என்ன வழி?" },
];

const PIPELINE_STAGES = [
  { id: "lang", iconName: "globe", label: "Language Detection" },
  { id: "trans", iconName: "translate", label: "Translation" },
  { id: "cls", iconName: "tag", label: "Classification" },
  { id: "ner", iconName: "entity", label: "NER" },
  { id: "qa", iconName: "question", label: "Question Answering" },
  { id: "rec", iconName: "check", label: "Recommendation" },
];

/* ============================== ICON WIRING ============================== */
safe("icon wiring", function(){
  const logo = document.getElementById("logoSprout");
  if (logo) logo.innerHTML = icon("leafDouble", 22);

  document.querySelectorAll("[data-icon]").forEach(el => {
    const name = el.dataset.icon;
    const size = parseInt(el.dataset.size, 10) || 18;
    el.innerHTML = icon(name, size);
  });
});

/* ============================== THEME MANAGEMENT ============================== */
function applyTheme(mode){
  currentTheme = mode;
  document.documentElement.setAttribute("data-theme", mode);
  const btn = document.getElementById("themeToggleBtn");
  if (btn){
    btn.innerHTML = icon(mode === "dark" ? "moon" : "sun", 18);
    btn.setAttribute("title", mode === "dark" ? "Switch to Light Mode" : "Switch to Dark Mode");
  }
  try { localStorage.setItem("agrovox-theme", mode); } catch(e){}
}

safe("theme init", function(){
  let saved = "light";
  try { saved = localStorage.getItem("agrovox-theme") || "light"; } catch(e){}
  applyTheme(saved);

  const toggleHandler = function(){
    applyTheme(currentTheme === "dark" ? "light" : "dark");
  };
  const tBtn = document.getElementById("themeToggleBtn");
  if (tBtn) tBtn.addEventListener("click", toggleHandler);

  const stBtn = document.getElementById("settingsThemeToggle");
  if (stBtn) stBtn.addEventListener("click", toggleHandler);
});

/* ============================== SPA VIEW SWITCHING ============================== */
function switchTab(tabId){
  activeTab = tabId;
  // Update sidebar active classes
  document.querySelectorAll(".sidebar-nav .nav-item").forEach(item => {
    if (item.dataset.tab === tabId){
      item.classList.add("active");
    } else {
      item.classList.remove("active");
    }
  });

  // Update views
  document.querySelectorAll(".spa-view").forEach(view => {
    view.classList.remove("active-view");
  });
  const targetView = document.getElementById("view-" + tabId);
  if (targetView){
    targetView.classList.add("active-view");
  }

  // Scroll to top of content
  const scrollContainer = document.querySelector(".content-scroll");
  if (scrollContainer) scrollContainer.scrollTop = 0;

  // Trigger sub-renderers if needed
  if (tabId === "dashboard") renderDashboard();
  if (tabId === "knowledge") renderKnowledge();
  if (tabId === "pipeline") renderPipelineView();
  if (tabId === "profile") renderProfileView();

  // Close mobile sidebar if open
  const sidebar = document.getElementById("appSidebar");
  if (sidebar) sidebar.classList.remove("open");
}

safe("spa navigation init", function(){
  document.querySelectorAll("[data-tab]").forEach(el => {
    el.addEventListener("click", function(e){
      e.preventDefault();
      const tab = this.dataset.tab;
      if (tab) switchTab(tab);
    });
  });

  // Mobile menu toggle
  const menuBtn = document.getElementById("mobileMenuBtn");
  const sidebar = document.getElementById("appSidebar");
  if (menuBtn && sidebar){
    menuBtn.addEventListener("click", function(){
      sidebar.classList.toggle("open");
    });
  }
});

/* ============================== FARMER PROFILE ============================== */
safe("farmer profile init", function(){
  let farmerId = "FMR_5U1d982";
  let farmerName = "";
  let farmerLang = "en";

  try {
    const savedId = localStorage.getItem("agrovox-farmer-id");
    if (savedId) farmerId = savedId;
    else localStorage.setItem("agrovox-farmer-id", farmerId);

    farmerName = localStorage.getItem("agrovox-farmer-name") || "";
    farmerLang = localStorage.getItem("agrovox-farmer-lang") || "en";
  } catch(e){}

  const idEl = document.getElementById("sidebarFarmerId");
  if (idEl) idEl.textContent = farmerId;

  const nameInput = document.getElementById("farmerNameInput");
  if (nameInput){
    nameInput.value = farmerName;
    nameInput.addEventListener("input", function(){
      try { localStorage.setItem("agrovox-farmer-name", this.value.trim()); } catch(e){}
    });
  }

  const langSelect = document.getElementById("farmerLangSelect");
  if (langSelect){
    langSelect.value = farmerLang;
    langSelect.addEventListener("change", function(){
      try { localStorage.setItem("agrovox-farmer-lang", this.value); } catch(e){}
    });
  }

  // Copy Farmer ID
  const copyBtn = document.getElementById("copyFarmerIdBtn");
  if (copyBtn){
    copyBtn.addEventListener("click", function(){
      navigator.clipboard.writeText(farmerId).then(() => {
        copyBtn.innerHTML = icon("check", 15);
        setTimeout(() => { copyBtn.innerHTML = icon("copy", 15); }, 2000);
      });
    });
  }

  // Resume Toggle & Submit
  const resumeSubmit = document.getElementById("resumeSubmitBtn");
  const resumeInput = document.getElementById("resumeIdInput");
  if (resumeSubmit && resumeInput){
    resumeSubmit.addEventListener("click", function(){
      const val = resumeInput.value.trim();
      if (val){
        farmerId = val;
        if (idEl) idEl.textContent = val;
        try { localStorage.setItem("agrovox-farmer-id", val); } catch(e){}
        resumeInput.value = "";
        alert("Resumed profile for " + val);
      }
    });
  }
});

/* ============================== NLP EVALUATION RESULTS TABLE ============================== */
function renderEvalTable(filterText = "", filterType = "all"){
  const tbody = document.getElementById("evalTableBody");
  if (!tbody) return;

  const q = filterText.toLowerCase();
  const filtered = EVAL_DATA.filter(row => {
    const matchesSearch = !q || row.ref.toLowerCase().includes(q) || row.hyp.toLowerCase().includes(q) || row.res.toLowerCase().includes(q);
    const matchesType = (filterType === "all") || (row.res.toLowerCase() === filterType.toLowerCase());
    return matchesSearch && matchesType;
  });

  const countBadge = document.getElementById("evalCountBadge");
  if (countBadge) countBadge.textContent = filtered.length + " queries";

  tbody.innerHTML = filtered.map(row => {
    const badgeClass = row.res.toLowerCase() === "good" ? "good" : (row.res.toLowerCase() === "exact" ? "exact" : "review");
    return `
      <tr>
        <td style="color:var(--text-dim); font-weight:600;">${row.id}</td>
        <td><strong>${escapeHtml(row.ref)}</strong></td>
        <td style="color:var(--text-muted);">${escapeHtml(row.hyp)}</td>
        <td style="font-family:var(--font-mono); font-size:0.82rem;">${row.wer.toFixed(4)}</td>
        <td style="font-family:var(--font-mono); font-size:0.82rem; font-weight:600;">${row.conf}%</td>
        <td><span class="result-badge ${badgeClass}">${row.res}</span></td>
        <td>
          <button class="table-more-btn" title="Inspect query in Assistant" onclick="testQueryInAssistant('${escapeHtml(row.ref)}')">
            ${icon("moreVertical", 16)}
          </button>
        </td>
      </tr>
    `;
  }).join("");
}

window.testQueryInAssistant = function(queryText){
  switchTab("assistant");
  const input = document.getElementById("queryInput");
  if (input){
    input.value = queryText;
    document.getElementById("analyzeBtn").click();
  }
};

safe("eval table init", function(){
  renderEvalTable();

  // Search input
  const searchInput = document.getElementById("evalQuerySearch");
  if (searchInput){
    searchInput.addEventListener("input", function(){
      renderEvalTable(this.value, evalFilterMode);
    });
  }

  // Filter button
  const filterBtn = document.getElementById("evalFilterBtn");
  if (filterBtn){
    filterBtn.addEventListener("click", function(){
      const modes = ["all", "good", "exact", "review"];
      const nextIdx = (modes.indexOf(evalFilterMode) + 1) % modes.length;
      evalFilterMode = modes[nextIdx];
      this.querySelector("span:last-child").textContent = evalFilterMode === "all" ? "Filter" : evalFilterMode.toUpperCase();
      const currentSearch = searchInput ? searchInput.value : "";
      renderEvalTable(currentSearch, evalFilterMode);
    });
  }

  // Export button
  const exportBtn = document.getElementById("evalExportBtn");
  if (exportBtn){
    exportBtn.addEventListener("click", function(){
      let csv = "ID,Reference Transcript,Hypothesis (Predicted),WER,Confidence,Result\n";
      EVAL_DATA.forEach(r => {
        csv += `"${r.id}","${r.ref}","${r.hyp}",${r.wer},${r.conf}%,"${r.res}"\n`;
      });
      const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.setAttribute("href", url);
      link.setAttribute("download", "agrovox_evaluation_results.csv");
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    });
  }

  // Sub-module pills
  document.querySelectorAll("#evalSubmodulePills .submodule-pill").forEach(pill => {
    pill.addEventListener("click", function(){
      document.querySelectorAll("#evalSubmodulePills .submodule-pill").forEach(p => p.classList.remove("active"));
      this.classList.add("active");
      activeSubtab = this.dataset.subtab;
      handleSubtabChange(activeSubtab);
    });
  });
});

function handleSubtabChange(subtab){
  const werCard = document.getElementById("cardAvgWer");
  const asrCard = document.getElementById("cardAsrStatus");
  const ttsCard = document.getElementById("cardTtsStatus");
  const queriesCard = document.getElementById("cardTotalQueries");

  if (subtab === "classification"){
    if (werCard) werCard.textContent = "98.96%";
    if (asrCard) asrCard.textContent = "MultiNB";
    if (ttsCard) ttsCard.textContent = "10 Classes";
    if (queriesCard) queriesCard.textContent = (AGROVOX_DATA.stats.total_intent_records || 300) + "";
  } else if (subtab === "ner"){
    if (werCard) werCard.textContent = "94.5% F1";
    if (asrCard) asrCard.textContent = "Gazetteer";
    if (ttsCard) ttsCard.textContent = "6 Types";
    if (queriesCard) queriesCard.textContent = "85 Sentences";
  } else if (subtab === "translation"){
    if (werCard) werCard.textContent = "88.4% Cov";
    if (asrCard) asrCard.textContent = "PhraseDict";
    if (ttsCard) ttsCard.textContent = "0.812 BLEU";
    if (queriesCard) queriesCard.textContent = AGROVOX_DATA.stats.translation_terms + " terms";
  } else if (subtab === "qa"){
    if (werCard) werCard.textContent = "94.1% Top-1";
    if (asrCard) asrCard.textContent = "TF-IDF";
    if (ttsCard) ttsCard.textContent = "0.961 MRR";
    if (queriesCard) queriesCard.textContent = AGROVOX_DATA.stats.qa_pairs + " pairs";
  } else if (subtab === "speech"){
    if (werCard) werCard.textContent = "18.3%";
    if (asrCard) asrCard.textContent = "Unavailable";
    if (ttsCard) ttsCard.textContent = "Unavailable";
    if (queriesCard) queriesCard.textContent = "128";
  }
}

/* ============================== AI ASSISTANT PIPELINE ============================== */
safe("assistant init", function(){
  // Setup sample chips
  const chipsContainer = document.getElementById("sampleChips");
  if (chipsContainer){
    chipsContainer.innerHTML = SAMPLE_QUERIES.map(q => `<span class="sample-chip">${escapeHtml(q.text)}</span>`).join("");
    chipsContainer.querySelectorAll(".sample-chip").forEach((chip, i) => {
      chip.addEventListener("click", function(){
        const input = document.getElementById("queryInput");
        if (input){
          input.value = SAMPLE_QUERIES[i].text;
          document.getElementById("analyzeBtn").click();
        }
      });
    });
  }

  // Render pipeline stages
  const pipeRow = document.getElementById("pipelineRow");
  if (pipeRow){
    pipeRow.innerHTML = PIPELINE_STAGES.map((s, idx) => `
      <div class="pipeline-node" id="step-${s.id}">
        <div class="node-circle">${icon(s.iconName, 18)}</div>
        <div class="node-label">${s.label}</div>
      </div>
      ${idx < PIPELINE_STAGES.length - 1 ? '<div class="pipeline-arrow">' + icon("arrowRight", 14) + '</div>' : ''}
    `).join("");
  }

  // Analyze button
  const analyzeBtn = document.getElementById("analyzeBtn");
  if (analyzeBtn){
    analyzeBtn.addEventListener("click", function(){
      const text = (document.getElementById("queryInput").value || "").trim();
      if (!text) return;
      runAssistantPipeline(text);
    });
  }

  // Mic Button (Web Speech API)
  const micBtn = document.getElementById("micBtn");
  const micStatus = document.getElementById("micStatus");
  if (micBtn && ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window)){
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    const recognition = new SpeechRecognition();
    recognition.continuous = false;
    recognition.interimResults = false;

    micBtn.addEventListener("click", function(){
      const lang = document.getElementById("langSelect").value;
      recognition.lang = lang === "ta" ? "ta-IN" : "en-US";
      try {
        recognition.start();
        micBtn.classList.add("listening");
        if (micStatus) micStatus.textContent = "Listening... speak your farming question now.";
      } catch(e){
        recognition.stop();
        micBtn.classList.remove("listening");
      }
    });

    recognition.onresult = function(event){
      const transcript = event.results[0][0].transcript;
      document.getElementById("queryInput").value = transcript;
      micBtn.classList.remove("listening");
      if (micStatus) micStatus.textContent = "";
      document.getElementById("analyzeBtn").click();
    };

    recognition.onerror = function(event){
      micBtn.classList.remove("listening");
      if (micStatus) micStatus.textContent = "Mic error: " + event.error;
    };
    recognition.onend = function(){
      micBtn.classList.remove("listening");
    };
  } else if (micBtn){
    micBtn.addEventListener("click", function(){
      alert("Speech recognition is not supported in this browser environment.");
    });
  }

  // Speak Answer Button (TTS)
  const speakBtn = document.getElementById("speakBtn");
  if (speakBtn && 'speechSynthesis' in window){
    speakBtn.addEventListener("click", function(){
      const text = document.getElementById("resAnswer").textContent;
      if (!text || text === "—") return;
      const utterance = new SpeechSynthesisUtterance(text);
      window.speechSynthesis.speak(utterance);
    });
  }

  // Feedback buttons
  const fbUp = document.getElementById("fbUp");
  const fbDown = document.getElementById("fbDown");
  if (fbUp && fbDown){
    fbUp.addEventListener("click", function(){
      this.classList.add("active");
      fbDown.classList.remove("active");
      alert("Thank you for your feedback! Marked as helpful.");
    });
    fbDown.addEventListener("click", function(){
      this.classList.add("active");
      fbUp.classList.remove("active");
      alert("Feedback received. We will improve our response ranking.");
    });
  }
});

function runAssistantPipeline(text){
  const stages = ["lang", "trans", "cls", "ner", "qa", "rec"];
  stages.forEach(s => {
    const el = document.getElementById("step-" + s);
    if (el){ el.classList.remove("active", "done"); }
  });

  // Animate stages step by step
  let currentIdx = 0;
  function nextStep(){
    if (currentIdx > 0){
      const prevEl = document.getElementById("step-" + stages[currentIdx - 1]);
      if (prevEl) { prevEl.classList.remove("active"); prevEl.classList.add("done"); }
    }
    if (currentIdx < stages.length){
      const currEl = document.getElementById("step-" + stages[currentIdx]);
      if (currEl) currEl.classList.add("active");
      currentIdx++;
      setTimeout(nextStep, 150);
    } else {
      finishPipelineInference(text);
    }
  }
  nextStep();
}

function finishPipelineInference(text){
  const langChoice = document.getElementById("langSelect").value;
  const result = runPipeline(text, langChoice === "auto" ? null : langChoice);

  const panel = document.getElementById("resultPanel");
  if (panel) panel.style.display = "block";

  // Intent
  const resIntent = document.getElementById("resIntent");
  if (resIntent) resIntent.textContent = result.intent;

  // Answer
  const resAnswer = document.getElementById("resAnswer");
  if (resAnswer) resAnswer.textContent = result.answer;

  const resConf = document.getElementById("resAnswerConf");
  const confPct = Math.round(result.qa_confidence * 100);
  if (resConf) resConf.textContent = confPct + "%";

  const resBar = document.getElementById("resAnswerBar");
  if (resBar) resBar.style.width = confPct + "%";

  // Language & Translation
  const resLang = document.getElementById("resLang");
  if (resLang) resLang.textContent = result.language === "ta" ? "Tamil (தமிழ்)" : "English";

  const resTrans = document.getElementById("resTranslated");
  if (resTrans){
    resTrans.textContent = result.language === "ta" && result.translated_text ? `(Translated: "${result.translated_text}")` : "";
  }

  // Entities
  const highlightEl = document.getElementById("resEntityHighlight");
  if (highlightEl){
    if (result.entities && result.entities.length > 0){
      highlightEl.innerHTML = result.entities.map(e => `<span class="ent-token ent-${e.type}">${escapeHtml(e.entity)} <small>(${e.type})</small></span>`).join(" ");
    } else {
      highlightEl.innerHTML = "<span style='color:var(--text-dim);'>No specific entities extracted</span>";
    }
  }

  // Save to history
  addConsultationHistory(text, result.intent, confPct);
}

function addConsultationHistory(query, intent, conf){
  const tbody = document.querySelector("#historyTable tbody");
  if (!tbody) return;
  const tr = document.createElement("tr");
  tr.innerHTML = `
    <td><strong>${escapeHtml(query)}</strong></td>
    <td><span class="result-badge good">${escapeHtml(intent)}</span></td>
    <td>${conf}%</td>
  `;
  tbody.insertBefore(tr, tbody.firstChild);
}

/* ============================== DASHBOARD & OTHER VIEWS ============================== */
function renderDashboard(){
  const grid = document.getElementById("moduleGrid");
  if (!grid || grid.children.length > 0) return;

  const modules = [
    { title: "Machine Translation", ic: "translate", desc: "Translates farming questions between Tamil and English using an agricultural glossary." },
    { title: "Text Classification", ic: "tag", desc: "Classifies queries into 10 agricultural intents with 98.96% held-out test accuracy." },
    { title: "Named Entity Recognition", ic: "entity", desc: "Extracts crops, diseases, pests, fertilizers, soil types and Tamil Nadu locations." },
    { title: "Question Answering", ic: "question", desc: "Cosine retrieval over curated agricultural knowledge base pairs." },
    { title: "Speech Processing", ic: "mic", desc: "Speech-to-text recognition and text-to-speech audio synthesis." },
    { title: "Multilingual Intelligence", ic: "globe", desc: "Automatic Unicode script detection for Tamil and English language routing." },
  ];

  grid.innerHTML = modules.map(m => `
    <div class="panel-card" style="padding:18px; margin-bottom:0;">
      <div style="display:flex; align-items:center; gap:10px; margin-bottom:10px;">
        <div class="summary-card-icon ic-green" style="width:36px; height:36px;">${icon(m.ic, 18)}</div>
        <h4 style="font-family:var(--font-main); font-size:0.95rem;">${m.title}</h4>
      </div>
      <p style="font-size:0.82rem; color:var(--text-muted);">${m.desc}</p>
    </div>
  `).join("");
}

function renderKnowledge(){
  const area = document.getElementById("kbContentArea");
  if (!area || area.children.length > 0) return;

  const sampleCrops = [
    { name: "Rice (Paddy / நெல்)", soil: "Clayey / Alluvial", season: "Kharif / Samba", water: "High (Standing water)", fert: "Urea, DAP, MOP" },
    { name: "Sugarcane (கரும்பு)", soil: "Deep loamy soil", season: "Year-round", water: "Medium-High", fert: "NPK 275:62.5:112.5 kg/ha" },
    { name: "Tomato (தக்காளி)", soil: "Sandy loam", season: "Rabi & Kharif", water: "Moderate", fert: "DAP, Potash, Vermicompost" },
    { name: "Cotton (பருத்தி)", soil: "Black cotton soil", season: "Kharif", water: "Moderate", fert: "Zinc sulphate, Nitrogen" },
  ];

  area.innerHTML = `
    <div style="display:grid; grid-template-columns:repeat(2, 1fr); gap:16px;">
      ${sampleCrops.map(c => `
        <div style="background:var(--bg-subtle); border:1px solid var(--border-color); border-radius:var(--radius-md); padding:16px;">
          <h4 style="font-family:var(--font-main); font-size:1rem; color:var(--brand-green); margin-bottom:6px;">${c.name}</h4>
          <p style="font-size:0.81rem; color:var(--text-muted); margin-bottom:3px;"><strong>Suitable Soil:</strong> ${c.soil}</p>
          <p style="font-size:0.81rem; color:var(--text-muted); margin-bottom:3px;"><strong>Season:</strong> ${c.season}</p>
          <p style="font-size:0.81rem; color:var(--text-muted); margin-bottom:3px;"><strong>Water Need:</strong> ${c.water}</p>
          <p style="font-size:0.81rem; color:var(--text-muted);"><strong>Recommended Fertilizer:</strong> ${c.fert}</p>
        </div>
      `).join("")}
    </div>
  `;
}

function renderPipelineView(){
  const container = document.getElementById("etlFunnelContainer");
  if (!container || container.children.length > 0) return;

  const stages = [
    { label: "1. Raw Agricultural Log Ingestion", count: "1,048 rows", pct: 100, color: "#2B75C2" },
    { label: "2. Missing Text & Blank Label Removal", count: "1,012 rows", pct: 96.5, color: "#1F8E96" },
    { label: "3. Language Detection & Unicode Cleaning", count: "985 rows", pct: 93.9, color: "#16A34A" },
    { label: "4. Punctuation Cleanup & Deduplication", count: "940 rows", pct: 89.6, color: "#D6972B" },
    { label: "5. Final Verified Training Corpus", count: "912 rows", pct: 87.0, color: "#1B7A43" },
  ];

  container.innerHTML = stages.map(s => `
    <div style="margin-bottom:14px;">
      <div style="display:flex; justify-content:space-between; font-size:0.83rem; font-weight:600; margin-bottom:4px;">
        <span>${s.label}</span>
        <span>${s.count} (${s.pct}%)</span>
      </div>
      <div class="conf-track" style="height:12px;"><div class="conf-fill" style="width:${s.pct}%; background:${s.color};"></div></div>
    </div>
  `).join("");
}

function renderProfileView(){
  const container = document.getElementById("profileDetailsCard");
  if (!container) return;
  const fid = localStorage.getItem("agrovox-farmer-id") || "FMR_5U1d982";
  const fname = localStorage.getItem("agrovox-farmer-name") || "Murugan";
  const flang = localStorage.getItem("agrovox-farmer-lang") || "en";

  container.innerHTML = `
    <p><strong>Farmer ID:</strong> <code style="font-family:var(--font-mono);">${escapeHtml(fid)}</code></p>
    <p><strong>Name:</strong> ${escapeHtml(fname || 'Not specified')}</p>
    <p><strong>Preferred Language:</strong> ${flang === 'ta' ? 'Tamil (தமிழ்)' : 'English'}</p>
    <p><strong>Location:</strong> Coimbatore, Tamil Nadu</p>
    <p><strong>Primary Crops:</strong> Paddy, Sugarcane, Tomato</p>
  `;
}

/* ============================== GLOBAL SEARCH ============================== */
safe("global search init", function(){
  const gSearch = document.getElementById("globalSearchInput");
  if (gSearch){
    gSearch.addEventListener("keydown", function(e){
      if (e.key === "Enter"){
        const val = this.value.trim();
        if (val){
          switchTab("assistant");
          const qInput = document.getElementById("queryInput");
          if (qInput){
            qInput.value = val;
            document.getElementById("analyzeBtn").click();
          }
        }
      }
    });
  }

  // Clear storage button
  const clearBtn = document.getElementById("clearStorageBtn");
  if (clearBtn){
    clearBtn.addEventListener("click", function(){
      if (confirm("Reset all session data and reload?")){
        localStorage.clear();
        location.reload();
      }
    });
  }
});

function escapeHtml(str){
  if (!str) return "";
  return String(str).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

})();
