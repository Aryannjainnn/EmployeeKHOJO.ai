/**
 * NexusSearch.jsx
 * Nexus — Talent Intelligence
 * Production-grade React component for intent-aware hybrid search UI.
 *
 * Key changes vs previous version:
 *   - "Why Selected" tab streams explanation word-by-word via SSE
 *   - Explanation only generated when tab is clicked (on-demand)
 *   - Score bars now map to actual reranker fields:
 *       BM25     → hybrid retrieval weight (alpha)
 *       Semantic → KG graph weight (beta)
 *       Hybrid   → fused final score
 *   - Signal Analysis tab shows the correct score field names
 */

const { useState, useEffect, useRef, useCallback } = React;

// ─────────────────────────────────────────────────────────────────────────────
// CSS-in-JS styles
// ─────────────────────────────────────────────────────────────────────────────
const GLOBAL_CSS = `
  @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  html, body, #root { height: 100%; }
  body {
    font-family: 'DM Sans', sans-serif;
    background: #06080d;
    color: #e2e8f4;
    overflow-x: hidden;
  }

  :root {
    --bg: #06080d;
    --panel: #0f1219;
    --panel2: #141820;
    --panel3: #191e28;
    --border: rgba(255,255,255,0.06);
    --border2: rgba(255,255,255,0.10);
    --border3: rgba(255,255,255,0.16);
    --accent: #00d4ff;
    --accent2: #7c5cfc;
    --accent3: #00ffb3;
    --accent4: #ff6b6b;
    --accent5: #ffd166;
    --text: #e2e8f4;
    --text2: #8892aa;
    --text3: #4a5268;
    --sidebar-w: 260px;
  }

  ::-webkit-scrollbar { width: 4px; height: 4px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 2px; }

  @keyframes nexus-slideUp {
    from { opacity: 0; transform: translateY(22px); }
    to   { opacity: 1; transform: translateY(0); }
  }
  @keyframes nexus-spin {
    to { transform: rotate(360deg); }
  }
  @keyframes nexus-pulse {
    0%, 100% { opacity: 1; box-shadow: 0 0 8px var(--accent); }
    50%       { opacity: 0.4; box-shadow: 0 0 3px var(--accent); }
  }
  @keyframes nexus-panelIn {
    from { opacity: 0; transform: translateY(-5px); }
    to   { opacity: 1; transform: translateY(0); }
  }
  @keyframes nexus-fadeIn {
    from { opacity: 0; }
    to   { opacity: 1; }
  }
  @keyframes nexus-cursor-blink {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0; }
  }

  .nexus-search-input:focus {
    border-color: rgba(0,212,255,0.5) !important;
    box-shadow: 0 0 0 4px rgba(0,212,255,0.07), 0 0 20px rgba(0,212,255,0.05) !important;
  }
  .nexus-topbar-input:focus {
    border-color: rgba(0,212,255,0.4) !important;
    box-shadow: 0 0 0 3px rgba(0,212,255,0.06) !important;
  }
  .nexus-search-btn:hover {
    opacity: 0.88;
    box-shadow: 0 0 26px rgba(124,92,252,0.4);
  }
  .nexus-search-btn:active { transform: scale(0.97); }
  .nexus-sug-chip:hover {
    border-color: rgba(124,92,252,0.4) !important;
    color: var(--accent2) !important;
    background: rgba(124,92,252,0.07) !important;
  }
  .nexus-card:hover {
    border-color: var(--border2) !important;
    box-shadow: 0 4px 32px rgba(0,0,0,0.35) !important;
  }
  .nexus-tab-btn:hover:not(.nexus-tab-active) {
    color: var(--text2) !important;
    background: var(--panel2) !important;
  }
  .nexus-kw-chip:hover {
    background: rgba(0,212,255,0.12) !important;
    border-color: rgba(0,212,255,0.3) !important;
  }
  .nexus-history-item:hover { background: var(--panel2) !important; border-color: var(--border) !important; }

  .streaming-cursor {
    display: inline-block;
    width: 2px;
    height: 1em;
    background: var(--accent2);
    margin-left: 2px;
    vertical-align: text-bottom;
    animation: nexus-cursor-blink 0.8s ease infinite;
  }
`;

function injectGlobalStyles() {
  if (document.getElementById("nexus-global-css")) return;
  const style = document.createElement("style");
  style.id = "nexus-global-css";
  style.textContent = GLOBAL_CSS;
  document.head.appendChild(style);
}

// ─────────────────────────────────────────────────────────────────────────────
// Demo data
// ─────────────────────────────────────────────────────────────────────────────
const DEMO_RESULTS = [
  {
    id: "c1", rank: 1, title: "Oracle PL/SQL Developer",
    industry: "Retail", location: "Pune / Madurai / Bengaluru", rrf_score: 0.0325,
    source: "both",
    preview: "Experienced Oracle PL/SQL Developer with 6+ years maintaining live production systems adhering to customer and HCL standards. Expert in database tuning, real-time data processing and ETL pipelines.",
    skills: ["PL/SQL", "Oracle DB", "SQL Tuning", "ETL", "Data Pipelines", "Stored Procedures"],
    explanation: {
      summary: "Strong keyword alignment via hybrid retrieval. Oracle PL/SQL maps directly to SQL requirements.",
      detail_bullets: [
        "Matched 'sql' → SQL Tuning via HAS_SKILL (1-hop, +3.000)",
        "Matched 'oracle' → Oracle DB via HAS_SKILL (1-hop, +3.000)",
        "Fusion: α=0.55·hybrid + β=0.45·kg",
      ],
      score_breakdown: { BM25: 0.55, Semantic: 0.45 },
      bm25_raw: 0.62, semantic_cosine: 0.38, hybrid_score: 0.9312,
      modifier_delta: 0,
      keyword_highlights: [{ term: "sql" }, { term: "oracle" }, { term: "data" }],
      skill_overlap: ["SQL", "Data Pipelines", "ETL"],
      transparency_notes: [],
      intent_alignment: "Intent: skill_search (confidence 0.87). Retrieved via: both.",
    },
  },
  {
    id: "c2", rank: 2, title: "Senior Data Analyst — BI & Reporting",
    industry: "FinTech", location: "Pune", rrf_score: 0.0301,
    source: "both",
    preview: "5 years driving data-driven decisions at a leading payments firm. Deep expertise in SQL, Power BI, Python scripting and stakeholder reporting across cross-functional teams.",
    skills: ["SQL", "Power BI", "Python", "Tableau", "Excel Advanced", "Data Storytelling"],
    explanation: {
      summary: "Direct title and skill match. Candidate is a senior data analyst with comprehensive SQL and BI toolkit.",
      detail_bullets: [
        "Matched 'python' → Python via HAS_SKILL (1-hop, +3.000)",
        "Matched 'sql' → SQL via HAS_SKILL (1-hop, +3.000)",
        "Fusion: α=0.50·hybrid + β=0.50·kg",
      ],
      score_breakdown: { BM25: 0.50, Semantic: 0.50 },
      bm25_raw: 0.55, semantic_cosine: 0.45, hybrid_score: 0.9781,
      modifier_delta: 0.05,
      keyword_highlights: [{ term: "data analyst" }, { term: "sql" }, { term: "python" }],
      skill_overlap: ["SQL", "Python", "Power BI"],
      transparency_notes: [],
      intent_alignment: "Intent: multi_skill (confidence 0.92). Retrieved via: both.",
    },
  },
];

const SUGGESTIONS = [
  { label: "Python ML Developer", q: "Python developer machine learning" },
  { label: "Senior Java Backend",  q: "Java backend engineer senior" },
  { label: "Data Analyst Pune",    q: "data analyst SQL Pune India" },
  { label: "DevOps Engineer",      q: "DevOps kubernetes docker cloud" },
  { label: "React Frontend",       q: "React frontend developer" },
  { label: "Project Manager",      q: "project manager agile scrum" },
];

// ─────────────────────────────────────────────────────────────────────────────
// Tiny helpers
// ─────────────────────────────────────────────────────────────────────────────
const rankMeta = (rank) => {
  if (rank === 1) return { bg: "rgba(255,209,102,0.12)", border: "rgba(255,209,102,0.25)", color: "var(--accent5)" };
  if (rank === 2) return { bg: "rgba(0,212,255,0.10)",   border: "rgba(0,212,255,0.22)",   color: "var(--accent)"  };
  if (rank === 3) return { bg: "rgba(124,92,252,0.10)",  border: "rgba(124,92,252,0.22)",  color: "var(--accent2)" };
  return           { bg: "var(--panel2)",                border: "var(--border2)",          color: "var(--text3)"   };
};

const stripGradient = (rank) => {
  if (rank <= 2) return "linear-gradient(90deg, var(--accent2), var(--accent), var(--accent3))";
  if (rank <= 4) return "linear-gradient(90deg, var(--accent), var(--accent2))";
  return                 "linear-gradient(90deg, var(--accent3), var(--accent))";
};

const tierOf = (pct) => (pct > 60 ? "strong" : pct > 30 ? "moderate" : "weak");

const TIER_STYLE = {
  strong:   { bg: "rgba(0,255,179,0.08)",   color: "var(--accent3)", border: "rgba(0,255,179,0.2)"   },
  moderate: { bg: "rgba(255,209,102,0.08)", color: "var(--accent5)", border: "rgba(255,209,102,0.2)" },
  weak:     { bg: "rgba(255,255,255,0.03)", color: "var(--text3)",   border: "var(--border)"         },
};

// ─────────────────────────────────────────────────────────────────────────────
// ScoreBar
// ─────────────────────────────────────────────────────────────────────────────
function ScoreBar({ label, pct, colorVar, gradient }) {
  const [w, setW] = useState(0);
  useEffect(() => {
    const t = setTimeout(() => setW(pct), 100);
    return () => clearTimeout(t);
  }, [pct]);

  return (
    <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
      <div style={{ fontFamily: "'Space Mono',monospace", fontSize: 9, color: "var(--text3)", width: 58, flexShrink: 0 }}>
        {label}
      </div>
      <div style={{ flex: 1, height: 3, background: "var(--panel2)", borderRadius: 2, overflow: "hidden" }}>
        <div style={{
          height: "100%", borderRadius: 2,
          background: gradient || colorVar || "var(--accent)",
          width: `${w}%`,
          transition: "width 0.9s cubic-bezier(0.16,1,0.3,1)",
        }} />
      </div>
      <div style={{ fontFamily: "'Space Mono',monospace", fontSize: 9, color: "var(--text3)", width: 32, textAlign: "right" }}>
        {pct}%
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// SignalBlock
// ─────────────────────────────────────────────────────────────────────────────
function SignalBlock({ topColor, topGradient, title, value, valueStyle, sub, pct }) {
  const ts = TIER_STYLE[tierOf(pct)] || TIER_STYLE.weak;
  return (
    <div style={{ padding: "14px 16px", borderRadius: 10, background: "var(--panel2)", border: "1px solid var(--border2)", position: "relative", overflow: "hidden" }}>
      <div style={{ position: "absolute", top: 0, left: 0, right: 0, height: 2, background: topGradient || topColor || "var(--accent)" }} />
      <div style={{ fontFamily: "'Space Mono',monospace", fontSize: 9, letterSpacing: "1.5px", color: "var(--text3)", textTransform: "uppercase", marginBottom: 8, fontWeight: 700 }}>
        {title}
      </div>
      <div style={{ fontFamily: "'Space Mono',monospace", fontSize: 22, fontWeight: 700, marginBottom: 3, letterSpacing: "-1px", ...valueStyle }}>
        {value}
      </div>
      <div style={{ fontSize: 10, color: "var(--text3)", marginBottom: 6 }}>{sub}</div>
      <span style={{ display: "inline-block", padding: "2px 8px", borderRadius: 4, fontFamily: "'Space Mono',monospace", fontSize: 8, fontWeight: 700, letterSpacing: 1, textTransform: "uppercase", background: ts.bg, color: ts.color, border: `1px solid ${ts.border}` }}>
        {tierOf(pct).toUpperCase()}
      </span>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// StreamingExplanation — fetches and renders SSE text word-by-word
// ─────────────────────────────────────────────────────────────────────────────
function StreamingExplanation({ rowData, onDone }) {
  const [text, setText]       = useState("");
  const [streaming, setStreaming] = useState(true);
  const [error, setError]     = useState(null);
  const abortRef              = useRef(null);

  useEffect(() => {
    const controller = new AbortController();
    abortRef.current = controller;
    let buffer = "";

    fetch("/explain", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ row_data: rowData }),
      signal: controller.signal,
    })
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const reader = res.body.getReader();
        const decoder = new TextDecoder();

        function pump() {
          reader.read().then(({ done, value }) => {
            if (done) {
              setStreaming(false);
              if (onDone) onDone();
              return;
            }
            buffer += decoder.decode(value, { stream: true });
            // Parse SSE lines
            const lines = buffer.split("\n");
            buffer = lines.pop(); // keep incomplete line
            for (const line of lines) {
              if (!line.startsWith("data: ")) continue;
              const chunk = line.slice(6);
              if (chunk === "[DONE]") {
                setStreaming(false);
                if (onDone) onDone();
                return;
              }
              setText((prev) => prev + (prev ? " " : "") + chunk);
            }
            pump();
          }).catch((err) => {
            if (err.name !== "AbortError") {
              setError("Stream interrupted.");
              setStreaming(false);
            }
          });
        }
        pump();
      })
      .catch((err) => {
        if (err.name !== "AbortError") {
          setError("Could not connect to the explainer service.");
          setStreaming(false);
        }
      });

    return () => controller.abort();
  }, []);

  if (error) {
    return (
      <div style={{ fontSize: 12, color: "var(--accent4)", padding: "10px 0" }}>
        {error}
      </div>
    );
  }

  return (
    <div style={{ fontSize: 13, color: "var(--text2)", lineHeight: 1.75 }}>
      {text || (streaming && <span style={{ color: "var(--text3)", fontSize: 12 }}>Generating…</span>)}
      {streaming && text && <span className="streaming-cursor" />}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// CandCard
// ─────────────────────────────────────────────────────────────────────────────
function CandCard({ r, idx }) {
  const [tab, setTab]           = useState("signals");
  const [visible, setVisible]   = useState(false);
  // null = not started, "loading" = fetching, "done" = complete
  const [explainState, setExplainState] = useState(null);

  useEffect(() => {
    const t = setTimeout(() => setVisible(true), idx * 75);
    return () => clearTimeout(t);
  }, [idx]);

  // Trigger streaming explanation when tab is opened for the first time
  const handleTabClick = (tabId) => {
    setTab(tabId);
    if (tabId === "explain" && explainState === null) {
      setExplainState("loading");
    }
  };

  const exp = r.explanation || {};
  const sb  = exp.score_breakdown || {};

  // BM25 slot = alpha (hybrid retrieval weight), Semantic slot = beta (KG weight)
  const alphaPct   = Math.round((sb.BM25     || 0) * 100);  // hybrid weight %
  const betaPct    = Math.round((sb.Semantic || 0) * 100);  // KG weight %
  // For the hybrid bar: show final score as % of 1.0
  const finalScore = exp.hybrid_score || r.rrf_score || 0;
  const finalPct   = Math.min(100, Math.round(finalScore * 100));

  // Raw values for Signal Analysis tab
  const hybridRaw  = (exp.bm25_raw        || 0).toFixed(4);  // hybrid_score
  const kgRaw      = (exp.semantic_cosine || 0).toFixed(4);  // kg_score
  const fusedScore = (exp.hybrid_score    || r.rrf_score || 0).toFixed(4);
  const modDelta   = exp.modifier_delta || 0;
  const sourceTag  = r.source || "both";

  const rk = rankMeta(r.rank);

  const TABS = [
    { id: "explain",  label: "Why Selected"    },
    { id: "signals",  label: "Signal Analysis" },
    { id: "keywords", label: "Keywords"        },
  ];

  return (
    <div
      className="nexus-card"
      style={{
        opacity:    visible ? 1 : 0,
        transform:  visible ? "translateY(0)" : "translateY(20px)",
        transition: "opacity 0.45s ease, transform 0.45s cubic-bezier(0.16,1,0.3,1)",
        background: "var(--panel)", border: "1px solid var(--border)", borderRadius: 18, overflow: "hidden",
      }}
    >
      {/* Colour strip */}
      <div style={{ height: 3, background: stripGradient(r.rank) }} />

      {/* Header */}
      <div style={{ display: "flex", alignItems: "flex-start", gap: 18, padding: "22px 24px 14px" }}>
        <div style={{ width: 38, height: 38, borderRadius: 10, display: "flex", alignItems: "center", justifyContent: "center", fontFamily: "'Space Mono',monospace", fontSize: 13, fontWeight: 700, flexShrink: 0, marginTop: 2, background: rk.bg, border: `1px solid ${rk.border}`, color: rk.color }}>
          #{r.rank}
        </div>
        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{ fontFamily: "'Syne',sans-serif", fontWeight: 700, fontSize: 20, letterSpacing: "-0.5px", color: "var(--text)", marginBottom: 8, lineHeight: 1.2 }}>
            {r.title}
          </div>
          <div style={{ display: "flex", alignItems: "center", flexWrap: "wrap", gap: 8, marginBottom: 10 }}>
            <span style={{ fontSize: 11, fontWeight: 500, color: "var(--text2)" }}>🏢 {r.industry}</span>
            {r.location && <>
              <span style={{ color: "var(--text3)", fontSize: 10 }}>·</span>
              <span style={{ fontSize: 11, color: "var(--text2)" }}>📍 {r.location}</span>
            </>}
            {/* Source badge */}
            <span style={{
              fontSize: 9, fontFamily: "'Space Mono',monospace",
              padding: "2px 7px", borderRadius: 4, fontWeight: 700, letterSpacing: 1,
              background: sourceTag === "both" ? "rgba(0,255,179,0.08)" : sourceTag === "hybrid_only" ? "rgba(0,212,255,0.08)" : "rgba(124,92,252,0.08)",
              color: sourceTag === "both" ? "var(--accent3)" : sourceTag === "hybrid_only" ? "var(--accent)" : "var(--accent2)",
              border: `1px solid ${sourceTag === "both" ? "rgba(0,255,179,0.2)" : sourceTag === "hybrid_only" ? "rgba(0,212,255,0.2)" : "rgba(124,92,252,0.2)"}`,
            }}>
              {sourceTag === "both" ? "H+KG" : sourceTag === "hybrid_only" ? "HYBRID" : "KG"}
            </span>
            <span style={{ marginLeft: "auto", fontFamily: "'Space Mono',monospace", fontSize: 9, color: "var(--text3)", padding: "2px 8px", background: "var(--panel2)", border: "1px solid var(--border)", borderRadius: 4 }}>
              {r.rrf_score.toFixed(4)}
            </span>
          </div>
          <div style={{ fontSize: 13, color: "var(--text2)", lineHeight: 1.65, display: "-webkit-box", WebkitLineClamp: 2, WebkitBoxOrient: "vertical", overflow: "hidden" }}>
            {r.preview}
          </div>
        </div>
      </div>

      {/* Skills */}
      {r.skills?.length > 0 && (
        <div style={{ padding: "0 24px 14px", display: "flex", flexWrap: "wrap", gap: 6 }}>
          {r.skills.slice(0, 8).map((s) => (
            <div key={s} style={{ padding: "3px 10px", borderRadius: 5, fontSize: 10, fontFamily: "'Space Mono',monospace", background: "rgba(0,255,179,0.05)", border: "1px solid rgba(0,255,179,0.13)", color: "var(--accent3)" }}>
              {s}
            </div>
          ))}
        </div>
      )}

      {/* Score bars — Hybrid Weight / KG Weight / Final Score */}
      <div style={{ padding: "10px 24px 14px", display: "flex", flexDirection: "column", gap: 5 }}>
        <ScoreBar label="Hybrid"   pct={alphaPct} colorVar="var(--accent)"  />
        <ScoreBar label="KG"       pct={betaPct}  colorVar="var(--accent2)" />
        <ScoreBar label="Fused"    pct={finalPct} gradient="linear-gradient(90deg,var(--accent2),var(--accent))" />
      </div>

      {/* Tab bar */}
      <div style={{ borderTop: "1px solid var(--border)", display: "flex" }}>
        {TABS.map((t, ti) => (
          <button
            key={t.id}
            className={`nexus-tab-btn${tab === t.id ? " nexus-tab-active" : ""}`}
            onClick={() => handleTabClick(t.id)}
            style={{
              flex: 1, padding: "11px 14px", textAlign: "center",
              fontFamily: "'Space Mono',monospace", fontSize: 9, letterSpacing: "1.2px",
              textTransform: "uppercase", cursor: "pointer",
              border: "none",
              borderRight: ti < TABS.length - 1 ? "1px solid var(--border)" : "none",
              borderBottom: `2px solid ${tab === t.id ? "var(--accent)" : "transparent"}`,
              background: tab === t.id ? "rgba(0,212,255,0.03)" : "transparent",
              color: tab === t.id ? "var(--accent)" : "var(--text3)",
              fontWeight: 700, transition: "all 0.15s",
            }}
          >
            {t.label}
            {t.id === "explain" && explainState === "loading" && (
              <span style={{ marginLeft: 5, display: "inline-block", width: 6, height: 6, borderRadius: "50%", background: "var(--accent2)", animation: "nexus-pulse 1.2s infinite", verticalAlign: "middle" }} />
            )}
          </button>
        ))}
      </div>

      {/* Tab panels */}
      <div style={{ borderTop: "1px solid var(--border)", background: "rgba(0,0,0,0.15)", animation: "nexus-panelIn 0.2s ease" }}>

        {/* ── Why Selected ── */}
        {tab === "explain" && (
          <div style={{ padding: "20px 24px 22px" }}>
            <div style={{ padding: "14px 16px", background: "var(--panel2)", borderRadius: 10, borderLeft: "3px solid var(--accent2)", marginBottom: 16 }}>
              <div style={{ fontFamily: "'Space Mono',monospace", fontSize: 9, letterSpacing: 2, color: "var(--accent2)", textTransform: "uppercase", marginBottom: 10, fontWeight: 700, display: "flex", alignItems: "center", gap: 8 }}>
                AI Explanation
                {explainState === "loading" && (
                  <div style={{ width: 12, height: 12, border: "1.5px solid var(--panel3)", borderTopColor: "var(--accent2)", borderRadius: "50%", animation: "nexus-spin 0.9s linear infinite" }} />
                )}
              </div>

              {explainState === null && (
                /* Not yet triggered — shouldn't happen since we auto-start on tab click */
                <div style={{ fontSize: 12, color: "var(--text3)" }}>
                  Loading explanation…
                </div>
              )}

              {(explainState === "loading" || explainState === "done") && (
                <StreamingExplanation
                  rowData={r}
                  onDone={() => setExplainState("done")}
                />
              )}
            </div>

            {/* Detail bullets (from match_reasons) */}
            {exp.detail_bullets?.length > 0 && (
              <ul style={{ listStyle: "none", marginTop: 12 }}>
                {exp.detail_bullets.map((b, i) => (
                  <li key={i} style={{ fontSize: 12, color: "var(--text2)", padding: "6px 0 6px 20px", position: "relative", borderBottom: "1px solid var(--border)", lineHeight: 1.6 }}>
                    <span style={{ position: "absolute", left: 4, color: "var(--accent2)", fontSize: 14, top: 5 }}>›</span>
                    {b}
                  </li>
                ))}
              </ul>
            )}

            {exp.transparency_notes?.length > 0 && (
              <div style={{ marginTop: 16 }}>
                <div style={{ fontFamily: "'Space Mono',monospace", fontSize: 9, letterSpacing: 2, color: "var(--text3)", textTransform: "uppercase", marginBottom: 8, fontWeight: 700 }}>
                  Transparency Notes
                </div>
                <ul style={{ listStyle: "none" }}>
                  {exp.transparency_notes.map((n, i) => (
                    <li key={i} style={{ fontSize: 11, color: "var(--text3)", padding: "5px 0 5px 16px", position: "relative", borderBottom: "1px solid var(--border)", lineHeight: 1.6 }}>
                      <span style={{ position: "absolute", left: 0, color: "var(--accent4)", fontSize: 10, top: 6 }}>→</span>
                      {n}
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        )}

        {/* ── Signal Analysis ── */}
        {tab === "signals" && (
          <div style={{ padding: "20px 24px 22px" }}>
            {/* Three score blocks */}
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 10, marginBottom: 16 }}>
              <SignalBlock
                topColor="var(--accent)"
                title="Hybrid Retrieval"
                value={hybridRaw}
                valueStyle={{ color: "var(--accent)" }}
                sub={`${alphaPct}% weight (α)`}
                pct={alphaPct}
              />
              <SignalBlock
                topColor="var(--accent2)"
                title="KG Graph Score"
                value={kgRaw}
                valueStyle={{ color: "var(--accent2)" }}
                sub={`${betaPct}% weight (β)`}
                pct={betaPct}
              />
              <SignalBlock
                topGradient="linear-gradient(90deg,var(--accent2),var(--accent))"
                title="Fused Score"
                value={fusedScore}
                valueStyle={{ background: "linear-gradient(90deg,var(--accent2),var(--accent))", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent", backgroundClip: "text" }}
                sub="α·hybrid + β·kg"
                pct={finalPct}
              />
            </div>

            {/* Modifier delta */}
            {modDelta !== 0 && (
              <div style={{ padding: "8px 12px", borderRadius: 6, background: modDelta > 0 ? "rgba(0,255,179,0.05)" : "rgba(255,107,107,0.05)", border: `1px solid ${modDelta > 0 ? "rgba(0,255,179,0.15)" : "rgba(255,107,107,0.15)"}`, fontSize: 11, color: modDelta > 0 ? "var(--accent3)" : "var(--accent4)", marginBottom: 12, fontFamily: "'Space Mono',monospace" }}>
                Modifier Δ: {modDelta > 0 ? "+" : ""}{modDelta.toFixed(4)} {modDelta > 0 ? "(experience / domain bonus)" : "(negated skill penalty)"}
              </div>
            )}

            {/* Source + intent */}
            <div style={{ padding: "12px 14px", background: "var(--panel2)", borderRadius: 8, border: "1px solid var(--border)", fontSize: 12, color: "var(--text2)", lineHeight: 1.6 }}>
              <strong style={{ fontSize: 10, fontFamily: "'Space Mono',monospace", color: "var(--text3)", letterSpacing: "1.5px", textTransform: "uppercase", display: "block", marginBottom: 6 }}>
                Intent Alignment
              </strong>
              {exp.intent_alignment || "General query match via hybrid + KG retrieval."}
            </div>
          </div>
        )}

        {/* ── Keywords ── */}
        {tab === "keywords" && (
          <div style={{ padding: "20px 24px 22px" }}>
            <div style={{ marginBottom: 16 }}>
              <div style={{ fontFamily: "'Space Mono',monospace", fontSize: 9, letterSpacing: 2, color: "var(--text3)", textTransform: "uppercase", marginBottom: 10, fontWeight: 700 }}>
                Matched Terms (from KG traversal)
              </div>
              <div style={{ display: "flex", flexWrap: "wrap", gap: 7 }}>
                {exp.keyword_highlights?.length > 0
                  ? exp.keyword_highlights.map((k, i) => (
                    <div key={i} className="nexus-kw-chip" style={{ padding: "4px 12px", borderRadius: 6, fontFamily: "'Space Mono',monospace", fontSize: 10, background: "rgba(0,212,255,0.06)", border: "1px solid rgba(0,212,255,0.16)", color: "var(--accent)", transition: "all 0.15s" }}>
                      {typeof k === "string" ? k : k.term}
                    </div>
                  ))
                  : <span style={{ color: "var(--text3)", fontSize: 11 }}>None detected</span>
                }
              </div>
            </div>
            {exp.skill_overlap?.length > 0 && (
              <div>
                <div style={{ fontFamily: "'Space Mono',monospace", fontSize: 9, letterSpacing: 2, color: "var(--text3)", textTransform: "uppercase", marginBottom: 10, fontWeight: 700 }}>
                  Skill Overlap with Query
                </div>
                <div style={{ display: "flex", flexWrap: "wrap", gap: 7 }}>
                  {exp.skill_overlap.map((s, i) => (
                    <div key={i} style={{ padding: "4px 12px", borderRadius: 6, fontFamily: "'Space Mono',monospace", fontSize: 10, background: "rgba(0,255,179,0.06)", border: "1px solid rgba(0,255,179,0.16)", color: "var(--accent3)" }}>
                      {s}
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Sidebar
// ─────────────────────────────────────────────────────────────────────────────
function Sidebar() {
  const INFO_BLOCKS = [
    {
      label: "Retrieval Mode",
      content: "Intent-aware fusion of BM25+dense hybrid retrieval and 4-hop Knowledge Graph traversal, re-ranked by learned α/β weights.",
      badge: "⬡ Hybrid + KG",
    },
    {
      label: "How it works",
      content: (
        <div style={{ fontSize: 12, color: "var(--text2)", lineHeight: 1.7 }}>
          {[
            "1. Query → intent + entity extraction",
            "2. Hybrid retrieval (BM25 + FAISS)",
            "3. KG traversal (4-hop graph)",
            "4. Intent-aware α/β re-ranking",
            "5. On-demand AI explanation",
          ].map((s, i) => (
            <div key={i} style={{ marginBottom: 6 }}>{s}</div>
          ))}
        </div>
      ),
    },
    {
      label: "Score Legend",
      content: (
        <div style={{ fontSize: 11, color: "var(--text2)", lineHeight: 1.8 }}>
          <div><span style={{ color: "var(--accent)" }}>Hybrid</span> — BM25+dense RRF signal weight (α)</div>
          <div><span style={{ color: "var(--accent2)" }}>KG</span> — graph traversal score weight (β)</div>
          <div><span style={{ color: "var(--accent3)" }}>Fused</span> — α·hybrid + β·kg + modifier Δ</div>
        </div>
      ),
    },
  ];

  return (
    <aside style={{ width: "var(--sidebar-w)", minHeight: "100vh", background: "var(--panel)", borderRight: "1px solid var(--border)", display: "flex", flexDirection: "column", position: "fixed", left: 0, top: 0, bottom: 0, zIndex: 50, overflow: "hidden" }}>
      <div style={{ position: "absolute", top: 0, left: 0, right: 0, height: 200, background: "radial-gradient(ellipse at top left, rgba(124,92,252,0.15), transparent 70%)", pointerEvents: "none" }} />

      <div style={{ padding: "20px 20px 16px", borderBottom: "1px solid var(--border)", position: "relative" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <div style={{ width: 34, height: 34, background: "linear-gradient(135deg,var(--accent2),var(--accent))", borderRadius: 8, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 16, flexShrink: 0 }}>⬡</div>
          <div>
            <div style={{ fontFamily: "'Syne',sans-serif", fontWeight: 800, fontSize: 18, letterSpacing: "-0.5px" }}>Nexus</div>
            <div style={{ fontFamily: "'Space Mono',monospace", fontSize: 9, color: "var(--text3)", letterSpacing: 2, marginTop: 1 }}>TALENT INTELLIGENCE v2</div>
          </div>
        </div>
      </div>

      <div style={{ flex: 1, padding: "24px 20px", display: "flex", flexDirection: "column", gap: 16, overflowY: "auto" }}>
        {INFO_BLOCKS.map((blk, i) => (
          <div key={i} style={{ padding: "14px 16px", borderRadius: 10, background: "var(--panel2)", border: "1px solid var(--border)" }}>
            <div style={{ fontFamily: "'Space Mono',monospace", fontSize: 9, letterSpacing: 2, textTransform: "uppercase", color: "var(--text3)", marginBottom: 8, fontWeight: 700 }}>
              {blk.label}
            </div>
            {typeof blk.content === "string"
              ? <div style={{ fontSize: 12, color: "var(--text2)", lineHeight: 1.6 }}>{blk.content}</div>
              : blk.content
            }
            {blk.badge && (
              <div style={{ display: "inline-flex", alignItems: "center", gap: 5, padding: "3px 10px", borderRadius: 5, fontFamily: "'Space Mono',monospace", fontSize: 9, marginTop: 6, background: "rgba(0,212,255,0.06)", border: "1px solid rgba(0,212,255,0.14)", color: "var(--accent)" }}>
                {blk.badge}
              </div>
            )}
          </div>
        ))}
      </div>

      <div style={{ padding: "14px 16px", borderTop: "1px solid var(--border)", fontFamily: "'Space Mono',monospace", fontSize: 9, color: "var(--text3)", letterSpacing: 1 }}>
        NEXUS · HYBRID+KG RETRIEVAL · 2025
      </div>
    </aside>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Hero
// ─────────────────────────────────────────────────────────────────────────────
function Hero({ onSearch }) {
  const [q, setQ] = useState("");
  const inputRef  = useRef(null);
  const submit    = (val) => { const v = (val ?? q).trim(); if (v) onSearch(v); };

  return (
    <div style={{ animation: "nexus-slideUp 0.5s ease both" }}>
      <div style={{ padding: "64px 0 48px", textAlign: "center" }}>
        <div style={{ display: "inline-flex", alignItems: "center", gap: 6, padding: "4px 12px 4px 8px", border: "1px solid rgba(0,212,255,0.2)", borderRadius: 20, background: "rgba(0,212,255,0.04)", fontFamily: "'Space Mono',monospace", fontSize: 9, letterSpacing: "2.5px", color: "var(--accent)", textTransform: "uppercase", marginBottom: 28 }}>
          <div style={{ width: 5, height: 5, borderRadius: "50%", background: "var(--accent)", boxShadow: "0 0 8px var(--accent)", animation: "nexus-pulse 2s infinite" }} />
          Hybrid + Knowledge Graph Retrieval
        </div>

        <h1 style={{ fontFamily: "'Syne',sans-serif", fontWeight: 800, fontSize: "clamp(40px, 5.5vw, 68px)", letterSpacing: "-2.5px", lineHeight: 1, marginBottom: 16 }}>
          <span style={{ color: "var(--text)", display: "block" }}>Find the Right</span>
          <span style={{ display: "block", background: "linear-gradient(90deg,var(--accent) 0%,var(--accent2) 50%,var(--accent3) 100%)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent", backgroundClip: "text" }}>
            Talent, Instantly
          </span>
        </h1>

        <p style={{ fontSize: 14, fontWeight: 400, color: "var(--text2)", maxWidth: 480, margin: "0 auto 40px", lineHeight: 1.7 }}>
          Intent-aware search combining BM25 keyword precision, FAISS semantic search, and 4-hop Knowledge Graph traversal — re-ranked by learned weights.
        </p>
      </div>

      <div style={{ maxWidth: 680, margin: "0 auto" }}>
        <div style={{ display: "flex", gap: 10, marginBottom: 14 }}>
          <div style={{ flex: 1, position: "relative" }}>
            <span style={{ position: "absolute", left: 18, top: "50%", transform: "translateY(-50%)", fontSize: 18, color: "var(--text3)", pointerEvents: "none" }}>⌕</span>
            <input
              ref={inputRef}
              className="nexus-search-input"
              value={q}
              onChange={(e) => setQ(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && submit()}
              placeholder="e.g. Python developer with ML experience…"
              autoComplete="off"
              style={{ width: "100%", padding: "16px 18px 16px 52px", background: "var(--panel)", border: "1px solid var(--border2)", borderRadius: 14, color: "var(--text)", fontFamily: "'DM Sans',sans-serif", fontSize: 14, outline: "none", transition: "border-color 0.2s, box-shadow 0.2s" }}
            />
          </div>
          <button
            className="nexus-search-btn"
            onClick={() => submit()}
            style={{ padding: "0 28px", background: "linear-gradient(135deg,var(--accent2) 0%,var(--accent) 100%)", border: "none", borderRadius: 14, color: "#fff", fontFamily: "'Syne',sans-serif", fontSize: 13, fontWeight: 700, letterSpacing: "0.5px", cursor: "pointer", whiteSpace: "nowrap", transition: "opacity 0.15s, transform 0.1s, box-shadow 0.2s" }}
          >
            Search →
          </button>
        </div>

        <div style={{ display: "flex", flexWrap: "wrap", gap: 7, justifyContent: "center" }}>
          {SUGGESTIONS.map((s) => (
            <div
              key={s.q}
              className="nexus-sug-chip"
              onClick={() => { setQ(s.q); submit(s.q); }}
              style={{ padding: "5px 14px", borderRadius: 20, border: "1px solid var(--border2)", background: "var(--panel)", color: "var(--text2)", fontSize: 11, fontWeight: 500, cursor: "pointer", transition: "all 0.15s" }}
            >
              {s.label}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// ResultsView
// ─────────────────────────────────────────────────────────────────────────────
function ResultsView({ query, onNewSearch }) {
  const [loading,  setLoading]  = useState(true);
  const [results,  setResults]  = useState([]);
  const [meta,     setMeta]     = useState(null);
  const [topQ,     setTopQ]     = useState(query);

  useEffect(() => {
    setLoading(true);
    setResults([]);
    const t0 = Date.now();

    fetch(`/search?q=${encodeURIComponent(query)}&k=10&mode=hybrid`)
      .then((r) => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json(); })
      .then((data) => {
        setMeta({
          intent:   data.intent?.primary || "talent_search",
          count:    data.results.length,
          total:    data.total_candidates || "—",
          ms:       data.timing_ms || (Date.now() - t0),
          expanded: data.expanded_queries || [],
        });
        setResults(data.results || []);
        setLoading(false);
      })
      .catch(() => {
        setMeta({ intent: "talent_search", count: DEMO_RESULTS.length, total: 142, ms: Date.now() - t0, expanded: [query] });
        setResults(DEMO_RESULTS);
        setLoading(false);
      });
  }, [query]);

  const handleTopSearch = () => { const v = topQ.trim(); if (v && v !== query) onNewSearch(v); };

  return (
    <div>
      {/* Sticky top search bar */}
      <div style={{ position: "sticky", top: 0, zIndex: 40, background: "rgba(6,8,13,0.88)", backdropFilter: "blur(12px)", borderBottom: "1px solid var(--border)", padding: "12px 0", marginBottom: 24, display: "flex", alignItems: "center", gap: 12 }}>
        <div style={{ flex: 1, position: "relative", maxWidth: 600 }}>
          <span style={{ position: "absolute", left: 14, top: "50%", transform: "translateY(-50%)", fontSize: 14, color: "var(--text3)", pointerEvents: "none" }}>⌕</span>
          <input
            className="nexus-topbar-input"
            value={topQ}
            onChange={(e) => setTopQ(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleTopSearch()}
            style={{ width: "100%", padding: "9px 16px 9px 38px", background: "var(--panel2)", border: "1px solid var(--border2)", borderRadius: 10, color: "var(--text)", fontFamily: "'DM Sans',sans-serif", fontSize: 13, outline: "none", transition: "border-color 0.2s, box-shadow 0.2s" }}
          />
        </div>
        <button
          onClick={handleTopSearch}
          style={{ padding: "8px 20px", background: "linear-gradient(135deg,var(--accent2),var(--accent))", border: "none", borderRadius: 10, color: "#fff", fontFamily: "'Syne',sans-serif", fontSize: 12, fontWeight: 700, cursor: "pointer", transition: "opacity 0.15s" }}
        >
          Search →
        </button>
      </div>

      {/* Meta bar */}
      {!loading && meta && (
        <div style={{ display: "flex", alignItems: "center", gap: 10, paddingBottom: 16, flexWrap: "wrap", borderBottom: "1px solid var(--border)", marginBottom: 24 }}>
          <div style={{ display: "inline-flex", alignItems: "center", gap: 7, padding: "5px 12px 5px 8px", borderRadius: 7, background: "rgba(124,92,252,0.1)", border: "1px solid rgba(124,92,252,0.22)", color: "var(--accent2)", fontSize: 11, fontWeight: 600 }}>
            <div style={{ width: 6, height: 6, borderRadius: "50%", background: "var(--accent2)", boxShadow: "0 0 6px var(--accent2)" }} />
            Intent: {meta.intent.replace(/_/g, " ")}
          </div>
          <span style={{ fontFamily: "'Space Mono',monospace", fontSize: 10, color: "var(--text3)" }}>
            {meta.count} results / {meta.total} candidates
          </span>
          <div style={{ marginLeft: "auto", fontFamily: "'Space Mono',monospace", fontSize: 10, color: "var(--text3)", display: "flex", alignItems: "center", gap: 5 }}>
            <div style={{ width: 5, height: 5, borderRadius: "50%", background: "var(--accent3)", boxShadow: "0 0 6px var(--accent3)" }} />
            {meta.ms}ms
          </div>
          {meta.expanded.length > 1 && (
            <div style={{ width: "100%", fontFamily: "'Space Mono',monospace", fontSize: 10, color: "var(--text3)", lineHeight: 1.8 }}>
              Query expanded →{" "}
              {meta.expanded.map((t, i) => (
                <span key={i} style={{ color: "var(--text2)", background: "var(--panel2)", padding: "1px 6px", borderRadius: 3, border: "1px solid var(--border)", margin: "0 2px" }}>{t}</span>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Loading spinner */}
      {loading && (
        <div style={{ textAlign: "center", padding: "80px 0" }}>
          <div style={{ width: 44, height: 44, border: "2px solid var(--panel3)", borderTopColor: "var(--accent)", borderRightColor: "var(--accent2)", borderRadius: "50%", animation: "nexus-spin 0.9s linear infinite", margin: "0 auto 20px" }} />
          <div style={{ fontFamily: "'Space Mono',monospace", fontSize: 11, color: "var(--text3)", letterSpacing: 1 }}>
            Processing intent · Graph traversal · Re-ranking…
          </div>
        </div>
      )}

      {/* Results */}
      {!loading && (
        <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
          {results.length === 0
            ? (
              <div style={{ textAlign: "center", padding: "80px 0", color: "var(--text3)" }}>
                <div style={{ fontSize: 48, marginBottom: 16, opacity: 0.4 }}>◎</div>
                <div style={{ fontFamily: "'Syne',sans-serif", fontSize: 18, fontWeight: 700, color: "var(--text2)", marginBottom: 8 }}>No candidates found</div>
                <div style={{ fontSize: 13 }}>Try refining your search.</div>
              </div>
            )
            : results.map((r, i) => <CandCard key={r.id || i} r={r} idx={i} />)
          }
        </div>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Root App
// ─────────────────────────────────────────────────────────────────────────────
function NexusSearch() {
  const [query, setQuery] = useState(null);
  useEffect(() => { injectGlobalStyles(); }, []);

  return (
    <div style={{ display: "flex", minHeight: "100vh" }}>
      <div style={{ position: "fixed", top: -150, left: "50%", transform: "translateX(-50%)", width: 600, height: 300, background: "radial-gradient(ellipse,rgba(124,92,252,0.07) 0%,transparent 70%)", pointerEvents: "none", zIndex: 0 }} />
      <div style={{ position: "fixed", inset: 0, backgroundImage: "repeating-linear-gradient(0deg,transparent,transparent 2px,rgba(0,0,0,0.03) 2px,rgba(0,0,0,0.03) 4px)", pointerEvents: "none", zIndex: 1000 }} />

      <Sidebar />

      <main style={{ marginLeft: "var(--sidebar-w)", flex: 1, minHeight: "100vh", display: "flex", flexDirection: "column" }}>
        <div style={{ flex: 1, padding: "0 32px 60px", maxWidth: 900, width: "100%", position: "relative", zIndex: 1 }}>
          {query === null
            ? <Hero onSearch={(q) => setQuery(q)} />
            : <ResultsView key={query} query={query} onNewSearch={(q) => setQuery(q)} />
          }
        </div>
      </main>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Mount
// ─────────────────────────────────────────────────────────────────────────────
const _rootEl = document.getElementById("root");
if (_rootEl) {
  ReactDOM.createRoot(_rootEl).render(<NexusSearch />);
}