import { useState, useRef } from "react";

const API = import.meta.env.VITE_API_URL || "http://localhost:8000";

function SourceCard({ s, i }) {
  const [open, setOpen] = useState(false);
  const pct = Math.round(s.confidence * 100);
  const color = pct > 80 ? "#22c55e" : pct > 50 ? "#f59e0b" : "#ef4444";
  return (
    <div style={{ border: "1px solid #27272a", borderRadius: 8, padding: "10px 14px", marginBottom: 8, background: "#18181b" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <span style={{ fontSize: 13, color: "#a1a1aa" }}>
          [{i}] {s.filename} — page {s.page}
        </span>
        <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
          <span style={{ fontSize: 11, padding: "2px 8px", borderRadius: 99, background: color + "22", color, border: `1px solid ${color}44` }}>
            {pct}%
          </span>
          <button onClick={() => setOpen(o => !o)}
            style={{ fontSize: 11, color: "#6366f1", background: "none", border: "none", cursor: "pointer" }}>
            {open ? "hide" : "snippet"}
          </button>
        </div>
      </div>
      {open && <p style={{ fontSize: 12, color: "#71717a", marginTop: 8, lineHeight: 1.6 }}>{s.snippet}</p>}
    </div>
  );
}

export default function App() {
  const [query, setQuery] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [docs, setDocs] = useState([]);
  const [uploading, setUploading] = useState(false);
  const fileRef = useRef();

  async function fetchDocs() {
    const r = await fetch(`${API}/documents`);
    const d = await r.json();
    setDocs(d.documents || []);
  }

  async function upload(e) {
    const file = e.target.files[0];
    if (!file) return;
    setUploading(true);
    const form = new FormData();
    form.append("file", file);
    try {
      await fetch(`${API}/upload`, { method: "POST", body: form });
      setTimeout(fetchDocs, 3000); // wait for indexing
    } catch { setError("Upload failed"); }
    setUploading(false);
    fileRef.current.value = "";
  }

  async function ask() {
    if (!query.trim()) return;
    setLoading(true); setError(""); setResult(null);
    try {
      const r = await fetch(`${API}/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, top_k: 5 }),
      });
      if (!r.ok) throw new Error((await r.json()).detail);
      setResult(await r.json());
    } catch (e) { setError(e.message); }
    setLoading(false);
  }

  return (
    <div style={{ minHeight: "100vh", background: "#09090b", color: "#fafafa", fontFamily: "system-ui, sans-serif" }}>
      {/* header */}
      <div style={{ borderBottom: "1px solid #27272a", padding: "16px 0" }}>
        <div style={{ maxWidth: 760, margin: "0 auto", padding: "0 20px", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <div>
            <span style={{ fontSize: 20, fontWeight: 700 }}>AskMyDocs</span>
            <span style={{ fontSize: 12, color: "#52525b", marginLeft: 10 }}>by Fardeen NS Khan · NIT Surat</span>
          </div>
          <label style={{ cursor: "pointer", background: "#6366f1", color: "#fff", borderRadius: 8, padding: "7px 16px", fontSize: 13, fontWeight: 600 }}>
            {uploading ? "Indexing..." : "+ Upload PDF"}
            <input ref={fileRef} type="file" accept=".pdf" onChange={upload} style={{ display: "none" }} disabled={uploading} />
          </label>
        </div>
      </div>

      <div style={{ maxWidth: 760, margin: "0 auto", padding: "28px 20px" }}>

        {/* indexed docs */}
        {docs.length > 0 && (
          <div style={{ marginBottom: 20, padding: "12px 16px", background: "#18181b", borderRadius: 10, border: "1px solid #27272a" }}>
            <p style={{ fontSize: 11, color: "#52525b", textTransform: "uppercase", letterSpacing: ".06em", marginBottom: 8 }}>Indexed</p>
            {docs.map(d => (
              <div key={d.filename} style={{ fontSize: 13, color: "#a1a1aa", padding: "4px 0", borderBottom: "1px solid #27272a" }}>
                {d.filename} <span style={{ color: "#52525b" }}>· {d.chunks} chunks · {d.doc_type}</span>
              </div>
            ))}
          </div>
        )}

        {/* query bar */}
        <div style={{ display: "flex", gap: 10, marginBottom: 20 }}>
          <input value={query} onChange={e => setQuery(e.target.value)}
            onKeyDown={e => e.key === "Enter" && !loading && ask()}
            placeholder="Ask a question about your documents..."
            style={{ flex: 1, padding: "12px 16px", borderRadius: 10, border: "1px solid #27272a", background: "#18181b", color: "#fafafa", fontSize: 14, outline: "none" }} />
          <button onClick={ask} disabled={loading || !query.trim()}
            style={{ padding: "12px 24px", borderRadius: 10, border: "none", background: loading ? "#3f3f46" : "#6366f1", color: "#fff", fontWeight: 600, fontSize: 14, cursor: "pointer" }}>
            {loading ? "..." : "Ask"}
          </button>
        </div>

        {/* error */}
        {error && <div style={{ padding: "10px 14px", background: "#450a0a", border: "1px solid #7f1d1d", borderRadius: 8, color: "#fca5a5", fontSize: 13, marginBottom: 16 }}>{error}</div>}

        {/* answer */}
        {result && (
          <div style={{ background: "#18181b", border: "1px solid #27272a", borderRadius: 12, padding: "20px 24px" }}>
            <p style={{ fontSize: 14, lineHeight: 1.8, color: "#e4e4e7", whiteSpace: "pre-wrap", marginBottom: 16 }}>{result.answer}</p>
            <div style={{ display: "flex", gap: 16, fontSize: 11, color: "#52525b", borderTop: "1px solid #27272a", paddingTop: 12, marginBottom: 12 }}>
              <span>⏱ {result.latency_ms}ms</span>
              <span>🔢 {result.tokens} tokens</span>
              <span>📄 {result.sources?.length} sources</span>
            </div>
            {result.sources?.map((s, i) => <SourceCard key={i} s={s} i={i + 1} />)}
          </div>
        )}

        {/* empty state */}
        {!result && !loading && !error && (
          <div style={{ textAlign: "center", padding: "60px 0", color: "#3f3f46" }}>
            <div style={{ fontSize: 48, marginBottom: 12 }}>📄</div>
            <p style={{ fontSize: 14 }}>Upload a PDF then ask anything about it</p>
            <p style={{ fontSize: 12, marginTop: 6 }}>Answers include exact page citations</p>
          </div>
        )}
      </div>
    </div>
  );
}