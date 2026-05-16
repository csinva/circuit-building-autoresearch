# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Build an interactive HTML report aggregating per-task results across runs/."""
import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
OUT = ROOT / "analyze" / "report.html"


def parse_params(s: str) -> float:
    try:
        return float(s)
    except (ValueError, TypeError):
        return float("nan")


def collect():
    records = []
    for run_dir in sorted(RUNS.iterdir()):
        if not run_dir.is_dir():
            continue
        csv_path = run_dir / "results" / "overall_results.csv"
        if not csv_path.exists():
            continue
        lib_dir = run_dir / "interpretable_transformers_lib"
        for order, row in enumerate(csv.DictReader(csv_path.open())):
            model_name = row.get("model_shorthand_name", "")
            code = ""
            if lib_dir.exists():
                code_file = lib_dir / f"{model_name}.py"
                if code_file.exists():
                    code = code_file.read_text()
            records.append({
                "run": run_dir.name,
                "task": row.get("task", "") or run_dir.name,
                "order": order,
                "accuracy": float(row["accuracy"]) if row.get("accuracy") else 0.0,
                "status": row.get("status", ""),
                "model": model_name,
                "n_params": parse_params(row.get("n_params", "")),
                "description": row.get("description", ""),
                "code": code,
            })
    return records


HTML_TEMPLATE = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>Runs Overall</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/prismjs@1.29.0/components/prism-core.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/prismjs@1.29.0/components/prism-python.min.js"></script>
<style>
  :root {
    --bg: #ffffff;
    --fg: #1a1a1a;
    --muted: #6b7280;
    --border: #e5e7eb;
    --panel: #fafafa;
    --accent: #2563eb;
    --code-bg: #fbfbfb;
    --code-fg: #1f2937;
  }
  * { box-sizing: border-box; }
  html, body { background: var(--bg); color: var(--fg); margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; font-size: 14px; line-height: 1.5; }
  .wrap { max-width: 1400px; margin: 0 auto; padding: 24px; }
  h1 { font-size: 22px; font-weight: 600; margin: 0 0 4px; }
  h2 { font-size: 16px; font-weight: 600; margin: 24px 0 8px; color: var(--fg); }
  p.sub { color: var(--muted); margin: 0 0 16px; }
  .split { display: grid; grid-template-columns: 1.1fr 1fr; gap: 20px; align-items: stretch; }
  .card { background: var(--bg); border: 1px solid var(--border); border-radius: 8px; padding: 16px; }
  .controls { display: flex; gap: 12px; align-items: center; margin-bottom: 8px; flex-wrap: wrap; }
  select, label { font-size: 13px; }
  select { padding: 6px 10px; border: 1px solid var(--border); border-radius: 6px; background: var(--bg); color: var(--fg); }
  #plot { width: 100%; height: 560px; }
  #summary { width: 100%; height: 600px; }
  .meta { font-size: 13px; color: var(--fg); }
  .meta .row { margin: 2px 0; }
  .meta b { color: var(--muted); font-weight: 500; margin-right: 4px; }
  .empty { color: var(--muted); font-style: italic; }
  pre { background: var(--code-bg); color: var(--code-fg); border: 1px solid var(--border); border-radius: 6px;
        padding: 12px; max-height: 460px; overflow: auto; font-size: 12px; margin: 0;
        font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
  /* Prism light theme overrides for white background */
  code[class*="language-"], pre[class*="language-"] { color: #1f2937; background: transparent; text-shadow: none; }
  .token.comment, .token.prolog, .token.doctype, .token.cdata { color: #6b7280; font-style: italic; }
  .token.punctuation { color: #374151; }
  .token.property, .token.tag, .token.boolean, .token.number, .token.constant, .token.symbol { color: #b45309; }
  .token.selector, .token.attr-name, .token.string, .token.char, .token.builtin { color: #047857; }
  .token.operator, .token.entity, .token.url, .token.variable { color: #1f2937; }
  .token.keyword { color: #2563eb; }
  .token.function, .token.class-name { color: #7c3aed; }
  .footer { color: var(--muted); font-size: 12px; margin-top: 24px; text-align: center; }
</style></head>
<body>
<div class="wrap">
  <h1>Circuit-building runs — overall results</h1>
  <p class="sub">Accuracy vs. parameter count per task. Each point is labeled with the order it was tried. Hover for details; click to view the model source.</p>

  <div class="split">
    <div class="card">
      <div class="controls">
        <label>Task: <select id="task-select"></select></label>
      </div>
      <div id="plot"></div>
    </div>
    <div class="card">
      <div id="meta" class="meta"><span class="empty">Click a point to view its model code.</span></div>
      <h2 style="margin-top:12px">Source</h2>
      <pre><code id="code" class="language-python"></code></pre>
    </div>
  </div>

  <h2>Summary across all tasks</h2>
  <p class="sub">Every method from every run on one plot. Color encodes task; size is constant.</p>
  <div class="card"><div id="summary"></div></div>

  <div class="footer">Generated from <code>runs/*/results/overall_results.csv</code>.</div>
</div>

<script>
const RECORDS = __DATA__;

const COMMON_LAYOUT = {
  paper_bgcolor: '#ffffff',
  plot_bgcolor: '#ffffff',
  font: { family: '-apple-system, system-ui, sans-serif', color: '#1a1a1a', size: 12 },
  margin: { l: 60, r: 20, t: 20, b: 50 },
  hovermode: 'closest',
  xaxis: { gridcolor: '#eef0f3', zerolinecolor: '#e5e7eb', linecolor: '#cbd5e1' },
  yaxis: { gridcolor: '#eef0f3', zerolinecolor: '#e5e7eb', linecolor: '#cbd5e1' },
};

const tasks = [...new Set(RECORDS.map(r => r.task))].sort();
const sel = document.getElementById('task-select');
tasks.forEach(t => {
  const o = document.createElement('option');
  o.value = t; o.textContent = t; sel.appendChild(o);
});

function recordsForTask(task) {
  return RECORDS.filter(r => r.task === task).slice().sort((a,b) => a.order - b.order);
}

function escapeHtml(s) {
  return String(s ?? '').replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'})[c]);
}

function render(task) {
  const recs = recordsForTask(task);
  const trace = {
    x: recs.map(r => r.n_params),
    y: recs.map(r => r.accuracy),
    mode: 'markers+text',
    type: 'scatter',
    text: recs.map((_, i) => String(i + 1)),
    textposition: 'top center',
    textfont: { size: 10, color: '#374151' },
    hovertext: recs.map(r =>
      `<b>${escapeHtml(r.model)}</b><br>` +
      `step ${r.order + 1} &nbsp; params: ${r.n_params.toExponential(2)}<br>` +
      `accuracy: ${r.accuracy.toFixed(4)}<br>` +
      `${escapeHtml(r.description).slice(0, 280)}`
    ),
    hoverinfo: 'text',
    marker: {
      size: 13,
      color: recs.map(r => r.accuracy),
      colorscale: [[0,'#e5e7eb'],[0.5,'#93c5fd'],[1,'#1d4ed8']],
      cmin: 0, cmax: 1,
      showscale: true,
      colorbar: { title: { text: 'accuracy', font: { size: 12 } }, tickfont: { size: 11 }, outlinewidth: 0 },
      line: { width: 1, color: '#1f2937' },
    },
    customdata: recs.map((_, i) => i),
  };
  const layout = Object.assign({}, COMMON_LAYOUT, {
    xaxis: Object.assign({}, COMMON_LAYOUT.xaxis, { title: 'n_params', type: 'log', autorange: true }),
    yaxis: Object.assign({}, COMMON_LAYOUT.yaxis, { title: 'accuracy', range: [-0.05, 1.05] }),
  });
  Plotly.newPlot('plot', [trace], layout, { responsive: true, displaylogo: false }).then(gd => {
    gd.on('plotly_click', evt => {
      const i = evt.points[0].customdata;
      showRecord(recs[i]);
    });
  });
}

function showRecord(r) {
  const rows = [
    ['Model', r.model],
    ['Task', r.task],
    ['Run', r.run],
    ['Step', String(r.order + 1)],
    ['Accuracy', r.accuracy.toFixed(4)],
    ['Params', isFinite(r.n_params) ? r.n_params.toExponential(3) : '—'],
    ['Status', r.status || '—'],
    ['Description', r.description || '—'],
  ];
  document.getElementById('meta').innerHTML =
    rows.map(([k,v]) => `<div class="row"><b>${k}:</b> ${escapeHtml(v)}</div>`).join('');
  const codeEl = document.getElementById('code');
  codeEl.textContent = r.code || '(no source file found in interpretable_transformers_lib)';
  if (window.Prism) Prism.highlightElement(codeEl);
}

function renderSummary() {
  const palette = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf','#3b82f6','#ef4444','#10b981','#a855f7','#f59e0b'];
  const traces = tasks.map((t, idx) => {
    const recs = recordsForTask(t);
    return {
      name: t,
      x: recs.map(r => r.n_params),
      y: recs.map(r => r.accuracy),
      mode: 'lines+markers',
      type: 'scatter',
      hovertext: recs.map(r => `<b>${escapeHtml(t)}</b><br>${escapeHtml(r.model)} (step ${r.order + 1})<br>params: ${r.n_params.toExponential(2)}<br>acc: ${r.accuracy.toFixed(4)}`),
      hoverinfo: 'text',
      line: { color: palette[idx % palette.length], width: 1.5, shape: 'linear' },
      marker: { size: 9, color: palette[idx % palette.length], line: { width: 0.5, color: '#1f2937' }, opacity: 0.85 },
    };
  });
  const layout = Object.assign({}, COMMON_LAYOUT, {
    xaxis: Object.assign({}, COMMON_LAYOUT.xaxis, { title: 'n_params', type: 'log', autorange: true }),
    yaxis: Object.assign({}, COMMON_LAYOUT.yaxis, { title: 'accuracy', range: [-0.05, 1.05] }),
    legend: { font: { size: 11 }, bgcolor: 'rgba(255,255,255,0)', orientation: 'v' },
    margin: { l: 60, r: 20, t: 20, b: 50 },
  });
  Plotly.newPlot('summary', traces, layout, { responsive: true, displaylogo: false });
}

sel.addEventListener('change', () => render(sel.value));
sel.value = tasks[0];
render(tasks[0]);
renderSummary();
</script>
</body></html>
"""


def main():
    records = collect()
    OUT.write_text(HTML_TEMPLATE.replace("__DATA__", json.dumps(records)))
    n_runs = len({r["run"] for r in records})
    print(f"Wrote {OUT} with {len(records)} records across {n_runs} runs.")


if __name__ == "__main__":
    main()
