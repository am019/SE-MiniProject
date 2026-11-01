"use strict";
/* IFAP dashboard - plain JS, no inline code (strict CSP). All dynamic text is set via textContent. */
const $ = (s, r = document) => r.querySelector(s);
const main = $("#main");
let token = null, role = null, username = null, poll = null;
try { token = sessionStorage.getItem("ifap_token"); role = sessionStorage.getItem("ifap_role"); username = sessionStorage.getItem("ifap_user"); } catch (e) { /* storage unavailable */ }

function h(tag, attrs, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (k === "class") e.className = v; else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else if (v === true) e.setAttribute(k, ""); else if (v !== false && v != null) e.setAttribute(k, v);
  }
  for (const c of kids.flat()) if (c != null && c !== false) e.append(c.nodeType ? c : document.createTextNode(String(c)));
  return e;
}
function toast(msg, err) { const t = $("#toast"); t.textContent = msg; t.className = "show" + (err ? " err" : ""); clearTimeout(toast.t); toast.t = setTimeout(() => t.className = "", 6000); }
const fmt = (v, d = 5) => v == null ? "n/a" : (typeof v === "number" ? Number(v.toPrecision(d)).toString() : String(v));
const tag = (text, cls) => h("span", { class: "tag " + cls }, text);
const statusTag = s => ({ COMPLETED: tag("✓ COMPLETED", "ok"), VALID: tag("✓ VALID", "ok"), UP: tag("✓ UP", "ok"), BLOCKED: tag("✗ BLOCKED", "bad"), FAILED: tag("✗ FAILED", "bad"), DOWN: tag("✗ DOWN", "bad"), TIMED_OUT: tag("⧖ TIMED OUT", "bad"), TRAINING: tag("▶ TRAINING", "info"), PREDICTING: tag("▶ PREDICTING", "info"), RUNNING: tag("▶ RUNNING", "info"), QUEUED: tag("… QUEUED", "warn") }[s] || tag(s, "info"));

async function api(path, opts = {}) {
  const o = { method: opts.method || "GET", headers: { ...(token ? { Authorization: "Bearer " + token } : {}) } };
  if (opts.json) { o.headers["Content-Type"] = "application/json"; o.body = JSON.stringify(opts.json); }
  if (opts.form) o.body = opts.form;
  const r = await fetch("/api" + path, o);
  if (opts.blob && r.ok) return r.blob();
  let body = null; try { body = await r.json(); } catch (e) { /* not json */ }
  if (r.status === 401 && path !== "/auth/login") { logout(); throw new Error("Session expired - please log in again."); }
  if (!r.ok) { const er = (body && body.error) || {}; const err = new Error(`${er.message || "Request failed"} (${er.code || r.status}, ${er.requestId || "-"})`); err.code = er.code; err.fields = er.fields; throw err; }
  return body;
}
const guard = fn => async (...a) => { try { await fn(...a); } catch (e) { toast(e.message, true); } };

function logout() { token = role = username = null; try { sessionStorage.clear(); } catch (e) { /* */ } clearInterval(poll); route(); }

/* ---------------------------------------------------------------- routing */
const PAGES = [["overview", "Overview"], ["datasets", "Datasets"], ["experiment", "Run experiment"], ["results", "Results"], ["predict", "Predict"], ["registry", "Registry"], ["health", "Admin", true]];
function route() {
  clearInterval(poll);
  const top = $("#top");
  if (!token) { top.hidden = true; return renderLogin(); }
  top.hidden = false; $("#who").textContent = `${username} (${role})`;
  const page = (location.hash.slice(2).split("?")[0]) || "overview";
  const nav = $("#nav"); nav.replaceChildren(...PAGES.filter(p => !p[2] || role === "Administrator").map(([k, l]) => h("a", { href: "#/" + k, "aria-current": k === page ? "page" : null }, l)));
  main.replaceChildren(h("p", { class: "muted" }, "Loading…"));
  (({ overview: pOverview, datasets: pDatasets, experiment: pExperiment, results: pResults, predict: pPredict, registry: pRegistry, health: pHealth })[page] || pOverview)().catch(e => { main.replaceChildren(h("div", { class: "note bad" }, e.message)); });
  main.focus();
}
window.addEventListener("hashchange", route);
$("#logout").addEventListener("click", logout);

function renderLogin() {
  const u = h("input", { type: "text", id: "u", autocomplete: "username", value: "analyst" }), p = h("input", { type: "password", id: "p", autocomplete: "current-password" });
  main.replaceChildren(h("div", { class: "card login" }, h("h1", {}, "IFAP sign in"),
    h("p", { class: "muted small" }, "Compare classical ML, deep learning and quantum ML for volatility and option-price forecasting. Credentials are printed by the start script (see README)."),
    h("form", { onsubmit: guard(async ev => { ev.preventDefault(); const r = await api("/auth/login", { method: "POST", json: { username: u.value, password: p.value } }); token = r.token; role = r.role; username = r.username; try { sessionStorage.setItem("ifap_token", token); sessionStorage.setItem("ifap_role", role); sessionStorage.setItem("ifap_user", username); } catch (e) { /* */ } location.hash = "#/overview"; route(); }) },
      h("label", { for: "u" }, "Username"), u, h("label", { for: "p" }, "Password"), p, h("p", {}, h("button", { class: "btn", type: "submit" }, "Sign in")))));
}

/* ---------------------------------------------------------------- overview (FR-18) */
async function pOverview() {
  const ch = await api("/models/champions").catch(() => ({}));
  main.replaceChildren(h("h1", {}, "Overview"),
    h("div", { class: "card" }, h("h2", {}, "Purpose"), h("p", {}, "IFAP is a microservice platform that trains and compares three families of models on the same data, the same features and the same test period, and automatically selects the best performer for financial volatility and derivatives (option price) prediction. Selection uses validation data only; test results are reported for honesty, never for choosing.")),
    h("div", { class: "grid g3" },
      ...[["Classical ML", "GARCH(1,1) and Black-Scholes baselines, Random Forest, Gradient Boosting. Fast, deterministic with a fixed seed."], ["Deep learning", "LSTM sequence model with a documented seed, epochs and early-stopping rule."], ["Quantum ML (simulator)", "Variational quantum regressor, up to 8 qubits, run on a classical state-vector simulator. No quantum advantage is claimed."]].map(([t, d]) => h("div", { class: "card" }, h("h2", {}, t), h("p", {}, d)))),
    h("div", { class: "grid g2" },
      h("div", { class: "card" }, h("h2", {}, "Supported tasks"), h("ul", {}, h("li", {}, h("b", {}, "Volatility: "), "forecast annualised realised volatility over the next h trading days (h = 1–30) from a daily price series (CSV: date, close)."), h("li", {}, h("b", {}, "Option price: "), "predict European option prices from an option chain (CSV: date, strike, expiry, option_type, underlying_price, option_price)."))),
      h("div", { class: "card" }, h("h2", {}, "Current champions"), ...["VOLATILITY", "OPTION_PRICE"].map(t => h("p", {}, h("b", {}, t + ": "), ch[t] ? `${ch[t].modelId} (promoted ${ch[t].promotedAt})` : "none yet - run an experiment")))),
    h("div", { class: "note" }, h("b", {}, "Limitations. "), "Financial predictions are uncertain; results describe one dataset and one chronological split and are not investment advice. Sample datasets are synthetic and labelled as such - their metrics are not real market results. Quantum results are simulator results."));
}

/* ---------------------------------------------------------------- datasets (FR-01..03) */
async function pDatasets() {
  const [list, samples] = await Promise.all([api("/datasets"), api("/samples")]);
  const out = h("div", {});
  const file = h("input", { type: "file", id: "csv", accept: ".csv,text/csv" });
  const show = d => out.replaceChildren(report(d));
  const reload = () => route();
  main.replaceChildren(h("h1", {}, "Datasets"),
    h("div", { class: "card" }, h("h2", {}, "Upload a CSV"), h("p", { class: "muted small" }, "UTF-8 .csv up to 25 MB. Price series need columns date, close. Option chains need date, strike, expiry, option_type, underlying_price, option_price (rate optional). Files are parsed as data only."),
      h("div", { class: "row" }, h("div", {}, h("label", { for: "csv" }, "CSV file"), file), h("div", {}, h("button", { class: "btn", onclick: guard(async () => { if (!file.files[0]) return toast("Choose a .csv file first.", true); const f = new FormData(); f.append("file", file.files[0]); try { show(await api("/datasets", { method: "POST", form: f })); toast("Imported - see the validation report below."); } catch (e) { out.replaceChildren(h("div", { class: "note bad" }, e.message, e.fields ? " Field(s): " + e.fields.join(", ") : "")); } }) }, "Upload and validate"))),
      h("h3", {}, "Or load a synthetic sample"), h("div", { class: "row" }, ...samples.samples.map(s => h("button", { class: "btn secondary", title: s.description, onclick: guard(async () => { show(await api(`/samples/${s.file}/load`, { method: "POST" })); toast("Sample loaded (synthetic data)."); }) }, s.file)))),
    out,
    h("div", { class: "card" }, h("h2", {}, "Imported datasets"), list.datasets.length ? table(["Name", "ID", "Type", "Status", "Rows", "Columns", "Date range", "Issues"], list.datasets.map(d => [d.isSample ? `${d.name} (SAMPLE)` : d.name, h("span", { class: "mono" }, d.datasetId), d.type.replace("_", " "), statusTag(d.status), d.rows, d.columns, `${d.dateMin} → ${d.dateMax}`, `${d.blockingIssues} blocking, ${d.warnings} warnings`]), [4, 5]) : h("p", { class: "muted" }, "No datasets yet.")));
  const last = list.datasets[0]; if (last && !out.firstChild) { /* leave empty until an upload */ }
}
function report(d) {
  const r = d.validationReport; const blocked = r.status !== "VALID";
  return h("div", { class: "card" }, h("h2", {}, "Validation report"), h("div", { class: "note " + (blocked ? "bad" : "ok") }, blocked ? "Blocked: feature building and model runs are disabled until the blocking issues are fixed." : "Valid: ready for experiments."),
    h("p", {}, `Rows: ${d.rows} · Columns: ${d.columns} · Date range: ${d.dateMin} → ${d.dateMax} · Detected type: ${d.type.replace("_", " ")} · Validated in ${d.validationSeconds}s`),
    h("p", { class: "mono" }, "SHA-256 " + d.datasetHash), d.ignoredColumns && d.ignoredColumns.length ? h("p", { class: "muted small" }, "Ignored extra columns: " + d.ignoredColumns.join(", ")) : null,
    r.issues.length ? table(["Severity", "Code", "Field", "Message", "Count", "Rows (first)"], r.issues.map(i => [i.severity === "BLOCKING" ? tag("✗ BLOCKING", "bad") : tag("⚠ WARNING", "warn"), i.code, i.field || "-", i.message, i.count, (i.rows || []).join(", ") || "-"])) : h("p", {}, "No issues found."));
}
function table(head, rows, numCols = []) {
  return h("div", { class: "tablewrap" }, h("table", {}, h("thead", {}, h("tr", {}, head.map((x, i) => h("th", { class: numCols.includes(i) ? "num" : "" }, x)))), h("tbody", {}, rows.map(r => h("tr", {}, r.map((c, i) => h("td", { class: numCols.includes(i) ? "num" : "" }, c)))))));
}

/* ---------------------------------------------------------------- experiment (FR-10, FR-12, FR-15) */
async function pExperiment() {
  const [ds, av, list] = await Promise.all([api("/datasets"), api("/models/available"), api("/experiments")]);
  const valid = ds.datasets.filter(d => d.status === "VALID");
  const dsSel = h("select", { id: "ds" }, valid.map(d => h("option", { value: d.datasetId }, `${d.name} – ${d.type.replace("_", " ").toLowerCase()}`)));
  const task = h("select", { id: "task" }, h("option", { value: "VOLATILITY" }, "Volatility forecast"), h("option", { value: "OPTION_PRICE" }, "Option price"));
  const hor = h("input", { type: "number", id: "hor", min: 1, max: 30, value: 5 }), seed = h("input", { type: "number", id: "seed", min: 0, value: 42 });
  const pm = h("select", { id: "pm" }); const from = h("input", { type: "date", id: "from" }), to = h("input", { type: "date", id: "to" });
  const margin = h("input", { type: "number", id: "margin", min: 0, max: 50, step: 0.5, value: 0 });
  const boxes = h("div", { class: "checks" }); const prog = h("div", {});
  const sync = () => {
    const t = task.value, d = valid.find(x => x.datasetId === dsSel.value);
    if (d) task.value = d.type === "OPTION_CHAIN" ? "OPTION_PRICE" : "VOLATILITY";
    const tk = task.value; hor.disabled = tk === "OPTION_PRICE";
    pm.replaceChildren(...av.primaryMetrics[tk].map(m => h("option", { value: m }, m)));
    boxes.replaceChildren(...av.models.filter(m => m.tasks.includes(tk)).map(m => { const base = av.baselines[tk] === m.name; return h("label", {}, h("input", { type: "checkbox", value: m.name, checked: true, disabled: base }), `${m.label} [${m.family}]`, base ? tag("baseline - always included", "info") : null, m.serviceStatus !== "UP" ? tag("✗ service " + (m.serviceStatus || "DOWN"), "bad") : null); }));
  };
  dsSel.addEventListener("change", sync); task.addEventListener("change", () => { sync(); });
  main.replaceChildren(h("h1", {}, "Run experiment"),
    valid.length ? h("div", { class: "card" }, h("div", { class: "grid g2" },
      h("div", {}, h("label", { for: "ds" }, "Dataset"), dsSel, h("label", { for: "task" }, "Task"), task, h("label", { for: "hor" }, "Forecast horizon (trading days, volatility task)"), hor, h("div", { class: "row" }, h("div", {}, h("label", { for: "from" }, "From (optional)"), from), h("div", {}, h("label", { for: "to" }, "To (optional)"), to))),
      h("div", {}, h("label", { for: "pm" }, "Primary metric (declared before the run; evaluated on validation data)"), pm, h("label", { for: "margin" }, "Required improvement over baseline (%)"), margin, h("label", { for: "seed" }, "Random seed"), seed, h("h3", {}, "Models"), boxes)),
      h("p", {}, h("button", { class: "btn", id: "go", onclick: guard(async () => {
        const models = [...boxes.querySelectorAll("input:checked:not(:disabled)")].map(x => x.value); if (!models.length) return toast("Select at least one model.", true);
        const body = { datasetId: dsSel.value, task: task.value, horizon: +hor.value, models, primaryMetric: pm.value, seed: +seed.value, margin: +margin.value / 100 }; if (from.value) body.dateFrom = from.value; if (to.value) body.dateTo = to.value;
        const r = await api("/experiments", { method: "POST", json: body }); toast("Experiment " + r.experimentId + " accepted."); watch(r.experimentId, prog);
      }) }, "Start experiment"))) : h("div", { class: "note" }, "No valid dataset yet - import one on the Datasets page."),
    prog, h("div", { class: "card" }, h("h2", {}, "Previous experiments"), list.experiments.length ? table(["ID", "Status", "Task", "Horizon", "Primary metric", "Created"], list.experiments.map(e => [h("a", { href: "#/results?" + e.experimentId }, e.experimentId), statusTag(e.status), e.task, e.horizon || "-", e.primaryMetric, e.createdAt])) : h("p", { class: "muted" }, "None yet.")));
  if (valid.length) sync();
  const running = list.experiments.find(e => e.status === "QUEUED" || e.status === "RUNNING"); if (running) watch(running.experimentId, prog);
}
function watch(eid, box) {
  const tick = guard(async () => {
    const e = await api("/experiments/" + eid);
    box.replaceChildren(h("div", { class: "card", role: "region", "aria-label": "Experiment progress" }, h("h2", {}, `Experiment ${eid} `, statusTag(e.status)),
      table(["Model", "Family", "Status", "Reason / ID"], Object.entries(e.models).map(([k, v]) => [k, v.family, statusTag(v.status), v.reason || v.modelId || ""])),
      e.status === "COMPLETED" ? h("p", {}, h("a", { class: "btn", href: "#/results?" + eid }, "View leaderboard")) : null, e.status === "FAILED" ? h("div", { class: "note bad" }, e.error || "Experiment failed.") : null));
    if (e.status === "COMPLETED" || e.status === "FAILED") clearInterval(poll);
  });
  clearInterval(poll); tick(); poll = setInterval(tick, 3000);
}

/* ---------------------------------------------------------------- results (FR-11..13, FR-17) */
async function pResults() {
  const list = await api("/experiments"); const done = list.experiments.filter(e => e.status === "COMPLETED");
  const want = location.hash.split("?")[1]; const sel = h("select", { id: "exp" }, done.map(e => h("option", { value: e.experimentId, selected: e.experimentId === want }, `${e.experimentId} – ${e.task} – ${e.createdAt}`)));
  const body = h("div", {});
  main.replaceChildren(h("h1", {}, "Leaderboard and selection"), done.length ? h("div", { class: "card" }, h("label", { for: "exp" }, "Experiment"), sel) : h("div", { class: "note" }, "No completed experiment yet."), body);
  const load = guard(async () => { if (!sel.value) return; const [lb, s] = await Promise.all([api(`/experiments/${sel.value}/leaderboard`), api(`/experiments/${sel.value}/selection`)]); body.replaceChildren(...renderBoard(lb, s)); });
  sel.addEventListener("change", load); load();
}
function renderBoard(lb, s) {
  const pm = lb.primaryMetric, mets = lb.task === "VOLATILITY" ? ["RMSE", "MAE", "QLIKE", "MAPE", "DA"] : ["RMSE", "MAE", "MAPE"];
  const rows = lb.leaderboard.map(r => [r.rank ?? "-", h("span", {}, r.model, " ", r.isChampion ? tag("★ CHAMPION", "ok") : null, r.isBaseline ? tag("baseline", "info") : null), r.family, statusTag(r.status), ...mets.map(m => r.status === "COMPLETED" ? `${fmt(r.val[m], 4)} / ${fmt(r.test[m], 4)}` : "-"), fmt(r.trainTimeSec, 3), fmt(r.inferenceTimeMs, 3), r.note || ""]);
  return [
    h("div", { class: "card" }, h("h2", {}, "Selected champion"), h("p", {}, h("b", {}, s.champion.model), ` (${s.champion.family}) – validation ${pm} = ${fmt(s.validationValue)}`, s.flags.length ? [" ", tag("⚠ " + s.flags.join(", "), "warn")] : null), h("p", {}, s.justification),
      h("p", { class: "muted small" }, `Policy: selection on validation data only; baseline ${s.baseline}; margin ${(s.policy.margin * 100).toFixed(1)}%; tie tolerance ${(s.policy.tieTolerance * 100).toFixed(1)}%.`),
      lb.isSample ? h("div", { class: "note" }, "Synthetic sample data – these metrics are not real market results.") : null),
    h("div", { class: "card" }, h("h2", {}, "Leaderboard"), h("p", { class: "small muted" }, `Task ${lb.task} · horizon ${lb.horizon || "n/a"} · validation ${lb.valPeriod.from} → ${lb.valPeriod.to} · test period ${lb.testPeriod.from} → ${lb.testPeriod.to} (${lb.testPeriod.rows} rows) · each cell shows validation / test. Test timestamp hash `, h("span", { class: "mono" }, lb.testTimestampHash.slice(0, 16)), " · feature set ", lb.featureSetVersion),
      table(["#", "Model", "Family", "Status", ...mets.map(m => m + " (val / test)"), "Train s", "Infer ms", "Note"], rows, [0]),
      h("details", {}, h("summary", {}, "Metric definitions"), h("ul", {}, Object.entries(lb.metricDefinitions).map(([k, v]) => h("li", {}, h("b", {}, k + ": "), v))))),
    chart(lb), h("div", { class: "card" }, h("h2", {}, "Export report"), h("p", { class: "muted small" }, "Aggregate results, selection justification, configuration, provenance hashes and limitations (no raw data)."), h("div", { class: "row" }, ...["pdf", "csv", "json"].map(f => h("button", { class: "btn secondary", onclick: guard(async () => { const b = await api(`/reports/${lb.experimentId}?format=${f}`, { blob: true }); const a = h("a", { href: URL.createObjectURL(b), download: `ifap-${lb.experimentId}.${f}` }); document.body.append(a); a.click(); a.remove(); }) }, "Download " + f.toUpperCase()))))];
}
function chart(lb) {
  const pm = lb.primaryMetric, ok = lb.leaderboard.filter(r => r.status === "COMPLETED"), W = 900, rowH = 46, L = 190, H = 96 + ok.length * rowH, R = 40, T = 40;
  const max = Math.max(...ok.flatMap(r => [r.val[pm], r.test[pm]])) * 1.1 || 1, x = v => L + (W - L - R) * v / max, ns = "http://www.w3.org/2000/svg";
  const el = (t, a, txt) => { const e = document.createElementNS(ns, t); for (const [k, v] of Object.entries(a || {})) e.setAttribute(k, v); if (txt != null) e.textContent = txt; return e; };
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, width: "100%", role: "img", "aria-label": `Bar chart of validation and test ${pm} per model` });
  const defs = el("defs"), pat = el("pattern", { id: "hatch", width: 6, height: 6, patternUnits: "userSpaceOnUse", patternTransform: "rotate(45)" }); pat.append(el("rect", { width: 6, height: 6, fill: "#fff" }), el("rect", { width: 3, height: 6, fill: "#b45309" })); defs.append(pat); svg.append(defs);
  svg.append(el("text", { x: W / 2, y: 20, "text-anchor": "middle", "font-size": 15, "font-weight": 700 }, `${pm} per model (lower is better)`));
  for (let i = 0; i <= 5; i++) { const v = max * i / 5; svg.append(el("line", { x1: x(v), x2: x(v), y1: T, y2: H - 50, stroke: "#cbd2dc" }), el("text", { x: x(v), y: H - 34, "text-anchor": "middle", "font-size": 11 }, fmt(v, 3))); }
  svg.append(el("text", { x: (L + W - R) / 2, y: H - 8, "text-anchor": "middle", "font-size": 12 }, `${pm} (${lb.task === "VOLATILITY" ? "annualised volatility, decimal" : "price units"})`));
  ok.forEach((r, i) => { const y = T + i * rowH; svg.append(el("text", { x: L - 8, y: y + 22, "text-anchor": "end", "font-size": 13 }, r.model + (r.isChampion ? " ★" : "")));
    [["val", "#0b5cad", r.val[pm]], ["test", "url(#hatch)", r.test[pm]]].forEach(([k, f, v], j) => { svg.append(el("rect", { x: L, y: y + j * 17 + 2, width: Math.max(1, x(v) - L), height: 15, fill: f, stroke: "#14213d" }), el("text", { x: x(v) + 4, y: y + j * 17 + 14, "font-size": 11 }, fmt(v, 4))); }); });
  return h("div", { class: "card" }, h("h2", {}, `${pm}: validation vs test`), svg, h("div", { class: "legend" }, h("span", {}, h("span", { class: "sw val" }), "Validation (decides selection)"), h("span", {}, h("span", { class: "sw test" }), "Test (reported only)"), h("span", {}, "★ = champion")));
}

/* ---------------------------------------------------------------- predict (FR-14) */
async function pPredict() {
  const task = h("select", { id: "t" }, h("option", { value: "VOLATILITY" }, "Volatility forecast"), h("option", { value: "OPTION_PRICE" }, "Option price"));
  const hor = h("input", { type: "number", id: "h", min: 1, max: 30, placeholder: "champion's horizon" }), k = h("input", { type: "number", id: "k", min: 0, step: "any" }), d = h("input", { type: "number", id: "d", min: 1, value: 30 }), ty = h("select", { id: "ty" }, h("option", { value: "call" }, "call"), h("option", { value: "put" }, "put"));
  const volBox = h("div", {}, h("label", { for: "h" }, "Horizon (trading days; leave empty to use the champion's)"), hor), optBox = h("div", {}, h("label", { for: "k" }, "Strike"), k, h("label", { for: "d" }, "Days to expiry"), d, h("label", { for: "ty" }, "Option type"), ty), out = h("div", {});
  const sync = () => { volBox.hidden = task.value !== "VOLATILITY"; optBox.hidden = task.value !== "OPTION_PRICE"; }; task.addEventListener("change", sync);
  main.replaceChildren(h("h1", {}, "Predict with the champion"), h("div", { class: "card" }, h("label", { for: "t" }, "Task"), task, volBox, optBox, h("p", {}, h("button", { class: "btn", onclick: guard(async () => {
    const b = { task: task.value }; if (b.task === "VOLATILITY" && hor.value) b.horizon = +hor.value; if (b.task === "OPTION_PRICE") Object.assign(b, { strike: +k.value, daysToExpiry: +d.value, optionType: ty.value });
    try { const p = await api("/predictions", { method: "POST", json: b }); out.replaceChildren(h("div", { class: "card" }, h("h2", {}, "Prediction"), h("p", { class: "big" }, h("b", {}, p.values.map(v => fmt(v, 5)).join(", ")), " ", p.unit), table(["Model", "Family", "Version", "Validation metric", "As of", "Horizon", "Inference"], [[p.model, p.family + (p.simulator ? " (simulator)" : ""), p.version, `${p.validationMetric.name} = ${fmt(p.validationMetric.value)}`, p.asOf, p.horizon || "n/a", fmt(p.inferenceTimeMs, 3) + " ms"]]), p.isSample ? h("div", { class: "note" }, "Trained on synthetic sample data.") : null, h("div", { class: "note" }, p.disclaimer))); }
    catch (e) { out.replaceChildren(h("div", { class: "note bad" }, e.message)); }
  }) }, "Predict"))), out); sync();
}

/* ---------------------------------------------------------------- registry (FR-16, FR-19) */
async function pRegistry() {
  const [champs, exps] = await Promise.all([api("/models/champions"), api("/experiments")]);
  const hist = await Promise.all(["VOLATILITY", "OPTION_PRICE"].map(t => api("/models/champion/history?task=" + t)));
  const models = h("div", {}); const done = exps.experiments;
  const sel = h("select", { id: "e" }, done.map(e => h("option", { value: e.experimentId }, `${e.experimentId} – ${e.task} – ${e.status}`)));
  const showModels = guard(async () => { if (!sel.value) return; const r = await api("/models?experimentId=" + sel.value); models.replaceChildren(...r.models.map(m => h("details", {}, h("summary", {}, `${m.name} – ${m.modelId} – `, statusTag(m.status), m.isChampion ? " ★ champion" : ""), h("div", { class: "grid g2 small" }, h("div", {}, h("p", {}, `Family ${m.family}, version ${m.version}, seed ${m.seed}, created ${m.createdAt}`), h("p", {}, `Train ${fmt(m.trainTimeSec, 3)} s · inference ${fmt(m.inferenceTimeMs, 3)} ms`), m.failureReason ? h("p", {}, "Reason: " + m.failureReason) : null, h("p", { class: "mono" }, "artifact SHA-256: " + (m.artifactHash || "n/a")), h("p", { class: "mono" }, "dataset SHA-256: " + m.datasetHash), h("p", {}, "feature set " + m.featureSetVersion)), h("div", {}, h("pre", { class: "mono" }, JSON.stringify({ config: m.config, metrics: m.metrics, libraries: m.libraryVersions, info: m.info }, null, 1)))), role === "Administrator" && !m.isChampion ? h("button", { class: "btn secondary", onclick: guard(async () => { if (!confirm("Delete model " + m.modelId + "?")) return; await api("/models/" + m.modelId, { method: "DELETE" }); toast("Deleted."); showModels(); }) }, "Delete model") : null))); });
  main.replaceChildren(h("h1", {}, "Model registry"),
    h("div", { class: "grid g2" }, ...["VOLATILITY", "OPTION_PRICE"].map((t, i) => h("div", { class: "card" }, h("h2", {}, t), champs[t] ? h("p", {}, "Active champion: ", h("b", {}, champs[t].modelId), ` (since ${champs[t].promotedAt})`) : h("p", { class: "muted" }, "No champion yet."),
      role === "Administrator" && champs[t] ? h("button", { class: "btn secondary", onclick: guard(async () => { if (!confirm("Restore the previous champion for " + t + "?")) return; await api("/models/champion/rollback", { method: "POST", json: { task: t } }); toast("Previous champion restored."); route(); }) }, "Roll back to previous champion") : (champs[t] ? h("p", { class: "muted small" }, "Roll-back requires the Administrator role.") : null),
      hist[i].history.length ? table(["Model", "Promoted", "Retired", "Active"], hist[i].history.map(x => [x.modelId, x.promotedAt, x.retiredAt || "-", x.active ? tag("✓ active", "ok") : "no"])) : null))),
    h("div", { class: "card" }, h("h2", {}, "Runs of an experiment"), exps.experiments.length ? [h("label", { for: "e" }, "Experiment"), sel, models] : h("p", { class: "muted" }, "No experiments yet.")));
  sel.addEventListener("change", showModels); showModels();
}

/* ---------------------------------------------------------------- admin health (FR-20, SR-08) */
async function pHealth() {
  if (role !== "Administrator") throw new Error("This page requires the Administrator role.");
  const box = h("div", { class: "card" }); const cfg = h("div", { class: "card" });
  const mg = h("input", { type: "number", id: "m", min: 0, max: 50, step: 0.5 }), tt = h("input", { type: "number", id: "tt", min: 0, max: 20, step: 0.5 }), to = h("input", { type: "number", id: "to", min: 1, max: 7200 });
  const current = await api("/admin/config"); const s = current.selection || {}; mg.value = (s.margin ?? 0) * 100; tt.value = (s.tieTolerance ?? 0.01) * 100; to.value = s.timeoutSec ?? 1200;
  cfg.append(h("h2", {}, "Platform configuration (Administrator)"), h("div", { class: "row" }, h("div", {}, h("label", { for: "m" }, "Default margin over baseline (%)"), mg), h("div", {}, h("label", { for: "tt" }, "Tie tolerance (%)"), tt), h("div", {}, h("label", { for: "to" }, "Model job timeout (s)"), to), h("div", {}, h("button", { class: "btn", onclick: guard(async () => { await api("/admin/config", { method: "PUT", json: { selection: { margin: +mg.value / 100, tieTolerance: +tt.value / 100, timeoutSec: +to.value } } }); toast("Configuration saved."); }) }, "Save"))));
  main.replaceChildren(h("h1", {}, "Service health"), box, cfg);
  const tick = guard(async () => { const r = await api("/admin/health"); box.replaceChildren(h("h2", {}, "Services ", tag(r.overall === "UP" ? "✓ ALL UP" : "⚠ DEGRADED", r.overall === "UP" ? "ok" : "warn")), table(["Service", "Status", "Latency (ms)", "Last checked (UTC)"], r.services.map(x => [x.service, statusTag(x.status), x.latencyMs, x.lastChecked]), [2]), h("p", { class: "muted small" }, "Refreshes every 10 seconds.")); });
  tick(); poll = setInterval(tick, 10000);
}
route();
