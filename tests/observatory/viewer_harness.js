// Executes the viewer's real script in Node against a stub DOM and a REAL snapshot (JSON on argv[2]).
// Catches syntax errors, undefined identifiers and drawing-path exceptions. It does NOT render pixels.
const fs = require("fs");
const html = fs.readFileSync(process.argv[3], "utf8");
const snap = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
const script = html.split("<script>")[1].split("</script>")[0];
const calls = {};
const ctx = new Proxy({}, { get: (t, k) => (k in t ? t[k] : (...a) => { calls[k] = (calls[k] || 0) + 1; }), set: (t, k, v) => { t[k] = v; return true; } });
const els = {};
const el = (id) => els[id] || (els[id] = { id, width: 760, height: 300, textContent: "", value: "", children: [],
  getContext: () => ctx, addEventListener() {}, appendChild(c) { this.children.push(c); },
  getBoundingClientRect: () => ({ left: 0, top: 0, width: 760, height: 300 }) });
global.document = { getElementById: el, createElement: () => ({}) };
global.window = { __UNIVERSE_TEST__: true };
global.location = { protocol: "http:", host: "x" };
global.WebSocket = function () {}; global.fetch = async () => ({ ok: true, json: async () => ({}) });
const run = new Function(script + "\nreturn { draw, state, fmt, drawPhase, drawDensity };");
const api = run();
api.draw(snap);
const out = { calls, steps: els["s-steps"].textContent, t: els["s-t"].textContent, hash: els["s-h"].textContent,
              fmtNaN: api.fmt(NaN), fmtNum: api.fmt(1234.5, 2) };
console.log(JSON.stringify(out));
