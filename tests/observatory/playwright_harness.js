// Stage 1.01 — Browser test harness (self-contained).
//
// Launches the real observatory server (python -m universe_engine.observatory.server),
// opens the viewer in headless Chromium, generates a universe, navigates to a star
// system, and saves: a screenshot, the console log, the network log, and any WebGL
// errors. Running one command produces all artifacts.
//
// Usage (PowerShell, from repo root):
//   $env:PYTHONPATH="src"; node tests/observatory/playwright_harness.js [outdir]
//
// Default outdir: audit_runs/stage1_harness
// Server port: 8011 (configurable via OBSERVATORY_PORT)

const { chromium } = require("playwright");
const { spawn } = require("child_process");
const fs = require("fs");
const path = require("path");
const http = require("http");

const ROOT = path.resolve(__dirname, "..", "..");
const OUTDIR = process.argv[2] || path.join(ROOT, "audit_runs", "stage1_harness");
const PORT = parseInt(process.env.OBSERVATORY_PORT || "8011", 10);
const BASE = `http://127.0.0.1:${PORT}`;
const PYTHON = process.env.PYTHON || "python";

fs.mkdirSync(OUTDIR, { recursive: true });

const consoleLog = [];
const networkLog = [];
const webglErrors = [];

function logAppend(arr, line) { arr.push(`[${new Date().toISOString()}] ${line}`); }

function writeArtifacts() {
  fs.writeFileSync(path.join(OUTDIR, "console.log"), consoleLog.join("\n") + "\n");
  fs.writeFileSync(path.join(OUTDIR, "network.log"), networkLog.join("\n") + "\n");
  fs.writeFileSync(path.join(OUTDIR, "webgl_errors.log"), webglErrors.join("\n") + "\n");
}

function serverReady() {
  return new Promise((resolve, reject) => {
    const req = http.get(`${BASE}/`, (res) => {
      res.resume();
      if (res.statusCode === 200) resolve();
      else reject(new Error(`server returned ${res.statusCode}`));
    });
    req.on("error", reject);
    req.setTimeout(2000, () => { req.destroy(); reject(new Error("server connect timeout")); });
  });
}

async function waitForServer(serverProc, timeoutMs = 60000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (serverProc.exitCode !== null && serverProc.exitCode !== undefined) {
      const stderr = serverProc.stderrRead || "";
      throw new Error(`server exited early (code ${serverProc.exitCode}): ${stderr}`);
    }
    try { await serverReady(); return; } catch (e) { /* keep waiting */ }
    await new Promise((r) => setTimeout(r, 500));
  }
  throw new Error(`server did not become ready within ${timeoutMs}ms`);
}

async function navigateToStar(page) {
  return await page.evaluate(() => {
    const app = window.__app;
    if (!app) return { ok: false, reason: "window.__app not available" };
    app.pause();
    app.setT(app.U.Cosmo.tOfA(1)); // present: galaxies/stars exist
    const uni = app.uni;
    const halo = uni.haloNode(0);
    const members = uni.members_of(halo);
    if (!members || members.length === 0) return { ok: false, reason: "no halo members" };
    const gal = members[0];
    app.stars.build(gal, 2000);
    const s = app.stars.stars;
    const tG = app.t / app.U.GYR;
    let spec = null;
    for (let i = 0; i < Math.min(s.n, 2000); i++) {
      const age = tG - s.attr[i * 4 + 2];
      if (age >= 2 && age <= s.attr[i * 4 + 3]) {
        spec = { pos: [s.pos[i * 3], s.pos[i * 3 + 1], s.pos[i * 3 + 2]], M: s.mass[i], tb: s.attr[i * 4 + 2], id: i };
        break;
      }
    }
    if (!spec) return { ok: false, reason: "no valid star" };
    const star = app.U.makeStarNode(gal, spec);
    app.U.makeSystem(star);
    app.focusOn(star);
    const planets = (star.children || []).filter((p) => !p.data.moon);
    const pMin = planets.length ? Math.min(...planets.map((p) => p.data.P)) : null;
    return {
      ok: true,
      rate: app.getRate(),
      customRate: app.playback ? app.playback.customRate : null,
      nPlanets: planets.length,
      pMin,
      expectedRateFor30s: pMin ? pMin / 30 : null,
    };
  }, { timeout: 60000 });
}

(async () => {
  // ---- 1. Launch the real observatory server ---------------------------
  const serverProc = spawn(PYTHON, ["-m", "universe_engine.observatory.server", "--port", String(PORT)], {
    cwd: ROOT,
    env: { ...process.env, PYTHONPATH: path.join(ROOT, "src") + (process.env.PYTHONPATH ? path.delimiter + process.env.PYTHONPATH : "") },
    stdio: ["ignore", "pipe", "pipe"],
  });
  let serverStdout = "";
  let serverStderr = "";
  serverProc.stdout.on("data", (d) => { serverStdout += d.toString(); });
  serverProc.stderr.on("data", (d) => { serverStderr += d.toString(); });
  serverProc.stderrRead = serverStderr;

  fs.writeFileSync(path.join(OUTDIR, "server_stdout.log"), ""); // placeholder, filled at exit
  fs.writeFileSync(path.join(OUTDIR, "server_stderr.log"), "");

  console.log(`[harness] launching observatory server on port ${PORT}...`);
  await waitForServer(serverProc);
  console.log("[harness] server ready");

  // ---- 2. Launch Chromium and open the viewer --------------------------
  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({ viewport: { width: 1400, height: 900 } });
  const page = await context.newPage();

  // Console capture (incl. WebGL errors reported as console errors).
  page.on("console", (m) => {
    const text = `${m.type()}: ${m.text()}`;
    logAppend(consoleLog, text);
    if (m.type() === "error" || /WebGL|webgl|GL_|gl\.|shader|program/i.test(m.text())) {
      logAppend(webglErrors, text);
    }
  });
  page.on("pageerror", (e) => logAppend(consoleLog, `PAGE ERROR: ${e.message}`));
  page.on("requestfailed", (req) => logAppend(networkLog, `FAILED ${req.method()} ${req.url()} ${req.failure ? req.failure.errorText : ""}`));
  page.on("response", (res) => logAppend(networkLog, `${res.status()} ${res.request().method()} ${res.url()}`));

  // ---- 3. Open viewer, generate universe -------------------------------
  await page.goto(`${BASE}/`, { waitUntil: "domcontentloaded", timeout: 30000 });
  // Screenshot of the initial / "Preparing..." state (Bug H reproduction).
  await page.screenshot({ path: path.join(OUTDIR, "screenshot-01-load.png"), fullPage: true });
  // Wait for generation to reach Ready (Bug H: not stuck on Preparing).
  await page.waitForFunction(() => {
    const el = document.getElementById("prog-label");
    return el && el.textContent.includes("Ready");
  }, { timeout: 300000 });
  await page.click("#btn-begin");
  await page.waitForFunction(() => {
    const i = document.getElementById("intro");
    return i && i.classList.contains("gone");
  }, { timeout: 120000 });

  // ---- 4. Navigate to a star system -----------------------------------
  const sys = await navigateToStar(page);
  console.log("[harness] star system:", JSON.stringify(sys));

  // Wait a beat for rendering, then screenshot.
  await page.waitForTimeout(1500);
  await page.screenshot({ path: path.join(OUTDIR, "screenshot.png"), fullPage: true });
  console.log(`[harness] screenshot saved to ${path.join(OUTDIR, "screenshot.png")}`);

  writeArtifacts();
  fs.writeFileSync(path.join(OUTDIR, "server_stdout.log"), serverStdout);
  fs.writeFileSync(path.join(OUTDIR, "server_stderr.log"), serverStderr);
  fs.writeFileSync(path.join(OUTDIR, "system.json"), JSON.stringify(sys, null, 2));

  await browser.close();
  serverProc.kill();

  console.log(`\n========== HARNESS ARTIFACTS in ${OUTDIR} ==========`);
  console.log("  screenshot.png  console.log  network.log  webgl_errors.log  system.json");
  console.log("  server_stdout.log  server_stderr.log");
  console.log(`console entries: ${consoleLog.length}, network entries: ${networkLog.length}, webgl errors: ${webglErrors.length}`);
  process.exit(sys.ok ? 0 : 1);
})().catch(async (e) => {
  console.error("FATAL:", e);
  try {
    await page.screenshot({ path: path.join(OUTDIR, "screenshot-failure.png"), fullPage: true });
    console.log(`[harness] failure screenshot saved`);
  } catch (_) { /* best effort */ }
  try { writeArtifacts(); } catch (_) { /* best effort */ }
  process.exit(2);
});
