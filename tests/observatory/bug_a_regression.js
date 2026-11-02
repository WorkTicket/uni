const { chromium } = require("playwright");

// Bug A regression tests: black star-system screen fix.
// Verifies the observatory boots, generates a universe, navigates to a star
// system, and renders visible content (not black).
//
// Root cause: Cosmo.aOfD() was missing from cosmo.js, causing catalog.js:65
// to throw during App.generate(), so no universe was ever generated.

const BASE = process.env.OBSERVATORY_URL || "http://127.0.0.1:8001";
const LOADER_TIMEOUT_MS = 60000;

const results = { pass: 0, fail: 0, errors: [], screenshots: [] };

async function check(cond, name, detail = "") {
  if (cond) { results.pass++; console.log("PASS:", name); }
  else { results.fail++; results.errors.push(name + ": " + detail); console.log("FAIL:", name, detail); }
}

async function waitForReady(page) {
  await page.waitForFunction(() => {
    const el = document.getElementById("prog-label");
    return el && el.textContent.includes("Ready");
  }, { timeout: LOADER_TIMEOUT_MS });
}

async function captureScreenshot(page, name) {
  const path = `bug_a_${name}.png`;
  await page.screenshot({ path, fullPage: true });
  results.screenshots.push({ name, path });
  console.log("Screenshot saved:", path);
}

// Sample canvas pixels via WebGL readPixels to verify non-black content.
async function sampleCanvasPixels(page) {
  return await page.evaluate(() => {
    const canvas = document.getElementById("gl");
    if (!canvas) return { error: "no canvas" };
    const gl = canvas.getContext("webgl2") || canvas.getContext("webgl");
    if (!gl) return { error: "no webgl context" };
    const w = canvas.width, h = canvas.height;
    const px = new Uint8Array(4);
    const samples = [];
    const positions = [
      { x: Math.floor(w / 2), y: Math.floor(h / 2), label: "center" },
      { x: Math.floor(w / 4), y: Math.floor(h / 4), label: "top-left" },
      { x: Math.floor(3 * w / 4), y: Math.floor(3 * h / 4), label: "bottom-right" },
    ];
    for (const p of positions) {
      gl.readPixels(p.x, p.y, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, px);
      samples.push({ ...p, r: px[0], g: px[1], b: px[2], a: px[3] });
    }
    return { width: w, height: h, samples };
  });
}

(async () => {
  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({ viewport: { width: 1400, height: 900 } });
  const page = await context.newPage();

  const consoleMsgs = [];
  page.on("console", msg => consoleMsgs.push({ type: msg.type(), text: msg.text() }));
  page.on("pageerror", err => consoleMsgs.push({ type: "pageerror", text: err.message }));

  // ---- Test 1: Boot succeeds ----
  console.log("=== Test 1: Boot ===");
  await page.goto(BASE + "/", { waitUntil: "domcontentloaded", timeout: 30000 });
  await waitForReady(page);
  const label = await page.locator("#prog-label").innerText();
  check(label.includes("Ready"), "Boot reaches Ready state", "got: " + label);
  check(!label.includes("Boot failed"), "No boot failure error", "got: " + label);
  await captureScreenshot(page, "overview");

  // ---- Test 2: Skip to today and verify overview renders ----
  console.log("=== Test 2: Overview view ===");
  await page.click("#btn-skip");
  await page.waitForTimeout(2000);
  const overviewDiag = await page.evaluate(() => {
    const app = window.__app;
    return app ? { nodeKind: app.node.kind, nodeName: app.node.name, t: app.t } : { error: "no app" };
  });
  check(overviewDiag.nodeKind === "universe", "Overview shows universe node", JSON.stringify(overviewDiag));

  const overviewPixels = await sampleCanvasPixels(page);
  check(!overviewPixels.error, "Overview canvas WebGL sample", overviewPixels.error || "");
  if (!overviewPixels.error) {
    const nonBlack = overviewPixels.samples.some(s => s.r > 0 || s.g > 0 || s.b > 0);
    check(nonBlack, "Overview canvas has non-black pixels", JSON.stringify(overviewPixels.samples));
  }

  // ---- Test 3: Navigate to star system ----
  console.log("=== Test 3: Star system ===");
  const navResult = await page.evaluate(() => {
    const app = window.__app;
    if (!app) return { error: "no __app" };
    app.pause();
    app.setT(app.U.Cosmo.tOfA(1));
    const uni = app.uni;
    const halo = uni.haloNode(0);
    const members = uni.members_of(halo);
    const gal = members.length ? members[0] : null;
    if (!gal) return { error: "no galaxy" };
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
    if (!spec) return { error: "no valid star" };
    const star = app.U.makeStarNode(gal, spec);
    app.U.makeSystem(star);
    app.flyTo(star);
    // Run autopilot to move camera close to the star (real user navigation)
    const V = app.U.V;
    for (let i = 0; i < 300; i++) { app.update(0.016); if (!app.autopilot) break; }
    const camDist = V.len(app.pos);
    const planets = (star.children || []).filter(p => !p.data.moon);
    const pMin = planets.length ? Math.min(...planets.map(p => p.data.P)) : null;
    return { ok: true, starName: star.name, nPlanets: planets.length, pMin, camDist };
  }, { timeout: 60000 });
  check(navResult.ok, "Reached a star system", navResult.error || "");
  if (navResult.ok) {
    check(navResult.nPlanets > 0, "Star system has planets", "found " + navResult.nPlanets);
    check(navResult.camDist < 1e15, "Camera is close to star after flyTo", `dist=${navResult.camDist}`);
    await page.waitForTimeout(3000);
    await captureScreenshot(page, "star_system");

    const starDiag = await page.evaluate(() => {
      const app = window.__app;
      return app ? { nodeKind: app.node.kind, nodeName: app.node.name, sysRate: app.sysRate, sysClk: app.sysClk } : { error: "no app" };
    });
    check(starDiag.nodeKind === "star", "Star system view shows star node", JSON.stringify(starDiag));
    check(starDiag.nodeName && starDiag.nodeName !== "Observable universe", "Star has a real catalog name", starDiag.nodeName);

    const starPixels = await sampleCanvasPixels(page);
    check(!starPixels.error, "Star system canvas WebGL sample", starPixels.error || "");
    if (!starPixels.error) {
      const nonBlack = starPixels.samples.some(s => s.r > 0 || s.g > 0 || s.b > 0);
      check(nonBlack, "Star system canvas has non-black pixels", JSON.stringify(starPixels.samples));
    }
  }

  // ---- Test 4: Orbital rate ----
  console.log("=== Test 4: Orbital playback speed ===");
  const rateDiag = await page.evaluate(() => {
    const app = window.__app;
    if (!app) return { error: "no app" };
    const node = app.node;
    if (node.kind !== "star" || !node.children) return { error: "not a star" };
    const planets = node.children.filter(p => p.data && !p.data.moon && Number.isFinite(p.data.P));
    if (!planets.length) return { error: "no planets" };
    const pMin = Math.min(...planets.map(p => p.data.P));
    const orbitWallSec = pMin / app.sysRate;
    return { pMin, sysRate: app.sysRate, expectedRate: pMin / 30, orbitWallSec };
  });
  if (!rateDiag.error) {
    check(Math.abs(rateDiag.sysRate - rateDiag.expectedRate) < 1e-3, "sysRate = innermost period / 30", `sysRate=${rateDiag.sysRate} expected=${rateDiag.expectedRate}`);
    check(rateDiag.orbitWallSec >= 29.9, "Innermost planet orbit >= 30 wall seconds", `got ${rateDiag.orbitWallSec}s`);
  } else {
    console.log("SKIP orbital rate test:", rateDiag.error);
  }

  // ---- Test 5: Mind Inspector loads ----
  console.log("=== Test 5: Mind Inspector ===");
  // Check that the inspector panel elements exist
  const inspCheck = await page.evaluate(() => {
    const inspKind = document.getElementById("insp-kind");
    const inspName = document.getElementById("insp-name");
    return { hasInspKind: !!inspKind, hasInspName: !!inspName };
  });
  check(inspCheck.hasInspKind, "Mind Inspector kind element present");
  check(inspCheck.hasInspName, "Mind Inspector name element present");

  // ---- Test 6: No WebGL errors in console ----
  console.log("=== Test 6: Console errors ===");
  const webglErrors = consoleMsgs.filter(m => m.type === "error" && (m.text.includes("WebGL") || m.text.includes("GL")));
  check(webglErrors.length === 0, "No WebGL errors in console", webglErrors.length ? webglErrors.map(e => e.text).join("; ") : "");

  await browser.close();
  console.log("\n=== RESULTS ===");
  console.log("PASS:", results.pass, "FAIL:", results.fail);
  if (results.errors.length) { console.log("ERRORS:"); results.errors.forEach((e) => console.log(" -", e)); }
  console.log("Screenshots:", results.screenshots.map(s => s.path).join(", "));
  process.exit(results.fail > 0 ? 1 : 0);
})().catch((e) => { console.error("FATAL:", e); process.exit(2); });
