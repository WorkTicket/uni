const { chromium } = require("playwright");

// Universe Observatory viewer emergency-fix tests (Stage 1).
// Each test opens a real Chromium instance against a running observatory
// server (http://127.0.0.1:8001 by default) and asserts behaviour in the
// running app, not by reading source.

const BASE = process.env.OBSERVATORY_URL || "http://127.0.0.1:8001";
const LOADER_TIMEOUT_MS = 30000;

const results = { pass: 0, fail: 0, errors: [] };

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

async function navigateToStar(page) {
  return await page.evaluate(() => {
    const app = window.__app;
    app.pause();
    app.setT(app.U.Cosmo.tOfA(1)); // present, so galaxies/stars exist
    const uni = app.uni;
    const halo = uni.haloNode(0);
    const gal = uni.members_of(halo)[0];
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
      customRate: app.playback.customRate,
      nPlanets: planets.length,
      pMin,
      expectedRateFor30s: pMin ? pMin / 30 : null,
      orbitWallSecAtRate: pMin ? pMin / (pMin / 30) : null,
    };
  }, { timeout: 60000 });
}

(async () => {
  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({ viewport: { width: 1400, height: 900 } });
  const page = await context.newPage();

  // Bug H: app must not hang forever on "Preparing..."; generation must reach Ready.
  await page.goto(BASE + "/", { waitUntil: "domcontentloaded", timeout: 30000 });
  await waitForReady(page);
  const label = await page.locator("#prog-label").innerText();
  check(label.includes("Ready"), "Bug H: generation reaches Ready (not stuck on Preparing)", "got: " + label);
  check((await page.locator("#btn-begin").isDisabled()) === false, "Bug H: Begin button enabled after generation");
  console.log("Bug H prog label:", label);

  // Bug B: entering a star system sets a level-appropriate rate so the innermost
  // planet's orbit takes at least 30 wall-seconds.
  const sys = await navigateToStar(page);
  check(sys.ok, "Bug B: reached a star system", sys.reason || "");
  if (sys.ok) {
    check(sys.nPlanets > 0, "Bug B: system has planets", "found " + sys.nPlanets);
    if (sys.nPlanets > 0) {
      check(Math.abs(sys.customRate - (sys.expectedRateFor30s || 0)) < 1e-6, "Bug B: rate = innermost period / 30s", "rate=" + sys.customRate + " expected=" + sys.expectedRateFor30s);
      check(sys.orbitWallSecAtRate >= 29.9, "Bug B: innermost orbit >= 30 wall-seconds at advised rate", "got " + sys.orbitWallSecAtRate + "s");
    }
  }

  await browser.close();
  console.log("\nPASS:", results.pass, "FAIL:", results.fail);
  if (results.errors.length) { console.log("ERRORS:"); results.errors.forEach((e) => console.log(" -", e)); }
  process.exit(results.fail > 0 ? 1 : 0);
})().catch((e) => { console.error("FATAL:", e); process.exit(2); });
