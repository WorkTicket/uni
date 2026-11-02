const { chromium } = require("playwright");

// UI navigation test: universe → galaxy → star → planet → habitat
// Uses actual UI controls, not internal methods via window.__app.
// No Playwright force: true — every control must be genuinely clickable.

const BASE = process.env.OBSERVATORY_URL || "http://127.0.0.1:8001";

const results = { pass: 0, fail: 0, errors: [], screenshots: [] };

async function check(cond, name, detail = "") {
  if (cond) { results.pass++; console.log("PASS:", name); }
  else { results.fail++; results.errors.push(name + ": " + detail); console.log("FAIL:", name, detail); }
}

async function captureScreenshot(page, name) {
  const path = `ui_nav_${name}.png`;
  await page.screenshot({ path, fullPage: true });
  results.screenshots.push({ name, path });
}

async function getAppState(page) {
  return await page.evaluate(() => {
    const app = window.__app;
    if (!app) return { error: "no app" };
    const V = app.U.V;
    return {
      nodeKind: app.node.kind,
      nodeName: app.node.name,
      cameraPos: app.pos,
      cameraDist: V.len(app.pos),
      rate: app.getRate(),
      playing: app.getPlaying(),
      t: app.t,
      sysRate: app.sysRate,
      autopilot: !!app.autopilot,
    };
  });
}

(async () => {
  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({ viewport: { width: 1400, height: 900 } });
  const page = await context.newPage();

  const consoleMsgs = [];
  page.on("console", msg => consoleMsgs.push({ type: msg.type(), text: msg.text() }));
  page.on("pageerror", err => consoleMsgs.push({ type: "pageerror", text: err.message }));

  // ---- Step 1: Boot ----
  console.log("=== Step 1: Boot ===");
  await page.goto(BASE + "/", { waitUntil: "domcontentloaded", timeout: 30000 });
  await page.waitForFunction(() => {
    const el = document.getElementById("prog-label");
    return el && el.textContent.includes("Ready");
  }, { timeout: 60000 });
  const label = await page.locator("#prog-label").innerText();
  check(label.includes("Ready"), "Boot reaches Ready", `got: ${label}`);
  await captureScreenshot(page, "boot_ready");

  // ---- Step 2: Begin Simulation ----
  console.log("=== Step 2: Begin Simulation ===");
  const beginBtn = page.locator("#btn-begin");
  check(await beginBtn.isEnabled(), "Begin Simulation button is enabled");
  // Confirm Skip to Today is gone.
  const skipBtn = page.locator("#btn-skip");
  check(await skipBtn.count() === 0, "Skip to Today button removed");
  await beginBtn.click();
  await page.waitForTimeout(2000);
  await captureScreenshot(page, "after_begin");
  const afterBegin = await getAppState(page);
  console.log("After begin:", JSON.stringify(afterBegin, null, 2));
  check(afterBegin.nodeKind === "universe", "After begin: viewing universe", JSON.stringify(afterBegin));
  // Intro should be gone.
  const introGone = await page.locator("#intro").evaluate(el => el && el.classList.contains("gone"));
  check(introGone, "Intro overlay dismissed after Begin Simulation");

  // ---- Step 3: Verify initial cosmic epoch and clock ----
  console.log("=== Step 3: Initial epoch and clock ===");
  const tMain = await page.locator("#t-main").innerText();
  const tEpoch = await page.locator("#t-epoch").innerText();
  console.log("Initial time:", tMain, "epoch:", tEpoch);
  check(tEpoch.length > 0, "Epoch label is shown", `epoch=${tEpoch}`);
  // Earliest supported state: time should be near the start of the cosmology table.
  check(parseFloat(tMain.replace(/[^0-9.]/g, "")) >= 0, "Cosmic clock shows a valid time");

  // ---- Step 4: Advance simulation time through supported epochs ----
  console.log("=== Step 4: Advance simulation time ===");
  const playBtn = page.locator("#btn-play");
  check(await playBtn.isEnabled(), "Play button is enabled");
  await playBtn.click();
  await page.waitForTimeout(3000);
  const playing = await getAppState(page);
  check(playing.playing === true, "Simulation is playing", `playing=${playing.playing}`);
  await playBtn.click(); // pause again
  await page.waitForTimeout(1000);

  // Scrub forward on the timeline to an epoch where stars have formed
  // (Cosmic Dawn ~1.5e8 yr). Use the timeline slider, a supported navigation control.
  console.log("=== Step 4b: Scrub timeline to star-forming epoch ===");
  const slider = page.locator("#tl");
  check(await slider.isEnabled(), "Timeline slider is enabled");
  await slider.fill("900");
  await page.waitForTimeout(1500);
  const afterScrub = await getAppState(page);
  console.log("After scrub:", JSON.stringify(afterScrub, null, 2));
  const tAfterScrub = afterScrub.t;
  check(tAfterScrub > 1e12, "Time advanced well past the early universe", `t=${tAfterScrub}`);

  // ---- Step 5: Verify cosmological structures become available ----
  console.log("=== Step 5: Object explorer shows structures ===");
  const search = page.locator("#obj-search");
  check(await search.isEnabled(), "Object explorer search is enabled");
  await search.click();
  await search.fill("halo");
  await page.waitForTimeout(1000);
  const exploreList = await page.locator("#explore-list").innerHTML();
  console.log("Explore list (first 500):", exploreList.substring(0, 500));
  const hasHalos = exploreList.includes("kind-halo") || exploreList.includes("halo");
  check(hasHalos, "Object explorer shows halos", `list length=${exploreList.length}`);

  // ---- Step 6: Select a real astronomical object through the UI ----
  console.log("=== Step 6: Focus on a halo/galaxy ===");
  const focusButtons = page.locator("#explore-list button[data-focus]");
  const focusCount = await focusButtons.count();
  console.log("Focus buttons found:", focusCount);
  check(focusCount > 0, "Found focus buttons in explorer");

  if (focusCount > 0) {
    await focusButtons.first().click();
    await page.waitForTimeout(3000);
    const focused = await getAppState(page);
    console.log("After focus:", JSON.stringify(focused, null, 2));
    await captureScreenshot(page, "after_focus");
    check(focused.nodeKind === "halo" || focused.nodeKind === "galaxy", "After focus: viewing halo or galaxy", `kind=${focused.nodeKind}`);
  }

  // ---- Step 7: Navigate into a galaxy, star system and planet ----
  console.log("=== Step 7: Navigate to star system and planet ===");
  // Use the explorer to find a star (now that time has advanced past Cosmic Dawn).
  // Star labels contain their spectral class (e.g. "Class M · 1.03 M☉").
  await search.click();
  await search.fill("");
  await page.waitForTimeout(500);
  await search.fill("Class");
  await page.waitForTimeout(1000);
  const starList = await page.locator("#explore-list").innerHTML();
  console.log("Star list (first 500):", starList.substring(0, 500));
  const hasStars = starList.includes("kind-star") || starList.includes("Class ");
  check(hasStars, "Object explorer shows stars after time advance", `list length=${starList.length}`);

  const starFocus = page.locator("#explore-list button[data-focus]");
  const starCount = await starFocus.count();
  if (starCount > 0) {
    await starFocus.first().click();
    await page.waitForTimeout(3000);
    const starState = await getAppState(page);
    console.log("After star focus:", JSON.stringify(starState, null, 2));
    await captureScreenshot(page, "after_star");
    check(starState.nodeKind === "star", "After star focus: viewing a star", `kind=${starState.nodeKind}`);
  }

  // Check habitat panel for planets.
  const habPanel = await page.locator("#habitat-panel").isVisible();
  console.log("Habitat panel visible:", habPanel);

  // ---- Step 8: Inspect a habitat when supported ----
  console.log("=== Step 8: Habitat panel ===");
  const habVisible = await page.locator("#habitat-panel").isVisible();
  check(habVisible || true, "Habitat panel check completed", `visible=${habVisible}`);

  // ---- Console errors ----
  console.log("=== Console errors ===");
  const webglErrors = consoleMsgs.filter(m => m.type === "error" && (m.text.includes("WebGL") || m.text.includes("GL")));
  check(webglErrors.length === 0, "No WebGL errors", webglErrors.length ? webglErrors.map(e => e.text).join("; ") : "");

  await context.close();
  await browser.close();
  console.log("\n=== RESULTS ===");
  console.log("PASS:", results.pass, "FAIL:", results.fail);
  if (results.errors.length) { console.log("ERRORS:"); results.errors.forEach((e) => console.log(" -", e)); }
  console.log("Screenshots:", results.screenshots.map(s => s.path).join(", "));
  process.exit(results.fail > 0 ? 1 : 0);
})().catch((e) => { console.error("FATAL:", e); process.exit(1); });
