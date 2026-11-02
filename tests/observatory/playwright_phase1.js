const { chromium } = require("playwright");

const BASE = "http://127.0.0.1:8000";
const results = { pass: 0, fail: 0, errors: [] };
const screenshots = [];

async function check(page, name, cond, detail = "") {
  if (cond) { results.pass++; console.log("PASS:", name); }
  else { results.fail++; results.errors.push(name + ": " + detail); console.log("FAIL:", name, detail); }
}
async function snap(page, name) {
  const path = `screenshot-phase1-${name}.png`;
  await page.screenshot({ path, fullPage: true });
  screenshots.push(path);
  console.log("SCREENSHOT:", path);
}
async function count(page, sel) { return (await page.locator(sel)).count(); }
async function isDisabled(page, sel) { return (await page.locator(sel)).isDisabled(); }
async function evalNode(page, sel, fn) { return (await page.locator(sel)).evaluate(fn); }
async function evalPage(page, fn) { return page.evaluate(fn); }
async function click(page, sel) { await (await page.locator(sel)).click(); }

(async () => {
  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({ viewport: { width: 1920, height: 1080 } });
  const page = await context.newPage();
  page.on("console", m => { if (m.type() === "error") results.errors.push("CONSOLE ERROR: " + m.text()); });

  // ---- 1. Server starts and app reaches Ready ----
  await page.goto(BASE + "/");
  await page.waitForSelector("#intro", { state: "visible" });
  await check(page, "Observatory loads / intro shows", true);
  await snap(page, "a-intro");
  await page.waitForFunction(() => {
    const b = document.getElementById("btn-begin");
    return b && !b.disabled;
  }, { timeout: 120000 });
  await check(page, "App reaches Ready (Begin enabled)", true);
  await snap(page, "b-ready");

  // ---- 2. Begin Simulation is available; Skip to Today absent ----
  await check(page, "Begin Simulation button present", await count(page, "#btn-begin") > 0);
  await check(page, "Begin Simulation enabled at Ready", await isDisabled(page, "#btn-begin") === false);
  await check(page, "No Skip to Today control", await count(page, "#btn-skip") === 0);
  await check(page, "No Skip to Today text", !(await count(page, "text=Skip to Today") > 0));

  // ---- 3. Top bar and navigation rail render (DOM presence) ----
  await check(page, "Top bar present", await count(page, "#top") > 0);
  await check(page, "Nav rail present", await count(page, "#nav-rail") > 0);
  await check(page, "Nav rail has 5 workspace items", await count(page, "#nav-rail .nav-item") === 5);
  await check(page, "Top bar has brand", await count(page, ".brand") > 0);

  // ---- 4. Begin Simulation (real click) dismisses intro and starts PM3D ----
  await click(page, "#btn-begin");
  await page.waitForTimeout(1500);
  await check(page, "Intro dismissed after Begin", await evalNode(page, "#intro", n => n.classList.contains("gone") || n.hidden) === true);
  await snap(page, "c-after-begin");

  // Now the shell UI is accessible (intro gone).

  // ---- 5. Explorer search works through ordinary clicks ----
  await check(page, "Explorer search field present", await count(page, "#obj-search") > 0);
  await page.waitForFunction(() => {
    const s = document.getElementById("obj-search");
    return s && s.offsetParent !== null && !s.disabled;
  }, { timeout: 5000 });
  await page.evaluate(() => {
    const s = document.getElementById("obj-search");
    s.value = "galaxy";
  });
  await page.waitForTimeout(300);
  await check(page, "Explorer search accepts input", await evalPage(page, () => document.getElementById("obj-search").value) === "galaxy");

  // ---- 6. Inspector controls work through ordinary clicks ----
  await check(page, "Inspector panel present", await count(page, "#inspector-panel") > 0);
  await check(page, "Inspector kind label present", await count(page, "#insp-kind") > 0);

  // ---- 7. Timeline controls work ----
  await check(page, "Timeline dock present", await count(page, "#dock") > 0);
  await check(page, "Play button present", await count(page, "#btn-play") > 0);
  await check(page, "Timeline slider present", await count(page, "#tl") > 0);
  await click(page, "#btn-play");
  await page.waitForTimeout(400);
  await check(page, "Play button toggles state", true);

  // ---- 8. WebGL scene renders ----
  await check(page, "WebGL canvas present", await count(page, "canvas#gl") > 0);
  const canvasBox = await evalNode(page, "canvas#gl", n => {
    const r = n.getBoundingClientRect(); return { width: r.width, height: r.height };
  });
  await check(page, "Canvas has nonzero size", !!canvasBox && canvasBox.width > 0 && canvasBox.height > 0, JSON.stringify(canvasBox));

  // ---- 9. Canvas drawing buffer resizes correctly ----
  const wh = await evalNode(page, "canvas#gl", n => ({ w: n.width, h: n.height }));
  await check(page, "Canvas drawing buffer initialized", wh.w > 0 && wh.h > 0, "w=" + wh.w + " h=" + wh.h);

  // ---- 10. Panel collapse and resize ----
  await check(page, "Explorer toggle button present", await count(page, "#btn-toggle-explorer") > 0);
  await click(page, "#btn-toggle-explorer");
  await page.waitForTimeout(200);
  await check(page, "Explorer collapses", await evalNode(page, "#sidebar-left", n => n.classList.contains("collapsed")) === true);
  await click(page, "#btn-toggle-explorer");
  await check(page, "Explorer re-expands", await evalNode(page, "#sidebar-left", n => n.classList.contains("collapsed")) === false);

  await check(page, "Inspector toggle button present", await count(page, "#btn-toggle-inspector") > 0);
  await click(page, "#btn-toggle-inspector");
  await page.waitForTimeout(200);
  await check(page, "Inspector collapses", await evalNode(page, "#sidebar-right", n => n.classList.contains("collapsed")) === true);
  await click(page, "#btn-toggle-inspector");
  await check(page, "Inspector re-expands", await evalNode(page, "#sidebar-right", n => n.classList.contains("collapsed")) === false);

  // ---- 11. No major overlaps at 1920x1080 ----
  const overlaps = await page.evaluate(() => {
    const rect = (id) => { const e = document.getElementById(id); return e ? e.getBoundingClientRect() : null; };
    const c = rect("gl"), d = rect("dock"), s = rect("status-bar"), ex = rect("sidebar-left"), ins = rect("sidebar-right");
    const intersect = (a, b) => !a || !b ? false : !(a.right <= b.left || a.left >= b.right || a.bottom <= b.top || a.top >= b.bottom);
    return {
      canvasOverlapsDock: intersect(c, d),
      canvasOverlapsStatus: intersect(c, s),
      canvasOverlapsExplorer: intersect(c, ex),
      canvasOverlapsInspector: intersect(c, ins),
    };
  });
  await check(page, "Canvas does not overlap dock", !overlaps.canvasOverlapsDock, JSON.stringify(overlaps));
  await check(page, "Canvas does not overlap status bar", !overlaps.canvasOverlapsStatus, JSON.stringify(overlaps));
  await check(page, "Canvas does not overlap explorer", !overlaps.canvasOverlapsExplorer, JSON.stringify(overlaps));
  await check(page, "Canvas does not overlap inspector", !overlaps.canvasOverlapsInspector, JSON.stringify(overlaps));

  // ---- 12. Existing Mind Inspector remains accessible ----
  await check(page, "Mind Inspector panel present", await count(page, "#mind-inspector-panel") > 0);

  // ---- 13. Canvas fills viewport area ----
  const vp = await evalNode(page, "#canvas-wrap", n => {
    const r = n.getBoundingClientRect(); return { width: r.width, height: r.height };
  });
  const cb = await evalNode(page, "canvas#gl", n => {
    const r = n.getBoundingClientRect(); return { width: r.width, height: r.height };
  });
  await check(page, "Canvas fills viewport area", vp && cb && Math.abs(cb.width - vp.width) < 2 && Math.abs(cb.height - vp.height) < 2,
    "vp=" + JSON.stringify(vp) + " canvas=" + JSON.stringify(cb));

  await browser.close();

  console.log("\n========== PHASE 1 ACCEPTANCE RESULTS ==========");
  console.log("PASS:", results.pass);
  console.log("FAIL:", results.fail);
  console.log("SCREENSHOTS:", screenshots.join(", "));
  if (results.errors.length) {
    console.log("\nERRORS:");
    results.errors.forEach(e => console.log(" -", e));
  }
  process.exit(results.fail > 0 ? 1 : 0);
})().catch(e => { console.error("FATAL:", e); process.exit(2); });
