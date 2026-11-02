const { chromium } = require("playwright");

const BASE = "http://127.0.0.1:8000";
const resolutions = [
  { name: "1280x720",  w: 1280, h: 720 },
  { name: "1600x900",  w: 1600, h: 900 },
  { name: "1920x1080", w: 1920, h: 1080 },
  { name: "3840x2160", w: 3840, h: 2160 },
];

const results = { pass: 0, fail: 0, errors: [] };
const screenshots = [];

async function check(name, cond, detail = "") {
  if (cond) { results.pass++; console.log("PASS:", name); }
  else { results.fail++; results.errors.push(name + ": " + detail); console.log("FAIL:", name, detail); }
}
async function snap(page, resName, name) {
  const path = `screenshot-phase1-${resName.toLowerCase().replace('x','-')}-${name}.png`;
  await page.screenshot({ path, fullPage: true });
  screenshots.push(path);
  console.log("SCREENSHOT:", path);
}
async function count(page, sel) { return (await page.locator(sel)).count(); }
async function evalNode(page, sel, fn) { return (await page.locator(sel)).evaluate(fn); }
async function click(page, sel) { await (await page.locator(sel)).click(); }

(async () => {
  const browser = await chromium.launch({ headless: true });
  for (const res of resolutions) {
    const context = await browser.newContext({ viewport: { width: res.w, height: res.h } });
    const page = await context.newPage();
    try {
      await page.goto(BASE + "/");
      await page.waitForSelector("#intro", { state: "visible" });
      await page.waitForFunction(() => {
        const b = document.getElementById("btn-begin");
        return b && !b.disabled;
      }, { timeout: 120000 });
      await click(page, "#btn-begin");
      await page.waitForTimeout(1000);

      await snap(page, res.name, "shell");
      await check(`${res.name} shell elements`, await count(page, "#top") > 0 && await count(page, "#nav-rail") > 0 && await count(page, "#breadcrumb") > 0 && await count(page, "#sidebar-left") > 0 && await count(page, "#canvas-wrap") > 0 && await count(page, "#sidebar-right") > 0 && await count(page, "#dock") > 0 && await count(page, "#status-bar") > 0, true);
      await check(`${res.name} nav rail 5 items`, await count(page, "#nav-rail .nav-item") === 5);
      await check(`${res.name} Begin present`, await count(page, "#btn-begin") > 0);
      await check(`${res.name} no Skip to Today`, await count(page, "#btn-skip") === 0);
      await check(`${res.name} canvas present`, await count(page, "canvas#gl") > 0);
      const wh = await evalNode(page, "canvas#gl", n => ({ w: n.width, h: n.height }));
      await check(`${res.name} canvas buffer`, wh.w > 0 && wh.h > 0, "w=" + wh.w + " h=" + wh.h);
      // overlaps
      const ov = await page.evaluate(() => {
        const r = (id) => { const e = document.getElementById(id); return e ? e.getBoundingClientRect() : null; };
        const c = r("gl"), d = r("dock"), ex = r("sidebar-left"), ins = r("sidebar-right"), s = r("status-bar");
        const inter = (a, b) => !a || !b ? false : !(a.right <= b.left || a.left >= b.right || a.bottom <= b.top || a.top >= b.bottom);
        return { dock: inter(c, d), explorer: inter(c, ex), inspector: inter(c, ins), status: inter(c, s) };
      });
      await check(`${res.name} no canvas/dock overlap`, !ov.dock, JSON.stringify(ov));
      await check(`${res.name} no canvas/explorer overlap`, !ov.explorer, JSON.stringify(ov));
      await check(`${res.name} no canvas/inspector overlap`, !ov.inspector, JSON.stringify(ov));
      await check(`${res.name} no canvas/status overlap`, !ov.status, JSON.stringify(ov));
      // collapse works
      await click(page, "#btn-toggle-explorer");
      await page.waitForTimeout(200);
      await check(`${res.name} explorer collapses`, await evalNode(page, "#sidebar-left", n => n.classList.contains("collapsed")) === true);
      await click(page, "#btn-toggle-explorer");
      await check(`${res.name} explorer re-expands`, await evalNode(page, "#sidebar-left", n => n.classList.contains("collapsed")) === false);
    } catch (e) {
      results.fail++; results.errors.push(res.name + ": " + e.message); console.log("FAIL:", res.name, e.message);
    } finally {
      await context.close();
    }
  }
  await browser.close();

  console.log("\n========== PHASE 1 RESPONSIVE RESULTS ==========");
  console.log("PASS:", results.pass);
  console.log("FAIL:", results.fail);
  console.log("SCREENSHOTS:", screenshots.join(", "));
  if (results.errors.length) { console.log("\nERRORS:"); results.errors.forEach(e => console.log(" -", e)); }
  process.exit(results.fail > 0 ? 1 : 0);
})().catch(e => { console.error("FATAL:", e); process.exit(2); });
