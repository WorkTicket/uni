/* main.js - boot sequence: load cosmology parameters from the engine (if served), generate the web, start the loop. */
(function (g) {
  "use strict";
  const U = g.U, $ = (id) => document.getElementById(id);
  const q = new URLSearchParams(location.search);

  async function boot() {
    let app;
    try { app = new U.App($("gl"), { quality: q.get("quality") || undefined }); }
    catch (e) { $("prog-label").textContent = ""; const el = $("intro-err"); el.hidden = false; el.textContent = "This viewer needs WebGL2 (" + e.message + "). Try a current Chrome, Edge, Firefox or Safari."; return; }
    g.__app = app;
    const ui = U.bindUI(app);
    // cosmology straight from the engine's config (falls back to built-in Planck-like values when opened from disk)
    try {
      const r = await fetch("/api/cosmology"); if (r.ok) { const c = await r.json(); U.Cosmo.build({ H0: c.H0_km_s_Mpc, Om: c.Omega_m, Or: c.Omega_r, ns: c.spectral_index || 0.965, source: "engine config/reference.yaml" }); app.cosmoSource = U.Cosmo.P.source; }
    } catch (e) { /* offline / file:// */ }
    const qq = q.get("quality"); if (qq && U.QUALITY[qq]) { document.querySelectorAll("#quality button").forEach((b) => b.classList.toggle("on", b.dataset.q === qq)); app.quality = qq; }
    const seed = q.has("seed") ? parseInt(q.get("seed"), 10) : Math.floor(Math.random() * 1e9);
    $("seed").value = seed;
    try { await app.generate(seed, ui.progress); } catch (e) { ui.fail(e); return; }
    $("prog-label").textContent = "Ready — " + app.uni.K.toLocaleString() + " halos, seed " + seed;
    $("btn-begin").disabled = false; $("btn-skip").disabled = false;
    let last = performance.now();
    function frame(now) { const dt = Math.min(0.1, (now - last) / 1000); last = now; app.update(dt); app.render(); requestAnimationFrame(frame); }
    if (!q.has("manual")) requestAnimationFrame(frame);
    // WebGL context loss / restoration handling
    const canvas = $("gl");
    let lost = false;
    canvas.addEventListener("webglcontextlost", (e) => { e.preventDefault(); lost = true; }, false);
    canvas.addEventListener("webglcontextrestored", () => { lost = false; app.resize(); }, false);
    const renderLoop = () => {
      if (lost) { requestAnimationFrame(renderLoop); return; }
      const now = performance.now(); const dt = Math.min(0.1, (now - last) / 1000); last = now;
      app.update(dt); app.render(); requestAnimationFrame(renderLoop);
    };
    if (!q.has("manual")) requestAnimationFrame(renderLoop);
    // test / scripting hook (also handy from the browser console)
    g.__universe = {
      app, U,
      tick(n, dt) { for (let i = 0; i < (n || 1); i++) { app.update(dt || 0.016); } app.render(); },
      begin() { $("intro").classList.add("gone"); },
      setT(t) { app.setT(t); }, setS(s) { app.setS(s); },
    };
    if (q.has("autostart")) { $("btn-skip").click(); }
  }
  boot();
})(window);
