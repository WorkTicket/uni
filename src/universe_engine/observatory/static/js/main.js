/* main.js - boot sequence: staged loader with health check, generate with timeout,
   and a non-blocking guide. The guide never blocks: Skip is always clickable and
   the guide state is saved so it is not repeated. */
(function (g) {
  "use strict";
  const U = g.U, $ = (id) => document.getElementById(id);
  const q = new URLSearchParams(location.search);
  const LOADER_TIMEOUT_MS = 30000;
  const GUIDE_KEY = "universe-guide-done";

  function guideDone() { return localStorage.getItem(GUIDE_KEY) === "1"; }
  function markGuideDone() { try { localStorage.setItem(GUIDE_KEY, "1"); } catch (e) { /* storage unavailable */ } }

  function setStep(id, done) {
    const el = $(id); if (!el) return;
    el.classList.toggle("done", done);
  }
  function showActions(show) { $("loader-actions").style.display = show ? "flex" : "none"; }
  function showSteps(show) { $("loader-steps").style.display = show ? "block" : "none"; }

  let msgLast = "";
  function failWith(msg) {
    msgLast = msg;
    $("prog-label").textContent = msg;
    $("intro-err").textContent = msg;
    $("intro-err").hidden = false;
    showActions(true);
    $("btn-begin").disabled = true;
    $("btn-skip").disabled = false; // Skip always remains clickable (guide never blocks)
  }
  function copyDiagnostics() {
    const info = {
      url: location.href,
      ua: navigator.userAgent,
      viewport: [window.innerWidth, window.innerHeight],
      webgl: !!$("gl"),
      time: new Date().toISOString(),
    };
    const text = "Universe Observatory diagnostics:\n" + JSON.stringify(info, null, 2);
    try { navigator.clipboard.writeText(text); } catch (e) { /* clipboard unavailable */ }
    $("intro-err").textContent = "Diagnostics copied to clipboard. " + msgLast;
    $("intro-err").hidden = false;
  }

  function buildUniverseHook(app) {
    g.__universe = {
      app, U,
      tick(n, dt) { for (let i = 0; i < (n || 1); i++) { app.update(dt || 0.016); } app.render(); },
      begin() { $("intro").classList.add("gone"); },
      setT(t) { app.setT(t); }, setS(s) { app.setS(s); },
    };
  }

  function startRenderLoop(app) {
    let last = performance.now();
    function frame(now) {
      const dt = Math.min(0.1, (now - last) / 1000); last = now;
      app.update(dt); app.render(); requestAnimationFrame(frame);
    }
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
  }

  async function runBoot() {
    let app;
    try { app = new U.App($("gl"), { quality: q.get("quality") || undefined }); }
    catch (e) { return failWith("This viewer needs WebGL2 (" + e.message + ")."); }
    g.__app = app;
    const ui = U.bindUI(app);

    try {
      const r = await fetch("/api/cosmology");
      if (r.ok) {
        const c = await r.json();
        U.Cosmo.build({ H0: c.H0_km_s_Mpc, Om: c.Omega_m, Or: c.Omega_r, ns: c.spectral_index || 0.965, source: "engine config/reference.yaml" });
        app.cosmoSource = U.Cosmo.P.source;
      }
    } catch (e) { /* offline / file:// */ }

    const seed = q.has("seed") ? parseInt(q.get("seed"), 10) : Math.floor(Math.random() * 1e9);
    const seedEl = $("seed"); if (seedEl) seedEl.value = seed;

    if (!guideDone()) {
      showSteps(true);
      $("btn-skip").disabled = false;
      setStep("step-server", true);

      let generationError = null;
      const genPromise = (async () => {
        try { await app.generate(seed, ui.progress); }
        catch (e) { generationError = e; }
      })();
      const timeoutPromise = new Promise((resolve) => setTimeout(resolve, LOADER_TIMEOUT_MS));
      await Promise.race([genPromise, timeoutPromise]);
      if (!app.uni) {
        if (generationError) return failWith("Universe generation failed: " + generationError.message);
        return failWith("Universe generation timed out after " + (LOADER_TIMEOUT_MS / 1000) + " seconds.");
      }
      setStep("step-universe", true);
      setStep("step-state", true);
      $("prog-label").textContent = "Ready — " + app.uni.K.toLocaleString() + " halos, seed " + seed;
      setStep("step-renderer", true);
      showSteps(false);

      $("btn-loader-retry").onclick = () => { location.reload(); };
      $("btn-loader-copy").onclick = copyDiagnostics;
      $("btn-loader-continue").onclick = () => { markGuideDone(); $("intro").classList.add("gone"); };
    } else {
      $("btn-skip").disabled = false;
      $("prog-label").textContent = "Ready";
      $("intro").classList.add("gone");
    }

    $("btn-begin").disabled = false;
    $("btn-skip").disabled = false;
    $("btn-begin").onclick = () => { markGuideDone(); app.homeView(); app.setS(0); $("intro").classList.add("gone"); app.reachedNow = false; app.pause(); };
    $("btn-skip").onclick = () => { markGuideDone(); app.homeView(); app.setS(U.Cosmo.NOW_S); $("intro").classList.add("gone"); app.pause(); };

    buildUniverseHook(app);
    startRenderLoop(app);
    if (q.has("autostart")) { $("btn-skip").click(); }
  }
  runBoot();
})(window);
