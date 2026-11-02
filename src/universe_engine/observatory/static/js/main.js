/* main.js - boot sequence: staged loader, load cosmology, generate the web, start the loop.
   Bug H fix: never dead-lock on "Preparing…". The loader polls /health, bounds
   app.generate with a timeout, and on failure offers Retry / Copy error /
   Continue without guide so the viewer always reaches a usable state. */
(function (g) {
  "use strict";
  const U = g.U, $ = (id) => document.getElementById(id);
  const q = new URLSearchParams(location.search);
  const loader = {
    overlay: $("loader-overlay"),
    steps: ["loader-step-1", "loader-step-2", "loader-step-3", "loader-step-4", "loader-step-5"],
    label: $("loader-prog-label"),
    fail: $("loader-fail"),
    actions: $("loader-actions"),
    retry: $("btn-loader-retry"),
    copy: $("btn-loader-copy"),
    cont: $("btn-loader-continue"),
    current: 0,
    done(id) {
      const el = $(id); if (el) { el.classList.add("done"); }
      this.current = Math.min(this.current + 1, this.steps.length);
    },
    setLabel(t) { if (this.label) this.label.textContent = t; },
    setFail(t) {
      if (!this.fail) return;
      this.fail.hidden = !t;
      if (t) this.fail.textContent = t;
    },
    showActions(show) { if (this.actions) this.actions.hidden = !show; },
    hide() { if (this.overlay) this.overlay.hidden = true; },
  };
  if (U.loader) Object.assign(U.loader, loader); else U.loader = loader;

  async function waitForHealth(timeoutMs = 30000, intervalMs = 500) {
    const deadline = Date.now() + timeoutMs;
    let lastErr = null;
    while (Date.now() < deadline) {
      try {
        const r = await fetch("/health", { cache: "no-store", signal: AbortSignal.timeout(3000) });
        if (r.ok) { const j = await r.json().catch(() => ({})); if (j.ok) return j; }
      } catch (e) { lastErr = e; }
      await new Promise((res) => setTimeout(res, intervalMs));
    }
    throw new Error("server /health did not respond in time (" + (lastErr && lastErr.message) + ")");
  }

  async function boot() {
    const intro = $("intro");
    if (intro) intro.hidden = true;
    const canvas = $("gl");
    let app;
    try {
      loader.setLabel("Waiting for observatory server…");
      loader.done("loader-step-1");
      await waitForHealth();
      loader.done("loader-step-2");

      loader.setLabel("Creating WebGL2 context…");
      try { app = new U.App(canvas, { quality: q.get("quality") || undefined }); }
      catch (e) {
        loader.setFail("WebGL2 unavailable: " + e.message);
        loader.showActions(true);
        U.loader.retryHandler = () => location.reload();
        return;
      }
      loader.done("loader-step-3");
      g.__app = app;

      loader.setLabel("Binding UI…");
      const ui = U.bindUI(app);
      loader.done("loader-step-4");

      // cosmology straight from the engine's config (falls back to built-in Planck-like values when opened from disk)
      try {
        const r = await fetch("/api/cosmology"); if (r.ok) { const c = await r.json(); U.Cosmo.build({ H0: c.H0_km_s_Mpc, Om: c.Omega_m, Or: c.Omega_r, ns: c.spectral_index || 0.965, source: "engine config/reference.yaml" }); app.cosmoSource = U.Cosmo.P.source; }
      } catch (e) { /* offline / file:// */ }

      const qq = q.get("quality"); if (qq && U.QUALITY[qq]) { document.querySelectorAll("#quality button").forEach((b) => b.classList.toggle("on", b.dataset.q === qq)); app.quality = qq; }
      const seed = q.has("seed") ? parseInt(q.get("seed"), 10) : Math.floor(Math.random() * 1e9);
      if ($("seed")) $("seed").value = seed;

      loader.setLabel("Generating world (seed " + seed + ")…");
      // Bound generate so a hung world build cannot dead-lock the boot.
      const genTimeoutMs = q.has("manual") ? 0 : 120000;
      let genPromise = app.generate(seed, (p) => { loader.setLabel("Generating world… " + Math.round(p * 100) + "%"); });
      if (genTimeoutMs > 0) {
        genPromise = Promise.race([
          genPromise,
          new Promise((_, rej) => setTimeout(() => rej(new Error("world generation timed out after " + (genTimeoutMs / 1000) + "s")), genTimeoutMs)),
        ]);
      }
      await genPromise;
      loader.done("loader-step-5");

      if ($("prog-label")) $("prog-label").textContent = "Ready — " + app.uni.K.toLocaleString() + " halos, seed " + seed;
      if ($("btn-begin")) $("btn-begin").disabled = false;
      if ($("btn-skip")) $("btn-skip").disabled = false;

      let last = performance.now();
      function frame(now) { const dt = Math.min(0.1, (now - last) / 1000); last = now; app.update(dt); app.render(); requestAnimationFrame(frame); }
      if (!q.has("manual")) requestAnimationFrame(frame);

      // WebGL context loss / restoration handling
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
        begin() { if ($("intro")) $("intro").classList.add("gone"); },
        setT(t) { app.setT(t); }, setS(s) { app.setS(s); },
      };
      if (q.has("autostart")) { if ($("btn-skip")) $("btn-skip").click(); }

      loader.hide();
      if (intro) intro.hidden = false;
    } catch (e) {
      loader.setFail("Boot failed: " + (e && e.message ? e.message : String(e)));
      loader.showActions(true);
      U.loader.retryHandler = () => location.reload();
      console.error("observatory boot failed", e);
    }
  }

  // Retry / Copy error / Continue without guide wiring.
  loader.retry.addEventListener("click", () => { if (typeof loader.retryHandler === "function") loader.retryHandler(); else location.reload(); });
  loader.copy.addEventListener("click", () => {
    const msg = loader.fail ? loader.fail.textContent : "";
    navigator.clipboard.writeText(msg).then(() => { loader.copy.textContent = "Copied!"; setTimeout(() => { loader.copy.textContent = "Copy error"; }, 1200); });
  });
  loader.cont.addEventListener("click", () => {
    // Skip generation/UI binding issues: reveal the shell and let the user
    // explore what loaded rather than staying stuck on the loader.
    loader.hide();
    if ($("intro")) { $("intro").hidden = false; $("intro").classList.remove("gone"); }
    if ($("btn-begin")) $("btn-begin").disabled = false;
    if ($("btn-skip")) $("btn-skip").disabled = false;
  });

  boot();
})(window);
