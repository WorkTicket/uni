/* ui.js - HUD, timeline, breadcrumbs, info card, labels, ruler and input handling. */
(function (g) {
  "use strict";
  const U = g.U, V = U.V, Q = U.Q;
  const $ = (id) => document.getElementById(id);

  U.bindUI = function (app) {
    const canvas = app.canvas, labelsEl = $("labels"), tip = $("tip");
    const ui = { labels: true, orbits: true, motion: true };
    let lastHud = 0, lastFocus = null, labelPool = [];
    const keys = {};

    // ---- timeline ticks (epoch landmarks) ----
    const ticks = $("tl-ticks");
    U.Cosmo.EPOCHS.forEach((e, i) => {
      if (i === 0) return;
      const el = document.createElement("i"); el.style.left = (U.Cosmo.sOfT(e.t) * 100) + "%";
      if (/Recombination|Cosmic Dawn|Present/.test(e.name)) el.className = /Present/.test(e.name) ? "now" : "major";
      el.title = e.name; ticks.appendChild(el);
    });

    // ---- transport ----
    const slider = $("tl"), play = $("btn-play");
    // Slider scrubs physical (cosmic) time within the validated horizon; it never resets the universe.
    slider.addEventListener("input", () => { app.pause(); app.setS(parseFloat(slider.value) / 1000); });
    play.addEventListener("click", () => { const r = app.togglePlayback(); updatePlaybackUI(); });
    app.on("play", (p) => { play.innerHTML = p ? "&#10074;&#10074;" : "&#9654;"; play.setAttribute("aria-label", p ? "Pause" : "Play"); updatePlaybackUI(); });
    app.on("playback-changed", () => { updatePlaybackUI(); });
    app.on("horizon-reached", () => { updatePlaybackUI(); showHorizonNotice(); });
    app.on("horizon-extended", (r) => { updatePlaybackUI(); showHorizonNotice(r); });
    // Safe binding: skip elements that are not present in the shell so a
    // missing DOM node can never abort the boot sequence (Bug H).
    function bind(id, ev, fn) { const el = $(id); if (el) el.addEventListener(ev, fn); }
    $("btn-step-back").addEventListener("click", () => { app.stepBackward(); });
    $("btn-step-fwd").addEventListener("click", () => { app.stepForward(); });
    // Playback presets: Paused / Observe / Explore / Fast Forward / Custom / Cinematic Tour.
    const speedSeg = $("speed");
    function renderPresets() {
      if (!speedSeg) return; // speed segment not present in this shell
      speedSeg.innerHTML = "";
      const names = app.playback.presetNames();
      names.forEach((key) => {
        const p = app.playback.preset(key);
        const b = document.createElement("button");
        b.dataset.s = p.rate; b.dataset.preset = key;
        b.textContent = (p.cinematic ? "[" : "") + p.label + (p.cinematic ? "]" : "");
        b.title = p.label + " · " + fmtRate(p.rate) + " sim s / wall s" + (p.cinematic ? " (compressed overview)" : "");
        b.classList.toggle("on", key === app.playback.preset && Math.abs(app.playback.rate - p.rate) < 1e-6);
        b.addEventListener("click", () => { app.pause(); const r = app.playback.setPreset(key); updatePlaybackUI(); });
        speedSeg.appendChild(b);
      });
      // Custom rate input.
      const custom = app.playback.preset("custom");
      const inp = document.createElement("input"); inp.type = "number"; inp.step = "any"; inp.min = "0";
      inp.value = fmtRate(app.playback.customRate);
      inp.title = "Custom playback rate (simulated seconds per wall-clock second)";
      inp.addEventListener("change", () => {
        const v = parseFloat(inp.value);
        if (Number.isFinite(v) && v >= 0) { app.pause(); app.playback.setCustomRate(v); updatePlaybackUI(); }
      });
      speedSeg.appendChild(inp);
      const lbl = document.createElement("span"); lbl.className = "fine"; lbl.textContent = " sim s/wall s";
      speedSeg.appendChild(lbl);
    }
    function fmtRate(r) {
      if (r === 0) return "paused";
      if (r >= 1e9) return (r / 1e9).toFixed(2) + "×10⁹";
      if (r >= 1e6) return (r / 1e6).toFixed(2) + "×10⁶";
      if (r >= 1e3) return (r / 1e3).toFixed(2) + "×10³";
      if (r >= 1) return r.toFixed(2);
      return r.toExponential(2);
    }
    function updatePlaybackUI() {
      renderPresets();
      const r = app.getRate();
      const p = app.getPreset();
      const playing = app.getPlaying();
      $("btn-play").innerHTML = playing ? "&#10074;&#10074;" : "&#9654;";
      $("btn-play").setAttribute("aria-label", playing ? "Pause" : "Play");
      const rateEl = $("tl-rate");
      if (rateEl) rateEl.textContent = "Rate: " + fmtRate(r) + " · " + p + (playing ? " · playing" : " · paused");
    }
    function showHorizonNotice(r) {
      const el = $("horizon-notice");
      if (!el) return;
      if (app.U.Cosmo.atBoundary()) {
        el.hidden = false;
        el.textContent = r && r.message ? r.message : "Reached current model horizon — extend to continue";
      } else {
        el.hidden = true;
      }
    }
    $("btn-home").addEventListener("click", () => { app.homeView(); });
    bind("btn-dive", "click", () => startDive());
    function startDive() { app.planDive((Date.now() & 0xffff) ^ app.seed); }
    // ---- horizon extension ----
    const horizonInfo = $("horizon-info"), horizonVal = $("horizon-val"), horizonT = $("horizon-t");
    function updateHorizonDisplay() {
      if (!horizonInfo) return;
      const aMax = app.U.Cosmo.a_max;
      const tMax = app.U.Cosmo.tOfA(aMax);
      horizonVal.textContent = aMax.toFixed(2);
      horizonT.textContent = (tMax / (1e9 * app.U.YEAR)).toFixed(1);
      horizonInfo.hidden = false;
    }
    updateHorizonDisplay();
    bind("btn-extend-horizon", "click", () => {
      const v = parseFloat($("horizon-a").value);
      if (!Number.isFinite(v) || v <= 0) { $("horizon-a").focus(); return; }
      try {
        const r = app.extendHorizon(v);
        if (r.ok) { updateHorizonDisplay(); updatePlaybackUI(); showHorizonNotice(r); }
        else { alert("Cannot extend horizon: " + r.error); }
      } catch (e) { alert("Extend horizon failed: " + e.message); }
    });
    bind("btn-reset-horizon", "click", () => {
      try {
        const r = app.extendHorizon(80);
        updateHorizonDisplay(); updatePlaybackUI(); showHorizonNotice(r);
      } catch (e) { alert("Reset horizon failed: " + e.message); }
    });
    // ---- menu ----
    const menu = $("menu");
    bind("btn-menu", "click", () => { menu.hidden = !menu.hidden; });
    bind("opt-labels", "change", (e) => { ui.labels = e.target.checked; });
    bind("opt-orbits", "change", (e) => { ui.orbits = e.target.checked; app.orbits = e.target.checked; });
    bind("opt-motion", "change", (e) => { app.userRate = e.target.checked ? 1 : 0; });
    document.querySelectorAll("#quality button").forEach((b) => b.addEventListener("click", () => {
      document.querySelectorAll("#quality button").forEach((x) => x.classList.toggle("on", x === b)); app.pendingQuality = b.dataset.q;
    }));
    // View mode: Illustrative (procedural) vs Scientific Simulation (actual PM3D state)
    document.querySelectorAll("#view-mode button").forEach((b) => b.addEventListener("click", () => {
      document.querySelectorAll("#view-mode button").forEach((x) => x.classList.toggle("on", x === b));
      app.setScientific(b.dataset.m === "scientific");
    }));
    // PM3D simulation controls
    const pm3dStatus = $("pm3d-status");
    bind("btn-pm3d-create", "click", async () => {
      const seed = parseInt($("pm3d-seed").value, 10) || 42;
      try {
        pm3dStatus.textContent = "Creating PM3D simulation…";
        await app.pm3dCreate("sim-1", seed, 16, 8, 0.2, 2.0, 0.02);
        pm3dStatus.textContent = "PM3D sim at a=" + app.pm3dState.a.toFixed(3) + " (z=" + app.pm3dState.z.toFixed(2) + ", t=" + app.pm3dState.t_Gyr.toFixed(2) + " Gyr)";
        app.setScientific(true);
        document.querySelectorAll("#view-mode button").forEach((x) => x.classList.toggle("on", x.dataset.m === "scientific"));
      } catch (e) { pm3dStatus.textContent = "Error: " + e.message; console.error(e); }
    });
    bind("btn-pm3d-step", "click", async () => {
      if (!app.pm3dId) { pm3dStatus.textContent = "Create a simulation first."; return; }
      try {
        pm3dStatus.textContent = "Advancing simulation…";
        await app.pm3dStep(2.0, 0.02);
        pm3dStatus.textContent = "PM3D sim at a=" + app.pm3dState.a.toFixed(3) + " (z=" + app.pm3dState.z.toFixed(2) + ", t=" + app.pm3dState.t_Gyr.toFixed(2) + " Gyr)";
      } catch (e) { pm3dStatus.textContent = "Error: " + e.message; console.error(e); }
    });
    bind("btn-pm3d-play", "click", async () => {
      if (!app.pm3dId) { pm3dStatus.textContent = "Create a simulation first."; return; }
      try {
        const speed = parseFloat($("pm3d-speed").value) || 1;
        const aMax = parseFloat($("pm3d-amax").value) || 10;
        await app.pm3dPlay(speed, aMax);
        pm3dStatus.textContent = "PM3D playback playing (speed=" + speed + ")";
      } catch (e) { pm3dStatus.textContent = "Error: " + e.message; console.error(e); }
    });
    bind("btn-pm3d-pause", "click", async () => {
      if (!app.pm3dId) return;
      try {
        await app.pm3dPause();
        pm3dStatus.textContent = "PM3D playback paused.";
      } catch (e) { pm3dStatus.textContent = "Error: " + e.message; console.error(e); }
    });
    bind("btn-pm3d-checkpoint", "click", async () => {
      if (!app.pm3dId) { pm3dStatus.textContent = "Create a simulation first."; return; }
      try {
        const cid = "cp-" + Date.now();
        await fetch("/api/pm3d/" + app.pm3dId + "/checkpoint", {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ id: cid }),
        });
        pm3dStatus.textContent = "Checkpoint saved: " + cid;
      } catch (e) { pm3dStatus.textContent = "Error: " + e.message; console.error(e); }
    });
    bind("btn-pm3d-restore", "click", async () => {
      if (!app.pm3dId) { pm3dStatus.textContent = "Create a simulation first."; return; }
      const cid = prompt("Checkpoint ID to restore:");
      if (!cid) return;
      try {
        const r = await fetch("/api/pm3d/" + app.pm3dId + "/checkpoints/" + cid + "/restore", { method: "POST" });
        if (!r.ok) throw new Error("restore failed: " + r.status);
        const j = await r.json();
        app.pm3dState = j.state;
        app.simParticles.upload(app.pm3dState);
        pm3dStatus.textContent = "Restored checkpoint " + cid + " at a=" + j.state.a.toFixed(3);
      } catch (e) { pm3dStatus.textContent = "Error: " + e.message; console.error(e); }
    });
    bind("btn-rand", "click", () => { $("seed").value = Math.floor(Math.random() * 1e9); });
    bind("btn-new", "click", async () => {
      menu.hidden = true; const wasPlaying = app.playing; app.pause();
      if (app.pendingQuality) { app.setQuality(app.pendingQuality); app.pendingQuality = null; }
      showIntro(true);
      try { await app.generate(parseInt($("seed").value, 10) || 0, progress); } catch (e) { fail(e); return; }
      $("prog-label").textContent = "Ready"; app.setS(0); $("btn-begin").disabled = false; $("btn-skip").disabled = false;
      renderObjectExplorer(); drawInfo();
    });
    function progress(f, label) { $("prog-bar").style.width = (f * 100).toFixed(0) + "%"; if (label) $("prog-label").textContent = label + "…"; }
    function showIntro(on) { const i = $("intro"); i.classList.toggle("gone", !on); if (on) { $("btn-begin").disabled = true; $("btn-skip").disabled = true; } }
    function fail(e) { const el = $("intro-err"); el.hidden = false; el.textContent = String(e && e.message || e); console.error(e); }
    ui.progress = progress; ui.fail = fail;
    $("btn-begin").addEventListener("click", () => { app.homeView(); app.setS(0); showIntro(false); app.reachedNow = false; app.pause(); });
    $("btn-skip").addEventListener("click", () => { app.homeView(); app.setS(U.Cosmo.NOW_S); showIntro(false); app.pause(); });

    // ---- input ----
    let drag = null; const pointers = new Map(); let pinch = 0, moved = 0;
    const rectOf = () => canvas.getBoundingClientRect();
    canvas.addEventListener("pointerdown", (e) => {
      canvas.setPointerCapture(e.pointerId); pointers.set(e.pointerId, { x: e.clientX, y: e.clientY }); moved = 0;
      drag = { x: e.clientX, y: e.clientY, btn: e.button, shift: e.shiftKey }; canvas.classList.add("dragging");
      if (pointers.size === 2) { const [a, b] = [...pointers.values()]; pinch = Math.hypot(a.x - b.x, a.y - b.y); }
    });
    canvas.addEventListener("pointermove", (e) => {
      const r = rectOf(); app.mouse.x = (e.clientX - r.left) / r.width; app.mouse.y = (e.clientY - r.top) / r.height; app.mouse.inside = true;
      if (pointers.has(e.pointerId)) pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
      if (pointers.size === 2) {
        const [a, b] = [...pointers.values()], d = Math.hypot(a.x - b.x, a.y - b.y); if (pinch) app.wheel(-(d - pinch) * 3.2); pinch = d; return;
      }
      if (drag) {
        const dx = e.clientX - drag.x, dy = e.clientY - drag.y; drag.x = e.clientX; drag.y = e.clientY; moved += Math.abs(dx) + Math.abs(dy);
        if (app.modeOf() === "free" && (drag.btn === 2 || drag.shift)) app.freeLook(dx, dy);
        else if (drag.btn === 2 || drag.shift) app.look(dx, dy); else app.orbit(dx, dy);
      }
    });
    const up = (e) => {
      pointers.delete(e.pointerId); pinch = 0; canvas.classList.remove("dragging");
      if (drag && moved < 5 && drag.btn === 0 && app.hover) flyToHover();
      drag = null;
    };
    canvas.addEventListener("pointerup", up); canvas.addEventListener("pointercancel", up);
    canvas.addEventListener("pointerleave", () => { app.mouse.inside = false; });
    canvas.addEventListener("contextmenu", (e) => e.preventDefault());
    canvas.addEventListener("wheel", (e) => { e.preventDefault(); app.wheel(e.deltaMode === 1 ? e.deltaY * 33 : e.deltaY); }, { passive: false });
    function flyToHover() { const n = app.nodeFromHover(app.hover); if (n) app.flyTo(n); }
    g.addEventListener("keydown", (e) => {
      keys[e.key] = true;
      if (/INPUT|SELECT|TEXTAREA/.test(e.target.tagName) && e.key !== "Escape") return;
      switch (e.key) {
        case " ": e.preventDefault(); play.click(); break;
        case "d": case "D":
          if (app.modeOf() !== "free" && app.modeOf() !== "free_flight") startDive();
          break;
        case "h": case "H": app.homeView(); break;
        case "Backspace": case "ArrowDown": e.preventDefault(); app.autopilot = null; app.popOut(); break;
        case "ArrowUp": app.wheel(-120); break;
        case "w": case "W": if (app.modeOf() === "free" || app.modeOf() === "free_flight") app.freeMove(0, 0, -1); else app.wheel(-90); break;
        case "s": case "S": if (app.modeOf() === "free" || app.modeOf() === "free_flight") app.freeMove(0, 0, 1); else app.wheel(90); break;
        case "a": case "A": if (app.modeOf() === "free" || app.modeOf() === "free_flight") app.freeMove(-1, 0, 0); else app.orbit(-14, 0); break;
        case "ArrowLeft": app.orbit(-14, 0); break; case "ArrowRight": app.orbit(14, 0); break;
        case "l": case "L": $("opt-labels").click(); break; case "o": case "O": $("opt-orbits").click(); break;
        case "f": case "F": if (!document.fullscreenElement) document.documentElement.requestFullscreen && document.documentElement.requestFullscreen(); else document.exitFullscreen(); break;
        case "p": case "P": app.surface(app.node); break;
        case "c": case "C": {
          const cur = app.modeOf();
          const next = cur === "orbit_target" ? "follow_target" :
                       cur === "follow_target" ? "planet_surface" :
                       cur === "planet_surface" ? "free_flight" :
                       "orbit_target";
          if (next === "follow_target") app.follow(app.node);
          else if (next === "planet_surface") app.surface(app.node);
          else if (next === "free_flight") app.freeMode();
          else app.orbitMode();
        } break;
        case "g": case "G": app.galaxyMap(); break;
        case "m": case "M": app.solarSystemMap(); break;
        case "v": case "V": app.microMode(!app.micro); break;
        case "Escape": app.autopilot = null; menu.hidden = true; break;
        case "]": bumpSpeed(1); break; case "[": bumpSpeed(-1); break;
        case ",": app.stepBackward(); break; case ".": app.stepForward(); break;
        case "n": case "N": app.jumpToFuture(0.5); break; case "j": case "J": app.jumpToFuture(0.5); break;
      }
    });
    function bumpSpeed(d) { const bs = [...document.querySelectorAll("#speed button")], i = bs.findIndex((b) => b.classList.contains("on")); const n = bs[U.clamp(i + d, 0, bs.length - 1)]; n.click(); }
    g.addEventListener("keyup", (e) => { keys[e.key] = false; });
    g.addEventListener("resize", () => app.resize());

    // ---- per-frame HUD ----
    app.on("update", () => {
      const now = performance.now();
      // WASD movement: free-flight and surface modes.
      const mode = app.modeOf();
      if (mode === "free" || mode === "free_flight") {
        const vx = (keys["d"] || keys["D"] ? 1 : 0) - (keys["a"] || keys["A"] ? 1 : 0);
        const vz = (keys["s"] || keys["S"] ? 1 : 0) - (keys["w"] || keys["W"] ? 1 : 0);
        app.freeMove(vx, 0, vz);
      } else if (mode === "planet_surface" || app.camera && app.camera.modeOf && app.camera.modeOf() === "planet_surface") {
        const step = 0.5; // metres per frame of altitude/heading
        if (keys["w"] || keys["W"]) app.camera.surface.altitude += step;
        if (keys["s"] || keys["S"]) app.camera.surface.altitude = Math.max(0, app.camera.surface.altitude - step);
        if (keys["a"] || keys["A"]) app.camera.surface.heading -= 0.02;
        if (keys["d"] || keys["D"]) app.camera.surface.heading += 0.02;
        app.camera.surface.altitude = Math.max(0, app.camera.surface.altitude);
      }
      // tooltip follows hover every frame
      const h = app.hover;
      if (h && h.label && app.mouse.inside) {
        tip.hidden = false; $("tip-name").textContent = h.label.name; $("tip-sub").textContent = h.label.sub || "";
        tip.style.left = (h.sx * 0.5 + 0.5) * canvas.clientWidth + "px"; tip.style.top = (0.5 - h.sy * 0.5) * canvas.clientHeight + "px"; canvas.classList.add("pointing");
      } else { tip.hidden = true; canvas.classList.remove("pointing"); }
      if (now - lastHud < 90) return; lastHud = now;
      updateHud(); updateLabels();
    });
    function updateHud() {
      const t = app.t, vis = U.Cosmo.visual(t), ep = U.Cosmo.epochAt(t);
      $("t-main").textContent = U.fmtTime(t);
      $("t-epoch").textContent = ep.name;
      // Temperature: adaptive precision / scientific notation, never "0.0 K" for a nonzero value.
      $("t-temp").textContent = fmtTempAdaptive(vis.T);
      // Redshift z = 1/a - 1. Negative in the future (a > 1); never clamped to zero.
      $("t-z").textContent = fmtZ(vis.z);
      // Label the region size precisely: comoving side length of the periodic representative region.
      $("t-size").textContent = app.uni ? "Region (comoving) " + U.fmtDist(app.uni.boxM * vis.a) : "-";
      // Playback rate + horizon notice.
      const rateEl = $("tl-rate");
      if (rateEl) rateEl.textContent = "Rate: " + fmtRate(app.getRate()) + " · " + app.getPreset() + (app.getPlaying() ? " · playing" : " · paused");
      // PM3D scientific simulation stats
      if (app.pm3dState) {
        const statsEl = $("pm3d-stats");
        const ds = app.pm3dState.density_stats || {};
        statsEl.textContent =
          "a=" + app.pm3dState.a.toFixed(3) +
          " z=" + fmtZ(app.pm3dState.z) +
          " t=" + app.pm3dState.t_Gyr.toFixed(2) + " Gyr" +
          " ρ_rms=" + (ds.rms !== undefined ? ds.rms.toFixed(3) : "–") +
          " pts=" + app.pm3dState.returned_points + "/" + app.pm3dState.n_particles;
        $("pm3d-play").textContent = app.pm3dPlaying ? "Pause" : "Play";
      }
      if (document.activeElement !== slider) slider.value = Math.round(app.scrubS * 1000);
      $("ep-name").textContent = ep.name; $("ep-text").textContent = ep.text;
      document.body.classList.toggle("cinema", app.getPlaying() && app.node.kind === "universe");
      $("epoch-card").style.opacity = app.node.kind === "universe" || app.node.kind === "halo" ? 1 : 0;
      showHorizonNotice();
      // breadcrumbs / info on focus change
      if (lastFocus !== app.node) { lastFocus = app.node; drawCrumbs(); drawInfo(); }
      if (app.node.kind === "star" || app.node.kind === "planet") { /* keep info fresh enough */ }
      // ruler: log10 of distance to focus
      const d = V.len(app.pos), lg = Math.log10(Math.max(d, 1)), lo = 3, hi = 26.5, f = U.clamp((lg - lo) / (hi - lo), 0, 1);
      $("ruler-mark").style.bottom = (f * 100) + "%"; $("ruler-val").style.display = "block"; $("ruler-val").textContent = U.fmtDist(d);
    }
    // Adaptive temperature formatting: avoid "0.0 K" for nonzero values; use scientific notation when tiny/huge.
    function fmtTempAdaptive(T) {
      if (!Number.isFinite(T)) return "–";
      if (T === 0) return "0 K";
      if (T < 1e-3 || T >= 1e6) return T.toExponential(2) + " K";
      if (T < 1) return T.toExponential(3) + " K";
      if (T < 100) return T.toFixed(2) + " K";
      return T.toFixed(1) + " K";
    }
    // Redshift formatting: z = 1/a - 1; negative in the future (a > 1).
    function fmtZ(z) {
      if (!Number.isFinite(z)) return "–";
      if (Math.abs(z) < 1e-6) return "0";
      if (Math.abs(z) < 1) return z.toFixed(3);
      if (Math.abs(z) < 1e4) return z.toFixed(2);
      return z.toExponential(2);
    }
    const RT = [[1e4, "10 km"], [1.27e7, "Earth"], [1.4e9, "Sun"], [1.5e11, "1 AU"], [9.46e15, "1 light-year"], [3.1e17, "10 pc"], [1e21, "Galaxy"], [3.09e22, "1 Mpc"], [3.09e24, "100 Mpc"], [3e25, "Region"]];
    const rt = $("ruler-ticks"); RT.forEach(([m, l]) => { const li = document.createElement("li"); li.textContent = l; li.style.bottom = (((Math.log10(m) - 3) / (26.5 - 3)) * 100) + "%"; rt.appendChild(li); });
    $("ruler-val").style.display = "none";

    function drawCrumbs() {
      const c = $("crumbs"); c.innerHTML = ""; const ch = app.node.chain();
      ch.forEach((n, i) => {
        if (i) { const s = document.createElement("i"); s.textContent = "›"; c.appendChild(s); }
        const b = document.createElement("button"); b.textContent = n.kind === "universe" ? "Universe" : n.name; if (i === ch.length - 1) b.className = "cur";
        else b.addEventListener("click", () => { app.autopilot = null; app.flyTo(n); });
        c.appendChild(b);
      });
    }
    function drawInfo() {
      const n = app.node, d = n.data;
      const KIND = { universe: "Cosmic web", halo: d.groupName || "Halo", galaxy: "Galaxy", star: "Star system", planet: "Planet", moon: "Moon" };
      const inspKind = $("insp-kind"), inspName = $("insp-name"), inspRows = $("insp-rows"), inspText = $("insp-text");
      if (inspKind) inspKind.textContent = KIND[n.kind] || n.kind;
      if (inspName) inspName.textContent = n.kind === "universe" ? "The universe" : n.name;
      if (inspRows) { inspRows.innerHTML = ""; app.describe(n).forEach(([k, v]) => { const a = document.createElement("dt"), b = document.createElement("dd"); a.textContent = k; b.textContent = v; inspRows.appendChild(a); inspRows.appendChild(b); }); }
      if (inspText) inspText.textContent = n.kind === "universe" ? "Scroll toward any glowing knot to fly into a cluster of galaxies." : n.kind === "galaxy" ? "Every point of light is a star. Scroll toward one to visit it." : n.kind === "star" ? "Click a planet or its label to fly there." : "";
      // show habitat panel for planets with a habitat
      const habPanel = $("habitat-panel");
      if (habPanel) habPanel.style.display = (n.kind === "planet" || n.kind === "moon") ? "block" : "none";
      if (n.kind === "planet" || n.kind === "moon") { app.maybeRefreshHabitat(n); renderHabitat(n); }
    }
    async function renderHabitat(node) {
      const rowsEl = $("hab-rows"); const textEl = $("hab-text"); const actsEl = $("hab-actions");
      if (!rowsEl) return;
      const hid = node && node.__habId;
      if (!hid) { rowsEl.innerHTML = ""; actsEl.innerHTML = ""; if (textEl) textEl.textContent = "Create a habitat to inspect this planet's environment."; return; }
      const st = app.habitatCache && app.habitatCache[hid];
      if (!st) { rowsEl.innerHTML = ""; actsEl.innerHTML = ""; if (textEl) textEl.textContent = "Loading habitat environment&hellip;"; return; }
      rowsEl.innerHTML = "";
      const rows = [
        ["Time (Gyr)", st.t_gyr !== undefined ? st.t_gyr.toFixed(3) : "–"],
        ["Sterile", String(!!st.sterile)],
        ["Replicators", st.replicators ? JSON.stringify(st.replicators) : "–"],
        ["Monomer pool", st.pool_monomer !== undefined ? String(st.pool_monomer) : "–"],
        ["Mass ledger error", st.mass_ledger_error !== undefined ? st.mass_ledger_error.toExponential(2) : "–"],
      ];
      for (const [k, v] of rows) { const a = document.createElement("dt"), b = document.createElement("dd"); a.textContent = k; b.textContent = v; rowsEl.appendChild(a); rowsEl.appendChild(b); }
      if (st.patches && st.patches.length) {
        const p = st.patches[0];
        const a = document.createElement("dt"); a.textContent = "Patch 0 temperature (K)"; const b = document.createElement("dd"); b.textContent = (p.temperature_K !== undefined ? p.temperature_K.toFixed(1) : "–"); rowsEl.appendChild(a); rowsEl.appendChild(b);
        const a2 = document.createElement("dt"); a2.textContent = "Patch 0 pH"; const b2 = document.createElement("dd"); b2.textContent = (p.pH !== undefined ? p.pH.toFixed(2) : "–"); rowsEl.appendChild(a2); rowsEl.appendChild(b2);
        const a3 = document.createElement("dt"); a3.textContent = "Liquid water"; const b3 = document.createElement("dd"); b3.textContent = String(!!p.liquid_water); rowsEl.appendChild(a3); rowsEl.appendChild(b3);
      }
      actsEl.innerHTML = "";
      const btnSeed = document.createElement("button"); btnSeed.textContent = "Seed replicator"; btnSeed.className = "pill secondary"; btnSeed.style.width = "auto";
      btnSeed.onclick = async () => { try { await fetch("/api/habitat/" + encodeURIComponent(hid), { method: "POST" }); await app.maybeRefreshHabitat(node); renderHabitat(node); } catch (e) {} };
      actsEl.appendChild(btnSeed);
      if (textEl) textEl.textContent = st.sterile ? "Habitat is sterile — seed replicators to begin life." : "Habitat active — chemistry and replicators are coupled to the planet environment.";
    }
    // ---- object explorer (PASS 2: searchable Universe Explorer) ----
    let catalog = [], catalogT = 0;
    function renderObjectExplorer() {
      const list = $("explore-list"); if (!list) return;
      const q = $("obj-search").value;
      const cat = app.buildObjectCatalog();
      catalog = cat; catalogT = app.time;
      list.innerHTML = "";
      const shown = cat.filter((o) => !q || o.name.toLowerCase().includes(q.toLowerCase()) || o.label.toLowerCase().includes(q.toLowerCase()));
      if (shown.length === 0) { list.innerHTML = "<div class='empty'>No objects match.</div>"; return; }
      const cur = app.node ? app.node : null;
      for (const o of shown.slice(0, 200)) {
        const el = document.createElement("div");
        el.className = "obj " + (o.kind) + (cur === o.node ? " selected" : "");
        el.innerHTML = `<div class="dot kind-${o.kind}"></div><div><div class="obj-name">${escapeHtml(o.name)}</div><div class="obj-sub">${escapeHtml(o.label)}</div></div><div class="obj-actions"><button data-kind="${o.kind}" data-focus>Focus</button></div>`;
        el.querySelector("[data-focus]").addEventListener("click", (ev) => { ev.stopPropagation(); app.focusObjectByNode(o.node); });
        el.addEventListener("click", () => { app.focusObjectByNode(o.node); renderObjectExplorer(); drawInfo(); });
        list.appendChild(el);
      }
    }
    function escapeHtml(t) { return String(t).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;", "'": "&#39;" })[c]); }
    // ---- navigation wiring ----
    function wireNav() {
      $("btn-focus").addEventListener("click", () => { const n = app.node; if (n) app.focusObjectByNode(n); });
      $("btn-follow").addEventListener("click", () => { app.follow(app.node); app.camMode = app.camera.modeOf(); });
      $("btn-surface").addEventListener("click", () => { app.surface(app.node); app.camMode = app.camera.modeOf(); });
      $("btn-orbit").addEventListener("click", () => { app.orbitMode(); app.camMode = app.camera.modeOf(); });
      $("btn-free").addEventListener("click", () => { app.freeMode(); app.camMode = app.camera.modeOf(); });
      $("btn-home").addEventListener("click", () => { app.homeView(); app.camMode = app.camera.modeOf(); renderObjectExplorer(); drawInfo(); });
      $("btn-back").addEventListener("click", () => { if (app.camera.back()) { app.camMode = app.camera.modeOf(); renderObjectExplorer(); drawInfo(); } });
      $("obj-search").addEventListener("input", () => { renderObjectExplorer(); });
    }
    function labelEl(i) {
      let el = labelPool[i];
      if (!el) { el = document.createElement("div"); el.className = "lab"; labelsEl.appendChild(el); el.addEventListener("click", () => el._node && app.flyTo(el._node)); labelPool[i] = el; }
      return el;
    }
    function updateLabels() {
      let used = 0;
      if (ui.labels && app.uni && !app.autopilot) {
        const W = canvas.clientWidth, H = canvas.clientHeight, items = app.labelTargets(), f = app.proj, b = Q.basis(app.q);
        for (const it of items) {
          if (used >= 16) break;
          const x = V.dot(b[0], it.rel), y = V.dot(b[1], it.rel), z = V.dot(b[2], it.rel), w = -z; if (w <= 0) continue;
          const sx = x * f[0] / w, sy = y * f[1] / w; if (Math.abs(sx) > 1.02 || Math.abs(sy) > 1.02) continue;
          const rpx = it.size / w * app.pxScale / (app.canvas.height / H);
          if (it.minPx && rpx < it.minPx) continue;
          const el = labelEl(used++); el._node = it.node; el.className = "lab " + (it.node.kind === "star" ? "star" : it.node.kind === "moon" ? "moon" : "");
          el.innerHTML = ""; const s1 = document.createElement("span"); s1.textContent = it.node.name; el.appendChild(s1);
          if (it.node.data && it.node.data.cls) { const sm = document.createElement("small"); sm.textContent = it.node.data.cls; el.appendChild(sm); }
          el.style.left = ((sx * 0.5 + 0.5) * W + Math.min(rpx, 60)) + "px"; el.style.top = ((0.5 - sy * 0.5) * H) + "px"; el.style.display = "flex";
        }
      }
      for (let i = used; i < labelPool.length; i++) labelPool[i].style.display = "none";
    }
    // Initialize the playback UI (presets, rate, horizon notice).
    updatePlaybackUI();
    wireNav();
    renderObjectExplorer();
    return ui;
  };
})(typeof window !== "undefined" ? window : globalThis);
