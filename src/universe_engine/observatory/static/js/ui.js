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
    slider.addEventListener("input", () => { app.pause(); app.setS(slider.value / 1000); });
    play.addEventListener("click", () => (app.playing ? app.pause() : app.play()));
    app.on("play", (p) => { play.innerHTML = p ? "&#10074;&#10074;" : "&#9654;"; play.setAttribute("aria-label", p ? "Pause" : "Play"); });
    $("btn-step-back").addEventListener("click", () => { app.stepBackward(); });
    $("btn-step-fwd").addEventListener("click", () => { app.stepForward(); });
    document.querySelectorAll("#speed button").forEach((b) => b.addEventListener("click", () => {
      app.speed = parseFloat(b.dataset.s); document.querySelectorAll("#speed button").forEach((x) => x.classList.toggle("on", x === b));
    }));
    $("btn-home").addEventListener("click", () => { app.homeView(); });
    $("btn-dive").addEventListener("click", () => startDive());
    function startDive() { app.planDive((Date.now() & 0xffff) ^ app.seed); }
    // ---- menu ----
    const menu = $("menu");
    $("btn-menu").addEventListener("click", () => { menu.hidden = !menu.hidden; });
    $("opt-labels").addEventListener("change", (e) => { ui.labels = e.target.checked; });
    $("opt-orbits").addEventListener("change", (e) => { ui.orbits = e.target.checked; app.orbits = e.target.checked; });
    $("opt-motion").addEventListener("change", (e) => { app.userRate = e.target.checked ? 1 : 0; });
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
    $("btn-pm3d-create").addEventListener("click", async () => {
      const seed = parseInt($("pm3d-seed").value, 10) || 42;
      try {
        pm3dStatus.textContent = "Creating PM3D simulation…";
        await app.pm3dCreate("sim-1", seed, 16, 8, 0.2, 2.0, 0.02);
        pm3dStatus.textContent = "PM3D sim at a=" + app.pm3dState.a.toFixed(3) + " (z=" + app.pm3dState.z.toFixed(2) + ", t=" + app.pm3dState.t_Gyr.toFixed(2) + " Gyr)";
        app.setScientific(true);
        document.querySelectorAll("#view-mode button").forEach((x) => x.classList.toggle("on", x.dataset.m === "scientific"));
      } catch (e) { pm3dStatus.textContent = "Error: " + e.message; console.error(e); }
    });
    $("btn-pm3d-step").addEventListener("click", async () => {
      if (!app.pm3dId) { pm3dStatus.textContent = "Create a simulation first."; return; }
      try {
        pm3dStatus.textContent = "Advancing simulation…";
        await app.pm3dStep(2.0, 0.02);
        pm3dStatus.textContent = "PM3D sim at a=" + app.pm3dState.a.toFixed(3) + " (z=" + app.pm3dState.z.toFixed(2) + ", t=" + app.pm3dState.t_Gyr.toFixed(2) + " Gyr)";
      } catch (e) { pm3dStatus.textContent = "Error: " + e.message; console.error(e); }
    });
    $("btn-rand").addEventListener("click", () => { $("seed").value = Math.floor(Math.random() * 1e9); });
    $("btn-new").addEventListener("click", async () => {
      menu.hidden = true; const wasPlaying = app.playing; app.pause();
      if (app.pendingQuality) { app.setQuality(app.pendingQuality); app.pendingQuality = null; }
      showIntro(true);
      try { await app.generate(parseInt($("seed").value, 10) || 0, progress); } catch (e) { fail(e); return; }
      $("prog-label").textContent = "Ready"; app.setS(0); $("btn-begin").disabled = false; $("btn-skip").disabled = false;
    });
    function progress(f, label) { $("prog-bar").style.width = (f * 100).toFixed(0) + "%"; if (label) $("prog-label").textContent = label + "…"; }
    function showIntro(on) { const i = $("intro"); i.classList.toggle("gone", !on); if (on) { $("btn-begin").disabled = true; $("btn-skip").disabled = true; } }
    function fail(e) { const el = $("intro-err"); el.hidden = false; el.textContent = String(e && e.message || e); console.error(e); }
    ui.progress = progress; ui.fail = fail;
    $("btn-begin").addEventListener("click", () => { app.homeView(); app.setS(0); showIntro(false); app.reachedNow = false; app.play(); });
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
        case "d": case "D": startDive(); break;
        case "h": case "H": app.homeView(); break;
        case "Backspace": case "ArrowDown": e.preventDefault(); app.autopilot = null; app.popOut(); break;
        case "ArrowUp": app.wheel(-120); break;
        case "w": case "W": if (app.modeOf() === "free") app.freeMove(0, 0, -1); else app.wheel(-90); break;
        case "s": case "S": if (app.modeOf() === "free") app.freeMove(0, 0, 1); else app.wheel(90); break;
        case "a": case "A": if (app.modeOf() === "free") app.freeMove(-1, 0, 0); else app.orbit(-14, 0); break;
        case "d": case "D": if (app.modeOf() === "free") app.freeMove(1, 0, 0); else startDive(); break;
        case "ArrowLeft": app.orbit(-14, 0); break; case "ArrowRight": app.orbit(14, 0); break;
        case "l": case "L": $("opt-labels").click(); break; case "o": case "O": $("opt-orbits").click(); break;
        case "f": case "F": if (!document.fullscreenElement) document.documentElement.requestFullscreen && document.documentElement.requestFullscreen(); else document.exitFullscreen(); break;
        case "p": case "P": app.surface(app.node); break;
        case "c": case "C": { const m = ["orbit","follow","surface","free"], i = m.indexOf(app.modeOf()); const next = m[(i + 1) % m.length]; if (next === "follow" || next === "surface") app[next === "follow" ? "follow" : "surface"](app.node); else app[next === "free" ? "freeMode" : "orbitMode"](); } break;
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
      // free-flight WASD movement applied every frame
      if (app.modeOf() === "free") {
        const vx = (keys["d"] || keys["D"] ? 1 : 0) - (keys["a"] || keys["A"] ? 1 : 0);
        const vz = (keys["s"] || keys["S"] ? 1 : 0) - (keys["w"] || keys["W"] ? 1 : 0);
        app.freeMove(vx, 0, vz);
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
      $("t-temp").textContent = U.fmtTemp(vis.T); $("t-z").textContent = vis.z > 1e4 ? U.fmtSci(vis.z) : vis.z < 0.005 ? "0" : vis.z.toFixed(vis.z < 10 ? 2 : 0);
      $("t-size").textContent = app.uni ? U.fmtDist(app.uni.boxM * vis.a) : "-";
      if (document.activeElement !== slider) slider.value = Math.round(app.s * 1000);
      $("ep-name").textContent = ep.name; $("ep-text").textContent = ep.text;
      document.body.classList.toggle("cinema", app.playing && app.node.kind === "universe");
      $("epoch-card").style.opacity = app.node.kind === "universe" || app.node.kind === "halo" ? 1 : 0;
      // breadcrumbs / info on focus change
      if (lastFocus !== app.node) { lastFocus = app.node; drawCrumbs(); drawInfo(); }
      if (app.node.kind === "star" || app.node.kind === "planet") { /* keep info fresh enough */ }
      // ruler: log10 of distance to focus
      const d = V.len(app.pos), lg = Math.log10(Math.max(d, 1)), lo = 3, hi = 26.5, f = U.clamp((lg - lo) / (hi - lo), 0, 1);
      $("ruler-mark").style.bottom = (f * 100) + "%"; $("ruler-val").style.display = "block"; $("ruler-val").textContent = U.fmtDist(d);
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
      $("info-kind").textContent = KIND[n.kind] || n.kind; $("info-name").textContent = n.kind === "universe" ? "The universe" : n.name;
      const dl = $("info-rows"); dl.innerHTML = "";
      app.describe(n).forEach(([k, v]) => { const a = document.createElement("dt"), b = document.createElement("dd"); a.textContent = k; b.textContent = v; dl.appendChild(a); dl.appendChild(b); });
      $("info-text").textContent = n.kind === "universe" ? "Scroll toward any glowing knot to fly into a cluster of galaxies." : n.kind === "galaxy" ? "Every point of light is a star. Scroll toward one to visit it." : n.kind === "star" ? "Click a planet or its label to fly there." : "";
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
    return ui;
  };
})(typeof window !== "undefined" ? window : globalThis);
