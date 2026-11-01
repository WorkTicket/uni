/* app.js - application core: scene state, hierarchical camera, picking, render orchestration, autopilot.
   The whole picture is a pure function of (seed, cosmic time, camera): scrubbing time backwards or forwards is exact. */
(function (g) {
  "use strict";
  const U = g.U, V = U.V, Q = U.Q, KPC = U.KPC, MPC = U.MPC, AU = U.AU, YEAR = U.YEAR;
  const FOV = 58 * Math.PI / 180;
  const QUALITY = {
    low:    { N: 64,  keep: 1,    stars: 120000,  scale: 0.75, maxHalos: 9000 },
    medium: { N: 128, keep: 0.25, stars: 240000, scale: 1.0,  maxHalos: 22000 },
    high:   { N: 128, keep: 0.7,  stars: 420000, scale: 1.0,  maxHalos: 30000 },
  };
  const TOTAL_SECONDS = 96;       // default length of "Big Bang -> today" at 1x

  class App {
    constructor(canvas, opts) {
      this.canvas = canvas; this.opts = opts || {};
      const { gl, hdr } = U.GL.create(canvas); this.gl = gl; this.hdr = hdr;
      this.R = { gl }; this.post = new U.GL.Post(gl); this.post.fmt = hdr ? "hdr" : "ldr";
      this.web = new U.Layers.WebLayer(this.R); this.sprites = new U.Layers.GalaxySprites(this.R);
      this.stars = new U.Layers.GalaxyStars(this.R); this.system = new U.Layers.SystemLayer(this.R);
      this.quality = this.opts.quality || "medium";       this.seed = 1; this.uni = null;
      this.s = 0; this.t = U.Cosmo.tOfS(0); this.playing = false; this.speed = 1; this.stopAtNow = false; this.reachedNow = false;
      this.time = 0; this.spin = 0; this.sysClk = 0; this.spinClk = 0; this.userRate = 1;
      this.node = null; this.pos = [0, 0, 0]; this.q = Q.ident(); this.zoomAcc = 0; this.recenter = 0;
      this.mouse = { x: 0.5, y: 0.5, inside: false }; this.hover = null; this.lastPick = 0;
      this.exposure = 1; this.bloom = 0.38; this.autopilot = null; this.listeners = {};
      this.camMode = "orbit";        // orbit | follow | surface | free
      this.followTarget = null;      // {node, t} for follow/surface tracking
      this.followDist = 0;           // metres from target centre
      this.resize();
    }
    on(ev, fn) { (this.listeners[ev] = this.listeners[ev] || []).push(fn); }
    emit(ev, a) { (this.listeners[ev] || []).forEach((f) => f(a)); }

    // ---- setup ------------------------------------------------------------------------------------------
    async generate(seed, onProgress) {
      this.seed = seed >>> 0; const Qn = QUALITY[this.quality];
      const h = U.Cosmo.P.H0 / 100;
      const web = await U.generateWeb({ N: Qn.N, L: 640, seed: this.seed, Om: U.Cosmo.P.Om, h, ns: U.Cosmo.P.ns, truncMpc: 5, psiRms: Qn.N === 64 ? 15 : 15, keep: Qn.keep, subgrid: 0.8, maxHalos: Qn.maxHalos }, onProgress);
      this.uni = new U.Universe(this.seed, web);
      this.web.upload(web); this.sprites.uploadUniverse(this.uni); this.stars.free(); this.groupHalo = null;
      this.haloPos = new Float64Array(this.uni.K * 3); this.haloD = -1;
      this.node = this.uni.root; this.homeView();
      this.emit("generated", this.uni);
    }
    homeView() {
      this.setNode(this.uni.root);
      const d = this.uni.boxM * 2.5, yaw = 0.62, pit = 0.34;
      this.pos = [d * Math.cos(pit) * Math.sin(yaw), d * Math.sin(pit), d * Math.cos(pit) * Math.cos(yaw)];
      this.q = this.lookQ(this.pos, [0, 0, 0]); this.zoomAcc = 0; this.autopilot = null;
    }
    setNode(n) { this.node = n; this.emit("focus", n); }
    resize() {
      const dpr = Math.min(window.devicePixelRatio || 1, 2) * (QUALITY[this.quality].scale || 1);
      const w = Math.max(2, Math.floor(this.canvas.clientWidth * dpr)), h = Math.max(2, Math.floor(this.canvas.clientHeight * dpr));
      if (this.canvas.width !== w || this.canvas.height !== h || !this.post.scene) { this.canvas.width = w; this.canvas.height = h; this.post.resize(w, h); }
      this.w = this.canvas.width; this.h = this.canvas.height;
    }
    setQuality(qn) { this.quality = qn; this.resize(); }
    lookQ(from, to) { return Q.lookAt(V.sub(to, from), [0, 1, 0]); }

    // ---- time ------------------------------------------------------------------------------------------
    setS(s) { this.s = U.clamp(s, 0, 1); this.t = U.Cosmo.tOfS(this.s); if (this.s < U.Cosmo.NOW_S - 1e-4) this.reachedNow = false; }
    setT(t) { this.setS(U.Cosmo.sOfT(t)); this.t = t; }
    play() { this.playing = true; this.emit("play", true); }
    pause() { this.playing = false; this.emit("play", false); }

    // ---- timeline control ----
    stepForward(dt = 0.05) { this.pause(); this.setS(U.clamp(this.s + dt / TOTAL_SECONDS, 0, 1)); }
    stepBackward(dt = 0.05) { this.pause(); this.setS(U.clamp(this.s - dt / TOTAL_SECONDS, 0, 1)); }
    jumpToPresent() { this.pause(); this.setS(U.Cosmo.NOW_S); this.reachedNow = false; }
    jumpToFuture(fraction = 0.5) { this.pause(); const f = U.clamp(fraction, 0, 1); this.setS(U.Cosmo.NOW_S + f * (1 - U.Cosmo.NOW_S)); }
    setSpeed(speed) { this.speed = speed; }

    // checkpoints (named saved timeline states)
    checkpoint(id) { this._checkpoints = this._checkpoints || {}; this._checkpoints[id] = { s: this.s, t: this.t, node: this.node, pos: this.pos.slice(), q: this.q.slice() }; return this._checkpoints[id]; }
    returnToCheckpoint(id) { const cp = this._checkpoints && this._checkpoints[id]; if (!cp) return false; this.pause(); this.setS(cp.s); this.t = cp.t; this.node = cp.node; this.pos = cp.pos.slice(); this.q = cp.q.slice(); this.recenter = 1.0; return true; }
    listCheckpoints() { return this._checkpoints ? Object.keys(this._checkpoints) : []; }

    // ---- chain & frames ---------------------------------------------------------------------------------
    chain() { return this.node.chain(); }
    updateDynamic(chain) {
      const D = U.Cosmo.D(U.Cosmo.aOfT(this.t));
      for (const n of chain) if (n.kind === "halo") this.uni.updateHalo(n, D);
      const st = chain.find((n) => n.kind === "star");
      if (st && st.children) this.system.layout(st, this.sysClk);
    }
    camFrames(chain) {                                  // camera position (metres) in the frame of every chain node
      const out = new Array(chain.length); let p = this.pos.slice();
      out[chain.length - 1] = p;
      for (let k = chain.length - 2; k >= 0; k--) { p = V.add(p, chain[k + 1].local); out[k] = p; }
      return out;
    }
    limits(n) {
      const d = n.data;
      switch (n.kind) {
        case "universe": return { min: 25 * MPC, pop: Infinity, max: this.uni.boxM * 9 };
        case "halo": return { min: Math.max(d.rvir * 0.02, 60 * KPC), pop: Math.max(d.rvir * 14, 4 * MPC), max: 1e30 };
        case "galaxy": return { min: Math.max(0.0008 * n.size, 0.4 * KPC), pop: n.size * 18, max: 1e30 };
        case "star": return { min: d.R * 2.4, pop: Math.max(n.size * 9, 40 * AU), max: 1e30 };
        case "planet": case "moon": return { min: n.size * 1.00045 + 1200, pop: Math.max(n.size * 70, (n.children && n.children.length ? n.children[n.children.length - 1].data.a * 4 : 0)), max: 1e30 };
      }
      return { min: 1, pop: Infinity, max: 1e30 };
    }

    // ---- focus changes ----------------------------------------------------------------------------------
    focusOn(n, keepCamera) {
      if (n === this.node) return;
      const a = this.node.chain(), b = n.chain(); let ca = null;
      for (let i = 0; i < Math.min(a.length, b.length) && a[i] === b[i]; i++) ca = a[i];
      this.updateDynamic(b);
      const camA = V.add(this.pos, this.node.posIn(ca));
      this.pos = V.sub(camA, n.posIn(ca));
      this.setNode(n); this.recenter = 1.6;
    }
    popOut() {
      const n = this.node; if (!n.parent) return;
      this.pos = V.add(this.pos, n.local); this.setNode(n.parent); this.recenter = 1.2;
    }
    // ---- camera modes ---------------------------------------------------------------
    follow(node, distM) { this.camMode = "follow"; this.followTarget = { node, t: this.t }; this.followDist = distM || this.limits(node).pop; this.recenter = 1.0; }
    surface(node) { this.camMode = "surface"; this.followTarget = { node, t: this.t }; const lim = this.limits(node); this.followDist = lim.min * 1.15; this.recenter = 1.0; }
    orbitMode() { this.camMode = "orbit"; this.followTarget = null; }
    freeMode() { this.camMode = "free"; this.followTarget = null; }
    modeOf() { return this.camMode; }

    // resolve the follow/surface target's current world position (metres, universe frame)
    followPosition() {
      const ft = this.followTarget; if (!ft) return null;
      const chain = ft.node.chain(); const cams = this.camFrames(chain);
      // the camera frame of the focused node is its position in the universe frame
      return cams[0];
    }
    // resolve a hover target into a node (creating it lazily)
    nodeFromHover(h) {
      const uni = this.uni;
      switch (h.kind) {
        case "halo": { const n = uni.haloNode(h.index); if (n.data.members === 1) { return uni.members_of(n)[0]; } return n; }
        case "galaxy": return h.node;
        case "cloud": {
          const gal = this.stars.gal, s = this.stars.stars, i = h.index;
          const spec = { pos: [s.pos[i * 3], s.pos[i * 3 + 1], s.pos[i * 3 + 2]], M: s.mass[i], tb: s.attr[i * 4 + 2], id: U.hash2(gal.data.seed, i + 1) };
          const n = U.makeStarNode(gal, spec); n.cloudIdx = i; U.makeSystem(n); return n;
        }
        case "local": {
          const gal = this.stars.gal, s = h.spec, n = U.makeStarNode(gal, { pos: s.pos, M: s.M, tb: s.tb, id: s.id }); U.makeSystem(n); return n;
        }
        case "body": return h.node;
      }
      return null;
    }
    zoomInto(hv) {
      const n = this.nodeFromHover(hv); if (!n || n === this.node) return;
      // a halo's single member / a member of the current group are children: focus them directly
      if (n.parent === this.node || n.parent === this.node.parent || !n.parent) { this.focusOn(n); return; }
      this.focusOn(n);
    }

    // ---- input ------------------------------------------------------------------------------------------
    orbit(dx, dy) {
      const k = 0.0045; this.autopilot = null;
      let up = [0, 1, 0]; if (this.node.kind === "galaxy") up = Q.rotate(this.node.q, [0, 1, 0]);
      const right = Q.rotate(this.q, [1, 0, 0]);
      const rot = Q.mul(Q.axisAngle(up, -dx * k), Q.axisAngle(right, -dy * k));
      this.pos = Q.rotate(rot, this.pos); this.q = Q.normalize(Q.mul(rot, this.q)); this.recenter = 0;
      this.tilt = this.tilt || 0;
    }
    look(dx, dy) { this.tilt = U.clamp((this.tilt || 0) + dy * 0.004, 0, 1.45); }
    wheel(deltaY, mx, my) {
      this.autopilot = null;
      const amt = -deltaY * 0.0016;
      if (amt > 0 && this.hover) {                       // zooming in on something that is not the current focus
        const h = this.hover; const n = this.nodeFromHover(h);
        if (n && n !== this.node) { this.focusOn(n); }
      }
      this.zoomAcc += amt;
    }

    // ---- picking ----------------------------------------------------------------------------------------
    pick() {
      const chain = this.chain(), node = this.node; if (!this.mouse.inside) { this.hover = null; return; }
      this.updateDynamic(chain);
      const cams = this.camFrames(chain), W = this.w, H = this.h;
      const basis = Q.basis(this.q), f = this.proj, mx = (this.mouse.x * 2 - 1), my = (1 - this.mouse.y * 2);
      const pxs = this.pxScale / (this.canvas.width / this.canvas.clientWidth || 1);       // not used for thresholds below
      const toScreen = (rel) => {
        const x = V.dot(basis[0], rel), y = V.dot(basis[1], rel), z = V.dot(basis[2], rel), w = -z; if (w <= 0) return null;
        return { sx: x * f[0] / w, sy: y * f[1] / w, w };
      };
      const thr = (px) => px / (H * 0.5);                      // pixels -> ndc units (vertical)
      let best = null, bestScore = 1;
      const consider = (cand, rel, sizeM, kind, extra) => {
        const s = toScreen(rel); if (!s) return;
        const rad = sizeM / s.w * f[1];                         // ndc radius
        const lim = Math.max(rad, thr(9));
        const dx = (s.sx - mx) * (W / H), dy = s.sy - my, d = Math.hypot(dx, dy) * 1.0;
        const sc = d / lim; if (sc < bestScore) { bestScore = sc; best = Object.assign({ kind, sx: s.sx, sy: s.sy, size: rad }, extra); }
      };
      const t = this.t, tG = t / U.GYR;
      if (node.kind === "universe") {
        const uni = this.uni, D = U.Cosmo.D(U.Cosmo.aOfT(t)), c = cams[0], out = [0, 0, 0];
        for (let i = 0; i < uni.K; i++) {
          if (uni.tForm[i] > t) continue;
          uni.haloPos(i, D, out);
          const rel = [out[0] - c[0], out[1] - c[1], out[2] - c[2]];
          const size = uni.members[i] > 1 ? Math.max(uni.rvir[i] * 0.5, uni.galR[i] * KPC * 3) : uni.galR[i] * KPC;
          consider(null, rel, size, "halo", { index: i });
        }
      } else if (node.kind === "halo") {
        const gals = this.uni.members_of(node), c = cams[chain.length - 1];
        for (const gal of gals) { if (gal.data.tForm > t) continue; consider(null, V.sub(gal.local, c), gal.size, "galaxy", { node: gal }); }
      } else if (node.kind === "galaxy" && this.stars.gal === node) {
        const s = this.stars.stars, c = cams[chain.length - 1], gq = Q.conj(node.q);
        const ca = Q.rotate(gq, c).map((x) => x / KPC), R = node.data.R, spin = this.spin;
        const rot = this.stars.n && this.starSpin !== 0 ? 1 : 0, b = basis;
        // project in aligned frame: camera basis rotated into aligned axes
        const A0 = Q.rotate(gq, basis[0]), A1 = Q.rotate(gq, basis[1]), A2 = Q.rotate(gq, basis[2]);
        let bd = 1e9, bi = -1;
        for (let i = 0; i < this.stars.n; i++) {
          const age = tG - s.attr[i * 4 + 2]; if (age < 0 || age > s.attr[i * 4 + 3]) continue;
          let x = s.pos[i * 3], y = s.pos[i * 3 + 1], z = s.pos[i * 3 + 2];
          { const r = Math.hypot(x, z), th = spin / (0.25 + r / R), cc = Math.cos(th), ss = Math.sin(th); const nx = cc * x - ss * z; z = ss * x + cc * z; x = nx; }
          const rx = x - ca[0], ry = y - ca[1], rz = z - ca[2];
          const vz = A2[0] * rx + A2[1] * ry + A2[2] * rz; if (vz >= 0) continue; const w = -vz;
          const vx = A0[0] * rx + A0[1] * ry + A0[2] * rz, vy = A1[0] * rx + A1[1] * ry + A1[2] * rz;
          const sx = vx * f[0] / w, sy = vy * f[1] / w, dx = (sx - mx) * (W / H), dy = sy - my, dd = Math.hypot(dx, dy);
          const lum = s.attr[i * 4] , bonus = 1 + 0.25 * Math.max(0, lum);
          const sc = dd / (thr(10) * bonus); if (sc < bestScore) { bestScore = sc; bi = i; best = { kind: "cloud", index: i, sx, sy, size: 0 }; }
        }
      } else if (node.kind === "star" || node.kind === "planet") {
        const c = cams[chain.length - 1], st = chain.find((n) => n.kind === "star"), kids = node.kind === "star" ? node.children : node.children || [];
        if (node.kind === "star") {
          for (const p of kids || []) { consider(null, V.sub(p.local, c), p.size, "body", { node: p }); }
        } else {
          for (const m of kids || []) consider(null, V.sub(m.local, c), m.size, "body", { node: m });
          // siblings
          for (const p of (node.parent.children || [])) if (p !== node) consider(null, V.sub(V.add(p.local, V.scale(node.local, -1)), c), p.size, "body", { node: p });
        }
      }
      // local stars (real neighbourhood) when inside a system
      const stIdx = chain.findIndex((n) => n.kind === "star");
      if (stIdx >= 0 && this.stars.localList) {
        const gIdx = chain.findIndex((n) => n.kind === "galaxy"), cg = cams[gIdx], gq = Q.conj(chain[gIdx].q);
        const ca = Q.rotate(gq, cg).map((x) => x / KPC), A0 = Q.rotate(gq, basis[0]), A1 = Q.rotate(gq, basis[1]), A2 = Q.rotate(gq, basis[2]);
        for (const s of this.stars.localList) {
          if (s.id === chain[stIdx].id) continue;
          if (tG < s.tb || tG > s.tb + s.life) continue;
          const rx = s.pos[0] - ca[0], ry = s.pos[1] - ca[1], rz = s.pos[2] - ca[2];
          const vz = A2[0] * rx + A2[1] * ry + A2[2] * rz; if (vz >= 0) continue; const w = -vz;
          const sx = (A0[0] * rx + A0[1] * ry + A0[2] * rz) * f[0] / w, sy = (A1[0] * rx + A1[1] * ry + A1[2] * rz) * f[1] / w;
          const dd = Math.hypot((sx - mx) * (W / H), sy - my), bonus = 1 + 0.5 * Math.max(0, s.logL), sc = dd / (thr(10) * bonus);
          if (sc < bestScore) { bestScore = sc; best = { kind: "local", spec: s, sx, sy, size: 0 }; }
        }
      }
      this.hover = best;
      if (best) best.label = this.hoverLabel(best);
    }
    hoverLabel(h) {
      const uni = this.uni;
      switch (h.kind) {
        case "halo": { const n = uni.haloNode(h.index); return { name: n.name, sub: n.data.groupName + " · " + U.fmtSci(Math.pow(10, n.data.logM), "M☉") }; }
        case "galaxy": return { name: h.node.name, sub: "radius " + U.fmtDist(h.node.size) };
        case "cloud": return { name: "Star", sub: "click or scroll to travel" };
        case "local": return { name: U.name(h.spec.id), sub: "Class " + U.StarPhys.spectralClass(h.spec.T) + " star · " + h.spec.M.toFixed(2) + " M☉" };
        case "body": return { name: h.node.name, sub: h.node.data.cls };
      }
      return null;
    }

    // ---- frame ------------------------------------------------------------------------------------------
    update(dt) {
      this.time += dt;
      // cosmic clock
      if (this.playing) {
        this.setS(this.s + dt * this.speed / TOTAL_SECONDS);
        if (this.s >= 1) { this.s = 1; this.t = U.Cosmo.tOfS(1); this.pause(); this.emit("future-end"); }
      }
      // local clocks (decoupled from cosmic time so planets and spiral arms move at a watchable pace)
      const kind = this.node.kind, ur = this.userRate;
      this.spin += dt * 0.012 * ur; this.sysClk += dt * (YEAR / 70) * ur; this.spinClk += dt * 2400 * ur;
      // autopilot
      if (this.autopilot) this.stepAutopilot(dt);
      // zoom
      const chain = this.chain(); this.updateDynamic(chain);
      if (Math.abs(this.zoomAcc) > 1e-5) {
        const take = this.zoomAcc * (1 - Math.exp(-dt * 7)); this.zoomAcc -= take;
        const lim = this.limits(this.node), dist = V.len(this.pos);
        let nd = dist * Math.exp(-take);
        if (take < 0 && nd > lim.pop && this.node.parent) { this.popOut(); }
        else {
          nd = U.clamp(nd, lim.min, lim.max); if (nd === lim.min && take > 0) this.zoomAcc = 0;
          if (nd !== dist) this.pos = V.scale(this.pos, nd / dist);
        }
      }
      // keep the camera outside the minimum distance (e.g. planet moved)
      { const lim = this.limits(this.node), d = V.len(this.pos); if (d < lim.min) this.pos = V.scale(this.pos, lim.min / Math.max(d, 1e-9)); }
      // follow / surface tracking: hold a stable offset from the tracked body, keep it centred
      if ((this.camMode === "follow" || this.camMode === "surface") && this.followTarget && this.followTarget.node) {
        const ft = this.followTarget.node;
        if (ft !== this.node) { this.setNode(ft); this.recenter = 1.0; }
        const lim = this.limits(ft);
        const d = V.len(this.pos);
        let mind = this.camMode === "surface" ? Math.max(lim.min * 1.1, ft.size * 1.02) : lim.min;
        if (d < mind) this.pos = V.scale(this.pos, mind / Math.max(d, 1e-9));
        if (d > lim.pop * 4) this.pos = V.scale(this.pos, (lim.pop * 4) / Math.max(d, 1e-9));
        this.recenter = Math.max(this.recenter, 0.3);
      }
      // recentre on the focus
      if (this.recenter > 0) {
        this.recenter -= dt;
        const target = Q.lookAt(V.scale(this.pos, -1), this.node.kind === "galaxy" ? Q.rotate(this.node.q, [0, 1, 0]) : [0, 1, 0]);
        this.q = Q.slerp(this.q, target, 1 - Math.exp(-dt * 4.5));
      } else if (this.node.kind !== "universe" || true) {
        // keep looking at the focus unless the user tilted away (planet horizon mode)
        const target = Q.lookAt(V.scale(this.pos, -1), [0, 1, 0]);
        const d = V.len(this.pos), nk = this.node.kind;
        if ((nk === "planet" || nk === "moon") && d < this.node.size * 6) { /* tilt mode: orientation stays user controlled */ }
      }
      // planets: keep the camera co-rotating with the surface when close
      // hover picking at ~20 Hz
      if (this.time - this.lastPick > 0.05) { this.lastPick = this.time; this.pick(); }
      this.emit("update", dt);
    }

    stepAutopilot(dt) {
      const ap = this.autopilot, st = ap.steps[ap.i]; if (!st) { this.autopilot = null; this.emit("dive-done"); return; }
      if (!st.started) {
        st.started = true; st.t = 0;
        const n = st.node(); if (n !== this.node) this.focusOn(n);
        st.d0 = V.len(this.pos); st.dir0 = V.norm(this.pos);
      }
      st.t += dt; const u = U.clamp(st.t / st.dur, 0, 1), e = u * u * (3 - 2 * u);
      const d1 = typeof st.dist === "function" ? st.dist() : st.dist;
      const ld = Math.log(st.d0) + (Math.log(Math.max(d1, 1)) - Math.log(st.d0)) * e;
      let dir = st.dir0;
      if (st.dirTo) { const want = V.norm(st.dirTo()); const a = Q.slerp(Q.lookAt(V.scale(st.dir0, -1), [0, 1, 0]), Q.lookAt(V.scale(want, -1), [0, 1, 0]), e); dir = V.norm(V.lerp(st.dir0, want, e)); }
      this.pos = V.scale(dir, Math.exp(ld));
      this.recenter = Math.max(this.recenter, 0.2);
      if (u >= 1) { ap.i++; if (ap.i >= ap.steps.length) { this.autopilot = null; this.emit("dive-done"); } }
    }

    // choose a destination procedurally and fly there from wherever the camera is
    planDive(rngSeed) {
      const uni = this.uni, r = U.rng((rngSeed >>> 0) ^ 0x1234567), tG = U.Cosmo.TNOW / U.GYR;
      this.setS(U.Cosmo.NOW_S); this.pause();
      let halo = null, gal = null;
      for (let tries = 0; tries < 400 && !gal; tries++) {
        const i = Math.floor(600 + r() * Math.min(uni.K - 700, 9000)); if (uni.kind[i] > 1 || uni.members[i] > 6 || uni.logM[i] < 11.6) continue;
        const h = uni.haloNode(i); const g = uni.members_of(h)[0]; if (g.data.kind > 1) continue; halo = h; gal = g;
      }
      if (!gal) { halo = uni.haloNode(500); gal = uni.members_of(halo)[0]; }
      // build the star cloud lazily to find a Sun-like star in the disc with planets
      const wasNode = this.node;
      this.stars.build(gal, QUALITY[this.quality].stars);
      const s = this.stars.stars, R = gal.data.R; let star = null;
      for (let k = 0; k < 6000 && !star; k++) {
        const i = Math.floor(r() * this.stars.n), rad = Math.hypot(s.pos[i * 3], s.pos[i * 3 + 2]);
        if (s.mass[i] < 0.7 || s.mass[i] > 1.4 || rad < 0.3 * R || rad > 0.6 * R || Math.abs(s.pos[i * 3 + 1]) > 0.5) continue;
        if (tG - s.attr[i * 4 + 2] < 2 || tG - s.attr[i * 4 + 2] > s.attr[i * 4 + 3]) continue;
        const spec = { pos: [s.pos[i * 3], s.pos[i * 3 + 1], s.pos[i * 3 + 2]], M: s.mass[i], tb: s.attr[i * 4 + 2], id: U.hash2(gal.data.seed, i + 1) };
        const n = U.makeStarNode(gal, spec); n.cloudIdx = i; const pl = U.makeSystem(n);
        if (pl.length >= 3 && pl.some((p) => /ocean|temperate|desert|cold/.test(p.data.biome)) && pl.some((p) => p.data.hasRings || p.data.biome === "gas")) star = n;
      }
      if (!star) { const i = 0; const spec = { pos: [s.pos[0], s.pos[1], s.pos[2]], M: s.mass[0], tb: s.attr[2], id: 77 }; star = U.makeStarNode(gal, spec); star.cloudIdx = 0; U.makeSystem(star); }
      const planets = star.children.filter((p) => !p.data.moon);
      const hab = planets.find((p) => /ocean|temperate/.test(p.data.biome)) || planets.find((p) => /desert|cold/.test(p.data.biome)) || planets[0];
      const giant = planets.find((p) => p.data.biome === "gas" || p.data.biome === "icegiant");
      const steps = [];
      const galDir = () => Q.rotate(gal.q, V.norm([0.55, 0.78, 0.3]));
      if (halo.data.members > 1) steps.push({ node: () => halo, dist: () => halo.data.rvir * 2.0, dur: 7 });
      steps.push({ node: () => gal, dist: () => gal.size * 2.2, dur: 8, dirTo: galDir });
      steps.push({ node: () => star, dist: () => star.size * 2.4, dur: 9, dirTo: () => V.norm([0.3, 0.55, 0.78]) });
      if (giant && giant !== hab) steps.push({ node: () => giant, dist: () => giant.size * 7, dur: 6, dirTo: () => V.norm(V.add(V.scale(giant.local, -1), [0.0, giant.size * 0, 0])) });
      steps.push({ node: () => hab, dist: () => hab.size * 5.5, dur: 8, dirTo: () => { const s = V.norm(V.scale(hab.local, -1)); return V.norm(V.add(V.scale(s, 0.75), [0.15, 0.28, 0.22])); } });
      steps.push({ node: () => hab, dist: () => hab.size * 1.5, dur: 7, dirTo: () => { const s = V.norm(V.scale(hab.local, -1)); return V.norm(V.add(V.scale(s, 0.7), [0.2, 0.3, 0.2])); } });
      this.autopilot = { i: 0, steps, target: { halo, gal, star, hab } };
      return this.autopilot.target;
    }

    // ---- rendering --------------------------------------------------------------------------------------
    render() {
      if (!this.uni) return;
      this.resize();
      const gl = this.gl, W = this.w, H = this.h, chain = this.chain();
      const aspect = W / H, f = 1 / Math.tan(FOV / 2);
      this.proj = [f / aspect, f]; this.pxScale = 0.5 * H * f;
      const b = Q.basis(this.q);
      const Vm = new Float32Array([b[0][0], b[1][0], b[2][0], b[0][1], b[1][1], b[2][1], b[0][2], b[1][2], b[2][2]]);
      const Vinv = new Float32Array([b[0][0], b[0][1], b[0][2], b[1][0], b[1][1], b[1][2], b[2][0], b[2][1], b[2][2]]);
      this.updateDynamic(chain);
      const cams = this.camFrames(chain);
      const t = this.t, vis = U.Cosmo.visual(t);
      const dRel = V.len(cams[0]) / this.uni.boxM, dim = U.mix(1.0, 0.22, U.smooth(0.9, 2.6, dRel));
      let tilt = this.tilt || 0; // reserved
      const ctx = { gl, w: W, h: H, V: Vm, Vinv, Vrows: b, proj: this.proj, pxScale: this.pxScale, vis, t, time: this.time, boxMpc: this.uni.boxMpc,
        webPlasmaGain: 0.012 * U.mix(1, 0.5, U.smooth(0.9, 2.6, dRel)), webGasGain: 0.0065 * dim, webSize: 0.55 };
      const iH = chain.findIndex((n) => n.kind === "halo"), iG = chain.findIndex((n) => n.kind === "galaxy"), iS = chain.findIndex((n) => n.kind === "star");
      this.post.scene.bind(); gl.clearColor(0, 0, 0, 1); gl.clear(gl.COLOR_BUFFER_BIT); gl.disable(gl.DEPTH_TEST);
      // --- universe frame
      const camU = cams[0], camMpc = [camU[0] / MPC, camU[1] / MPC, camU[2] / MPC]; ctx.camMpc = camMpc;
      let webFade = 1, skipMul = 1, skip = -1, grpFade = 0, halo = iH >= 0 ? chain[iH] : null;
      if (halo) {
        const dH = V.len(cams[iH]), Hs = halo.data.members > 1 ? halo.data.rvir : this.uni.galR[halo.id] * KPC;
        webFade = U.smooth(Math.max(halo.data.rvir * 1.5, 0.6 * MPC), halo.data.rvir * 6 + 3 * MPC, dH);
        skipMul = U.smooth(0.8 * Hs, 4 * Hs, dH); skip = halo.id; grpFade = 1 - skipMul;
      }
      this.web.draw(ctx, webFade);
      this.sprites.draw(ctx, 0, camMpc, { fade: 1, skip, skipMul, minPx: 1.7, boost: 0.3 });
      // --- group frame
      if (halo && grpFade > 0.003) {
        if (this.groupHalo !== halo) { this.groupHalo = halo; this.sprites.uploadGroup(halo, this.uni.members_of(halo)); }
        const cg = cams[iH], camKpc = [cg[0] / KPC, cg[1] / KPC, cg[2] / KPC];
        let sk2 = -1, sm2 = 1;
        if (iG >= 0) { const G = chain[iG]; sk2 = G.index + this.sprites.groupGalOffset; sm2 = U.smooth(0.35 * G.size, 1.6 * G.size, V.len(cams[iG])) * 1.0; }
        this.sprites.draw(ctx, 1, camKpc, { fade: grpFade, skip: sk2, skipMul: sm2, minPx: 2.0, boost: 1.0 });
      }
      // --- galaxy frame
      let starFade = 0;
      if (iG >= 0) {
        const G = chain[iG], dG = V.len(cams[iG]);
        starFade = 1 - U.smooth(4 * G.size, 14 * G.size, dG);
        if (starFade > 0.003) {
          this.stars.build(G, QUALITY[this.quality].stars);
          const Rm = U.matFromQuat(G.q), cg = Q.rotate(Q.conj(G.q), cams[iG]).map((x) => x / KPC);
          ctx.galV = new Float32Array(U.mat3Mul(Array.from(Vm), Rm)); ctx.galCam = cg;
          const spin = this.spin;
          const st = iS >= 0 ? chain[iS] : null;
          this.stars.drawCloud(ctx, { fade: starFade, spin, skipIdx: st && st.cloudIdx !== undefined ? st.cloudIdx : -1 });
          if (st) { this.stars.updateLocal(st.alignedKpc, 70, st.id); this.stars.drawLocal(ctx, { fade: 1 }); }
          this.stars.drawNebulae(ctx, { fade: starFade, spin });
        }
      }
      // --- system frame
      if (iS >= 0) {
        const S = chain[iS], tG = t / U.GYR, alive = tG >= S.data.tb && tG < S.data.tb + S.data.life;
        const fade = alive ? 1 : 0;
        this.system.draw(ctx, { star: S, camStar: cams[iS], t: this.t, spinClk: this.spinClk }, { fade, orbits: this.orbits !== false });
      }
      // --- post
      const k = this.node.kind, tgt = (k === "planet" || k === "moon") ? [0.16, 1.0] : k === "star" ? [0.24, 1.0] : k === "galaxy" ? [0.5, 1.0] : [0.4, 1.0];
      const ease = 1 - Math.exp(-0.06 * 60 * 0.016);
      this.bloom += (tgt[0] - this.bloom) * ease; this.exposure += (tgt[1] - this.exposure) * ease;
      gl.disable(gl.BLEND);
      this.post.render({ bloom: this.bloom, exposure: this.exposure * (this.opts.exposure || 1), vignette: 0.32, time: this.time, bloomRadius: 1.0 }, W, H);
    }
    flyTo(n) {
      const lim = (k) => ({ universe: () => this.uni.boxM * 2, halo: () => Math.max(n.data.rvir * 2, 200 * KPC), galaxy: () => n.size * 2.2, star: () => n.size * 2.4, planet: () => n.size * 5.5, moon: () => n.size * 6 }[k]());
      this.autopilot = { i: 0, steps: [{ node: () => n, dist: () => lim(n.kind), dur: n.kind === "universe" ? 3 : 4.5 }] };
    }
    labelTargets() {
      const n = this.node, out = [], p = this.pos;
      if (n.kind === "star") { for (const c of n.children || []) if (!c.data.moon) out.push({ node: c, rel: V.sub(c.local, p), size: Math.max(c.size, 1) , minPx: 0 }); }
      else if (n.kind === "planet") {
        for (const m of n.children || []) out.push({ node: m, rel: V.sub(m.local, p), size: m.size, minPx: 0 });
        out.push({ node: n.parent, rel: V.sub(V.scale(n.local, -1), p), size: n.parent.data.R, minPx: 0 });
        for (const q of n.parent.children || []) if (q !== n) out.push({ node: q, rel: V.sub(V.sub(q.local, n.local), p), size: q.size, minPx: 0 });
      } else if (n.kind === "moon") { out.push({ node: n.parent, rel: V.sub(V.scale(n.local, -1), p), size: n.parent.size, minPx: 0 }); }
      else if (n.kind === "halo") { for (const gl of this.uni.members_of(n)) if (gl.data.tForm <= this.t) out.push({ node: gl, rel: V.sub(gl.local, p), size: gl.size, minPx: 6 }); }
      return out;
    }
    describe(n) {
      const d = n.data, rows = [];
      switch (n.kind) {
        case "universe": rows.push(["Simulated region", U.fmtDist(this.uni.boxM) + " cube (periodic)"], ["Halos tracked", String(this.uni.K)], ["Seed", String(this.seed)]); break;
        case "halo": rows.push(["Type", d.groupName], ["Mass", U.fmtSci(Math.pow(10, d.logM), "M☉")], ["Virial radius", U.fmtDist(d.rvir)], ["Member galaxies", String(d.members)]); break;
        case "galaxy": rows.push(["Type", U.GALAXY_KINDS[d.kind] + " galaxy"], ["Halo mass", U.fmtSci(Math.pow(10, d.logM + 0.6), "M☉")], ["Radius", U.fmtDist(n.size)], ["Formed", U.fmtTime(d.tForm)], ["Spiral arms", d.kind < 2 ? String(d.arms) : "-"]); break;
        case "star": rows.push(["Spectral class", d.cls], ["Mass", d.M.toFixed(2) + " M☉"], ["Luminosity", d.L < 100 ? d.L.toFixed(2) + " L☉" : U.fmtSci(d.L, "L☉")], ["Temperature", d.T.toFixed(0) + " K"], ["Radius", (d.R / U.RSUN).toFixed(2) + " R☉"], ["Age", d.age.toFixed(2) + " Gyr"], ["Planets", String((n.children || []).filter((p) => !p.data.moon).length)]); break;
        case "planet": case "moon": rows.push(["Class", d.cls], ["Radius", (d.R / U.REARTH).toFixed(2) + " R⊕"], ["Mass", U.fmtMass(d.Mass)], ["Orbit", n.kind === "planet" ? (d.a / AU).toFixed(2) + " AU · " + (d.P / YEAR).toFixed(2) + " yr" : U.fmtDist(d.a) + " · " + (d.P / 86400).toFixed(1) + " d"], ["Equilibrium temp", d.Teq.toFixed(0) + " K"], ["Moons", String((n.children || []).length)]); break;
      }
      return rows;
    }
  }
  U.App = App; U.QUALITY = QUALITY;
})(typeof window !== "undefined" ? window : globalThis);
