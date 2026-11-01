/* layers2.js - galaxy sprites, star cloud / local stars / nebulae, and the planetary system layer. */
(function (g) {
  "use strict";
  const U = g.U, V = U.V, Q = U.Q, setCam = U.setCam;
  const KPC = U.KPC, MPC = U.MPC, AU = U.AU;

  function attr(gl, prog, name, size, stride, offset, divisor) {
    const a = prog.attrib(name); if (a < 0) return;
    gl.enableVertexAttribArray(a); gl.vertexAttribPointer(a, size, gl.FLOAT, false, stride, offset); if (divisor) gl.vertexAttribDivisor(a, divisor);
  }

  // ===========================================================================================
  // Galaxy sprites: used for every halo in the universe frame (mode 0) and for the members of the focused halo (mode 1)
  class GalaxySprites {
    constructor(R) {
      this.R = R; const gl = R.gl;
      this.prog = new U.GL.Program(gl, U.SH.galVS, U.SH.galFS, "galaxy-sprites");
      this.vao = gl.createVertexArray(); this.buf = null; this.count = 0;
      this.vao2 = gl.createVertexArray(); this.buf2 = null; this.count2 = 0;
    }
    static pack(arr, o, q, psi, n, rad, kind, tF, tau, seed, arms, pitch, bulge, lum) {
      arr.set([q[0], q[1], q[2], psi[0], psi[1], psi[2], n[0], n[1], n[2], rad, kind, tF, tau, seed, arms, pitch, bulge, lum], o);
    }
    _fill(gl, vao, data) {
      gl.bindVertexArray(vao);
      const buf = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, buf); gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
      const S = 72, p = this.prog;
      attr(gl, p, "iQ", 3, S, 0, 1); attr(gl, p, "iPsi", 3, S, 12, 1); attr(gl, p, "iN", 3, S, 24, 1);
      attr(gl, p, "iP", 4, S, 36, 1); attr(gl, p, "iX", 4, S, 52, 1);
      const a = p.attrib("iLum"); gl.enableVertexAttribArray(a); gl.vertexAttribPointer(a, 1, gl.FLOAT, false, S, 68); gl.vertexAttribDivisor(a, 1);
      return buf;
    }
    uploadUniverse(uni) {
      const gl = this.R.gl, K = uni.K, H = uni.web.halos, data = new Float32Array(K * 18);
      for (let i = 0; i < K; i++) {
        const r = U.rng(U.hash2(uni.seed, i * 31 + 3)), th = Math.acos(2 * r() - 1), ph = r() * U.TAU;
        const n = [Math.sin(th) * Math.cos(ph), Math.cos(th), Math.sin(th) * Math.sin(ph)];
        const m = uni.members[i], lm = uni.logM[i], cluster = m > 1;
        const kind = cluster ? 4 : uni.kind[i];
        const radMpc = cluster ? Math.max(uni.rvir[i] * 0.55 / MPC, uni.galR[i] * 3e-3) : uni.galR[i] * 1e-3;
        const tau = kind === 2 || kind === 4 ? 0.8 : kind === 3 ? 5 : 6;
        const lum = 0.4 + 0.6 * U.clamp((lm - 11) / 3.5, 0, 1);
        GalaxySprites.pack(data, i * 18, [H.q[i * 3], H.q[i * 3 + 1], H.q[i * 3 + 2]], [H.psi[i * 3], H.psi[i * 3 + 1], H.psi[i * 3 + 2]], n, radMpc, kind,
          uni.tForm[i] / U.GYR, tau, r(), 2 + Math.floor(r() * 3), 0.2 + 0.25 * r(), 0.15 + 0.4 * r(), lum * (cluster ? 1.15 : 1));
      }
      this.buf = this._fill(gl, this.vao, data); this.count = K;
    }
    uploadGroup(halo, gals) {
      const gl = this.R.gl, n = gals.length + (gals.length > 8 ? 1 : 0), data = new Float32Array(n * 18);
      let o = 0;
      const uni = halo.data.uni;
      if (gals.length > 8) {             // intracluster glow
        GalaxySprites.pack(data, o, [0, 0, 0], [0, 0, 0], [0, 1, 0], halo.data.rvir / KPC * 0.6, 4, uni.tForm[halo.id] / U.GYR, 1, 0.37, 2, 0.3, 0.3, 0.55); o += 18;
      }
      for (const gl_ of gals) {
        const d = gl_.data, nrm = Q.rotate(gl_.q, [0, 1, 0]), r = U.rng(d.seed);
        GalaxySprites.pack(data, o, [gl_.local[0] / KPC, gl_.local[1] / KPC, gl_.local[2] / KPC], [0, 0, 0], nrm, d.R, d.kind, d.tForm / U.GYR, d.tau, (d.seed % 1000) / 1000, d.arms, d.pitch, d.bulge,
          0.45 + 0.55 * U.clamp((d.logM - 9.5) / 2.5, 0, 1));
        o += 18;
      }
      if (this.buf2) gl.deleteBuffer(this.buf2);
      this.buf2 = this._fill(gl, this.vao2, data); this.count2 = n; this.groupGalOffset = gals.length > 8 ? 1 : 0;
    }
    // mode 0: universe, mode 1: group
    draw(ctx, mode, camUnits, opt) {
      const gl = this.R.gl, p = this.prog.use();
      const count = mode === 0 ? this.count : this.count2;
      if (!count || opt.fade <= 0.002) return;
      setCam(p, ctx, camUnits);
      p.i1("uMode", mode).f1("uD", ctx.vis.D).f1("uBox", mode === 0 ? ctx.boxMpc : 1).f1("uTGyr", ctx.t / U.GYR).f1("uMinPx", opt.minPx || 2.4)
        .f1("uFade", opt.fade).f1("uSkipMul", opt.skipMul === undefined ? 1 : opt.skipMul).f1("uBoost", opt.boost || 1);
      gl.uniform1i(p.loc("uSkip"), opt.skip === undefined ? -1 : opt.skip);
      gl.enable(gl.BLEND); gl.blendFunc(gl.ONE, gl.ONE);
      gl.bindVertexArray(mode === 0 ? this.vao : this.vao2);
      gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, count);
    }
  }

  // ===========================================================================================
  // Stars of one galaxy (cloud), nebulae and dust, plus the local star field around the camera
  class GalaxyStars {
    constructor(R) {
      this.R = R; const gl = R.gl;
      this.prog = new U.GL.Program(gl, U.SH.starVS, U.SH.starFS, "stars");
      this.nebProg = new U.GL.Program(gl, U.SH.nebVS, U.SH.nebFS, "nebulae");
      this.vao = gl.createVertexArray(); this.nebVao = gl.createVertexArray(); this.locVao = gl.createVertexArray();
      this.gal = null; this.n = 0; this.nNeb = 0; this.locN = 0; this.cellCache = new Map(); this.locOrigin = [0, 0, 0]; this.locKey = "";
    }
    build(gal, N) {
      const gl = this.R.gl;
      if (this.gal === gal) return;
      this.free(); this.gal = gal;
      const s = U.sampleGalaxyStars(gal, N); this.stars = s;
      const buf = new Float32Array(N * 7);
      for (let i = 0; i < N; i++) { buf[i * 7] = s.pos[i * 3]; buf[i * 7 + 1] = s.pos[i * 3 + 1]; buf[i * 7 + 2] = s.pos[i * 3 + 2]; buf[i * 7 + 3] = s.attr[i * 4]; buf[i * 7 + 4] = s.attr[i * 4 + 1]; buf[i * 7 + 5] = s.attr[i * 4 + 2]; buf[i * 7 + 6] = s.attr[i * 4 + 3]; }
      gl.bindVertexArray(this.vao); this.buf = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, this.buf); gl.bufferData(gl.ARRAY_BUFFER, buf, gl.STATIC_DRAW);
      attr(gl, this.prog, "aPos", 3, 28, 0); attr(gl, this.prog, "aAttr", 4, 28, 12); this.n = N;
      // nebulae and dust from the young / massive population
      const r = U.rng(gal.data.seed ^ 0xabc), d = gal.data, R = d.R, neb = [];
      if (d.kind !== 2) for (let i = 0; i < N && neb.length < 4500 * 7; i++) {
        const M = s.mass[i], u = r(); let type = -1, size = 0;
        if (M > 9 && u < 0.25) { type = 0; size = R * (0.006 + 0.016 * r()); }
        else if (M > 4 && M <= 9 && u < 0.1) { type = 1; size = R * (0.01 + 0.02 * r()); }
        else if (M > 2.2 && M < 7 && u < 0.08 && d.kind !== 3) { type = 2; size = R * (0.03 + 0.07 * r()); }
        if (type < 0) continue;
        neb.push(s.pos[i * 3] + r.gauss() * R * 0.01, s.pos[i * 3 + 1] * (type === 2 ? 0.4 : 1), s.pos[i * 3 + 2] + r.gauss() * R * 0.01, size, type, type === 2 ? s.attr[i * 4 + 2] * 0.5 : s.attr[i * 4 + 2], r());
      }
      gl.bindVertexArray(this.nebVao); this.nebBuf = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, this.nebBuf); gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(neb), gl.STATIC_DRAW);
      attr(gl, this.nebProg, "aPos", 3, 28, 0); attr(gl, this.nebProg, "aNeb", 4, 28, 12); this.nNeb = neb.length / 7;
    }
    free() {
      const gl = this.R.gl;
      if (this.buf) gl.deleteBuffer(this.buf); if (this.nebBuf) gl.deleteBuffer(this.nebBuf); if (this.locBuf) gl.deleteBuffer(this.locBuf);
      this.buf = this.nebBuf = this.locBuf = null; this.gal = null; this.n = this.nNeb = this.locN = 0; this.cellCache.clear(); this.locKey = "";
    }
    // local field around an aligned-kpc centre; `skipId` is the focus star's id (kept out of the list)
    updateLocal(centerKpc, radiusPc, skipId) {
      const gl = this.R.gl, key = centerKpc.map((x) => Math.round(x * 1000 / 4)).join(",") + "|" + radiusPc;
      if (key === this.locKey) return; this.locKey = key;
      const list = U.localStarsAround(this.gal, centerKpc[0], centerKpc[1], centerKpc[2], radiusPc, this.cellCache).filter((s) => s.id !== skipId);
      this.localList = list;
      const origin = centerKpc.map((x) => Math.round(x * 100) / 100); this.locOrigin = origin;
      const n = Math.min(list.length, 120000), buf = new Float32Array(n * 7);
      for (let i = 0; i < n; i++) { const s = list[i]; buf[i * 7] = s.pos[0] - origin[0]; buf[i * 7 + 1] = s.pos[1] - origin[1]; buf[i * 7 + 2] = s.pos[2] - origin[2]; buf[i * 7 + 3] = s.logL; buf[i * 7 + 4] = s.T; buf[i * 7 + 5] = s.tb; buf[i * 7 + 6] = s.life; }
      gl.bindVertexArray(this.locVao); if (this.locBuf) gl.deleteBuffer(this.locBuf);
      this.locBuf = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, this.locBuf); gl.bufferData(gl.ARRAY_BUFFER, buf, gl.DYNAMIC_DRAW);
      attr(gl, this.prog, "aPos", 3, 28, 0); attr(gl, this.prog, "aAttr", 4, 28, 12); this.locN = n;
    }
    // ctx.galV (mat3), ctx.galCam (aligned kpc camera); opt: fade, spin, skipIdx
    drawCloud(ctx, opt) {
      if (!this.n || opt.fade <= 0.002) return;
      const gl = this.R.gl, p = this.prog.use(), d = this.gal.data;
      const c = { V: ctx.galV, proj: ctx.proj, pxScale: ctx.pxScale }; setCam(p, c, ctx.galCam);
      const Lw = Math.pow(10, d.logM - 11.5);
      p.f1("uTGyr", ctx.t / U.GYR).f1("uSpin", opt.spin).f1("uR", d.R).f1("uW", 3.2e6 * Lw * (150000 / this.n)).f1("uK", 1.6).f1("uLexp", 0.55).f1("uPsz", 0.9).f1("uExpo", 0.5 * (opt.expo || 1))
        .f1("uFade", opt.fade).f1("uNearPc", 14000).f1("uUnitPc", 1000).f1("uSkipIdx", opt.skipIdx === undefined ? -1 : opt.skipIdx).f3("uOrigin", 0, 0, 0);
      gl.enable(gl.BLEND); gl.blendFunc(gl.ONE, gl.ONE);
      gl.bindVertexArray(this.vao); gl.drawArrays(gl.POINTS, 0, this.n);
    }
    drawLocal(ctx, opt) {
      if (!this.locN) return;
      const gl = this.R.gl, p = this.prog.use();
      const c = { V: ctx.galV, proj: ctx.proj, pxScale: ctx.pxScale };
      const o = this.locOrigin; setCam(p, c, [ctx.galCam[0] - o[0], ctx.galCam[1] - o[1], ctx.galCam[2] - o[2]]);
      p.f1("uTGyr", ctx.t / U.GYR).f1("uSpin", 0).f1("uR", 1).f1("uW", 1).f1("uK", 500).f1("uLexp", 1.0).f1("uPsz", 1.3).f1("uExpo", 0.55 * (opt.expo || 1)).f1("uFade", opt.fade).f1("uNearPc", 0)
        .f1("uUnitPc", 1000).f1("uSkipIdx", -1).f3("uOrigin", 0, 0, 0);
      gl.enable(gl.BLEND); gl.blendFunc(gl.ONE, gl.ONE);
      gl.bindVertexArray(this.locVao); gl.drawArrays(gl.POINTS, 0, this.locN);
    }
    drawNebulae(ctx, opt) {
      if (!this.nNeb || opt.fade <= 0.002) return;
      const gl = this.R.gl, p = this.nebProg.use(), d = this.gal.data;
      const c = { V: ctx.galV, proj: ctx.proj, pxScale: ctx.pxScale }; setCam(p, c, ctx.galCam);
      p.f1("uTGyr", ctx.t / U.GYR).f1("uSpin", opt.spin).f1("uR", d.R).f1("uFade", opt.fade);
      gl.bindVertexArray(this.nebVao);
      gl.enable(gl.BLEND); gl.blendFunc(gl.ONE, gl.ONE); p.f1("uPass", 0); gl.drawArrays(gl.POINTS, 0, this.nNeb);       // emission
      gl.blendFunc(gl.ZERO, gl.ONE_MINUS_SRC_ALPHA); p.f1("uPass", 1); gl.drawArrays(gl.POINTS, 0, this.nNeb);            // dust absorption
    }
  }

  // ===========================================================================================
  // Planetary system: star, planets, moons, orbits, belt, rings (inside the body shader)
  const BIOME = { temperate: 0, ocean: 1, desert: 2, mars: 3, cold: 4, ice: 5, lava: 6, barren: 7 };
  function paletteFor(d) {
    const r = U.rng(d.seed), v = (a, s) => a.map((x) => U.clamp(x * (1 + (r() - 0.5) * s), 0, 1.5));
    const P = { type: 1, biome: BIOME[d.biome] || 0, A: [.3, .3, .3], B: [.5, .5, .5], C: [.7, .7, .7], D: [.1, .2, .4], p0: [0.45, 0.85, 0.5, 1], p1: [0, 0.012, 0, 0], atm: [0.3, 0.55, 1.0] };
    switch (d.biome) {
      case "temperate": Object.assign(P, { A: v([.05, .21, .04], .5), B: v([.30, .25, .10], .4), C: v([.28, .25, .22], .2), D: v([.01, .09, .26], .4), p0: [0.44 + 0.08 * r(), 0.86, 0.42, 1.0], p1: [0.7, 0.016, 0, 0] }); break;
      case "ocean": Object.assign(P, { A: v([.06, .24, .06], .4), B: v([.32, .28, .12], .3), C: v([.4, .38, .35], .2), D: v([.008, .08, .26], .4), p0: [0.62 + 0.06 * r(), 0.88, 0.5, 0.8], p1: [0.8, 0.017, 0, 0] }); break;
      case "desert": Object.assign(P, { A: v([.62, .43, .22], .3), B: v([.78, .58, .32], .3), C: v([.45, .30, .20], .3), p0: [0, 0.97, 0.1, 0.8], p1: [0.7, 0.014, 0, 0], atm: [0.85, 0.65, 0.4] }); break;
      case "mars": Object.assign(P, { A: v([.50, .25, .12], .3), B: v([.62, .34, .18], .3), C: v([.30, .17, .12], .3), p0: [0, 0.93, 0.04, 1.0], p1: [0.3, 0.012, 0, 0], atm: [0.8, 0.55, 0.4] }); break;
      case "cold": Object.assign(P, { A: v([.22, .28, .20], .3), B: v([.50, .50, .45], .2), C: v([.62, .62, .64], .1), D: v([.10, .20, .30], .3), p0: [0.38, 0.55, 0.45, 0.9], p1: [0.9, 0.015, 0, 0], atm: [0.45, 0.6, 1.0] }); break;
      case "ice": Object.assign(P, { A: v([.78, .86, .95], .1), B: v([.55, .70, .86], .2), C: [1, 1, 1], p0: [0, 0.4, 0.1, 0.5], p1: [0.18, 0.01, 0, 0], atm: [0.6, 0.75, 1.0] }); break;
      case "lava": Object.assign(P, { A: [.12, .07, .05], B: [.26, .13, .08], C: [.35, .2, .15], p0: [0, 0.99, 0, 1.0], p1: [0.7, 0.02, 1.0, 0], atm: [1.0, 0.45, 0.15] }); break;
      case "barren": Object.assign(P, { A: v([.28, .27, .26], .2), B: v([.5, .48, .44], .2), C: v([.72, .70, .66], .1), p0: [0, 1, 0, 1.0], p1: [0, 0.0, 0, 0] }); break;
      case "gas": { const t = Math.floor(r() * 3); const pal = [[[.75, .6, .42], [.92, .85, .70], [.55, .38, .25], [.72, .32, .15]], [[.86, .78, .55], [.95, .9, .75], [.7, .62, .45], [.8, .5, .3]], [[.45, .38, .32], [.62, .5, .4], [.3, .25, .25], [.8, .4, .25]]][t];
        Object.assign(P, { type: 2, biome: 0, A: pal[0], B: pal[1], C: pal[2], D: pal[3], p0: [0, 0, 0, 1], p1: [0.9, 0.03, 0, 0], atm: [.7, .62, .45] }); break; }
      case "icegiant": { const t = r() < 0.5; Object.assign(P, { type: 2, biome: 0, A: t ? [.30, .58, .86] : [.22, .55, .6], B: t ? [.5, .75, .92] : [.4, .75, .72], C: t ? [.3, .48, .7] : [.2, .45, .5], D: [.85, .93, 1.0], p0: [0, 0, 0, 1], p1: [1.0, 0.03, 0, 0], atm: t ? [.35, .6, 1.0] : [.3, .75, .7] }); break; }
    }
    if (d.moon && d.biome === "ice") { P.A = [.75, .8, .86]; P.B = [.55, .62, .72]; }
    return P;
  }
  U.bodyPalette = paletteFor;

  class SystemLayer {
    constructor(R) {
      this.R = R; const gl = R.gl;
      this.body = new U.GL.Program(gl, U.SH.bodyVS, U.SH.bodyFS, "bodies");
      this.mark = new U.GL.Program(gl, U.SH.markVS, U.SH.markFS, "markers");
      this.glow = new U.GL.Program(gl, U.SH.glowVS, U.SH.glowFS, "star-glow");
      this.orb = new U.GL.Program(gl, U.SH.orbitVS, U.SH.orbitFS, "orbits");
      this.vao = gl.createVertexArray(); this.orbVao = gl.createVertexArray();
      this.orbBuf = gl.createBuffer(); gl.bindVertexArray(this.orbVao); gl.bindBuffer(gl.ARRAY_BUFFER, this.orbBuf);
      attr(gl, this.orb, "aPos", 3, 12, 0);
      this.bodies = [];
    }
    // returns the list of bodies with star-frame positions (metres) for the orbital clock
    layout(star, clk) {
      const list = [{ node: star, pos: [0, 0, 0], R: star.data.R, kind: "star", parent: null }];
      const tmp = [0, 0, 0];
      for (const p of star.children || []) {
        const pp = U.orbitPos(p.data, clk, tmp.slice()); p.local = pp;
        list.push({ node: p, pos: pp, R: p.data.R, kind: "planet", parent: null });
        for (const m of p.children || []) {
          const mp = U.orbitPos(m.data, clk, [0, 0, 0]); m.local = mp;
          list.push({ node: m, pos: V.add(pp, mp), R: m.data.R, kind: "moon", parent: p, rel: mp });
        }
      }
      return list;
    }
    drawOrbit(ctx, a, ecc, inc, node, parentPos, camStar, alpha, col) {
      const gl = this.R.gl, n = 200, v = new Float32Array((n + 1) * 3), d = { a, ecc, inc, node };
      for (let i = 0; i <= n; i++) {
        const E = i / n * U.TAU, x = a * (Math.cos(E) - ecc), z = a * Math.sqrt(1 - ecc * ecc) * Math.sin(E);
        const ci = Math.cos(inc), si = Math.sin(inc), cn = Math.cos(node), sn = Math.sin(node), x1 = x * cn - z * sn, z1 = x * sn + z * cn;
        v[i * 3] = x1; v[i * 3 + 1] = z1 * si; v[i * 3 + 2] = z1 * ci;
      }
      gl.bindBuffer(gl.ARRAY_BUFFER, this.orbBuf); gl.bufferData(gl.ARRAY_BUFFER, v, gl.DYNAMIC_DRAW);
      const rel = V.sub(parentPos, camStar), p = this.orb.use();
      p.m3("uV", ctx.V).f2("uProj", ctx.proj[0], ctx.proj[1]).f3("uRel", rel[0], rel[1], rel[2]).f4("uColor", col[0] * alpha, col[1] * alpha, col[2] * alpha, 1);
      gl.bindVertexArray(this.orbVao); gl.bindBuffer(gl.ARRAY_BUFFER, this.orbBuf); attr(gl, this.orb, "aPos", 3, 12, 0);
      gl.drawArrays(gl.LINE_STRIP, 0, n + 1);
    }
    // sys: {star, camStar (metres, star frame), clk, spinClk}
    draw(ctx, sys, opt) {
      const gl = this.R.gl, star = sys.star, camS = sys.camStar, fade = opt.fade;
      if (fade <= 0.002) return;
      const bodies = this.layout(star, sys.clk); this.bodies = bodies;
      const W = ctx.w, H = ctx.h, sd = star.data;
      const hot = Math.pow(sd.T / 5772, 1.4);
      const starI = U.clamp(7 * hot, 3, 60);
      const sc = (() => { const c = U.blackbodyRGB(sd.T); const m = Math.max(c[0], c[1], c[2]); return [c[0] / m, c[1] / m, c[2] / m]; })();
      gl.enable(gl.BLEND); gl.blendFunc(gl.ONE, gl.ONE);
      // orbits
      const maxAU = (star.size || 5 * AU);
      for (const b of bodies) {
        if (b.kind === "star") continue;
        const dd = b.node.data, base = b.kind === "planet" ? [0, 0, 0] : bodies.find((x) => x.node === b.parent).pos;
        const dist = V.len(V.sub(b.pos, camS)), dpar = V.len(V.sub(base, camS));
        const near = U.smooth(b.R * 8, b.R * 60, dist);
        const a = (b.kind === "planet" ? 0.16 : 0.10) * near * fade * (opt.orbits === false ? 0 : 1);
        if (a > 0.004) this.drawOrbit(ctx, dd.a, dd.ecc, dd.inc, dd.node, base, camS, a, b.kind === "planet" ? [0.5, 0.65, 0.9] : [0.55, 0.55, 0.65]);
      }
      // sort far -> near
      const items = bodies.map((b) => { const rel = V.sub(b.pos, camS); return { b, rel, d: V.len(rel) }; }).sort((x, y) => y.d - x.d);
      for (const it of items) {
        const { b, rel, d } = it, R = b.R;
        const angPx = R / Math.max(d, 1) * ctx.pxScale;                    // radius in pixels
        if (b.kind === "star") {
          // glow billboard, then the disc
          const half = Math.max(R * 26, d * 110 / ctx.pxScale);
          const core = R / half;
          gl.blendFunc(gl.ONE, gl.ONE);
          const gp = this.glow.use(); gp.m3("uV", ctx.V).f2("uProj", ctx.proj[0], ctx.proj[1]).f3("uRel", rel[0], rel[1], rel[2]).f1("uHalf", half)
            .f3("uCol", sc[0], sc[1], sc[2]).f1("uI", 0.9 * fade).f1("uCore", Math.max(core, 0.004)).f1("uTime", ctx.time);
          gl.bindVertexArray(this.vao); gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
          if (angPx > 0.8) this.drawBody(ctx, b, rel, d, sys, { type: 0, starI: starI * fade, sc, sdata: sd }, true);
          continue;
        }
        if (angPx < 1.6) {                                                  // point marker
          const dstar = V.len(b.pos), lit = 0.3 + 0.7 * Math.max(0, V.dot(V.norm(V.sub([0, 0, 0], b.pos)), V.norm(V.sub([0, 0, 0], rel)))) ;
          const pal = paletteFor(b.node.data), c = pal.B, k = 1.2 * fade;
          const mp = this.mark.use(); mp.m3("uV", ctx.V).f2("uProj", ctx.proj[0], ctx.proj[1]).f3("uRel", rel[0], rel[1], rel[2]).f1("uSizePx", b.kind === "planet" ? 7 : 4)
            .f3("uCol", c[0] * k * 1.4, c[1] * k * 1.4, c[2] * k * 1.4);
          gl.blendFunc(gl.ONE, gl.ONE); gl.bindVertexArray(this.vao); gl.drawArrays(gl.POINTS, 0, 1);
          continue;
        }
        this.drawBody(ctx, b, rel, d, sys, null, false);
      }
    }
    drawBody(ctx, b, rel, d, sys, starOpt, isStar) {
      const gl = this.R.gl, node = b.node, dd = node.data, W = ctx.w, H = ctx.h;
      const p = this.body.use();
      const sunRel = V.sub([0, 0, 0], b.pos), sunDist = V.len(sunRel), sunDir = V.norm(sunRel);
      const R = b.R, hasRing = !isStar && dd.hasRings;
      const ext = isStar ? R * 1.05 : R * (hasRing ? 2.5 : 1 + 0.03 + 0.0);
      // scissor rectangle from the projected bounding sphere
      const vv = [V.dot(ctx.Vrows[0], rel), V.dot(ctx.Vrows[1], rel), V.dot(ctx.Vrows[2], rel)];
      let x0 = 0, y0 = 0, x1 = W, y1 = H;
      if (-vv[2] > ext * 1.05) {
        const w = -vv[2], rp = ext / Math.sqrt(Math.max(1 - (ext / d) * (ext / d), 0.0001)), cx = (vv[0] * ctx.proj[0] / w * 0.5 + 0.5) * W, cy = (vv[1] * ctx.proj[1] / w * 0.5 + 0.5) * H;
        const rx = rp / w * ctx.proj[0] * 0.5 * W, ry = rp / w * ctx.proj[1] * 0.5 * H;
        x0 = Math.max(0, Math.floor(cx - rx - 2)); x1 = Math.min(W, Math.ceil(cx + rx + 2)); y0 = Math.max(0, Math.floor(cy - ry - 2)); y1 = Math.min(H, Math.ceil(cy + ry + 2));
        if (x1 <= x0 || y1 <= y0) return;
      }
      gl.enable(gl.SCISSOR_TEST); gl.scissor(x0, y0, x1 - x0, y1 - y0);
      p.m3("uVinv", ctx.Vinv).f2("uProj", ctx.proj[0], ctx.proj[1]).f1("uPxScale", ctx.pxScale).f3("uC", rel[0], rel[1], rel[2]).f1("uR", R).f1("uTime", ctx.time);
      // body orientation: tilt about world Z, spin about body Y
      const spin = isStar ? sys.clk / 2.0e6 : U.TAU * sys.spinClk / (dd.day || 86400);
      const tilt = isStar ? 0 : dd.tilt, ct = Math.cos(tilt), st = Math.sin(tilt), cs = Math.cos(spin), ss = Math.sin(spin);
      // Rtilt(z) * Ry(spin), column-major
      const Ry = [cs, 0, -ss, 0, 1, 0, ss, 0, cs], Rz = [ct, st, 0, -st, ct, 0, 0, 0, 1];
      p.m3("uBody", U.mat3Mul(Rz, Ry)).v3("uSun", sunDir);
      if (isStar) {
        const sdt = starOpt.sdata;
        p.i1("uType", 0).f1("uSunI", starOpt.starI).f3("uStarCol", starOpt.sc[0], starOpt.sc[1], starOpt.sc[2]).f1("uSeed", (sdt.seed % 1000) / 100).f1("uStarT", sdt.T);
        p.f4("uP0", 0, 0, 0, 1).f4("uP1", 0, 0, 0, 0).f1("uBiome", 0);
        ["uPalA", "uPalB", "uPalC", "uPalD", "uAtmCol"].forEach((n) => p.f3(n, 0, 0, 0));
      } else {
        const pal = paletteFor(dd), irr = 0.62 * U.clamp(Math.pow(star_lum(sys) / Math.pow(sunDist / AU, 2), 0.28), 0.45, 1.5);
        p.i1("uType", pal.type).f1("uSunI", irr).f1("uSeed", (dd.seed % 1000) / 37).f1("uBiome", pal.biome)
          .v3("uPalA", pal.A).v3("uPalB", pal.B).v3("uPalC", pal.C).v3("uPalD", pal.D).v3("uAtmCol", pal.atm);
        p.f4("uP0", pal.p0[0], pal.p0[1], pal.p0[2], pal.p0[3] * (dd.moon ? 1.2 : 1));
        p.f4("uP1", pal.p1[0] * (dd.atm > 0 ? 1 : 0) * (dd.moon ? 0 : 1), pal.p1[1], pal.p1[2], hasRing ? 1 : 0);
        p.f3("uStarCol", 1, 1, 1).f1("uStarT", 5772);
      }
      gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA); gl.enable(gl.BLEND);
      gl.bindVertexArray(this.vao); gl.drawArrays(gl.TRIANGLES, 0, 3);
      gl.disable(gl.SCISSOR_TEST);
    }
  }
  function star_lum(sys) { return sys.star.data.L; }

  U.Layers.GalaxySprites = GalaxySprites; U.Layers.GalaxyStars = GalaxyStars; U.Layers.SystemLayer = SystemLayer;
})(typeof window !== "undefined" ? window : globalThis);
