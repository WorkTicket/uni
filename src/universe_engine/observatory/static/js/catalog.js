/* catalog.js - the procedural hierarchy: universe -> halo (group/cluster) -> galaxy -> star (system) -> planet -> moon.
   Everything is a pure function of (seed, ids). Nothing is stored that cannot be regenerated, and detail is only
   generated for what the camera is near. Positions in node frames are metres (doubles); frames are translation-only. */
(function (g) {
  "use strict";
  const U = g.U, V = U.V, Q = U.Q;
  const KPC = U.KPC, MPC = U.MPC, PC = U.PC, AU = U.AU;

  // ---- rotation helpers (3x3 column-major as Float32Array / Array) ---------------------------------------
  U.matFromQuat = function (q) { const b = Q.basis(q); return [b[0][0], b[0][1], b[0][2], b[1][0], b[1][1], b[1][2], b[2][0], b[2][1], b[2][2]]; };
  U.mat3Mul = function (a, b) {          // a*b, column-major
    const o = new Array(9);
    for (let c = 0; c < 3; c++) for (let r = 0; r < 3; r++) o[c * 3 + r] = a[r] * b[c * 3] + a[3 + r] * b[c * 3 + 1] + a[6 + r] * b[c * 3 + 2];
    return o;
  };
  function randQuat(r) { const u1 = r(), u2 = r() * U.TAU, u3 = r() * U.TAU, a = Math.sqrt(1 - u1), b = Math.sqrt(u1); return [a * Math.sin(u2), a * Math.cos(u2), b * Math.sin(u3), b * Math.cos(u3)]; }

  // ---- stellar relations ----------------------------------------------------------------------------------
  const Star = (U.StarPhys = {
    lum(M) { return M < 0.43 ? 0.23 * Math.pow(M, 2.3) : M < 2 ? Math.pow(M, 4) : M < 55 ? 1.4 * Math.pow(M, 3.5) : 32000 * M; },
    radius(M) { return M < 1 ? Math.pow(M, 0.8) : Math.pow(M, 0.57); },
    teff(L, R) { return 5772 * Math.pow(L / (R * R), 0.25); },
    life(M) { return Math.max(0.004, Math.min(2000, 10 * M / Star.lum(M))); },          // Gyr
    imf(r) {                                                                               // Kroupa-like, [0.1, 60] Msun
      const u = r();
      if (u < 0.62) { const a = 0.1, b = 0.5, e = -0.3; return Math.pow(Math.pow(a, e) + u / 0.62 * (Math.pow(b, e) - Math.pow(a, e)), 1 / e); }
      const a = 0.5, b = 60, e = -1.3, w = (u - 0.62) / 0.38; return Math.pow(Math.pow(a, e) + w * (Math.pow(b, e) - Math.pow(a, e)), 1 / e);
    },
    // main-sequence star + a late red-giant swelling; returns observable properties at `age` Gyr
    at(M, age) {
      let L = Star.lum(M), R = Star.radius(M), T = Star.teff(L, R);
      const lf = Star.life(M), f = age / lf;
      if (f > 0.88) { const k = U.smooth(0.88, 1.0, f); L *= 1 + 60 * k; R *= 1 + 40 * k; T = Star.teff(L, R); }
      return { L, R, T, life: lf, alive: f < 1 };
    },
    spectralClass(T) { return T > 30000 ? "O" : T > 10000 ? "B" : T > 7500 ? "A" : T > 6000 ? "F" : T > 5200 ? "G" : T > 3700 ? "K" : "M"; },
  });

  // ============================================================================================================
  class Node {
    constructor(kind, id, parent) { this.kind = kind; this.id = id; this.parent = parent || null; this.name = ""; this.size = 1; this.local = [0, 0, 0]; this.data = {}; this.q = Q.ident(); this.children = null; }
    chain() { const c = []; for (let n = this; n; n = n.parent) c.unshift(n); return c; }
    // position of this node expressed in the frame of `anc` (ancestor); metres
    posIn(anc) { let p = [0, 0, 0]; for (let n = this; n && n !== anc; n = n.parent) p = V.add(p, n.local); return p; }
  }
  U.Node = Node;

  // ============================================================================================================
  class Universe {
    constructor(seed, web) {
      this.seed = seed >>> 0; this.web = web;
      const H = web.halos, K = (this.K = H.count), h = U.Cosmo.P.H0 / 100;
      this.boxMpc = web.L / h; this.boxM = this.boxMpc * MPC;
      this.root = new Node("universe", 0, null); this.root.name = "Observable universe"; this.root.size = this.boxM; this.root.data.universe = this;
      this.logM = new Float32Array(K); this.rvir = new Float32Array(K); this.members = new Uint16Array(K);
      this.tForm = new Float32Array(K); this.kind = new Uint8Array(K); this.galR = new Float32Array(K);
      const sigCal = 3.9;
      for (let i = 0; i < K; i++) {
        const r = U.rng(U.hash2(this.seed, i + 17));
        const lm = Math.log10(1e15 / (1 + i / 18)) + (r() - 0.5) * 0.2;
        this.logM[i] = lm;
        this.rvir[i] = 0.26 * Math.pow(Math.pow(10, lm - 12), 1 / 3) * MPC;
        this.members[i] = Math.max(1, Math.min(150, Math.round(Math.pow(Math.pow(10, lm) / 4e11, 0.75))));
        const Dc = Math.min(0.95, 1.686 / (Math.max(H.nu[i], 1.5) * sigCal));
        this.tForm[i] = U.Cosmo.tOfA(U.Cosmo.aOfD(Dc)) * (0.92 + 0.16 * r());
        const k = this.centralKind(lm, r);
        this.kind[i] = k;
        this.galR[i] = this.galRadiusKpc(lm - 0.6, k);
      }
      this.haloNodes = new Map();
    }
    centralKind(lm, r) {
      const u = r();
      if (lm > 13.2) return u < 0.88 ? 2 : u < 0.95 ? 0 : 3;
      if (lm > 12.3) return u < 0.5 ? 2 : u < 0.88 ? (r() < 0.5 ? 1 : 0) : 3;
      if (lm > 11.4) return u < 0.1 ? 2 : u < 0.78 ? (r() < 0.45 ? 1 : 0) : 3;
      return u < 0.55 ? 3 : u < 0.9 ? 0 : 2;
    }
    galRadiusKpc(lmStellarHalo, kind) {
      const R = 9 * Math.pow(Math.pow(10, lmStellarHalo - 11), 0.33) * (kind === 2 ? 0.85 : kind === 3 ? 0.55 : 1);
      return U.clamp(R, 1.2, 70);
    }
    // position of halo i in the universe frame (metres, centred box), at cosmic time t
    haloPos(i, D, out) {
      const H = this.web.halos, L = this.boxM;
      for (let k = 0; k < 3; k++) { let x = H.q[i * 3 + k] + D * H.psi[i * 3 + k]; x -= Math.floor(x); out[k] = (x - 0.5) * L; }
      return out;
    }
    haloNode(i) {
      let n = this.haloNodes.get(i);
      if (n) return n;
      n = new Node("halo", i, this.root); n.name = "Halo " + U.name(U.hash2(this.seed, i)) + " " + (1000 + (i % 9000));
      n.size = Math.max(this.rvir[i], this.galR[i] * KPC * 3); n.data = { index: i, logM: this.logM[i], members: this.members[i], rvir: this.rvir[i], uni: this };
      n.data.groupName = this.members[i] > 20 ? "Galaxy cluster" : this.members[i] > 1 ? "Galaxy group" : "Galaxy";
      this.haloNodes.set(i, n);
      return n;
    }
    updateHalo(n, D) { this.haloPos(n.id, D, n.local); }
    // ---- galaxies inside a halo: member 0 is the central galaxy -------------------------------------------
    members_of(n) {
      if (n.children) return n.children;
      const i = n.id, M = this.members[i], lmH = this.logM[i], list = [];
      const rr = U.rng(U.hash2(this.seed, i * 977 + 5));
      for (let k = 0; k < M; k++) {
        const r = U.rng(U.hash3(this.seed, i, k + 101));
        const gal = new Node("galaxy", U.hash3(this.seed, i, k), n);
        gal.index = k;
        let lmG, kind, off;
        if (k === 0) { lmG = lmH - 0.6; kind = this.kind[i]; off = [0, 0, 0]; }
        else {
          lmG = Math.max(9.3, lmH - 1.7 - 2.4 * Math.pow(r(), 0.7));
          const u = r(), cl = lmH > 13;
          kind = cl ? (u < 0.5 ? 2 : u < 0.8 ? 0 : 3) : (u < 0.15 ? 2 : u < 0.55 ? 0 : u < 0.7 ? 1 : 3);
          const rad = this.rvir[i] * 0.85 * Math.pow(r(), 0.8), th = Math.acos(2 * r() - 1), ph = r() * U.TAU;
          off = [rad * Math.sin(th) * Math.cos(ph), rad * Math.cos(th) * 0.8, rad * Math.sin(th) * Math.sin(ph)];
        }
        gal.local = off;
        const Rk = this.galRadiusKpc(lmG, kind);
        const arms = r() < 0.55 ? 2 : r() < 0.6 ? 4 : 3;
        gal.q = randQuat(r);
        gal.size = Rk * KPC;
        const tF = this.tForm[i] * (1 + 0.5 * r() * (k ? 1 : 0.2));
        gal.data = {
          kind, logM: lmG, R: Rk, arms, pitch: 0.2 + 0.28 * r(), bulge: kind === 2 ? 1 : 0.14 + 0.38 * r() * (kind === 1 ? 1.2 : 1),
          tForm: tF, tau: kind === 2 ? 0.5 + r() : kind === 3 ? 3 + 4 * r() : 4 + 5 * r(), armAmp: kind === 2 ? 0 : 0.5 + 0.45 * r(),
          halo: i, seed: U.hash3(this.seed, i, k + 7), phase: r() * U.TAU, uni: this,
        };
        gal.name = this.galName(gal);
        list.push(gal);
      }
      n.children = list; return list;
    }
    galName(gal) { const d = gal.data; return U.name(d.seed) + " " + ["Spiral galaxy", "Barred spiral galaxy", "Elliptical galaxy", "Irregular galaxy"][d.kind]; }
  }
  U.Universe = Universe;
  U.GALAXY_KINDS = ["Spiral", "Barred spiral", "Elliptical", "Irregular"];

  // ============================================================================================================
  // Star cloud of one galaxy (aligned frame: disc in the XZ plane, +Y is the spin axis). Units: kpc.
  // Returns Float32 arrays pos(3), attr(4: logL, Teff, tBirthGyr, lifeGyr) and mass.
  function sampleStars(gal, N, kind) {
    const d = gal.data, R = d.R, r = U.rng(U.hash2(d.seed, 0x5bd1)), tNow = U.Cosmo.TNOW / U.GYR;
    const pos = new Float32Array(N * 3), attr = new Float32Array(N * 4), mass = new Float32Array(N);
    const hr = R / 3.2, hz = 0.13 * hr, bulgeA = R * 0.10, ell = d.kind === 2, irr = d.kind === 3, bar = d.kind === 1;
    const barAng = d.phase, barL = R * 0.32;
    const tF = d.tForm / U.GYR;
    const clumps = []; for (let k = 0; k < 14; k++) clumps.push([r.gauss() * R * 0.45, r.gauss() * R * 0.08, r.gauss() * R * 0.45, R * (0.08 + 0.12 * r())]);
    const armPhase = (rad) => Math.log(Math.max(rad, 0.2 * R * 0.25) / (0.28 * R)) / Math.tan(d.pitch);
    let n = 0;
    while (n < N) {
      let M = Star.imf(r); if (M < 0.8) M = 0.8 + 0.5 * r();             // the cloud shows luminous representatives
      let comp = r();
      const fb = d.bulge * 0.55, fh = 0.035;
      let x, y, z, young = Math.min(1, Math.max(0, (Math.log10(M) + 0.1) / 0.9));
      if (ell) {
        const a = R * 0.22, u = Math.sqrt(r()), rad = Math.min(a * u / (1 - u + 1e-3), R * 1.6), th = Math.acos(2 * r() - 1), ph = r() * U.TAU;
        x = rad * Math.sin(th) * Math.cos(ph); y = rad * Math.cos(th) * 0.72; z = rad * Math.sin(th) * Math.sin(ph) * 0.86;
        young = 0;
      } else if (comp < fh) {
        const rad = R * 0.5 * Math.pow(r(), -0.45), th = Math.acos(2 * r() - 1), ph = r() * U.TAU;
        if (rad > R * 2.6) continue;
        x = rad * Math.sin(th) * Math.cos(ph); y = rad * Math.cos(th); z = rad * Math.sin(th) * Math.sin(ph); young = 0;
      } else if (comp < fh + fb) {
        const u = Math.sqrt(r()), rad = Math.min(bulgeA * u / (1 - u + 1e-3), R * 0.6), th = Math.acos(2 * r() - 1), ph = r() * U.TAU;
        x = rad * Math.sin(th) * Math.cos(ph); y = rad * Math.cos(th) * 0.7; z = rad * Math.sin(th) * Math.sin(ph); young *= 0.15;
        if (bar && r() < 0.55) { const t = (r() * 2 - 1) * barL, w = 0.07 * R; const cx = t, cz = r.gauss() * w; x = cx * Math.cos(barAng) - cz * Math.sin(barAng); z = cx * Math.sin(barAng) + cz * Math.cos(barAng); y = r.gauss() * 0.05 * R; }
      } else if (irr) {
        const c = clumps[Math.floor(r() * clumps.length)];
        x = c[0] + r.gauss() * c[3]; y = c[1] + r.gauss() * c[3] * 0.5; z = c[2] + r.gauss() * c[3];
      } else {
        // exponential disc with spiral-arm modulation (stronger for massive, short-lived stars)
        let rad, phi, ok = false, tries = 0;
        while (!ok && tries++ < 40) {
          rad = -hr * (Math.log(r() + 1e-9) + Math.log(r() + 1e-9)); if (rad > R * 1.45) continue;
          phi = r() * U.TAU;
          const A = d.armAmp * (0.45 + 0.55 * young) * Math.min(1, rad / (0.25 * R) + 0.15) * U.smooth(0.15 * R, 0.45 * R, rad + 0.2 * R);
          const m = d.arms, mod = (1 + A * (Math.cos(m * (phi - armPhase(rad) - d.phase)) * 1.0 + 0.0) * 2.6) / (1 + A * 2.6);
          ok = r() < Math.max(0.02, mod);
        }
        if (!ok) continue;
        x = rad * Math.cos(phi); z = rad * Math.sin(phi);
        const thick = 1 + 2.5 * (1 - young);
        y = -hz * thick * Math.log(r() + 1e-9) * (r() < 0.5 ? 1 : -1);
        if (rad < 0.1 * R) continue;
      }
      // birth time: truncated exponentially declining SFR since formation (spheroids: a short early burst)
      const span = Math.max(0.5, tNow * 1.4 - tF), tau = (ell || (comp < fh + fb && !irr)) ? 0.5 + 0.4 * r() : d.tau;
      const old = (ell || comp < fh + fb) ? 1 : 0;
      let tb = tF - tau * Math.log(1 - r() * (1 - Math.exp(-span / tau)));
      if (comp < fh) tb = tF + r() * 0.2;
      const life = Star.life(M), L = Star.lum(M), Rr = Star.radius(M), T = Star.teff(L, Rr);
      pos[n * 3] = x; pos[n * 3 + 1] = y; pos[n * 3 + 2] = z;
      attr[n * 4] = Math.log10(L); attr[n * 4 + 1] = T; attr[n * 4 + 2] = tb; attr[n * 4 + 3] = life; mass[n] = M;
      n++;
    }
    return { pos, attr, mass, n };
  }
  U.sampleGalaxyStars = sampleStars;

  // relative stellar number density of the galaxy at aligned position (kpc): 1.0 ~ "solar neighbourhood"
  U.galaxyDensity = function (gal, x, y, z) {
    const d = gal.data, R = d.R, rad = Math.hypot(x, z), hr = R / 3.2, hz = 0.13 * hr * 2.0;
    if (d.kind === 2) { const s = Math.hypot(x, y / 0.75, z) / (R * 0.22); return 40 * Math.exp(-s * 0.8) + 0.01; }
    const disc = Math.exp(-(rad - 0.5 * R) / hr) * Math.pow(1 / Math.cosh(y / hz), 2);
    const bulge = 25 * Math.exp(-Math.hypot(x, y / 0.7, z) / (R * 0.06));
    return Math.max(0.0005, disc + bulge * d.bulge) * (R > 0 ? 1 : 0);
  };

  // Real stars near a point: generated per 6 pc cell from a hash, so the neighbourhood is identical every visit.
  const CELL_PC = 6;
  U.localStarsAround = function (gal, cx_kpc, cy_kpc, cz_kpc, radiusPc, cache) {
    const d = gal.data, tNow = U.Cosmo.TNOW / U.GYR;
    const ix0 = Math.floor((cx_kpc * 1000 - radiusPc) / CELL_PC), ix1 = Math.floor((cx_kpc * 1000 + radiusPc) / CELL_PC);
    const iy0 = Math.floor((cy_kpc * 1000 - radiusPc) / CELL_PC), iy1 = Math.floor((cy_kpc * 1000 + radiusPc) / CELL_PC);
    const iz0 = Math.floor((cz_kpc * 1000 - radiusPc) / CELL_PC), iz1 = Math.floor((cz_kpc * 1000 + radiusPc) / CELL_PC);
    const out = [];
    for (let ix = ix0; ix <= ix1; ix++) for (let iy = iy0; iy <= iy1; iy++) for (let iz = iz0; iz <= iz1; iz++) {
      const key = ix + "," + iy + "," + iz;
      let cell = cache && cache.get(key);
      if (!cell) {
        cell = [];
        const h0 = U.hash3(d.seed ^ 0x77, ix, U.hash2(iy, iz)), r = U.rng(h0);
        const px = (ix + 0.5) * CELL_PC / 1000, py = (iy + 0.5) * CELL_PC / 1000, pz = (iz + 0.5) * CELL_PC / 1000;
        const rho = 0.022 * U.galaxyDensity(gal, px, py, pz), lam = rho * CELL_PC * CELL_PC * CELL_PC;
        let k = 0, L0 = Math.exp(-lam), p = 1, cnt = 0; do { cnt++; p *= r(); } while (p > L0 && cnt < 200); cnt = Math.min(cnt - 1, 12);   // Poisson, capped
        const tF = d.tForm / U.GYR;
        for (let s = 0; s < cnt; s++) {
          let M = Star.imf(r); if (M < 0.3 && r() < 0.8) M = 0.3 + 0.5 * r(); const tb = tF + (tNow - tF) * (-Math.log(1 - r() * (1 - Math.exp(-3))) / 3 * 1.0) * (d.kind === 2 ? 0.2 : 1.0) - 0 + r() * 0.01;
          const L = Star.lum(M), Rr = Star.radius(M), T = Star.teff(L, Rr);
          cell.push({ pos: [(ix + r()) * CELL_PC / 1000, (iy + r()) * CELL_PC / 1000, (iz + r()) * CELL_PC / 1000], M, tb: Math.max(tF, Math.min(tNow, tb)), life: Star.life(M), logL: Math.log10(L), T, id: U.hash3(h0, s, 0x33) });
        }
        if (cache) cache.set(key, cell);
      }
      for (const s of cell) out.push(s);
    }
    return out;
  };

  // ============================================================================================================
  // Planetary system of a star (deterministic from the star id).
  U.makeStarNode = function (gal, spec) {   // spec: {pos(aligned kpc), M, tb, id}
    const n = new Node("star", spec.id, gal);
    const age = Math.max(0, U.Cosmo.TNOW / U.GYR - spec.tb);
    const sp = Star.at(spec.M, age);
    n.q = Q.ident();
    n.local = V.scale(Q.rotate(gal.q, spec.pos), KPC);                 // aligned kpc -> galaxy frame metres
    n.local = [Math.fround(n.local[0]), Math.fround(n.local[1]), Math.fround(n.local[2])];
    n.alignedKpc = spec.pos;
    n.name = U.name(spec.id) + (spec.M > 8 ? "" : "");
    n.data = { M: spec.M, tb: spec.tb, age, L: sp.L, R: sp.R * U.RSUN, T: sp.T, cls: Star.spectralClass(sp.T), life: sp.life, seed: spec.id, alive: sp.alive };
    n.size = 120 * AU;
    return n;
  };

  U.makeSystem = function (star) {
    if (star.children) return star.children;
    const d = star.data, r = U.rng(U.hash2(d.seed, 0x51ed));
    const M = d.M, L = d.L, list = [];
    star.children = list;
    const hasPlanets = M < 2.2 && r() < 0.85 && M > 0.1;
    const frost = 2.7 * Math.sqrt(L) * AU;
    let a = (0.04 + 0.35 * r()) * Math.sqrt(Math.max(L, 0.02)) * AU * (0.5 + M);
    const np = hasPlanets ? 1 + Math.floor(r() * 8 * Math.min(1, 0.4 + M)) : 0;
    const roman = ["b", "c", "d", "e", "f", "g", "h", "i", "j", "k"];
    for (let k = 0; k < np; k++) {
      const pr = U.rng(U.hash3(d.seed, k, 0xa11)), p = new Node("planet", U.hash3(d.seed, k, 9), star);
      const giant = a > frost * 0.8 ? pr() < 0.7 : pr() < 0.06;
      let Rp, Mp, cls;
      if (giant) { const ice = pr() < 0.4; Rp = (ice ? 3.2 + 1.6 * pr() : 7 + 4 * pr()) * U.REARTH; Mp = (ice ? 12 + 8 * pr() : 90 + 250 * pr()) * 5.972e24; cls = ice ? "Ice giant" : "Gas giant"; }
      else { Rp = (0.35 + 1.7 * Math.pow(pr(), 1.4)) * U.REARTH; Mp = Math.pow(Rp / U.REARTH, 3.2) * 5.972e24; cls = null; }
      const albedo = giant ? 0.35 : 0.1 + 0.4 * pr();
      const Teq = 278 * Math.pow(L, 0.25) / Math.sqrt(a / AU) * Math.pow(1 - albedo, 0.25);
      let biome = "";
      if (!giant) {
        const hasAtm = Mp > 0.2 * 5.972e24 * (1 + 0.5 * pr());
        if (!hasAtm) biome = "barren"; else if (Teq > 520) biome = "lava"; else if (Teq > 330) biome = "desert"; else if (Teq > 240) biome = pr() < 0.6 ? "ocean" : "temperate"; else if (Teq > 170) biome = pr() < 0.5 ? "cold" : "mars"; else biome = "ice";
        cls = { barren: "Airless rocky world", lava: "Molten lava world", desert: "Hot desert world", ocean: "Ocean world", temperate: "Temperate world with oceans", cold: "Cold world", mars: "Cold desert world", ice: "Frozen world" }[biome];
      } else biome = cls === "Gas giant" ? "gas" : "icegiant";
      p.name = U.name(d.seed) + " " + roman[k];
      p.size = Rp;
      const ecc = Math.min(0.35, Math.pow(pr(), 3) * 0.5);
      const P = Math.sqrt(Math.pow(a / AU, 3) / M) * U.YEAR;          // Kepler's third law
      p.data = {
        a, ecc, P, M0: pr() * U.TAU, inc: (pr() - 0.5) * 0.12, node: pr() * U.TAU, R: Rp, Mass: Mp, cls, biome, Teq, albedo, seed: U.hash3(d.seed, k, 0xbee),
        tilt: pr() < 0.5 ? pr() * 0.45 : pr() * 0.12, day: (giant ? 0.3 + 0.4 * pr() : 0.4 + 2.2 * pr()) * 86400, hasRings: giant && pr() < 0.4, index: k,
        atm: giant ? 1 : biome === "barren" ? 0 : Math.min(1, 0.4 + Mp / 5.972e24 * 0.4), star: star,
      };
      p.data.rings = p.data.hasRings;
      list.push(p);
      // moons
      p.children = [];
      const nm = giant ? Math.floor(pr() * 5) : (pr() < 0.25 ? 1 : 0);
      let ma = Rp * (giant ? 4 + 3 * pr() : 14 + 20 * pr());
      for (let m = 0; m < nm; m++) {
        const mr = U.rng(U.hash3(p.data.seed, m, 0x300)), mo = new Node("moon", U.hash3(p.data.seed, m, 3), p);
        const Rm = (giant ? 0.04 + 0.22 * mr() : 0.1 + 0.2 * mr()) * U.REARTH * (giant ? 1 : 1);
        mo.name = p.name + "-" + U.ROMAN[m + 1].toLowerCase(); mo.size = Rm;
        const Mp2 = p.data.Mass, Pm = 2 * Math.PI * Math.sqrt(Math.pow(ma, 3) / (6.674e-11 * Mp2));
        mo.data = { a: ma, ecc: 0.02, P: Pm, M0: mr() * U.TAU, inc: (mr() - 0.5) * 0.2, node: mr() * U.TAU, R: Rm, Mass: Rm * Rm * Rm * 3.3e3 * 4.19, cls: "Moon", biome: mr() < 0.5 ? "barren" : "ice", Teq: Teq, albedo: 0.3, seed: U.hash3(p.data.seed, m, 0x301), tilt: 0, day: Pm, hasRings: false, atm: 0, index: m, star: star, moon: true };
        p.children.push(mo); ma *= 1.7 + 1.2 * mr();
      }
      a *= 1.35 + 0.95 * pr();
    }
    // outermost orbit sets the system size
    const last = list[list.length - 1];
    star.size = Math.max(last ? last.data.a * 1.4 : 5 * AU, 0.5 * AU, d.R * 60);
    // optional belt
    star.data.belt = list.length > 2 && r() < 0.5 ? { a: list[Math.floor(list.length / 2)].data.a * 1.15, w: 0.12 } : null;
    return list;
  };

  // Keplerian position (metres) of a planet/moon in its parent's frame at orbital clock `clk` (seconds)
  U.orbitPos = function (d, clk, out) {
    const Mn = d.M0 + U.TAU * clk / d.P;
    let E = Mn; for (let i = 0; i < 6; i++) E -= (E - d.ecc * Math.sin(E) - Mn) / (1 - d.ecc * Math.cos(E));
    const x = d.a * (Math.cos(E) - d.ecc), z = d.a * Math.sqrt(1 - d.ecc * d.ecc) * Math.sin(E);
    const ci = Math.cos(d.inc), si = Math.sin(d.inc), cn = Math.cos(d.node), sn = Math.sin(d.node);
    const x1 = x * cn - z * sn, z1 = x * sn + z * cn;
    out[0] = x1; out[1] = z1 * si; out[2] = z1 * ci; return out;
  };
})(typeof window !== "undefined" ? window : globalThis);
