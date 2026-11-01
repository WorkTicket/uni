/* webgen.js - cosmic web generator (runs in a Web Worker, with a synchronous fallback).

   Method (truncated Zel'dovich approximation, the same first-order Lagrangian perturbation theory the engine's
   linear-growth module is built on):
     1. Gaussian random field delta(k) = white_noise(k) * sqrt(P(k)) * W(kR)     P(k) = k^ns T_BBKS(k)^2
     2. displacement field  psi(k) = i k delta(k) / k^2          (so that  div psi = -delta)
     3. particles x(q, a) = q + D(a) psi(q)    -- D(a) comes from the engine's LCDM growth ODE (cosmo.js)
     4. density from the deformation tensor  rho = 1 / det(I + D dpsi/dq)
   Halo candidates are the local maxima of delta plus an unresolved-small-scale term (sub-grid power).
   This is NOT an N-body simulation; see docs/SCIENTIFIC_LIMITS.md. Shell-crossed regions are only schematic. */
(function (g) {
  "use strict";
  const U = g.U;

  function webgenCore(p, progress) {
    const N = p.N, N3 = N * N * N, L = p.L;               // L in Mpc/h
    function mulberry32(a) { let s = a >>> 0; return function () { s = (s + 0x6d2b79f5) >>> 0; let t = s; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
    function hash32(x) { x |= 0; x = Math.imul(x ^ (x >>> 16), 0x7feb352d); x = Math.imul(x ^ (x >>> 15), 0x846ca68b); return (x ^ (x >>> 16)) >>> 0; }
    const rnd = mulberry32((p.seed ^ 0x9e3779b9) >>> 0);
    function gauss() { let u = 0; while (u === 0) u = rnd(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(6.283185307179586 * rnd()); }

    // ---- 3D FFT (radix-2, per-axis line buffers) ----
    const rev = new Uint32Array(N), bits = Math.round(Math.log2(N));
    for (let i = 0; i < N; i++) { let r = 0, x = i; for (let b = 0; b < bits; b++) { r = (r << 1) | (x & 1); x >>= 1; } rev[i] = r; }
    const cs = new Float64Array(N / 2), sn = new Float64Array(N / 2);
    for (let k = 0; k < N / 2; k++) { cs[k] = Math.cos(6.283185307179586 * k / N); sn[k] = Math.sin(6.283185307179586 * k / N); }
    const lr = new Float64Array(N), li = new Float64Array(N);
    function fft1(inv) {
      for (let i = 0; i < N; i++) { const j = rev[i]; if (j > i) { let t = lr[i]; lr[i] = lr[j]; lr[j] = t; t = li[i]; li[i] = li[j]; li[j] = t; } }
      for (let size = 2; size <= N; size <<= 1) {
        const half = size >> 1, step = N / size;
        for (let i = 0; i < N; i += size) {
          for (let j = 0, k = 0; j < half; j++, k += step) {
            const wr = cs[k], wi = inv ? sn[k] : -sn[k], a = i + j, b = a + half;
            const xr = lr[b] * wr - li[b] * wi, xi = lr[b] * wi + li[b] * wr;
            lr[b] = lr[a] - xr; li[b] = li[a] - xi; lr[a] += xr; li[a] += xi;
          }
        }
      }
    }
    function fft3(re, im, inv) {
      const strides = [N * N, N, 1];
      for (let ax = 0; ax < 3; ax++) {
        const st = strides[ax], o1 = strides[(ax + 1) % 3], o2 = strides[(ax + 2) % 3];
        for (let a = 0; a < N; a++) for (let b = 0; b < N; b++) {
          const base = a * o1 + b * o2;
          for (let i = 0, idx = base; i < N; i++, idx += st) { lr[i] = re[idx]; li[i] = im[idx]; }
          fft1(inv);
          for (let i = 0, idx = base; i < N; i++, idx += st) { re[idx] = lr[i]; im[idx] = li[i]; }
        }
      }
    }

    // ---- 1. white noise -> delta(k) ----
    progress(0.02, "Seeding primordial fluctuations");
    const wr = new Float32Array(N3), wi = new Float32Array(N3);
    for (let i = 0; i < N3; i++) wr[i] = gauss();
    fft3(wr, wi, false);
    progress(0.14, "Applying the matter power spectrum");
    const Gam = p.Om * p.h, ns = p.ns, kf = 6.283185307179586 / L;
    const cell = L / N, Rs = Math.max(p.truncMpc, 1.25 * cell);
    function bbks(k) {
      const q = k / Gam; if (q < 1e-6) return 1;
      const t = Math.log(1 + 2.34 * q) / (2.34 * q), c = 1 + 3.89 * q + Math.pow(16.1 * q, 2) + Math.pow(5.46 * q, 3) + Math.pow(6.71 * q, 4);
      return t * Math.pow(c, -0.25);
    }
    const dr = new Float32Array(N3), di = new Float32Array(N3);       // delta(k)
    for (let ix = 0; ix < N; ix++) {
      const kx = (ix <= N / 2 ? ix : ix - N) * kf;
      for (let iy = 0; iy < N; iy++) {
        const ky = (iy <= N / 2 ? iy : iy - N) * kf;
        for (let iz = 0; iz < N; iz++) {
          const kz = (iz <= N / 2 ? iz : iz - N) * kf, k2 = kx * kx + ky * ky + kz * kz, idx = (ix * N + iy) * N + iz;
          if (k2 === 0) continue;
          const k = Math.sqrt(k2), T = bbks(k), amp = Math.pow(k, ns * 0.5) * T * Math.exp(-0.5 * k2 * Rs * Rs);
          dr[idx] = wr[idx] * amp; di[idx] = wi[idx] * amp;
        }
      }
    }
    // ---- 2. displacement fields psi = i k delta / k^2 ----
    const psi = [new Float32Array(N3), new Float32Array(N3), new Float32Array(N3)];
    for (let ax = 0; ax < 3; ax++) {
      progress(0.2 + 0.18 * ax, "Computing displacement field " + "xyz"[ax]);
      for (let ix = 0; ix < N; ix++) {
        const kx = (ix <= N / 2 ? ix : ix - N) * kf;
        for (let iy = 0; iy < N; iy++) {
          const ky = (iy <= N / 2 ? iy : iy - N) * kf;
          for (let iz = 0; iz < N; iz++) {
            const kz = (iz <= N / 2 ? iz : iz - N) * kf, k2 = kx * kx + ky * ky + kz * kz, idx = (ix * N + iy) * N + iz;
            if (k2 === 0) { wr[idx] = 0; wi[idx] = 0; continue; }
            const kk = (ax === 0 ? kx : ax === 1 ? ky : kz) / k2;
            wr[idx] = -di[idx] * kk; wi[idx] = dr[idx] * kk;          // i * (a + ib) = -b + ia
          }
        }
      }
      fft3(wr, wi, true);
      const f = 1 / N3, out = psi[ax];
      for (let i = 0; i < N3; i++) out[i] = wr[i] * f;
    }
    // delta(x) at a=1 for halo finding
    progress(0.76, "Locating collapsing peaks");
    fft3(dr, di, true);
    const delta = dr; { const f = 1 / N3; for (let i = 0; i < N3; i++) delta[i] *= f; }

    // ---- 3. normalise so that the 3D rms displacement at a=1 equals p.psiRms (Mpc/h) ----
    let s2 = 0; for (let i = 0; i < N3; i++) s2 += psi[0][i] * psi[0][i] + psi[1][i] * psi[1][i] + psi[2][i] * psi[2][i];
    const sc = p.psiRms / Math.sqrt(s2 / N3);
    const toFrac = sc / L;                                              // Mpc/h -> box fraction after scaling
    for (let ax = 0; ax < 3; ax++) { const a = psi[ax]; for (let i = 0; i < N3; i++) a[i] *= toFrac; }
    let sd = 0; for (let i = 0; i < N3; i++) { delta[i] *= sc; sd += delta[i] * delta[i]; }
    const sigmaDelta = Math.sqrt(sd / N3);

    // strain eigenvalues at a grid cell: S_ij = d psi_i / d x_j (x in box fractions)
    const eig = new Float64Array(3);
    function strainEig(ix, iy, iz) {
      const xp = (ix + 1) % N, xm = (ix + N - 1) % N, yp = (iy + 1) % N, ym = (iy + N - 1) % N, zp = (iz + 1) % N, zm = (iz + N - 1) % N;
      const id = (a, b, c) => (a * N + b) * N + c, h = N / 2;
      const A = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
      for (let i = 0; i < 3; i++) {
        const f = psi[i];
        A[i][0] = (f[id(xp, iy, iz)] - f[id(xm, iy, iz)]) * h;
        A[i][1] = (f[id(ix, yp, iz)] - f[id(ix, ym, iz)]) * h;
        A[i][2] = (f[id(ix, iy, zp)] - f[id(ix, iy, zm)]) * h;
      }
      const a00 = A[0][0], a11 = A[1][1], a22 = A[2][2], a01 = 0.5 * (A[0][1] + A[1][0]), a02 = 0.5 * (A[0][2] + A[2][0]), a12 = 0.5 * (A[1][2] + A[2][1]);
      const p1 = a01 * a01 + a02 * a02 + a12 * a12, q = (a00 + a11 + a22) / 3;
      if (p1 < 1e-18) { const e = [a00, a11, a22].sort((x, y) => x - y); eig[0] = e[0]; eig[1] = e[1]; eig[2] = e[2]; return; }
      const p2 = (a00 - q) ** 2 + (a11 - q) ** 2 + (a22 - q) ** 2 + 2 * p1, pp = Math.sqrt(p2 / 6);
      const b00 = (a00 - q) / pp, b11 = (a11 - q) / pp, b22 = (a22 - q) / pp, b01 = a01 / pp, b02 = a02 / pp, b12 = a12 / pp;
      const det = b00 * (b11 * b22 - b12 * b12) - b01 * (b01 * b22 - b12 * b02) + b02 * (b01 * b12 - b11 * b02);
      const r = Math.max(-1, Math.min(1, det / 2)), phi = Math.acos(r) / 3;
      const e1 = q + 2 * pp * Math.cos(phi), e3 = q + 2 * pp * Math.cos(phi + 2.0943951023931953), e2 = 3 * q - e1 - e3;
      eig[0] = Math.min(e1, e2, e3); eig[2] = Math.max(e1, e2, e3); eig[1] = 3 * q - eig[0] - eig[2];
    }

    // ---- 4. particle subsample ----
    progress(0.8, "Placing particles");
    const keepP = p.keep, jit = 0.32 / N;
    let count = 0;
    const keepMask = new Uint8Array(N3);
    for (let i = 0; i < N3; i++) { if (keepP >= 1 || hash32(i * 2654435761 + p.seed) / 4294967296 < keepP) { keepMask[i] = 1; count++; } }
    const pq = new Float32Array(count * 3), pp = new Float32Array(count * 3), pl = new Float32Array(count * 3);
    let n = 0;
    for (let ix = 0; ix < N; ix++) for (let iy = 0; iy < N; iy++) for (let iz = 0; iz < N; iz++) {
      const idx = (ix * N + iy) * N + iz;
      if (!keepMask[idx]) continue;
      const h1 = hash32(idx * 3 + p.seed), h2 = hash32(idx * 3 + 1 + p.seed), h3 = hash32(idx * 3 + 2 + p.seed);
      pq[n * 3] = (ix + 0.5) / N + (h1 / 4294967296 - 0.5) * 2 * jit;
      pq[n * 3 + 1] = (iy + 0.5) / N + (h2 / 4294967296 - 0.5) * 2 * jit;
      pq[n * 3 + 2] = (iz + 0.5) / N + (h3 / 4294967296 - 0.5) * 2 * jit;
      pp[n * 3] = psi[0][idx]; pp[n * 3 + 1] = psi[1][idx]; pp[n * 3 + 2] = psi[2][idx];
      strainEig(ix, iy, iz);
      pl[n * 3] = eig[0]; pl[n * 3 + 1] = eig[1]; pl[n * 3 + 2] = eig[2];
      n++;
    }
    // ---- 5. halo candidates ----
    progress(0.92, "Finding halos");
    const sub = p.subgrid * sigmaDelta, vv = new Float32Array(N3);
    for (let i = 0; i < N3; i++) vv[i] = delta[i] + sub * gauss();
    const cand = [];
    for (let ix = 0; ix < N; ix++) for (let iy = 0; iy < N; iy++) for (let iz = 0; iz < N; iz++) {
      const idx = (ix * N + iy) * N + iz, v = vv[idx];
      if (v < sigmaDelta * 1.2) continue;
      let ismax = true;
      for (let dx = -1; dx <= 1 && ismax; dx++) for (let dy = -1; dy <= 1 && ismax; dy++) for (let dz = -1; dz <= 1; dz++) {
        if (!dx && !dy && !dz) continue;
        if (vv[(((ix + dx + N) % N) * N + (iy + dy + N) % N) * N + (iz + dz + N) % N] > v) { ismax = false; break; }
      }
      if (ismax) cand.push(idx);
    }
    cand.sort((a, b) => vv[b] - vv[a]);
    const K = Math.min(cand.length, p.maxHalos);
    const hq = new Float32Array(K * 3), hp = new Float32Array(K * 3), hl = new Float32Array(K * 3), hn = new Float32Array(K), hd = new Float32Array(K);
    for (let k = 0; k < K; k++) {
      const idx = cand[k], iz = idx % N, iy = ((idx - iz) / N) % N, ix = (idx - iz - iy * N) / (N * N);
      hq[k * 3] = (ix + 0.5) / N; hq[k * 3 + 1] = (iy + 0.5) / N; hq[k * 3 + 2] = (iz + 0.5) / N;
      hp[k * 3] = psi[0][idx]; hp[k * 3 + 1] = psi[1][idx]; hp[k * 3 + 2] = psi[2][idx];
      strainEig(ix, iy, iz);
      hl[k * 3] = eig[0]; hl[k * 3 + 1] = eig[1]; hl[k * 3 + 2] = eig[2];
      hn[k] = vv[idx] / sigmaDelta; hd[k] = delta[idx] / sigmaDelta;
    }
    // rms displacement in grid cells (diagnostic)
    progress(1, "Ready");
    return { N, L, count, q: pq, psi: pp, lam: pl, halos: { count: K, q: hq, psi: hp, lam: hl, nu: hn, delta: hd, total: cand.length },
      stats: { sigmaDelta, psiRmsMpcH: p.psiRms, cellMpcH: cell, truncMpcH: Rs } };
  }

  function workerSource() {
    return "const webgenCore=" + webgenCore.toString() + ";\n" +
      "onmessage=function(e){const p=e.data;try{const r=webgenCore(p,function(f,l){postMessage({progress:f,label:l});});" +
      "const tr=[r.q.buffer,r.psi.buffer,r.lam.buffer,r.halos.q.buffer,r.halos.psi.buffer,r.halos.lam.buffer,r.halos.nu.buffer,r.halos.delta.buffer];" +
      "postMessage({result:r},tr);}catch(err){postMessage({error:String(err&&err.stack||err)});}};";
  }

  // returns a promise; onProgress(fraction, label)
  U.generateWeb = function (params, onProgress) {
    return new Promise((resolve, reject) => {
      const done = (r) => resolve(r);
      let w = null;
      try {
        const url = URL.createObjectURL(new Blob([workerSource()], { type: "text/javascript" }));
        w = new Worker(url);
        w.onmessage = (e) => {
          const m = e.data;
          if (m.progress !== undefined) onProgress && onProgress(m.progress, m.label);
          else if (m.error) { w.terminate(); reject(new Error(m.error)); }
          else { w.terminate(); URL.revokeObjectURL(url); done(m.result); }
        };
        w.onerror = (e) => { w.terminate(); fallback(); };
        w.postMessage(params);
      } catch (e) { fallback(); }
      function fallback() {
        // synchronous fallback (blocks the UI; only used when workers are unavailable)
        setTimeout(() => { try { resolve(webgenCore(params, onProgress || function () {})); } catch (err) { reject(err); } }, 30);
      }
    });
  };
  U._webgenCore = webgenCore;     // exposed for tests / Node
  if (typeof module !== "undefined") module.exports = { webgenCore };
})(typeof window !== "undefined" ? window : globalThis);
