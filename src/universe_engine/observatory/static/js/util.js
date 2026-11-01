/* util.js - constants, deterministic hashing/RNG, small vector/quaternion math, formatting, names.
   Everything here is plain classic-script JS (no modules) so the viewer also runs from file://. */
(function (g) {
  "use strict";
  const U = (g.U = g.U || {});

  // ---- physical constants (SI) ------------------------------------------------------------
  U.PC = 3.0856775814913673e16; U.KPC = U.PC * 1e3; U.MPC = U.PC * 1e6; U.GPC = U.PC * 1e9;
  U.AU = 1.495978707e11; U.LY = 9.4607304725808e15; U.RSUN = 6.957e8; U.MSUN = 1.98847e30;
  U.REARTH = 6.371e6; U.RJUP = 7.1492e7; U.YEAR = 3.15576e7; U.GYR = U.YEAR * 1e9; U.DAY = 86400;
  U.TAU = Math.PI * 2;
  U.G = 6.67430e-11;      // gravitational constant, m^3 kg^-1 s^-2
  U.C = 299792458;        // speed of light, m/s

  // ---- hashing / RNG (deterministic: the whole universe is a pure function of the seed) -----
  U.hash32 = function (x) {
    x |= 0;
    x = Math.imul(x ^ (x >>> 16), 0x7feb352d);
    x = Math.imul(x ^ (x >>> 15), 0x846ca68b);
    return (x ^ (x >>> 16)) >>> 0;
  };
  U.hash2 = function (a, b) { return U.hash32((a | 0) ^ Math.imul(U.hash32(b | 0) + 0x9e3779b9, 0x85ebca6b)); };
  U.hash3 = function (a, b, c) { return U.hash2(U.hash2(a, b), c); };
  U.hashf = function (h) { return (h >>> 0) / 4294967296; };
  U.rng = function (seed) {                       // mulberry32
    let s = seed >>> 0;
    const f = function () {
      s = (s + 0x6d2b79f5) >>> 0;
      let t = s;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
    f.gauss = function () {
      let u = 0, v = 0;
      while (u === 0) u = f();
      v = f();
      return Math.sqrt(-2 * Math.log(u)) * Math.cos(U.TAU * v);
    };
    f.range = function (a, b) { return a + (b - a) * f(); };
    f.pick = function (arr) { return arr[Math.floor(f() * arr.length) % arr.length]; };
    return f;
  };

  // ---- scalar helpers ---------------------------------------------------------------------
  U.clamp = function (x, a, b) { return x < a ? a : x > b ? b : x; };
  U.mix = function (a, b, t) { return a + (b - a) * t; };
  U.smooth = function (a, b, x) { const t = U.clamp((x - a) / (b - a), 0, 1); return t * t * (3 - 2 * t); };
  U.log10 = Math.log10;

  // ---- vec3 (plain arrays, doubles) -------------------------------------------------------
  const V = (U.V = {
    add: (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]],
    sub: (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]],
    scale: (a, s) => [a[0] * s, a[1] * s, a[2] * s],
    dot: (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2],
    cross: (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]],
    len: (a) => Math.hypot(a[0], a[1], a[2]),
    norm: (a) => { const l = Math.hypot(a[0], a[1], a[2]) || 1; return [a[0] / l, a[1] / l, a[2] / l]; },
    lerp: (a, b, t) => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t],
  });

  // ---- quaternion [x,y,z,w] ---------------------------------------------------------------
  const Q = (U.Q = {
    ident: () => [0, 0, 0, 1],
    mul(a, b) {
      return [
        a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
        a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
        a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
        a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2],
      ];
    },
    axisAngle(axis, ang) { const s = Math.sin(ang / 2), n = V.norm(axis); return [n[0] * s, n[1] * s, n[2] * s, Math.cos(ang / 2)]; },
    normalize(q) { const l = Math.hypot(q[0], q[1], q[2], q[3]) || 1; return [q[0] / l, q[1] / l, q[2] / l, q[3] / l]; },
    rotate(q, v) {                                // rotate vector v by q
      const u = [q[0], q[1], q[2]], w = q[3];
      const c1 = V.cross(u, v), c2 = V.cross(u, c1);
      return [v[0] + 2 * (w * c1[0] + c2[0]), v[1] + 2 * (w * c1[1] + c2[1]), v[2] + 2 * (w * c1[2] + c2[2])];
    },
    conj: (q) => [-q[0], -q[1], -q[2], q[3]],
    slerp(a, b, t) {
      let d = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
      if (d < 0) { b = [-b[0], -b[1], -b[2], -b[3]]; d = -d; }
      if (d > 0.9995) return Q.normalize([a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t, a[3] + (b[3] - a[3]) * t]);
      const th = Math.acos(d), s = Math.sin(th), w1 = Math.sin((1 - t) * th) / s, w2 = Math.sin(t * th) / s;
      return [a[0] * w1 + b[0] * w2, a[1] * w1 + b[1] * w2, a[2] * w1 + b[2] * w2, a[3] * w1 + b[3] * w2];
    },
    // camera orientation looking from `eye` at `target` with approximate up
    lookAt(dir, up) {                              // dir: forward direction (unit-ish)
      const f = V.norm(dir);
      let r = V.cross(f, up); if (V.len(r) < 1e-6) r = V.cross(f, [1, 0, 0]);
      r = V.norm(r);
      const u = V.cross(r, f), b = V.scale(f, -1);
      return Q.fromBasis(r, u, b);
    },
    fromBasis(r, u, b) {                           // columns r,u,b -> quaternion
      const m00 = r[0], m01 = u[0], m02 = b[0], m10 = r[1], m11 = u[1], m12 = b[1], m20 = r[2], m21 = u[2], m22 = b[2];
      const tr = m00 + m11 + m22; let x, y, z, w;
      if (tr > 0) { const s = Math.sqrt(tr + 1) * 2; w = 0.25 * s; x = (m21 - m12) / s; y = (m02 - m20) / s; z = (m10 - m01) / s; }
      else if (m00 > m11 && m00 > m22) { const s = Math.sqrt(1 + m00 - m11 - m22) * 2; w = (m21 - m12) / s; x = 0.25 * s; y = (m01 + m10) / s; z = (m02 + m20) / s; }
      else if (m11 > m22) { const s = Math.sqrt(1 + m11 - m00 - m22) * 2; w = (m02 - m20) / s; x = (m01 + m10) / s; y = 0.25 * s; z = (m12 + m21) / s; }
      else { const s = Math.sqrt(1 + m22 - m00 - m11) * 2; w = (m10 - m01) / s; x = (m02 + m20) / s; y = (m12 + m21) / s; z = 0.25 * s; }
      return Q.normalize([x, y, z, w]);
    },
    basis(q) { return [Q.rotate(q, [1, 0, 0]), Q.rotate(q, [0, 1, 0]), Q.rotate(q, [0, 0, 1])]; }, // right, up, back
  });

  // ---- formatting -------------------------------------------------------------------------
  function sig(x, d) { d = d || 3; const a = Math.abs(x); if (a >= 100) return x.toFixed(0); if (a >= 10) return x.toFixed(Math.max(0, d - 2)); return x.toFixed(Math.max(0, d - 1)); }
  U.fmtDist = function (m) {
    if (!isFinite(m)) return "-";
    const a = Math.abs(m);
    if (a < 1) return sig(m * 100, 3) + " cm";
    if (a < 1e4) return sig(m, 3) + " m";
    if (a < 5e8) return sig(m / 1e3, 3).replace(/\B(?=(\d{3})+(?!\d))/g, ",") + " km";
    if (a < 0.3 * U.AU * 1) return sig(m / U.RSUN, 3) + " R☉";
    if (a < 0.5 * U.LY) return sig(m / U.AU, 3) + " AU";
    if (a < 3 * U.PC) return sig(m / U.LY, 3) + " ly";
    if (a < 1e3 * U.PC) return sig(m / U.PC, 3) + " pc";
    if (a < 1e6 * U.PC) return sig(m / U.KPC, 3) + " kpc";
    if (a < 1e9 * U.PC) return sig(m / U.MPC, 3) + " Mpc";
    return sig(m / U.GPC, 3) + " Gpc";
  };
  U.fmtTime = function (s) {   // cosmic time (seconds since the Big Bang) -> text
    if (!(s > 0)) return "0";
    const yr = s / U.YEAR;
    if (s < 1e-3) { const e = Math.floor(Math.log10(s)); return (s / Math.pow(10, e)).toFixed(1) + "×10" + sup(e) + " s"; }
    if (s < 1) return (s * 1000).toFixed(1) + " ms";
    if (s < 120) return s.toFixed(1) + " s";
    if (s < 7200) return (s / 60).toFixed(1) + " min";
    if (s < 86400 * 3) return (s / 3600).toFixed(1) + " hours";
    if (yr < 2) return (s / 86400).toFixed(0) + " days";
    if (yr < 1e4) return yr.toFixed(0) + " years";
    if (yr < 1e6) return (yr / 1e3).toFixed(1) + " thousand yr";
    if (yr < 1e9) return (yr / 1e6).toFixed(yr < 1e7 ? 2 : 0) + " million yr";
    return (yr / 1e9).toFixed(yr < 1e10 ? 2 : 1) + " billion yr";
  };
  function sup(n) { const m = { "-": "⁻", 0: "⁰", 1: "¹", 2: "²", 3: "³", 4: "⁴", 5: "⁵", 6: "⁶", 7: "⁷", 8: "⁸", 9: "⁹" }; return String(n).split("").map((c) => m[c]).join(""); }
  U.fmtTemp = function (T) {
    if (T >= 1e9) return (T / 1e9).toFixed(1) + "×10⁹ K";
    if (T >= 1e4) { const e = Math.floor(Math.log10(T)); return (T / Math.pow(10, e)).toFixed(1) + "×10" + sup(e) + " K"; }
    if (T >= 100) return T.toFixed(0) + " K";
    return T.toFixed(1) + " K";
  };
  U.fmtSci = function (x, unit) { if (!(x > 0)) return "0" + (unit ? " " + unit : ""); const e = Math.floor(Math.log10(x)); return (x / Math.pow(10, e)).toFixed(1) + "×10" + sup(e) + (unit ? " " + unit : ""); };
  U.fmtMass = function (kg) {
    const ms = kg / U.MSUN;
    if (ms >= 1e6) return U.fmtSci(ms, "M☉");
    if (ms >= 0.05) return (ms >= 100 ? ms.toFixed(0) : ms.toFixed(2)) + " M☉";
    const me = kg / 5.972e24;
    if (me >= 50) return (kg / 1.898e27).toFixed(2) + " M♃";
    return me.toFixed(me < 0.1 ? 3 : 2) + " M⊕";
  };

  // ---- procedural names -------------------------------------------------------------------
  const SYL_A = ["Ae", "Ba", "Ce", "Da", "El", "Fa", "Ga", "Ha", "Il", "Ja", "Ka", "Lu", "Ma", "Ne", "Or", "Pa", "Qi", "Ra", "Se", "Ta", "Ur", "Va", "We", "Xe", "Yo", "Ze"];
  const SYL_B = ["bar", "cor", "dan", "gel", "han", "ion", "kar", "lun", "mor", "nar", "pho", "rin", "sol", "tur", "van", "xis", "zan", "thi", "lor", "mer"];
  const SYL_C = ["a", "is", "on", "ae", "us", "ia", "o", "ex", "ar", "ys"];
  U.name = function (h) {
    const r = U.rng(h);
    let n = SYL_A[Math.floor(r() * SYL_A.length)] + SYL_B[Math.floor(r() * SYL_B.length)];
    if (r() < 0.6) n += SYL_C[Math.floor(r() * SYL_C.length)];
    return n;
  };
  U.ROMAN = ["", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII"];

  // ---- misc -------------------------------------------------------------------------------
  U.blackbodyRGB = function (T) {    // Tanner Helland approximation, returns linear-ish [0..1]
    T = U.clamp(T, 1000, 40000) / 100; let r, gg, b;
    r = T <= 66 ? 255 : 329.698727446 * Math.pow(T - 60, -0.1332047592);
    gg = T <= 66 ? 99.4708025861 * Math.log(T) - 161.1195681661 : 288.1221695283 * Math.pow(T - 60, -0.0755148492);
    b = T >= 66 ? 255 : T <= 19 ? 0 : 138.5177312231 * Math.log(T - 10) - 305.0447927307;
    const f = (x) => Math.pow(U.clamp(x, 0, 255) / 255, 2.2);
    return [f(r), f(gg), f(b)];
  };

  // ---- Schwarzschild black-hole physics (analytic, non-spinning) --------------------------
  // r_s = 2GM/c^2 ; photon sphere r_ph = 3GM/c^2 = 1.5 r_s ; critical impact parameter
  // b_crit = 3*sqrt(3)*GM/c^2 = (3*sqrt(3)/2) * r_s. Approximations: static Schwarzschild
  // spacetime, no frame dragging, no accretion-disk back-reaction. Lensing is the
  // weak-deflection / weak-lensing analytic approximation, not a full ray-traced GR image.
  U.BH = {
    schwarzschildRadius(massKg) { return 2 * U.G * massKg / (U.C * U.C); },
    photonSphere(massKg) { return 3 * U.G * massKg / (U.C * U.C); },
    criticalImpact(massKg) { return 3 * Math.sqrt(3) * U.G * massKg / (U.C * U.C); },
    // Weak-lensing deflection angle (radians) for a light ray with impact parameter b:
    // alpha = 4GM/(c^2 b) = 2 r_s / b. Valid for b >> r_s; not valid near the photon sphere.
    lensingAngle(massKg, b) { return 2 * U.BH.schwarzschildRadius(massKg) / Math.max(b, 1e-6); },
    // Mass in solar masses -> Schwarzschild radius in metres
    schwarzschildRadiusMsun(mSun) { return 2 * U.G * mSun * U.MSUN / (U.C * U.C); },
  };
})(typeof window !== "undefined" ? window : globalThis);
