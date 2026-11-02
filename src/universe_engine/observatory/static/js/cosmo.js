/* cosmo.js - flat LCDM background (same equations as src/universe_engine/physics.py), linear growth factor,
   epoch table and the AUTHORITATIVE cosmic background clock.

   The background clock owns the scientifically meaningful state:
     - cosmic time t (seconds), scale factor a, redshift z = 1/a - 1,
     - background temperature T = T0 / a,
     - the validated future horizon a_max (a model-validity bound, NOT the end of the universe).

   The presentation playback controller (in app.js) is a SEPARATE system that decides how
   fast the viewer advances through simulated physical time. It must not be the physics timebase.

   Parameters come from the engine's config/reference.yaml via /api/cosmology when served by
   the observatory, else built-in Planck 2018-like defaults. */
(function (g) {
  "use strict";
  const U = g.U;
  const MPC_KM = 3.0856775814913673e19;
  const YEAR = U.YEAR || 3.15576e7;

  const P = { H0: 67.4, Om: 0.315, Or: 9.2e-5, OL: 0.684908, ns: 0.965, T0: 2.7255,
              source: "built-in defaults (Planck 2018-like)" };

  // Authoritative background state (the physics clock).
  let state = {
    a: 1e-13,          // scale factor (dimensionless, a=1 today)
    t: 0,              // cosmic proper time (seconds)
    a_max: 80.0,       // validated future horizon (finite; extendable)
    a0: 1e-13,         // start of the table
    n: 9000,           // table resolution
    h: 0,              // d ln a per table step
    lna: null, logt: null, D: null,  // tables
    built: false,
  };

  function E2(a) { return P.Or / (a * a * a * a) + P.Om / (a * a * a) + P.OL; }
  function Hs(a) { return (P.H0 / MPC_KM) * Math.sqrt(E2(a)); }          // 1/s

  // Build (or rebuild) the background table on a log-spaced a grid from a0 to a_max.
  // Preserves the current (t, a) if the table already exists; otherwise starts at a0.
  function build(params) {
    if (params) Object.assign(P, params);
    P.OL = 1 - P.Om - P.Or;                                              // enforce flatness

    const a0 = state.a0;
    const a1 = state.a_max;
    const n = state.n;
    const lna0 = Math.log(a0), lna1 = Math.log(a1), h = (lna1 - lna0) / n;
    const lna = new Float64Array(n + 1), logt = new Float64Array(n + 1), D = new Float64Array(n + 1);
    // t(a) = int dln a / H ; radiation era start t0 = 1/(2H)
    let t = 1 / (2 * Hs(a0));
    for (let i = 0; i <= n; i++) {
      lna[i] = lna0 + i * h;
      if (i > 0) { const ha = Hs(Math.exp(lna[i - 1])), hb = Hs(Math.exp(lna[i])); t += 0.5 * h * (1 / ha + 1 / hb); }
      logt[i] = Math.log(t);
    }
    // growth factor: D'' + (2 + dlnH/dlna) D' - 1.5 Om(a) D = 0, in x = ln a (RK4), D ~ a when a is small
    const f = (x, d, dp) => {
      const a = Math.exp(x), e2 = E2(a), omA = (P.Om / (a * a * a)) / e2;
      const dlnH = (-4 * P.Or / (a * a * a * a) - 3 * P.Om / (a * a * a)) / (2 * e2);
      return [dp, -(2 + dlnH) * dp + 1.5 * omA * d];
    };
    let iStart = 0; while (iStart < n && Math.exp(lna[iStart]) < 1e-6) iStart++;
    for (let i = 0; i < iStart; i++) D[i] = Math.exp(lna[i]);
    let d = Math.exp(lna[iStart]), dp = d; D[iStart] = d;
    for (let i = iStart; i < n; i++) {
      const x = lna[i];
      const k1 = f(x, d, dp), k2 = f(x + h / 2, d + h * k1[0] / 2, dp + h * k1[1] / 2),
        k3 = f(x + h / 2, d + h * k2[0] / 2, dp + h * k2[1] / 2), k4 = f(x + h, d + h * k3[0], dp + h * k3[1]);
      d += h * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]) / 6; dp += h * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]) / 6;
      D[i + 1] = d;
    }
    const i1 = Math.round((0 - lna0) / h), d1 = D[i1] + (D[i1 + 1] - D[i1]) * (((0 - lna0) / h) - i1);
    for (let i = 0; i <= n; i++) D[i] /= d1;
    state.lna = lna; state.logt = logt; state.D = D; state.h = h; state.lna0 = lna0; state.n = n;
    state.built = true;
    // Re-sync t to the table at current a (keeps t consistent after a horizon change).
    if (state.a >= a0 && state.a <= a1) { state.t = API.tOfA(state.a); }
    else if (state.a > a1) { state.t = API.tOfA(a1); state.a = a1; } // clamp into the new table
    API.ageNow = API.tOfA(1);
  }

  function interp(arr, x) {                 // arr sampled on uniform lna grid; x = ln a
    const u = (x - state.lna0) / state.h;
    if (u <= 0) return arr[0];
    if (u >= state.n) return arr[state.n];
    const i = Math.floor(u), f = u - i;
    return arr[i] + (arr[i + 1] - arr[i]) * f;
  }

  // ---- authoritative background clock -----------------------------------------
  const API = (U.Cosmo = {
    P, build,
    // a -> t (seconds), within the current validated table only.
    tOfA(a) {
      if (!state.built) build();
      if (a <= 0) return 0;
      const aa = Math.min(Math.max(a, Math.exp(state.lna0)), Math.exp(state.lna0 + state.h * state.n));
      return Math.exp(interp(state.logt, Math.log(aa)));
    },
    // t (seconds) -> a, within the current validated table only.
    aOfT(t) {
      if (!state.built) build();
      const lt = Math.log(t), lt0 = state.logt[0];
      if (lt <= lt0) return Math.exp(state.lna0) * Math.sqrt(t / Math.exp(lt0));
      let lo = 0, hi = state.n;
      while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (state.logt[mid] <= lt) lo = mid; else hi = mid; }
      const f = (lt - state.logt[lo]) / (state.logt[hi] - state.logt[lo] || 1);
      return Math.exp(state.lna[lo] + (state.lna[hi] - state.lna[lo]) * f);
    },
    D(a) { return interp(state.D, Math.log(a)); },
    // a(D): inverse of the monotonic growth factor D(a). Returns the scale
    // factor a at which the linear growth factor reaches D (clamped to table).
    aOfD(D) {
      if (!state.built) build();
      const Ds = state.D, las = state.lna;
      if (Ds.length < 2) return Math.exp(las[0] || 0);
      if (D <= Ds[0]) return Math.exp(las[0]);
      if (D >= Ds[Ds.length - 1]) return Math.exp(las[las.length - 1]);
      let lo = 0, hi = Ds.length - 1;
      while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (Ds[mid] <= D) lo = mid; else hi = mid; }
      const f = (D - Ds[lo]) / (Ds[hi] - Ds[lo] || 1);
      return Math.exp(las[lo] + (las[hi] - las[lo]) * f);
    },
    Dt(t) { return API.D(API.aOfT(t)); },
    // Background temperature T = T0 / a. Always positive for finite a; never 0.
    TofA(a) { return P.T0 / a; },
    // Cosmological redshift z = 1/a - 1. Negative in the future for a > 1 (not clamped to zero).
    z(a) { return 1 / a - 1; },
    H0() { return P.H0; },
    ageNow: 0,

    // ---- authoritative mutable state ----
    get a() { return state.a; },
    get t() { return state.t; },
    get a_max() { return state.a_max; },
    get horizonReached() { return state.a >= state.a_max; },

    // Set the authoritative state (called by the presentation controller when advancing,
    // and by the extend-horizon operation). Time and scale factor stay consistent.
    setState(a, t) {
      if (a <= 0) throw new Error("scale factor must be positive");
      const aMax = state.a_max;
      const aa = Math.min(Math.max(a, state.a0), aMax);
      state.a = aa;
      state.t = t !== undefined ? t : API.tOfA(aa);
      if (state.a >= aMax) state.t = API.tOfA(aMax);
    },

    // Extend the validated future horizon to a later finite value and continue from
    // the current (t, a). Returns a status record. Never promises infinite runtime.
    extendHorizon(new_a_max) {
      if (!state.built) build();
      const next = Number(new_a_max);
      if (!Number.isFinite(next) || next <= state.a_max) {
        return { ok: false, error: "new horizon must be finite and greater than the current horizon" };
      }
      const curA = state.a, curT = state.t;
      state.a_max = next;
      build(); // rebuild table to the new horizon, re-syncing (t, a)
      // Clamp current state into the new table (should already be within).
      if (curA > next) { state.a = next; state.t = API.tOfA(next); }
      else { state.a = curA; state.t = curT; }
      return {
        ok: true,
        a_max: state.a_max,
        t_Gyr: state.t / (1e9 * YEAR),
        horizon_reached: state.a >= state.a_max,
        message: "Horizon extended to a=" + state.a_max.toFixed(2) + "; simulation can continue.",
      };
    },

    // Whether advancing further in a would leave the validated domain.
    atBoundary() { return state.a >= state.a_max; },
  });

  // ---- epochs (labels for the HUD) --------------------------------------------------------
  // start time in seconds; the epoch named at time t is the last one whose start <= t
  const EPOCHS = [
    { t: 0,        name: "Planck epoch",               text: "Spacetime itself is quantum. All forces are one." },
    { t: 1e-36,    name: "Cosmic inflation",           text: "Space expands by a factor of 10²⁶ or more in a fraction of a second; quantum ripples become the seeds of every future galaxy." },
    { t: 1e-32,    name: "Quark–gluon plasma",     text: "A seething soup of free quarks, gluons and leptons, hotter than 10²⁰ K." },
    { t: 1e-5,     name: "Hadron epoch",               text: "Quarks condense into protons and neutrons." },
    { t: 1,        name: "Lepton epoch",               text: "Neutrinos decouple; electrons and positrons annihilate." },
    { t: 10,       name: "Big Bang nucleosynthesis",   text: "Protons and neutrons fuse into hydrogen, helium-4 and traces of lithium." },
    { t: 1200,     name: "Photon epoch",               text: "An opaque, glowing fog of plasma. Sound waves ripple through it." },
    { t: 3.8e5 * YEAR, name: "Recombination",             text: "At about 3,000 K electrons join nuclei. The universe turns transparent and the cosmic microwave background is released." },
    { t: 5e6 * YEAR,  name: "The Dark Ages",              text: "No stars yet. Gravity quietly gathers dark matter and gas into a web of filaments." },
    { t: 1.5e8 * YEAR, name: "Cosmic Dawn",               text: "The first stars ignite and the first galaxies assemble; ultraviolet light begins to re-ionise the gas." },
    { t: 1e9 * YEAR,  name: "Galaxy assembly",            text: "Galaxies merge and grow along the filaments; star formation climbs toward its peak." },
    { t: 4e9 * YEAR,  name: "Cosmic noon",                text: "Star formation peaks, then slowly declines as the gas supply thins." },
    { t: 9e9 * YEAR,  name: "Dark-energy era",            text: "Cosmic expansion begins to accelerate. Structure growth stalls." },
    { t: 13.78e9 * YEAR, name: "Present day",             text: "13.8 billion years after the Big Bang." },
    { t: 15e9 * YEAR, name: "The far future",              text: "Expansion keeps accelerating. Star formation fades; massive stars die." },
    { t: 30e9 * YEAR, name: "Trillion-year future",        text: "Star formation ends; the last red dwarfs fade. Galaxies drift beyond the observable horizon." },
    { t: 100e9 * YEAR, name: "Dark-energy era",            text: "The expansion approaches a de Sitter phase. Bound structures remain isolated in an ever-larger void." },
  ];
  API.EPOCHS = EPOCHS;
  API.epochAt = function (t) { let e = EPOCHS[0]; for (const x of EPOCHS) if (t >= x.t) e = x; return e; };
  API.epochIndex = function (t) { let k = 0; EPOCHS.forEach((x, i) => { if (t >= x.t) k = i; }); return k; };

  // ---- presentation-only timeline map (legacy scrubber). NOT the physics timebase.
  // Maps a normalized s in [0,1] to cosmic time within the *current* validated horizon.
  // This is only for the UI scrubber; the authoritative timebase is the background clock above.
  API.NOW_S = 0.5; // presentation reference point (today, a=1) within the scrubber range.
  API.tOfS = function (s) {
    s = U.clamp(s, 0, 1);
    // Map s linearly onto the log-time range of the current table: t(a0) -> t(a_max).
    const t0 = state.built ? state.logt && Math.exp(state.logt[0]) : API.tOfA(state.a0);
    const t1 = state.built ? Math.exp(state.logt[state.n]) : API.tOfA(state.a_max);
    const lt0 = Math.log(Math.max(t0, 1e-44)), lt1 = Math.log(Math.max(t1, 1e-44));
    return Math.exp(lt0 + s * (lt1 - lt0));
  };
  API.sOfT = function (t) {
    const t0 = state.built ? Math.exp(state.logt[0]) : API.tOfA(state.a0);
    const t1 = state.built ? Math.exp(state.logt[state.n]) : API.tOfA(state.a_max);
    const lt0 = Math.log(Math.max(t0, 1e-44)), lt1 = Math.log(Math.max(t1, 1e-44));
    const lt = Math.log(Math.max(t, 1e-44));
    return U.clamp((lt - lt0) / (lt1 - lt0), 0, 1);
  };
  API.TNOW = API.tOfA(1);

  // ---- visual state as a pure function of cosmic time -------------------------------------
  API.visual = function (t) {
    const a = API.aOfT(t), D = API.D(a), T = API.TofA(a);
    const L = Math.log10(Math.max(t, 1e-44));
    const boxScale = 0.012 + 0.988 * U.smooth(-37.5, -31.0, L);                    // the region inflates out of a point
    const fogT = Math.min(T, 4e4);                                                  // colour temperature of the glowing plasma
    const rec = Math.exp(-Math.max(0, L - 13.1) * 6.5);                             // plasma goes dark after recombination
    const plasma = (0.22 + 0.5 * U.smooth(-43, -33, L)) * rec;                      // plasma brightness
    const jitter = 0.055 * (1 - U.smooth(-6, 13.0, L)) * U.smooth(-42, -36.5, L);   // thermal chaos, calms toward recombination
    const quantum = U.smooth(-37, -31, L);                                          // visibility of the primordial ripples
    const dvis = Math.max(D, 0.2 * quantum);                                       // contrast exaggeration while D is tiny
    const gas = U.smooth(15.2, 16.6, L) * 0.55 + U.smooth(16.6, 17.4, L) * 0.45;    // web gas glows after cosmic dawn
    return { t, L, a, z: 1 / a - 1, D, T, boxScale, fogT, plasma, jitter, quantum, dvis, gas };
  };

  // star formation history (relative), peaks near cosmic noon (z ~ 2, t ~ 3.4 Gyr), Madau & Dickinson-like shape
  API.sfrRel = function (t) {
    const a = API.aOfT(t), z = 1 / a - 1;
    return Math.pow(1 + z, 2.7) / (1 + Math.pow((1 + z) / 2.9, 5.6));
  };

  build();
})(typeof window !== "undefined" ? window : globalThis);
