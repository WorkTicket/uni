/* layers.js - render layers. Each layer owns its GL buffers and draws itself into the HDR scene target. */
(function (g) {
  "use strict";
  const U = g.U;

  // helper: set the shared camera uniforms of a program. `camUnits` is the camera position in the layer's frame
  // (double precision, in the layer's unit), split into hi/lo floats so the subtraction in the shader is exact.
  function setCam(prog, ctx, camUnits) {
    const hi = [Math.fround(camUnits[0]), Math.fround(camUnits[1]), Math.fround(camUnits[2])];
    prog.m3("uV", ctx.V).f2("uProj", ctx.proj[0], ctx.proj[1]).f1("uPxScale", ctx.pxScale)
      .f3("uCamHi", hi[0], hi[1], hi[2]).f3("uCamLo", camUnits[0] - hi[0], camUnits[1] - hi[1], camUnits[2] - hi[2]);
  }
  U.setCam = setCam;

  // ===========================================================================================
  class WebLayer {
    constructor(R) {
      this.R = R; const gl = R.gl;
      this.prog = new U.GL.Program(gl, U.SH.webVS, U.SH.webFS, "web");
      this.vao = gl.createVertexArray(); this.buf = null; this.count = 0;
      this.lineProg = new U.GL.Program(gl, U.SH.lineVS, U.SH.lineFS, "box-lines");
      const c = [-.5, -.5, -.5, .5, -.5, -.5, .5, .5, -.5, -.5, .5, -.5, -.5, -.5, .5, .5, -.5, .5, .5, .5, .5, -.5, .5, .5];
      const e = [0, 1, 1, 2, 2, 3, 3, 0, 4, 5, 5, 6, 6, 7, 7, 4, 0, 4, 1, 5, 2, 6, 3, 7], v = [];
      e.forEach((i) => v.push(c[i * 3], c[i * 3 + 1], c[i * 3 + 2]));
      this.boxVao = gl.createVertexArray(); gl.bindVertexArray(this.boxVao);
      this.boxBuf = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, this.boxBuf); gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(v), gl.STATIC_DRAW);
      const la = this.lineProg.attrib("aPos"); gl.enableVertexAttribArray(la); gl.vertexAttribPointer(la, 3, gl.FLOAT, false, 0, 0);
      this.boxCount = e.length;
    }
    upload(web) {
      const gl = this.R.gl, n = web.count, data = new Float32Array(n * 9);
      for (let i = 0; i < n; i++) {
        data[i * 9] = web.q[i * 3]; data[i * 9 + 1] = web.q[i * 3 + 1]; data[i * 9 + 2] = web.q[i * 3 + 2];
        data[i * 9 + 3] = web.psi[i * 3]; data[i * 9 + 4] = web.psi[i * 3 + 1]; data[i * 9 + 5] = web.psi[i * 3 + 2];
        data[i * 9 + 6] = web.lam[i * 3]; data[i * 9 + 7] = web.lam[i * 3 + 1]; data[i * 9 + 8] = web.lam[i * 3 + 2];
      }
      gl.bindVertexArray(this.vao);
      if (this.buf) gl.deleteBuffer(this.buf);
      this.buf = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, this.buf); gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
      const names = ["aQ", "aPsi", "aLam"];
      names.forEach((nm, k) => { const a = this.prog.attrib(nm); gl.enableVertexAttribArray(a); gl.vertexAttribPointer(a, 3, gl.FLOAT, false, 36, k * 12); });
      this.count = n; this.boxMpc = web.L / U.Cosmo.P.h * 1.0; // Mpc
      this.boxMpc = web.L / (U.Cosmo.P.H0 / 100);
      this.spacing = this.boxMpc / Math.cbrt(n);
    }
    // ctx.camMpc: camera in the universe frame (Mpc); ctx.vis: Cosmo.visual(t); fade: 0..1
    draw(ctx, fade) {
      if (!this.count || fade <= 0.001) return;
      const gl = this.R.gl, vis = ctx.vis, p = this.prog.use();
      setCam(p, ctx, ctx.camMpc);
      const fogRGB = U.blackbodyRGB(vis.fogT);
      // overlap-normalised gain: keep exposure roughly constant across particle budgets
      const gain = 262144 / this.count;
      p.f1("uD", vis.D).f1("uDvis", vis.dvis).f1("uBox", this.boxMpc).f1("uBoxScale", vis.boxScale).f1("uJit", vis.jitter)
        .f1("uTime", ctx.time).f1("uSize", this.spacing * ctx.webSize).f1("uPlasma", vis.plasma * ctx.webPlasmaGain).f1("uGas", vis.gas * ctx.webGasGain)
        .f1("uGain", gain).f1("uFade", fade).f1("uContrast", 1.0 + 2.4 * Math.min(1, vis.quantum)).f3("uFogCol", fogRGB[0], fogRGB[1], fogRGB[2]);
      gl.enable(gl.BLEND); gl.blendFunc(gl.ONE, gl.ONE);
      gl.bindVertexArray(this.vao); gl.drawArrays(gl.POINTS, 0, this.count);
      // frame of the simulated region
      const l = this.lineProg.use(); setCam(l, ctx, ctx.camMpc);
      const boxA = 0.16 * fade * (0.25 + 0.75 * U.smooth(-38, -33, vis.L));
      l.f1("uScale", this.boxMpc * vis.boxScale).f3("uCenter", 0, 0, 0).f4("uColor", 0.30 * boxA, 0.50 * boxA, 0.85 * boxA, 1);
      gl.bindVertexArray(this.boxVao); gl.drawArrays(gl.LINES, 0, this.boxCount);
    }
  }

  // ------------------------------------------------------------------
  // ScientificParticleLayer: renders actual PM3D particle positions from the
  // numerical simulation state (SIMULATED classification, not illustrative).
  // Positions are comoving box fractions; the shader scales by a to get proper.
  // ------------------------------------------------------------------
  class ScientificParticleLayer {
    constructor(R) {
      this.R = R; const gl = R.gl;
      this.prog = new U.GL.Program(gl, U.SH.simPointVS, U.SH.simPointFS, "sim-points");
      this.vao = gl.createVertexArray(); this.buf = null; this.count = 0;
      this.boxMpc = 100; this.a = 1;
      gl.bindVertexArray(this.vao);
      const la = this.prog.attrib("aPos"); gl.enableVertexAttribArray(la);
      gl.vertexAttribPointer(la, 3, gl.FLOAT, false, 0, 0);
    }
    upload(state) {
      const gl = this.R.gl;
      const x = state.x_comoving;      // (N,3) box fraction
      const n = x.length;
      const data = new Float32Array(n * 3);
      for (let i = 0; i < n; i++) {
        data[i * 3] = x[i][0]; data[i * 3 + 1] = x[i][1]; data[i * 3 + 2] = x[i][2];
      }
      gl.bindVertexArray(this.vao);
      if (this.buf) gl.deleteBuffer(this.buf);
      this.buf = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, this.buf);
      gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
      const la = this.prog.attrib("aPos"); gl.enableVertexAttribArray(la);
      gl.vertexAttribPointer(la, 3, gl.FLOAT, false, 0, 0);
      this.count = n;
      this.boxMpc = 100;  // reference box, from config
      this.a = state.a;
    }
    draw(ctx, fade) {
      if (!this.count || fade <= 0.001) return;
      const gl = this.R.gl, p = this.prog.use();
      setCam(p, ctx, ctx.camMpc);
      p.f1("uA", this.a).f1("uBox", this.boxMpc).f1("uFade", fade)
        .f1("uTime", ctx.time).f1("uSize", 2.0);
      gl.enable(gl.BLEND); gl.blendFunc(gl.ONE, gl.ONE);
      gl.bindVertexArray(this.vao); gl.drawArrays(gl.POINTS, 0, this.count);
    }
  }

  U.Layers = { WebLayer, ScientificParticleLayer };
})(typeof window !== "undefined" ? window : globalThis);
