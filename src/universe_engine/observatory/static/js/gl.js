/* gl.js - WebGL2 helpers: programs, render targets and the HDR post-processing chain (bloom + tonemap). */
(function (g) {
  "use strict";
  const U = g.U;

  const FS_TRI_VS = `#version 300 es
  out vec2 vUv;
  void main(){ vec2 p = vec2((gl_VertexID<<1)&2, gl_VertexID&2); vUv = p; gl_Position = vec4(p*2.0-1.0, 0.0, 1.0); }`;

  class Program {
    constructor(gl, vs, fs, name) {
      this.gl = gl; this.name = name || "program"; this.locs = {};
      const mk = (type, src) => {
        const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s);
        if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
          const log = gl.getShaderInfoLog(s);
          const lines = src.split("\n").map((l, i) => (i + 1) + ": " + l).join("\n");
          console.error("[" + this.name + "] shader compile error:\n" + log + "\n" + lines.slice(0, 6000));
          throw new Error(this.name + ": " + log);
        }
        return s;
      };
      const v = mk(gl.VERTEX_SHADER, vs), f = mk(gl.FRAGMENT_SHADER, fs);
      const p = gl.createProgram(); gl.attachShader(p, v); gl.attachShader(p, f); gl.linkProgram(p);
      if (!gl.getProgramParameter(p, gl.LINK_STATUS)) { const log = gl.getProgramInfoLog(p); console.error("[" + this.name + "] link error: " + log); throw new Error(this.name + ": " + log); }
      gl.deleteShader(v); gl.deleteShader(f);
      this.prog = p;
    }
    use() { this.gl.useProgram(this.prog); return this; }
    loc(n) { let l = this.locs[n]; if (l === undefined) { l = this.gl.getUniformLocation(this.prog, n); this.locs[n] = l; } return l; }
    attrib(n) { return this.gl.getAttribLocation(this.prog, n); }
    f1(n, a) { this.gl.uniform1f(this.loc(n), a); return this; }
    f2(n, a, b) { this.gl.uniform2f(this.loc(n), a, b); return this; }
    f3(n, a, b, c) { this.gl.uniform3f(this.loc(n), a, b, c); return this; }
    f4(n, a, b, c, d) { this.gl.uniform4f(this.loc(n), a, b, c, d); return this; }
    v3(n, v) { this.gl.uniform3f(this.loc(n), v[0], v[1], v[2]); return this; }
    i1(n, a) { this.gl.uniform1i(this.loc(n), a); return this; }
    m3(n, arr) { this.gl.uniformMatrix3fv(this.loc(n), false, arr); return this; }
  }

  class RT {
    constructor(gl, w, h, fmt) {
      this.gl = gl; this.w = Math.max(1, w | 0); this.h = Math.max(1, h | 0);
      this.tex = gl.createTexture(); gl.bindTexture(gl.TEXTURE_2D, this.tex);
      const hdr = fmt !== "ldr";
      gl.texImage2D(gl.TEXTURE_2D, 0, hdr ? gl.RGBA16F : gl.RGBA8, this.w, this.h, 0, gl.RGBA, hdr ? gl.HALF_FLOAT : gl.UNSIGNED_BYTE, null);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      this.fbo = gl.createFramebuffer(); gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbo);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.tex, 0);
      this.ok = gl.checkFramebufferStatus(gl.FRAMEBUFFER) === gl.FRAMEBUFFER_COMPLETE;
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    }
    bind() { const gl = this.gl; gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbo); gl.viewport(0, 0, this.w, this.h); }
    free() { this.gl.deleteTexture(this.tex); this.gl.deleteFramebuffer(this.fbo); }
  }

  const PRE = "#version 300 es\nprecision highp float;precision highp int;\n";

  class Post {
    constructor(gl) {
      this.gl = gl; this.scene = null; this.mips = [];
      this.vao = gl.createVertexArray();
      this.down = new Program(gl, FS_TRI_VS, PRE + `
        uniform sampler2D uTex; uniform vec2 uTexel; uniform float uKaris; in vec2 vUv; out vec4 o;
        vec3 s(vec2 d){ return texture(uTex, vUv + d*uTexel).rgb; }
        float kw(vec3 c){ return 1.0/(1.0+dot(c, vec3(0.2126,0.7152,0.0722))); }
        void main(){
          vec3 a=s(vec2(-2,2)),b=s(vec2(0,2)),c=s(vec2(2,2)),d=s(vec2(-2,0)),e=s(vec2(0,0)),f=s(vec2(2,0)),g1=s(vec2(-2,-2)),h=s(vec2(0,-2)),i=s(vec2(2,-2));
          vec3 j=s(vec2(-1,1)),k=s(vec2(1,1)),l=s(vec2(-1,-1)),m=s(vec2(1,-1));
          vec3 r;
          if(uKaris>0.5){
            vec3 g0=(d+e+a+b)*0.25, g2=(b+c+e+f)*0.25, g3=(d+e+g1+h)*0.25, g4=(e+f+h+i)*0.25, g5=(j+k+l+m)*0.25;
            r = g0*kw(g0)*0.125 + g2*kw(g2)*0.125 + g3*kw(g3)*0.125 + g4*kw(g4)*0.125 + g5*kw(g5)*0.5;
            r /= (0.125*(kw(g0)+kw(g2)+kw(g3)+kw(g4))+0.5*kw(g5));
          } else r = e*0.125 + (a+c+g1+i)*0.03125 + (b+d+f+h)*0.0625 + (j+k+l+m)*0.125;
          o = vec4(r,1.0);
        }`, "bloom-down");
      this.up = new Program(gl, FS_TRI_VS, PRE + `
        uniform sampler2D uTex; uniform vec2 uTexel; uniform float uRadius; in vec2 vUv; out vec4 o;
        void main(){
          vec2 t = uTexel*uRadius;
          vec3 r = texture(uTex, vUv+vec2(-1,1)*t).rgb + 2.0*texture(uTex, vUv+vec2(0,1)*t).rgb + texture(uTex, vUv+vec2(1,1)*t).rgb
                 + 2.0*texture(uTex, vUv+vec2(-1,0)*t).rgb + 4.0*texture(uTex, vUv).rgb + 2.0*texture(uTex, vUv+vec2(1,0)*t).rgb
                 + texture(uTex, vUv+vec2(-1,-1)*t).rgb + 2.0*texture(uTex, vUv+vec2(0,-1)*t).rgb + texture(uTex, vUv+vec2(1,-1)*t).rgb;
          o = vec4(r/16.0, 1.0);
        }`, "bloom-up");
      this.comp = new Program(gl, FS_TRI_VS, PRE + `
        uniform sampler2D uScene; uniform sampler2D uBloom; uniform float uBloomAmt; uniform float uExposure; uniform float uVig;
        uniform vec2 uRes; uniform float uTime; uniform float uSat; in vec2 vUv; out vec4 o;
        vec3 aces(vec3 x){ return clamp((x*(2.51*x+0.03))/(x*(2.43*x+0.59)+0.14), 0.0, 1.0); }
        float hash12(vec2 p){ vec3 p3 = fract(vec3(p.xyx)*.1031); p3 += dot(p3, p3.yzx+33.33); return fract((p3.x+p3.y)*p3.z); }
        void main(){
          vec2 uv = vUv;
          vec3 c = texture(uScene, uv).rgb + texture(uBloom, uv).rgb * uBloomAmt;
          c *= uExposure;
          float l = dot(c, vec3(0.2126,0.7152,0.0722));
          c = mix(vec3(l), c, uSat);
          c = aces(c);
          vec2 q = uv-0.5; float v = 1.0 - uVig*dot(q,q)*1.6; c *= v;
          c = pow(c, vec3(1.0/2.2));
          c += (hash12(gl_FragCoord.xy + fract(uTime)*100.0) - 0.5) / 255.0;
          o = vec4(c, 1.0);
        }`, "composite");
      this.blit = new Program(gl, FS_TRI_VS, PRE + `uniform sampler2D uTex; in vec2 vUv; out vec4 o; void main(){ o = texture(uTex, vUv); }`, "blit");
    }
    resize(w, h) {
      const gl = this.gl;
      if (this.scene) { this.scene.free(); this.mips.forEach((m) => m.free()); }
      this.scene = new RT(gl, w, h, this.fmt || "hdr");
      this.mips = [];
      let mw = w, mh = h;
      for (let i = 0; i < 6; i++) { mw = Math.max(2, mw >> 1); mh = Math.max(2, mh >> 1); this.mips.push(new RT(gl, mw, mh, this.fmt || "hdr")); }
    }
    tri() { const gl = this.gl; gl.bindVertexArray(this.vao); gl.drawArrays(gl.TRIANGLES, 0, 3); }
    tex(unit, tex) { const gl = this.gl; gl.activeTexture(gl.TEXTURE0 + unit); gl.bindTexture(gl.TEXTURE_2D, tex); }
    render(opt, canvasW, canvasH) {
      const gl = this.gl;
      gl.disable(gl.BLEND); gl.disable(gl.DEPTH_TEST);
      // downsample chain
      let src = this.scene;
      for (let i = 0; i < this.mips.length; i++) {
        const dst = this.mips[i]; dst.bind();
        this.down.use().i1("uTex", 0).f2("uTexel", 1 / src.w, 1 / src.h).f1("uKaris", i === 0 ? 1 : 0);
        this.tex(0, src.tex); this.tri(); src = dst;
      }
      // upsample + accumulate
      gl.enable(gl.BLEND); gl.blendFunc(gl.ONE, gl.ONE);
      for (let i = this.mips.length - 1; i > 0; i--) {
        const dst = this.mips[i - 1]; dst.bind();
        this.up.use().i1("uTex", 0).f2("uTexel", 1 / this.mips[i].w, 1 / this.mips[i].h).f1("uRadius", opt.bloomRadius || 1.0);
        this.tex(0, this.mips[i].tex); this.tri();
      }
      gl.disable(gl.BLEND);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null); gl.viewport(0, 0, canvasW, canvasH);
      this.comp.use().i1("uScene", 0).i1("uBloom", 1).f1("uBloomAmt", opt.bloom).f1("uExposure", opt.exposure).f1("uVig", opt.vignette)
        .f2("uRes", canvasW, canvasH).f1("uTime", opt.time || 0).f1("uSat", opt.saturation === undefined ? 1 : opt.saturation);
      this.tex(0, this.scene.tex); this.tex(1, this.mips[0].tex); this.tri();
    }
  }

  U.GL = {
    Program, RT, Post, PRE, FS_TRI_VS,
    create(canvas) {
      const gl = canvas.getContext("webgl2", { antialias: false, alpha: false, depth: false, stencil: false, powerPreference: "high-performance", preserveDrawingBuffer: true });
      if (!gl) throw new Error("WebGL2 is not available in this browser.");
      const cbf = gl.getExtension("EXT_color_buffer_float"), half = gl.getExtension("EXT_color_buffer_half_float");
      gl.getExtension("OES_texture_float_linear");
      return { gl, hdr: !!(cbf || half) };
    },
  };
})(typeof window !== "undefined" ? window : globalThis);
