/* shaders.js - GLSL sources shared by the render layers. No depth buffer is used anywhere: layers are painted
   far-to-near and every position is camera-relative (high/low float split of a double-precision camera). */
(function (g) {
  "use strict";
  const U = g.U;
  const PRE = U.GL.PRE;

  const COMMON = `
  uniform mat3 uV;          // world -> view rotation
  uniform vec2 uProj;       // (f/aspect, f)
  uniform float uPxScale;   // pixels per unit at distance 1  (= 0.5*H*f)
  uniform vec3 uCamHi; uniform vec3 uCamLo;
  vec4 toClip(vec3 rel){ vec3 v = uV*rel; return vec4(v.x*uProj.x, v.y*uProj.y, 0.0, -v.z); }
  float hash11(float p){ p = fract(p*.1031); p *= p+33.33; p *= p+p; return fract(p); }
  vec3 hash31(float p){ vec3 p3 = fract(vec3(p)*vec3(.1031,.1030,.0973)); p3 += dot(p3,p3.yzx+33.33); return fract((p3.xxy+p3.yzz)*p3.zyx); }
  float hash13(vec3 p3){ p3 = fract(p3*.1031); p3 += dot(p3, p3.zyx+31.32); return fract((p3.x+p3.y)*p3.z); }
  `;
  U.GLSL = { COMMON };

  // ----- cosmic web particles (Zel'dovich displacement evaluated on the GPU) -------------------
  U.SH = {};
  U.SH.webVS = PRE + COMMON + `
  in vec3 aQ; in vec3 aPsi; in vec3 aLam;
  uniform float uD, uDvis, uBox, uBoxScale, uJit, uTime, uSize, uPlasma, uGas, uGain, uFade, uContrast;
  uniform vec3 uFogCol;
  out vec3 vCol;
  vec3 gasRamp(float rho){
    // voids stay dark; filaments indigo -> violet; clusters orange -> white-gold
    vec3 c = vec3(0.0);
    c += vec3(0.12,0.20,0.70) * smoothstep(0.8, 2.6, rho);
    c += vec3(0.40,0.14,0.62) * smoothstep(2.5, 9.0, rho);
    c += vec3(0.85,0.40,0.10) * smoothstep(8.0, 30.0, rho);
    c += vec3(0.55,0.55,0.45) * smoothstep(25.0, 90.0, rho);
    return c;
  }
  void main(){
    float id = float(gl_VertexID);
    vec3 h = hash31(id*0.7311 + 3.17);
    vec3 x = aQ + uD*aPsi;
    float tt = uTime*(0.6 + h.x*1.4);
    x += uJit*vec3(sin(tt + h.y*40.0), sin(tt*1.13 + h.z*40.0), sin(tt*0.87 + h.x*40.0));
    x = fract(x);
    vec3 J = max(vec3(1.0) + uDvis*aLam, vec3(0.07));
    float rho = clamp(1.0/(J.x*J.y*J.z), 0.03, 90.0);
    vec3 pos = (x - 0.5)*uBox*uBoxScale;
    vec3 rel = (pos - uCamHi) - uCamLo;
    vec4 cp = toClip(rel);
    gl_Position = cp;
    float edge = smoothstep(0.0, 0.035, min(min(x.x,1.0-x.x), min(min(x.y,1.0-x.y), min(x.z,1.0-x.z))));
    float c = pow(rho, uContrast);
    vec3 col = uFogCol * uPlasma * (0.35 + 0.65*c) + gasRamp(rho) * uGas * pow(rho, 1.3);
    float sizeW = uSize*uBoxScale*clamp(pow(rho, -0.3), 0.3, 1.5);
    float px = 2.0*sizeW/max(cp.w, 1e-9)*uPxScale;
    gl_PointSize = clamp(px, 1.0, 72.0);
    float near = px > 20.0 ? pow(20.0/px, 1.7) : 1.0;  // sprites right in front of the camera must not flood the view
    vCol = col*uGain*uFade*edge*near*(px < 1.0 ? px*px : 1.0);
  }`;
  U.SH.webFS = PRE + `
  in vec3 vCol; out vec4 o;
  void main(){ vec2 d = gl_PointCoord*2.0-1.0; float r2 = dot(d,d); if(r2>1.0) discard; o = vec4(vCol*exp(-3.2*r2), 1.0); }`;

  // ----- box frame (line list) -----------------------------------------------------------------
  U.SH.lineVS = PRE + COMMON + `
  in vec3 aPos; uniform float uScale; uniform vec3 uCenter;
  void main(){ vec3 rel = (aPos*uScale + uCenter - uCamHi) - uCamLo; gl_Position = toClip(rel); }`;
  U.SH.lineFS = PRE + `uniform vec4 uColor; out vec4 o; void main(){ o = uColor; }`;
})(typeof window !== "undefined" ? window : globalThis);
