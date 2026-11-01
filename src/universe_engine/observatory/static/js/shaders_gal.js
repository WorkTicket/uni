/* shaders_gal.js - galaxy sprites (procedural spiral / elliptical / irregular / cluster images), star points, nebulae. */
(function (g) {
  "use strict";
  const U = g.U, PRE = U.GL.PRE, COMMON = U.GLSL.COMMON;

  const NOISE = `
  float vn(vec2 p){ vec2 i=floor(p), f=fract(p); f=f*f*(3.0-2.0*f);
    float a=hash13(vec3(i,1.0)), b=hash13(vec3(i+vec2(1,0),1.0)), c=hash13(vec3(i+vec2(0,1),1.0)), d=hash13(vec3(i+vec2(1,1),1.0));
    return mix(mix(a,b,f.x), mix(c,d,f.x), f.y); }
  float fbm2(vec2 p){ float s=0.0, a=0.5; for(int i=0;i<5;i++){ s+=a*vn(p); p=p*2.03+vec2(17.1,9.7); a*=0.5; } return s; }
  vec3 bbRGB(float T){
    T = clamp(T, 1000.0, 40000.0)/100.0; float r,g,b;
    r = T<=66.0 ? 1.0 : clamp(1.2929*pow(T-60.0,-0.1332), 0.0, 1.0);
    g = T<=66.0 ? clamp(0.3901*log(T)-0.6318, 0.0, 1.0) : clamp(1.1298*pow(T-60.0,-0.0755), 0.0, 1.0);
    b = T>=66.0 ? 1.0 : (T<=19.0 ? 0.0 : clamp(0.5432*log(T-10.0)-1.1962, 0.0, 1.0));
    return pow(vec3(r,g,b), vec3(2.2));
  }`;
  U.GLSL.NOISE = NOISE;

  // ---------------- galaxy sprites (instanced billboards) ----------------
  U.SH.galVS = PRE + COMMON + `
  in vec3 iQ; in vec3 iPsi; in vec3 iN; in vec4 iP; in vec4 iX; in float iLum;
  uniform int uMode; uniform float uD, uBox, uTGyr, uMinPx, uFade, uSkipMul; uniform int uSkip;
  out vec2 vUv; flat out vec3 vN; flat out vec4 vP; flat out vec4 vX; flat out vec4 vE;  // vE: radPx, age, lumGrowth, fade
  void main(){
    vec2 c = vec2((gl_VertexID & 1), (gl_VertexID >> 1)) * 2.0 - 1.0;
    float age = uTGyr - iP.z;
    float growth = 1.0 - exp(-max(age, 0.0)/0.6);
    float fade = uFade * (gl_InstanceID == uSkip ? uSkipMul : 1.0);
    if (age <= 0.0 || fade <= 0.001){ gl_Position = vec4(2.0,2.0,2.0,1.0); vUv=vec2(0); vN=vec3(0,1,0); vP=iP; vX=iX; vE=vec4(0); return; }
    vec3 pos;
    if (uMode == 0){ vec3 x = fract(iQ + uD*iPsi); pos = (x - 0.5)*uBox; } else pos = iQ;
    vec3 v = uV*((pos - uCamHi) - uCamLo);
    float w = -v.z;
    if (w <= 1e-12){ gl_Position = vec4(2.0,2.0,2.0,1.0); vUv=vec2(0); vN=vec3(0,1,0); vP=iP; vX=iX; vE=vec4(0); return; }
    float radPx = iP.x / w * uPxScale;
    float halfPx = max(radPx*1.3, uMinPx);
    float hs = halfPx * w / uPxScale;
    vec3 vv = v + vec3(c*hs, 0.0);
    gl_Position = vec4(vv.x*uProj.x, vv.y*uProj.y, 0.0, w);
    vUv = c * (halfPx / max(radPx, 1e-6));
    vN = normalize(uV*iN); vP = iP; vX = iX; vE = vec4(radPx, age, growth*iLum, fade);
  }`;

  U.SH.galFS = PRE + COMMON + NOISE + `
  in vec2 vUv; flat in vec3 vN; flat in vec4 vP; flat in vec4 vX; flat in vec4 vE; out vec4 o;
  uniform float uBoost;
  void main(){
    float radPx = max(vE.x, 1e-3), age = vE.y, lum = vE.z, fade = vE.w;
    if (lum <= 0.0) discard;
    float kind = vP.y, tau = vP.w, seed = vX.x, arms = vX.y, pitch = vX.z, bulgeF = vX.w;
    float youth = exp(-max(0.0, age-0.25)/max(tau,0.2));
    vec3 oldC = vec3(1.0,0.74,0.46), blueC = vec3(0.50,0.70,1.15);
    float I0 = 1.15*lum;
    // unresolved: soft sparkle with distance-independent brightness
    vec2 px = vUv*radPx;
    float blob = exp(-dot(px,px)/(2.0*1.1*1.1));
    vec3 blobC = mix(oldC, blueC, 0.55*youth) * (kind > 3.5 ? 0.9 : 1.0);
    float res = smoothstep(1.8, 5.5, radPx);
    float rr = length(vUv);
    vec3 col = vec3(0.0);
    float S0 = I0 * clamp(pow(3.0/radPx, 0.62), 0.10, 1.0);
    if (res > 0.0 && rr < 1.4){
      if (kind > 3.5){                       // cluster glow: hot gas + member sparkles
        float core = pow(1.0 + (rr/0.22)*(rr/0.22), -1.35);
        vec2 gp = vUv*18.0 + seed*50.0; vec2 cell = floor(gp); float hh = hash13(vec3(cell, seed));
        float sp = (hh > 0.80) ? exp(-dot(fract(gp)-0.5+ (hash13(vec3(cell,3.0))-0.5)*0.5, fract(gp)-0.5 + (hash13(vec3(cell,3.0))-0.5)*0.5)*38.0) * (0.4+hh) * core * 3.0 : 0.0;
        col = (vec3(0.95,0.62,0.35)*core*0.7 + vec3(1.0,0.82,0.6)*sp) * S0 * (1.0 - smoothstep(0.9,1.3,rr));
      } else {
        // plane coordinates from the screen offset (parallel projection of the disc plane)
        vec3 n = vN; vec3 a0 = abs(n.y) < 0.9 ? vec3(0,1,0) : vec3(1,0,0);
        vec3 e1 = normalize(cross(a0, n)); vec3 e2 = cross(n, e1);
        float det = e1.x*e2.y - e2.x*e1.y; float sg = det < 0.0 ? -1.0 : 1.0; float dc = sg*max(abs(det), 0.09);
        vec2 ab = vec2(e2.y*vUv.x - e2.x*vUv.y, -e1.y*vUv.x + e1.x*vUv.y)/dc;
        float r = length(ab), ph = atan(ab.y, ab.x);
        float nz = abs(n.z);
        if (kind > 1.5 && kind < 2.5){        // elliptical: de Vaucouleurs profile with a hashed ellipticity
          float q = 0.62 + 0.35*fract(seed*7.0), th = seed*6.28;
          vec2 u = vec2(cos(th)*vUv.x + sin(th)*vUv.y, -sin(th)*vUv.x + cos(th)*vUv.y);
          float rs = length(vec2(u.x, u.y/q));
          float I = exp(-7.67*(pow(max(rs,1e-3)/0.30, 0.25) - 1.0)) * 0.35;
          col = vec3(1.0,0.76,0.5) * mix(1.0, 0.8, youth) * I * S0 * 2.2 * (1.0 - smoothstep(0.8,1.2,rs));
        } else if (kind > 2.5){               // irregular
          float f = fbm2(vUv*4.0 + seed*40.0); float I = pow(f, 2.4)*2.2*exp(-rr*2.2);
          col = mix(vec3(1.0,0.8,0.6), blueC, 0.35+0.5*youth) * I * S0 * (1.0 - smoothstep(0.8, 1.2, rr));
        } else {                              // spiral / barred spiral
          float armAng = log(max(r, 0.06)/0.26)/tan(max(pitch,0.12)) + seed*6.28;
          float arm = 0.5 + 0.5*cos(arms*(ph - armAng));
          if (kind > 0.5){ float bl = 0.30; float ba = ph - seed*6.28; float bar = exp(-pow(r/bl, 2.0)*0.9) * pow(abs(cos(ba)), 6.0); arm = max(arm*smoothstep(0.1,0.35,r), bar*0.0); arm += bar*1.2; }
          float knots = fbm2(ab*9.0 + seed*30.0);
          float disc = exp(-r*3.2) * (0.30 + 1.25*pow(arm, 2.0)) * (0.55 + 0.9*knots) * (1.0 - smoothstep(0.95, 1.18, r));
          float young = pow(arm, 3.0) * exp(-r*2.2) * knots * youth * 1.6;
          // vertical thickness: blur perpendicular to the major axis when seen edge-on
          vec2 pdir = normalize(n.xy + vec2(1e-5)); float minor = dot(vUv, pdir);
          float thick = exp(-pow(minor/(0.04 + 0.5*nz), 2.0)) ;
          float pf = mix(thick, 1.0, smoothstep(0.15, 0.6, nz));
          float lane = 1.0 - 0.72*pow(1.0-nz, 1.5)*exp(-pow(minor/0.035, 2.0));
          float rb = length(vUv);
          float bulge = bulgeF*(1.7*exp(-rb/0.06) + 0.45*exp(-rb/0.22));
          vec3 dc3 = mix(oldC, blueC, clamp(0.85*youth,0.0,1.0)) * disc + vec3(0.55,0.8,1.2)*young + vec3(1.0,0.35,0.55)*young*0.35;
          col = (dc3*pf*1.15 + vec3(1.0,0.76,0.5)*bulge) * lane * S0 * 1.1;
        }
      }
    }
    vec3 outc = mix(blobC*blob*I0*0.55, col, res);
    outc *= fade * uBoost;
    o = vec4(outc, 1.0);
  }`;

  // ---------------- star points (cloud + local field) ----------------
  U.SH.starVS = PRE + COMMON + NOISE + `
  in vec3 aPos; in vec4 aAttr;
  uniform float uTGyr, uSpin, uR, uW, uK, uExpo, uFade, uNearPc, uUnitPc, uSkipIdx, uLexp, uPsz;
  uniform vec3 uOrigin;
  out vec3 vCol;
  void main(){
    float age = uTGyr - aAttr.z, life = aAttr.w;
    vCol = vec3(0.0); gl_Position = vec4(2.0,2.0,2.0,1.0); gl_PointSize = 1.0;
    if (age < 0.0 || age > life || uFade <= 0.001 || float(gl_VertexID) == uSkipIdx) return;
    vec3 p = aPos + uOrigin;
    if (uSpin != 0.0){ float r = length(p.xz); float th = uSpin/(0.25 + r/uR); float c = cos(th), s = sin(th); p = vec3(c*p.x - s*p.z, p.y, s*p.x + c*p.z); }
    vec3 rel = (p - uCamHi) - uCamLo;
    vec4 cp = toClip(rel);
    if (cp.w <= 1e-12) return;
    float dpc = length(rel)*uUnitPc;
    float f = clamp(age/life, 0.0, 1.0), k = smoothstep(0.88, 1.0, f);
    float L = pow(10.0, aAttr.x*uLexp) * (1.0 + 60.0*k);
    float T = aAttr.y * pow((1.0+60.0*k)/pow(1.0+40.0*k, 2.0), 0.25);
    float born = smoothstep(0.0, 0.02, age);
    float flux = L * uW / max(dpc*dpc, 1e-6);
    float I = uExpo * asinh(flux * uK) * born;
    float nearF = uNearPc > 0.0 ? smoothstep(0.35*uNearPc, uNearPc, dpc) : 1.0;
    vCol = bbRGB(T) * I * uFade * nearF * 1.0;
    gl_Position = cp;
    gl_PointSize = clamp(1.0 + uPsz*log2(1.0 + flux*uK*4.0), 1.0, 14.0);
  }`;
  U.SH.starFS = PRE + `in vec3 vCol; out vec4 o; void main(){ vec2 d = gl_PointCoord*2.0-1.0; float r2 = dot(d,d); if(r2>1.0) discard; o = vec4(vCol*exp(-3.0*r2), 1.0); }`;

  // ---------------- nebulae / dust (soft billboards) ----------------
  U.SH.nebVS = PRE + COMMON + `
  in vec3 aPos; in vec4 aNeb;   // size(kpc), type(0 HII,1 blue,2 dust), birthGyr, seed
  uniform float uTGyr, uSpin, uR, uFade, uPass;
  out vec4 vC;
  void main(){
    gl_Position = vec4(2.0,2.0,2.0,1.0); gl_PointSize = 1.0; vC = vec4(0.0);
    float age = uTGyr - aNeb.z; if (age < 0.0 || uFade <= 0.001) return;
    bool dust = aNeb.y > 1.5; if ((uPass < 0.5) == dust) return;
    vec3 p = aPos;
    if (uSpin != 0.0){ float r = length(p.xz); float th = uSpin/(0.25 + r/uR); float c = cos(th), s = sin(th); p = vec3(c*p.x - s*p.z, p.y, s*p.x + c*p.z); }
    vec4 cp = toClip((p - uCamHi) - uCamLo);
    if (cp.w <= 1e-12) return;
    float px = 2.0*aNeb.x/cp.w*uPxScale;
    if (px < 1.5) return;
    float dist = length((p - uCamHi) - uCamLo);
    float nearF = smoothstep(0.5*aNeb.x, 2.6*aNeb.x, dist);
    gl_Position = cp; gl_PointSize = clamp(px, 2.0, 220.0);
    float grow = smoothstep(0.0, 0.8, age) * nearF;
    float edge = clamp(px/6.0, 0.0, 1.0);
    if (aNeb.y < 0.5) vC = vec4(vec3(1.0,0.28,0.45)*0.09*edge, 0.0)*grow*uFade;
    else if (aNeb.y < 1.5) vC = vec4(vec3(0.35,0.55,1.0)*0.05*edge, 0.0)*grow*uFade;
    else vC = vec4(0.0,0.0,0.0, 0.12*edge*grow*uFade);
  }`;
  U.SH.nebFS = PRE + `in vec4 vC; out vec4 o; void main(){ vec2 d = gl_PointCoord*2.0-1.0; float r2 = dot(d,d); if(r2>1.0) discard; float a = exp(-3.2*r2); o = vec4(vC.rgb*a, vC.a*a); }`;
})(typeof window !== "undefined" ? window : globalThis);
