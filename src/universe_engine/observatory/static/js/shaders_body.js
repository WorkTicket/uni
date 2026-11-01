/* shaders_body.js - analytic sphere impostors for stars, rocky worlds, gas/ice giants and moons.
   Terrain is a pure function of the direction on the sphere, evaluated with a pixel-footprint-driven octave count,
   so detail keeps appearing as the camera descends (no textures, no tiles). */
(function (g) {
  "use strict";
  const U = g.U, PRE = U.GL.PRE;

  U.SH.bodyVS = `#version 300 es
  out vec2 vNdc;
  void main(){ vec2 p = vec2((gl_VertexID<<1)&2, gl_VertexID&2); vNdc = p*2.0-1.0; gl_Position = vec4(vNdc, 0.0, 1.0); }`;

  U.SH.bodyFS = PRE + `
  in vec2 vNdc; out vec4 o;
  uniform mat3 uVinv;            // view -> world
  uniform vec2 uProj; uniform float uPxScale;
  uniform vec3 uC;               // body centre relative to camera (world axes), metres
  uniform float uR;              // radius, metres
  uniform mat3 uBody;            // body -> world rotation (spin axis = body +Y)
  uniform vec3 uSun;             // direction from body to the star (world), unit
  uniform float uSunI;           // star irradiance scale at this body
  uniform int uType;             // 0 star, 1 rocky, 2 gas giant, 3 moon
  uniform float uSeed, uTime;
  uniform vec3 uPalA, uPalB, uPalC, uPalD;   // palettes (low / mid / high / extra)
  uniform vec4 uP0;              // x sea level, y ice latitude, z cloud cover, w relief
  uniform vec4 uP1;              // x atmosphere density, y atmosphere height (frac of R), z lava/emissive, w ring flag
  uniform vec3 uAtmCol;          // rayleigh-ish scattering colour
  uniform vec3 uStarCol; uniform float uStarT;
  uniform float uBiome;          // 0 temperate 1 ocean 2 desert 3 mars 4 cold 5 ice 6 lava 7 barren

  // ---- noise with analytic derivatives (value noise on a hashed lattice) ----
  vec3 h3(vec3 p){ p = vec3(dot(p,vec3(127.1,311.7,74.7)), dot(p,vec3(269.5,183.3,246.1)), dot(p,vec3(113.5,271.9,124.6))); return fract(sin(p)*43758.5453123); }
  float hs(vec3 p3){ p3 = fract(p3*0.1031); p3 += dot(p3, p3.zyx+31.32); return fract((p3.x+p3.y)*p3.z); }
  vec4 noised(vec3 x){
    vec3 i = floor(x), f = fract(x);
    vec3 u = f*f*f*(f*(f*6.0-15.0)+10.0), du = 30.0*f*f*(f*(f-2.0)+1.0);
    float a=hs(i+vec3(0,0,0)), b=hs(i+vec3(1,0,0)), c=hs(i+vec3(0,1,0)), d=hs(i+vec3(1,1,0));
    float e=hs(i+vec3(0,0,1)), ff=hs(i+vec3(1,0,1)), g=hs(i+vec3(0,1,1)), h=hs(i+vec3(1,1,1));
    float k0=a, k1=b-a, k2=c-a, k3=e-a, k4=a-b-c+d, k5=a-c-e+g, k6=a-b-e+ff, k7=-a+b+c-d+e-ff-g+h;
    return vec4(k0 + k1*u.x + k2*u.y + k3*u.z + k4*u.x*u.y + k5*u.y*u.z + k6*u.z*u.x + k7*u.x*u.y*u.z,
      du.x*(k1 + k4*u.y + k6*u.z + k7*u.y*u.z), du.y*(k2 + k5*u.z + k4*u.x + k7*u.z*u.x), du.z*(k3 + k6*u.x + k5*u.y + k7*u.x*u.y));
  }
  float vnoise(vec3 x){ return noised(x).x; }
  float fbm(vec3 p, float oct){
    float s=0.0, a=0.5, tot=0.0;
    for(int i=0;i<16;i++){ if(float(i) >= oct) break; float w = clamp(oct-float(i), 0.0, 1.0); s += a*w*vnoise(p); tot += a*w; p = p*2.02 + vec3(11.7,3.1,7.3); a *= 0.5; }
    return s/max(tot,1e-4);
  }
  // domain-warped continental height in [0,1]
  float terrain(vec3 n, float oct){
    vec3 p = n*1.7 + uSeed*13.37;
    float w = fbm(p*1.3 + 5.0, 4.0);
    float c = fbm(p + 2.5*vec3(w,-w,w*0.7), min(oct, 16.0));
    // ridged mountains where land is high
    float r = 0.0, a = 0.5; vec3 q = p*3.1;
    for(int i=0;i<12;i++){ if(float(i) >= oct-2.0) break; float v = 1.0 - abs(vnoise(q)*2.0-1.0); r += a*v*v; q = q*2.1 + 3.7; a *= 0.5; }
    float land = smoothstep(0.35, 0.65, c);
    return clamp(c*0.78 + r*0.55*land*uP0.w, 0.0, 1.0);
  }
  // crater field (cellular), 3 scales
  float craters(vec3 n, float oct){
    float h = 0.0;
    for(int s=0;s<5;s++){
      if(float(s)*1.7 > oct) break;
      float sc = 5.0*pow(2.35, float(s)); vec3 p = n*sc + uSeed*7.0; vec3 ip = floor(p), fp = fract(p);
      float best = 1e9, rad = 0.0;
      for(int x=-1;x<=1;x++) for(int y=-1;y<=1;y++) for(int z=-1;z<=1;z++){
        vec3 g = vec3(x,y,z); vec3 r = h3(ip+g+float(s)*31.7); vec3 d = g + r - fp; float dd = dot(d,d);
        if(dd < best && r.x > 0.45){ best = dd; rad = 0.18 + 0.3*r.y; }
      }
      float d = sqrt(best)/rad;
      float bowl = -smoothstep(1.0, 0.0, d)*0.9, rim = smoothstep(0.75, 1.0, d)*smoothstep(1.35, 1.0, d)*0.45;
      h += (bowl + rim) / pow(1.5, float(s));
    }
    return h*0.18;
  }
  // ground height used for shading, returns [0,1] and fills "land" flag
  float groundH(vec3 n, float oct){
    if(uBiome > 6.5) return clamp(0.5 + craters(n, oct) + (fbm(n*3.0+uSeed, min(oct,10.0))-0.5)*0.35, 0.0, 1.0);
    if(uBiome > 4.5 && uBiome < 5.5) return clamp(fbm(n*2.5 + uSeed, min(oct,14.0))*0.8 + 0.1, 0.0, 1.0);
    return terrain(n, oct);
  }
  vec3 tint(vec3 a, float s){ return a*(0.85 + 0.3*s); }

  vec3 skyTrans(float tau, vec3 beta){ return exp(-tau*beta); }

  // sphere intersection that stays precise when |c| >> R:  returns (t0,t1) or (-1,-1)
  vec2 isect(vec3 rd, vec3 c, float R){
    float tca = dot(rd, c); vec3 cr = cross(rd, c); float d2 = dot(cr,cr);
    float R2 = R*R; if(d2 > R2) return vec2(-1.0);
    float thc = sqrt(R2 - d2); return vec2(tca - thc, tca + thc);
  }

  void main(){
    vec3 rdv = normalize(vec3(vNdc.x/uProj.x, vNdc.y/uProj.y, -1.0));
    vec3 rd = uVinv*rdv;
    float R = uR;
    float Ratm = R*(1.0 + uP1.y);
    float Rext = (uP1.w > 0.5) ? R*2.45 : (uType==0 ? R : Ratm);
    vec3 col = vec3(0.0); float alpha = 0.0;
    float dist = length(uC);
    vec2 hit = isect(rd, uC, R);
    bool onBody = hit.x > 0.0 || (hit.x < 0.0 && hit.y > 0.0);
    float t0 = max(hit.x, 0.0);

    // pixel footprint in unit-sphere units
    float pxAng = 1.0/uPxScale;

    if(uType == 0){                        // ---- star ----
      if(!onBody) discard;
      vec3 pw = rd*t0; vec3 nW = normalize(pw - uC); float mu = clamp(dot(nW, -rd), 0.0, 1.0);
      vec3 nb = nW*uBody;
      float gran = fbm(nb*18.0 + vec3(uTime*0.02), 6.0), gran2 = fbm(nb*60.0 + uTime*0.05, 4.0);
      float spots = smoothstep(0.62, 0.78, fbm(nb*4.0 + uSeed, 4.0))*0.55;
      float limb = 1.0 - 0.62*(1.0 - mu) - 0.2*(1.0-mu)*(1.0-mu);
      vec3 base = uStarCol*(1.0 + 0.22*(gran-0.5) + 0.12*(gran2-0.5))*(1.0-spots);
      col = base*limb*uSunI; alpha = 1.0;
      o = vec4(col, 1.0); return;
    }

    // ---- ring geometry (body space: equatorial plane y = 0) ----
    float ringT = -1.0; vec3 ringHitB = vec3(0.0); bool hasRing = uP1.w > 0.5;
    vec3 ob = (-uC)*uBody, db = rd*uBody, sunB = uSun*uBody;
    if(hasRing && abs(db.y) > 1e-6){
      float tr = -ob.y/db.y;
      if(tr > 0.0){ vec3 ph = ob + db*tr; float rr = length(ph)/R; if(rr > 1.32 && rr < 2.35){ ringT = tr; ringHitB = ph; } }
    }

    // ---- ground ----
    vec3 ground = vec3(0.0); float groundHit = 0.0; vec3 nW = vec3(0.0); float tG = 1e30;
    if(onBody){
      tG = t0; groundHit = 1.0;
      vec3 pw = rd*t0; nW = normalize(pw - uC);
      vec3 nb = nW*uBody;
      float fp = t0*pxAng/R;                                   // unit-sphere size of one pixel
      float oct = clamp(log2(0.55/(max(fp,1e-7)*1.7)), 2.0, 16.0);
      vec3 sunN = uSun;
      float sunb = dot(nW, sunN);
      float diff = max(sunb, 0.0);
      float lat = abs(nb.y);
      if(uType == 2){                                          // ---- gas / ice giant ----
        float warp = fbm(vec3(nb.x*2.0, nb.y*7.0, nb.z*2.0) + uSeed, 4.0);
        float y = nb.y + (warp - 0.5)*0.12 + 0.03*sin(nb.y*30.0 + uSeed);
        float turb = fbm(vec3(nb.x*9.0 + uTime*0.01, y*26.0, nb.z*9.0) + uSeed*3.0, min(oct, 8.0));
        float b = 0.5 + 0.5*sin(y*(14.0 + 6.0*fract(uSeed*3.7)) + uSeed*6.0 + 1.5*(turb-0.5));
        float b2 = fbm(vec3(y*9.0, 0.5, uSeed), 3.0);
        vec3 c1 = mix(uPalA, uPalB, smoothstep(0.2, 0.8, b));
        c1 = mix(c1, uPalC, smoothstep(0.55, 0.9, b2)*0.7);
        c1 = mix(c1, uPalD, smoothstep(0.65, 0.8, turb)*0.45);
        // storm
        vec3 sc = normalize(vec3(cos(uSeed*40.0), 0.25*sin(uSeed*13.0)-0.25, sin(uSeed*40.0)));
        float sd = acos(clamp(dot(nb, sc), -1.0, 1.0));
        float storm = smoothstep(0.17, 0.05, sd);
        c1 = mix(c1, uPalD*1.1, storm*0.8);
        ground = c1*c1*(0.12 + diff*0.8)*uSunI;
        ground += c1*0.01;
      } else {                                                 // ---- rocky / moon ----
        float h = groundH(nb, oct);
        // finite-difference bump normal
        vec3 tA = normalize(cross(nb, abs(nb.y) < 0.99 ? vec3(0,1,0) : vec3(1,0,0))), tB = cross(nb, tA);
        float e = max(fp*0.9, 4e-6);
        float hx = groundH(normalize(nb + tA*e), oct), hy = groundH(normalize(nb + tB*e), oct);
        float relief = uP0.w*0.012;                           // height of full range relative to R
        vec3 gradT = (tA*(hx-h) + tB*(hy-h))/e;
        vec3 nbump = normalize(nb - gradT*relief*(uBiome > 6.5 ? 5.0 : 2.5)*smoothstep(1e-6, 6e-6, fp*3.0));
        vec3 nWb = uBody*nbump;
        float sea = uP0.x;
        float ocean = (uBiome < 5.5 && uBiome != 2.0 && uBiome != 3.0) ? 1.0 : 0.0;
        if(uBiome == 4.0) ocean = 1.0;
        float isSea = ocean > 0.5 ? smoothstep(sea+0.004, sea-0.004, h) : 0.0;
        float moist = fbm(nb*2.2 + uSeed*5.0 + 9.0, 5.0);
        float temp = 1.0 - lat*1.25 - max(0.0, h-sea)*0.8 + (moist-0.5)*0.15;
        vec3 land;
        float hh = clamp((h-sea)/max(1.0-sea,0.05), 0.0, 1.0);
        float detail = fbm(nb*80.0, min(oct, 8.0));
        if(uBiome == 7.0){ land = mix(uPalA, uPalB, smoothstep(0.35,0.7,h))*(0.75+0.5*detail); land = mix(land, uPalC, smoothstep(0.7,0.9,h)); }
        else if(uBiome == 5.0){ land = mix(uPalA, uPalB, detail); float cr = smoothstep(0.7,0.74, abs(h-0.5)*2.0); land = mix(land, uPalC, cr*0.6); }
        else if(uBiome == 6.0){ land = mix(uPalA*0.35, uPalB*0.5, hh)*(0.7+0.5*detail); }
        else {
          vec3 low = mix(uPalA, uPalB, smoothstep(0.3, 0.7, 1.0-moist + (1.0-temp)*0.4));
          land = mix(low, uPalC, smoothstep(0.35, 0.8, hh));
          land *= (0.8 + 0.4*detail);
          if(uBiome == 0.0 || uBiome == 1.0 || uBiome == 4.0){ land = mix(land, uPalA*0.9, smoothstep(0.55, 0.2, hh)*smoothstep(0.35,0.7,temp)*moist); }
          float snow = smoothstep(uP0.y - 0.06, uP0.y + 0.02, lat + (hh-0.3)*0.25 - (temp-0.3)*0.1);
          snow = max(snow, smoothstep(0.78, 0.95, hh)*smoothstep(0.1,0.5,1.0-temp+0.2));
          if(uBiome == 3.0) snow *= 0.5;
          land = mix(land, vec3(0.92,0.95,1.0), snow);
        }
        vec3 seaCol = mix(uPalD*0.35, uPalD, smoothstep(sea-0.25, sea, h));
        float ice = (uBiome == 4.0) ? smoothstep(0.1, 0.35, lat+0.2*(moist-0.5)+0.2) : smoothstep(uP0.y-0.05, uP0.y+0.03, lat);
        seaCol = mix(seaCol, vec3(0.9,0.94,1.0), ice);
        vec3 alb = mix(land, seaCol, isSea);
        float diffB = max(dot(isSea > 0.5 ? nWb*0.0 + nW : nWb, uSun), 0.0);
        float diffG = diff;
        float lightD = mix(diffB, diffG, 0.15);
        ground = alb*lightD*uSunI;
        // ocean specular glint
        if(isSea > 0.5 && ice < 0.5){
          vec3 g = noised(nb*900.0 + uTime*0.03).yzw; vec3 nWv = normalize(nW + (uBody*g)*0.035*smoothstep(2e-4, 2e-6, fp)); vec3 hv = normalize(uSun - rd); float sp = pow(max(dot(nWv, hv), 0.0), 260.0);
          ground += vec3(1.0,0.95,0.85)*min(sp,1.0)*0.5*uSunI*(1.0-ice);
        }
        // lava glow
        if(uP1.z > 0.0){
          float crack = pow(1.0 - abs(vnoise(nb*9.0 + uSeed)*2.0-1.0), 6.0) * smoothstep(0.35, 0.6, fbm(nb*4.0+7.0, 6.0));
          float crack2 = pow(1.0 - abs(vnoise(nb*40.0 + uSeed)*2.0-1.0), 8.0);
          vec3 em = vec3(1.0,0.32,0.05)*(crack*0.55 + crack2*0.12*smoothstep(0.4,0.7,fbm(nb*4.0+7.0,5.0)))*uP1.z;
          ground += em;
        }
        // clouds
        if(uP0.z > 0.0 && uP1.x > 0.0){
          vec3 cp = nb*3.2 + vec3(uTime*0.004, 0.0, 0.0) + uSeed*2.0;
          float cw = fbm(cp*0.8, 4.0);
          float cl = fbm(cp + cw*1.4, min(oct, 9.0));
          float camAlt = length(uC) - R;
          float cover = smoothstep(1.0 - uP0.z*0.8, 1.0 - uP0.z*0.8 + 0.2, cl + (1.0 - lat)*0.04) * smoothstep(0.01*R, 0.12*R, camAlt);
          vec3 cc = vec3(1.0)*(0.1 + 0.95*max(dot(nW, uSun), 0.0))*uSunI;
          ground = mix(ground, cc*0.62, cover*0.8);
        }
        ground += alb*0.0;
      }
      // terminator softness is handled by diffuse; add faint earthshine
      ground += vec3(0.002)*uSunI;
    }

    // ---- atmosphere (single scattering, plane-parallel sun transmittance) ----
    vec3 atm = vec3(0.0); vec3 trans = vec3(1.0);
    if(uP1.x > 0.0){
      vec2 ha = isect(rd, uC, Ratm);
      if(ha.y > 0.0){
        float s0 = max(ha.x, 0.0), s1 = ha.y;
        if(groundHit > 0.5) s1 = min(s1, tG);
        float len = s1 - s0;
        float Hs = uP1.y*0.33*R;                               // scale height
        vec3 beta = uAtmCol*uP1.x;
        const int NS = 14;
        float ds = len/float(NS);
        float tau = 0.0;
        float mu = dot(rd, uSun);
        float phaseR = 0.75*(1.0 + mu*mu);
        float g = 0.76, phaseM = (1.0-g*g)/pow(1.0 + g*g - 2.0*g*mu, 1.5)*0.1;
        for(int i=0;i<NS;i++){
          float s = s0 + (float(i)+0.5)*ds;
          vec3 p = rd*s - uC;
          float hgt = length(p) - R;
          float dens = exp(-max(hgt,0.0)/Hs);
          float cosS = dot(normalize(p), uSun);
          float tauSun = dens*Hs/max(cosS + 0.12, 0.04) * 0.9;
          // shadow of the planet itself
          float sh = (cosS < -0.15 && length(p) < Ratm) ? 0.0 : 1.0;
          if(cosS < 0.0){ float perp = length(p)*sqrt(max(1.0 - cosS*cosS, 0.0)); sh = smoothstep(R*0.985, R*1.0, perp); }
          vec3 sunT = exp(-beta * (tauSun / (R*0.02)));
          vec3 viewT = exp(-beta*(tau/(R*0.02)));
          atm += (beta*phaseR*0.5 + vec3(phaseM*uP1.x*0.6)) * dens * ds/(R*0.02) * sunT * viewT * sh;
          tau += dens*ds;
        }
        trans = exp(-beta*(tau/(R*0.02)));
        atm *= uSunI*0.62;
      }
    }
    // ---- compose ----
    vec3 base = vec3(0.0); float a = 0.0;
    if(groundHit > 0.5){ base = ground; a = 1.0; }
    base = base*trans + atm; a = max(a, clamp(max(atm.r, max(atm.g, atm.b))*1.5, 0.0, 1.0));
    // rings
    if(ringT > 0.0 && (groundHit < 0.5 || ringT < tG)){
      float rr = length(ringHitB)/R;
      float u = (rr - 1.32)/(2.35-1.32);
      float bands = 0.5 + 0.5*sin(u*70.0 + 4.0*fbm(vec3(u*25.0, uSeed, 1.0), 3.0)) ;
      float dens = smoothstep(0.0,0.06,u)*smoothstep(1.0,0.9,u)*(0.35 + 0.65*fbm(vec3(u*40.0, uSeed*3.0, 0.0), 4.0))*(0.4+0.6*bands);
      dens *= 1.0 - 0.9*smoothstep(0.50, 0.52, u)*smoothstep(0.58,0.56,u);       // a gap
      // planet shadow on the rings
      vec3 toSun = sunB; float tc = dot(-ringHitB, toSun); float perp = length(ringHitB + toSun*tc);
      float shadow = (tc < 0.0) ? smoothstep(R*0.98, R*1.02, perp) : 1.0;
      float lit = (0.15 + 0.85*abs(sunB.y)*0.5 + 0.35)*shadow*uSunI;
      vec3 rc = mix(uPalA, uPalB, bands)*lit;
      float ra = clamp(dens*0.9, 0.0, 0.95);
      if(groundHit > 0.5){ base = mix(base, rc, ra); a = max(a, ra); }
      else { base = rc*ra; a = ra; }
    } else if(ringT > 0.0 && groundHit > 0.5){ /* ring behind the planet: hidden */ }
    if(a <= 0.001 && max(base.r, max(base.g, base.b)) <= 0.0005) discard;
    o = vec4(base, clamp(a, 0.0, 1.0));
  }`;

  // ---------------- distant markers, glows, orbit lines ----------------
  U.SH.markVS = PRE + `
  uniform mat3 uV; uniform vec2 uProj; uniform vec3 uRel; uniform float uSizePx;
  void main(){ vec3 v = uV*uRel; gl_Position = vec4(v.x*uProj.x, v.y*uProj.y, 0.0, -v.z); gl_PointSize = uSizePx; }`;
  U.SH.markFS = PRE + `uniform vec3 uCol; out vec4 o; void main(){ vec2 d = gl_PointCoord*2.0-1.0; float r2 = dot(d,d); if(r2>1.0) discard; o = vec4(uCol*exp(-2.2*r2), 1.0); }`;

  // star glow: a view-space billboard centred on the star
  U.SH.glowVS = PRE + `
  uniform mat3 uV; uniform vec2 uProj; uniform vec3 uRel; uniform float uHalf;   // half-size in metres
  out vec2 vUv;
  void main(){ vec2 c = vec2((gl_VertexID & 1), (gl_VertexID >> 1))*2.0 - 1.0; vec3 v = uV*uRel; float w = -v.z;
    vec3 vv = v + vec3(c*uHalf, 0.0); gl_Position = vec4(vv.x*uProj.x, vv.y*uProj.y, 0.0, w); if(w <= 0.0) gl_Position = vec4(2.0,2.0,2.0,1.0); vUv = c; }`;
  U.SH.glowFS = PRE + `
  in vec2 vUv; uniform vec3 uCol; uniform float uI, uCore, uTime; out vec4 o;
  void main(){
    float r = length(vUv);
    float s = uCore;                      // star disc radius in glow-quad units
    float g = 1.0/(1.0 + pow(max(r - s*0.9, 0.0)/(s*0.9), 1.55)*1.0);
    float halo = exp(-r*r*(1.0/max(s*s*9.0,1e-6)));
    float ang = atan(vUv.y, vUv.x);
    float spikes = pow(abs(cos(ang*2.0)), 220.0)*0.5 + pow(abs(cos(ang*2.0 + 1.5708)), 220.0)*0.5;
    float spk = spikes*exp(-r*3.5/ (1.0 + s*8.0))*0.5;
    float v = (g*0.9*smoothstep(1.0, 0.0, r) + halo*0.6 + spk)*smoothstep(1.0, 0.82, r);
    o = vec4(uCol*v*uI, 1.0);
  }`;

  U.SH.orbitVS = PRE + `
  in vec3 aPos; uniform mat3 uV; uniform vec2 uProj; uniform vec3 uRel;      // uRel = parent centre relative to camera
  out float vW;
  void main(){ vec3 v = uV*(aPos + uRel); gl_Position = vec4(v.x*uProj.x, v.y*uProj.y, 0.0, -v.z); vW = -v.z; }`;
  U.SH.orbitFS = PRE + `uniform vec4 uColor; out vec4 o; in float vW; void main(){ o = uColor; }`;
})(typeof window !== "undefined" ? window : globalThis);
