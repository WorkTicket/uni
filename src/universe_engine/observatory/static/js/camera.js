/* camera.js - centralized camera controller with explicit reference frames.
   Each mode owns its movement behavior; modes never fight over position/rotation.
   Authoritative positions are double-precision; the renderer receives
   camera-relative coordinates for the focused node's frame.

   Modes:
     FREE_FLIGHT    - WASD + look in an inertial frame; no target.
     ORBIT_TARGET   - left-drag orbit around the current focus; wheel zooms.
     FOLLOW_TARGET  - co-moving offset behind a tracked body (orbit/follow).
     PLANET_SURFACE - ground-level navigation over a planet/moon surface;
                      altitude is relative to the surface, not an orbital clamp.
     SYSTEM_MAP     - frame a planetary system (star + planets).
     GALAXY_MAP     - frame a galaxy / the whole simulated region.
     MICROSCOPE     - chemical-field / close-up view on the focused planet.
*/
(function (g) {
  "use strict";
  const U = g.U, V = U.V, Q = U.Q;
  const AU = U.AU, KPC = U.KPC, MPC = U.MPC, YEAR = U.YEAR;

  const MODES = {
    FREE_FLIGHT: "free_flight",
    ORBIT_TARGET: "orbit_target",
    FOLLOW_TARGET: "follow_target",
    PLANET_SURFACE: "planet_surface",
    SYSTEM_MAP: "system_map",
    GALAXY_MAP: "galaxy_map",
    MICROSCOPE: "microscope",
  };

  // framing distance from FOV so the object's bounding sphere just fits the view
  const FOV = 58 * Math.PI / 180;
  function frameDist(radiusM) { const f = 1 / Math.tan(FOV / 2); return radiusM / Math.tan(FOV / 2) * 1.15; }

  class CameraController {
    constructor(app) {
      this.app = app;
      this.mode = MODES.ORBIT_TARGET;
      this.target = null;          // {node, chain}
      this.previousTarget = null;  // for Back navigation
      this.position = new Float64Array([0, 0, 0]);   // world/universe frame (metres)
      this.orientation = Q.ident();                  // quaternion
      this.free = { yaw: 0, pitch: 0, vel: [0, 0, 0], speed: 20 };
      this.orbit = { yaw: 0, pitch: 0.34, dist: 0 };
      this.surface = { altitude: 0, heading: 0, pitch: -0.1 };
      this.bounds = { min: 1, pop: Infinity, max: 1e30 };
      this.smooth = 0;
      this.autopilot = null;
      this.lastTarget = null;
    }

    // ---- mode access --------------------------------------------------------
    static MODES() { return MODES; }
    modeOf() { return this.mode; }

    // ---- bounds for a node (metres) ----------------------------------------
    boundsFor(node) {
      const app = this.app;
      if (!node || !app.uni) return { min: 1, pop: Infinity, max: 1e30 };
      const d = node.data;
      switch (node.kind) {
        case "universe": return { min: 25 * MPC, pop: Infinity, max: app.uni.boxM * 9 };
        case "halo": return { min: Math.max(d.rvir * 0.02, 60 * KPC), pop: Math.max(d.rvir * 14, 4 * MPC), max: 1e30 };
        case "galaxy": return { min: Math.max(0.0008 * node.size, 0.4 * KPC), pop: node.size * 18, max: 1e30 };
        case "star": return { min: d.R * 2.4, pop: Math.max(node.size * 9, 40 * AU), max: 1e30 };
        case "planet": case "moon": {
          const kids = node.children || [];
          const lastMoon = kids.length ? kids[kids.length - 1] : null;
          const moonDist = lastMoon ? (lastMoon.data.a || 0) * 4 : 0;
          return { min: node.size * 1.00045 + 1200, pop: Math.max(node.size * 70, moonDist), max: 1e30 };
        }
      }
      return { min: 1, pop: Infinity, max: 1e30 };
    }

    // ---- target resolution --------------------------------------------------
    setTarget(node) {
      if (!node) return;
      this.previousTarget = this.target;
      this.target = { node, chain: node.chain() };
      this.bounds = this.boundsFor(node);
      this.app.setNode(node);
      this.app.recenter = 1.6;
    }
    getTarget() { return this.target ? this.target.node : null; }
    getPreviousTarget() { return this.previousTarget ? this.previousTarget.node : null; }
    back() {
      const prev = this.getPreviousTarget();
      if (prev) { this.setTarget(prev); this.previousTarget = null; this.app.recenter = 1.2; }
      return !!prev;
    }

    // ---- world position of the focus in the universe frame ------------------
    focusWorldPos() {
      const app = this.app;
      if (!this.target) return app.pos.slice();
      const cams = app.camFrames(this.target.chain);
      return cams[0];
    }

    // ---- mode entry ---------------------------------------------------------
    orbitTarget(node, distM) {
      this.mode = MODES.ORBIT_TARGET;
      this.free.vel = [0, 0, 0];
      this.smooth = 0;
      if (node) this.setTarget(node);
      const n = this.getTarget();
      const lim = this.bounds;
      this.orbit.dist = distM || lim.pop || 10 * MPC;
      this.app.autopilot = null;
    }
    followTarget(node, distM) {
      this.mode = MODES.FOLLOW_TARGET;
      this.free.vel = [0, 0, 0];
      this.smooth = 0;
      if (node) this.setTarget(node);
      const n = this.getTarget();
      const lim = this.bounds;
      this.orbit.dist = distM || lim.pop || 10 * MPC;
      this.app.autopilot = null;
    }
    surface(node) {
      this.mode = MODES.PLANET_SURFACE;
      this.free.vel = [0, 0, 0];
      if (!node || node.kind !== "planet" && node.kind !== "moon") {
        // fall back to orbit if there's no surface to stand on
        this.orbitTarget(node); return;
      }
      this.setTarget(node);
      const lim = this.bounds;
      // Ground-level: start just above the surface (surface altitude ~0).
      // Altitude is stored separately; position is computed in update() from
      // surface heading/pitch so the camera can reach the surface.
      this.surface.altitude = Math.max(lim.min - node.size, 50);
      this.surface.heading = this.orbit.yaw || 0;
      this.surface.pitch = -0.15;
      this.app.autopilot = null;
    }
    freeFlight() {
      this.mode = MODES.FREE_FLIGHT;
      this.target = null;
      this.bounds = { min: 1, pop: Infinity, max: 1e30 };
      this.smooth = 0;
      this.app.autopilot = null;
    }
    systemMap() {
      this.mode = MODES.SYSTEM_MAP;
      this.free.vel = [0, 0, 0];
      const app = this.app;
      const st = app.node.kind === "star" ? app.node : (app.node.parent && app.node.parent.kind === "star" ? app.node.parent : null);
      if (!st) return;
      this.setTarget(st);
      this.orbit.dist = st.size * 2.4;
      this.app.autopilot = null;
    }
    galaxyMap() {
      this.mode = MODES.GALAXY_MAP;
      this.free.vel = [0, 0, 0];
      const app = this.app;
      const n = app.uni ? app.uni.root : app.node;
      this.setTarget(n);
      this.orbit.dist = (n && n.data ? Math.max(n.data.rvir * 2, 200 * KPC) : app.uni ? app.uni.boxM * 2 : 10 * MPC);
      this.app.autopilot = null;
    }
    microscope(node) {
      this.mode = MODES.MICROSCOPE;
      this.free.vel = [0, 0, 0];
      if (node) this.setTarget(node);
      this.orbit.dist = (node && node.size ? node.size * 1.2 : 10 * AU);
      this.app.autopilot = null;
    }

    // ---- focus / fly (smooth autopilot transition) --------------------------
    focus(node) {
      if (!node) return;
      this.previousTarget = this.target;
      this.target = { node, chain: node.chain() };
      this.bounds = this.boundsFor(node);
      this.app.setNode(node);
      const lim = this.bounds;
      const dist = frameDist(Math.max(lim.min, node.size || 1) * 2);
      this.orbit.dist = dist;
      this.smooth = 1.6;
      this.app.autopilot = null;
      this.app.recenter = 1.6;
      return node;
    }
    flyTo(node, dur) {
      // Reuse the existing autopilot-based flyTo for a smooth transition.
      const app = this.app;
      const lim = this.boundsFor(node);
      const dist = frameDist(Math.max(lim.min, node.size || 1) * 2);
      app.autopilot = { i: 0, steps: [{ node: () => node, dist: () => dist, dur: dur || (node.kind === "universe" ? 3 : 4.5) }] };
      this.setTarget(node);
      this.smooth = 1.2;
    }

    // ---- input --------------------------------------------------------------
    orbit(dx, dy) {
      const app = this.app;
      app.autopilot = null;
      const k = 0.0045;
      let up = [0, 1, 0];
      if (app.node && app.node.kind === "galaxy") up = Q.rotate(app.node.q, [0, 1, 0]);
      const right = Q.rotate(app.q, [1, 0, 0]);
      const rot = Q.mul(Q.axisAngle(up, -dx * k), Q.axisAngle(right, -dy * k));
      app.pos = Q.rotate(rot, app.pos);
      app.q = Q.normalize(Q.mul(rot, app.q));
      this.orbit.yaw = (this.orbit.yaw || 0) - dx * k;
      this.orbit.pitch = U.clamp((this.orbit.pitch || 0) - dy * k, -1.45, 1.45);
      app.recenter = 0;
    }
    look(dx, dy) {
      this.free.yaw = (this.free.yaw || 0) + dx * 0.004;
      this.free.pitch = U.clamp((this.free.pitch || 0) + dy * 0.004, -1.45, 1.45);
      this.app.q = Q.mul(Q.axisAngle([0, 1, 0], this.free.yaw), Q.axisAngle([1, 0, 0], this.free.pitch));
      this.app.recenter = 0;
    }
    pan(dx, dy) {
      // Pan the camera perpendicular to the view direction, scale-aware.
      const app = this.app;
      const b = Q.basis(app.q);
      const dist = V.len(app.pos) || 1;
      const scale = dist * 0.001;
      const right = V.scale(b[0], -dx * scale);
      const up = V.scale(b[1], dy * scale);
      app.pos = V.add(app.pos, V.add(right, up));
      app.recenter = 0;
    }
    wheel(deltaY, mx, my) {
      const app = this.app;
      app.autopilot = null;
      const amt = -deltaY * 0.0016;
      if (amt > 0 && app.hover) {
        const n = app.nodeFromHover(app.hover);
        if (n && n !== app.node) { this.focus(n); return; }
      }
      app.zoomAcc += amt;
    }
    freeMove(x, y, z) { this.free.vel = [x, y, z]; }

    // ---- key-driven focus ---------------------------------------------------
    focusSelection() {
      const app = this.app;
      const n = app.node;
      if (!n) return;
      if (n.kind === "universe") { this.galaxyMap(); return; }
      if (n.kind === "halo") { const gals = app.uni.members_of(n); if (gals.length) this.focus(gals[0]); else this.orbitTarget(n); return; }
      if (n.kind === "galaxy") { this.orbitTarget(n); return; }
      if (n.kind === "star") { this.systemMap(); return; }
      if (n.kind === "planet" || n.kind === "moon") { this.surface(n); return; }
      this.orbitTarget(n);
    }

    // ---- per-frame update ---------------------------------------------------
    // Returns the camera position (world frame) and orientation for the renderer.
    update(dt) {
      const app = this.app;
      const chain = app.chain();
      app.updateDynamic(chain);
      const mode = this.mode;
      const n = this.getTarget() || app.node;
      const lim = this.boundsFor(n);

      if (mode === MODES.FREE_FLIGHT) {
        const basis = Q.basis(app.q);
        const fwd = basis[2]; const rgt = basis[0]; const up = basis[1];
        const move = V.add(V.add(V.scale(fwd, this.free.vel[2]), V.scale(rgt, this.free.vel[0])), V.scale(up, this.free.vel[1]));
        app.pos = V.add(app.pos, V.scale(move, dt * this.free.speed));
        // orientation is user-controlled via look(); no recentre.
      } else if (mode === MODES.PLANET_SURFACE && n && (n.kind === "planet" || n.kind === "moon")) {
        this._updateSurface(dt, n, lim);
      } else if (mode === MODES.FOLLOW_TARGET && n) {
        this._updateFollow(dt, n, lim);
      } else {
        // ORBIT_TARGET / SYSTEM_MAP / GALAXY_MAP / MICROSCOPE: zoom + recentre on target.
        this._updateOrbit(dt, n, lim);
      }

      // Keep camera outside the absolute minimum (hard floor), but allow
      // PLANET_SURFACE to descend to the surface (altitude >= 0).
      if (mode !== MODES.PLANET_SURFACE) {
        const d = V.len(app.pos);
        if (d < lim.min) app.pos = V.scale(app.pos, lim.min / Math.max(d, 1e-9));
      }

      // Recentre orientation on the focus unless the user is steering.
      if (app.recenter > 0 && mode !== MODES.FREE_FLIGHT) {
        app.recenter -= dt;
        const target = Q.lookAt(V.scale(app.pos, -1), [0, 1, 0]);
        app.q = Q.slerp(app.q, target, 1 - Math.exp(-dt * 4.5));
      } else if (mode !== MODES.FREE_FLIGHT && mode !== MODES.PLANET_SURFACE) {
        const target = Q.lookAt(V.scale(app.pos, -1), [0, 1, 0]);
        app.q = Q.slerp(app.q, target, 1 - Math.exp(-dt * 4.5));
      }
      return { position: app.pos.slice(), orientation: app.q.slice() };
    }

    _updateSurface(dt, node, lim) {
      const app = this.app;
      const R = node.size;
      // Surface navigation: heading (yaw) + pitch (elevation above horizon) + altitude.
      // Altitude 0 = at the surface (just above the terrain radius).
      const alt = Math.max(0, this.surface.altitude);
      const heading = this.surface.heading || 0;
      const pitch = U.clamp(this.surface.pitch, -1.4, 1.4);
      // Position relative to planet centre: outward radial direction.
      const radial = [Math.cos(pitch) * Math.sin(heading), Math.sin(pitch), Math.cos(pitch) * Math.cos(heading)];
      const dist = R + alt + 10; // 10 m standoff above the surface
      app.pos[0] = radial[0] * dist; app.pos[1] = radial[1] * dist; app.pos[2] = radial[2] * dist;
      // Look toward the planet centre.
      app.q = Q.lookAt(V.scale(app.pos, -1), [0, 1, 0]);
      app.recenter = 0;
    }

    _updateFollow(dt, node, lim) {
      const app = this.app;
      const root = app.uni ? app.uni.root : app.node;
      const rootRef = root ? root : app.node;
      const ft = node;
      // Camera position in the universe frame, expressed relative to the follow target.
      const camU = V.add(app.pos.slice(), app.node.posIn(rootRef));
      app.pos = V.sub(camU, ft.posIn(rootRef));
      app.setNode(ft);
      const d = V.len(app.pos);
      const mind = lim.min;
      if (d < mind) app.pos = V.scale(app.pos, mind / Math.max(d, 1e-9));
      if (d > lim.pop * 4) app.pos = V.scale(app.pos, (lim.pop * 4) / Math.max(d, 1e-9));
      app.recenter = Math.max(app.recenter, 0.3);
    }

    _updateOrbit(dt, node, lim) {
      const app = this.app;
      // Zoom handling (shared by orbit/system/galaxy/map/microscope modes).
      if (Math.abs(app.zoomAcc) > 1e-5) {
        const take = app.zoomAcc * (1 - Math.exp(-dt * 7)); app.zoomAcc -= take;
        const dist = V.len(app.pos);
        let nd = dist * Math.exp(-take);
        if (take < 0 && nd > lim.pop && app.node && app.node.parent) { app.popOut(); }
        else {
          nd = U.clamp(nd, lim.min, lim.max);
          if (nd === lim.min && take > 0) app.zoomAcc = 0;
          if (nd !== dist) app.pos = V.scale(app.pos, nd / dist);
        }
      }
      // Re-frame on the current target's position.
      const n = this.getTarget() || app.node;
      if (n && n !== app.node) {
        const cams = app.camFrames(n.chain());
        app.pos = cams[0].slice();
        app.setNode(n);
        app.recenter = 1.0;
      }
    }
  }
  U.CameraController = CameraController;
  U.CameraModes = MODES;
})(typeof window !== "undefined" ? window : globalThis);
