/* mind-inspector.js - Mind Inspector for persistent organisms (PASS 3).
 *
 * Connects the existing organism API (create / step / inspect / checkpoint /
 * save / load) to the observatory frontend. Shows actual habitat/environment
 * state, organism physiology, sensory observations, episodic memory, controller
 * and self-model state, and save/load controls. Reads only real modeled state;
 * does not fabricate internal thoughts or feelings.
 */
(function (g) {
  "use strict";
  const U = g.U, $ = (id) => document.getElementById(id);

  const PANEL = $("mind-inspector-panel");
  const ORG_LIST = $("org-list");
  const ORG_HINT = $("org-hint");
  const ORG_SAVE_LIST = $("org-save-list");
  const OID_BADGE = $("mind-oid");
  const MIND_STATUS = $("mind-status");
  const MIND_ROWS = $("mind-rows");
  const MIND_MEMORY = $("mind-memory");
  const MIND_ACTIONS = $("mind-actions");
  const MIND_CONTROLS = $("mind-controls");
  const MIND_TEXT = $("mind-text");

  let app = null;
  let currentOid = null;
  let currentHid = null;

  function ready() { return !!app; }

  // ---- organism API helpers (delegates to app.js methods) ----
  async function listOrganisms() {
    const cache = app && app.organismCache || {};
    return Object.keys(cache).map((oid) => ({ id: oid, state: cache[oid] }));
  }
  async function listSaves() {
    return await app.organismListSaves();
  }
  async function selectOrganism(oid) {
    currentOid = oid;
    currentHid = null;
    PANEL.style.display = "block";
    ORG_HINT.hidden = true;
    renderList();
    await inspect();
  }
  async function inspect() {
    if (!currentOid) return;
    const st = await app.organismInspect(currentOid);
    if (!st) { MIND_TEXT.textContent = "Organism not found."; return; }
    currentHid = st.habitat_id || null;
    render(st);
  }
  async function stepN(n) {
    if (!currentOid) return;
    const j = await app.organismStep(currentOid, n);
    render(j.state || {});
  }
  async function saveCurrent(id) {
    if (!currentOid) return;
    const j = await app.organismSave(currentOid, id);
    MIND_TEXT.textContent = "Saved checkpoint " + j.id + " (" + j.size_bytes + " bytes).";
    await renderSaves();
  }
  async function loadSave(cid, organismId) {
    const j = await app.organismLoad(cid, organismId);
    currentOid = j.id;
    currentHid = j.habitat_id;
    renderList();
    render(j.state || {});
    MIND_TEXT.textContent = "Restored organism " + j.id + " from checkpoint " + cid + ".";
  }

  // ---- rendering ----
  function fmt(n, d) { return n === null || n === undefined || Number.isNaN(n) ? "–" : Number(n).toFixed(d || 3); }

  function renderList() {
    if (!ORG_LIST) return;
    ORG_LIST.innerHTML = "";
    const items = [];
    const cache = app.organismCache || {};
    Object.keys(cache).forEach((oid) => {
      const st = cache[oid];
      items.push({ id: oid, live: true, state: st, t_gyr: st && st.t_gyr !== undefined ? st.t_gyr : 0 });
    });
    items.sort((a, b) => (b.t_gyr || 0) - (a.t_gyr || 0));
    if (items.length === 0) {
      ORG_LIST.innerHTML = "<div class='empty'>No organisms yet.</div>";
    } else {
      for (const it of items.slice(0, 50)) {
        const el = document.createElement("div");
        el.className = "obj" + (it.id === currentOid ? " selected" : "");
        el.innerHTML = "<div class='dot'></div><div><div class='obj-name'>" + escapeHtml(it.id) + "</div><div class='obj-sub'>t=" + fmt(it.t_gyr, 3) + " Gyr &middot; " + (it.state && it.state.controller_type ? it.state.controller_type : "?") + "</div></div>";
        el.addEventListener("click", () => { selectOrganism(it.id); });
        ORG_LIST.appendChild(el);
      }
    }
  }

  function escapeHtml(t) { return String(t).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;", "'": "&#39;" })[c]); }

  function render(st) {
    if (!st) return;
    const oid = st.identity && st.identity.organism_id ? st.identity.organism_id : (currentOid || "organism");
    if (OID_BADGE) OID_BADGE.textContent = oid;
    if (MIND_STATUS) {
      const parts = [
        st.alive !== undefined ? ("alive=" + st.alive) : null,
        st.n_steps !== undefined ? ("steps=" + st.n_steps) : null,
        st.t_gyr !== undefined ? ("t=" + fmt(st.t_gyr, 3) + " Gyr") : null,
      ].filter(Boolean).join("  &middot;  ");
      MIND_STATUS.textContent = parts;
    }

    // Core rows: identity, physiology, habitat (actual environment state).
    const rows = [];
    if (st.identity) {
      rows.push(["Architecture", st.identity.architecture || "–"]);
      rows.push(["Generation", st.identity.generation !== undefined ? st.identity.generation : "–"]);
      rows.push(["Provenance", st.identity.provenance || "–"]);
    }
    if (st.body) {
      rows.push(["Energy", fmt(st.body.energy, 3)]);
      rows.push(["Nutrients", fmt(st.body.nutrients, 3)]);
      rows.push(["Stress", fmt(st.body.stress, 3)]);
      rows.push(["Age (Gyr)", fmt(st.body.age, 6)]);
      rows.push(["Lifetime reward", fmt(st.lifetime_reward, 4)]);
      rows.push(["Total cost paid", fmt(st.total_cost_paid, 4)]);
    }
    if (st.patch) {
      rows.push(["Patch", st.patch.index]);
      rows.push(["Patch temperature (K)", fmt(st.patch.temperature_K, 1)]);
      rows.push(["Patch pH", fmt(st.patch.pH, 2)]);
      rows.push(["Patch nutrient (mol)", fmt(st.patch.nutrient_mol, 4)]);
      rows.push(["Patch waste (mol)", fmt(st.patch.waste_mol, 4)]);
      rows.push(["Liquid water", String(!!st.patch.liquid_water)]);
    }
    if (MIND_ROWS) {
      MIND_ROWS.innerHTML = "";
      for (const [k, v] of rows) { const a = document.createElement("dt"); a.textContent = k; const b = document.createElement("dd"); b.textContent = v; MIND_ROWS.appendChild(a); MIND_ROWS.appendChild(b); }
    }

    // Sensory observations (what the organism actually senses).
    renderSensory(st);

    // Episodic memory timeline.
    renderMemory(st);

    // Controller / self-model / workspace / learning state.
    renderController(st);

    // Save/load controls.
    renderControls(st);

    if (MIND_TEXT) MIND_TEXT.textContent = "Real modeled state only. No fabricated internal thoughts or feelings.";
  }

  function renderSensory(st) {
    // Sensory observation lives inside the latest memory entry (observation).
    const mem = st.memory && st.memory.entries && st.memory.entries.length ? st.memory.entries[st.memory.entries.length - 1] : null;
    const obs = mem && mem.observation ? mem.observation : null;
    if (!obs) return;
    const rows = [];
    rows.push(["Time (Gyr)", fmt(obs.time, 3)]);
    rows.push(["Position (patch, –)", obs.position && obs.position[0] !== undefined ? obs.position[0] : "–"]);
    rows.push(["Energy level (0–1)", fmt(obs.energy_level, 3)]);
    rows.push(["Nutrient level", fmt(obs.nutrient_level, 3)]);
    rows.push(["Local temperature (K)", fmt(obs.local_temperature, 1)]);
    rows.push(["Local light (0–1)", fmt(obs.local_light, 2)]);
    rows.push(["Nearby resources (0–1)", fmt(obs.nearby_resources, 3)]);
    rows.push(["Nearby hazards (0–1)", fmt(obs.nearby_hazards, 3)]);
    rows.push(["Nearby agents", obs.nearby_agents !== undefined ? obs.nearby_agents : "–"]);
    // extra channels (truthful habitat/patch representation)
    const extra = obs.extra || {};
    if (extra.patch_index !== undefined) rows.push(["Sensed patch index", extra.patch_index]);
    if (extra.pH !== undefined) rows.push(["Sensed pH", fmt(extra.pH, 2)]);
    if (extra.liquid_water !== undefined) rows.push(["Sensed liquid water", String(!!extra.liquid_water)]);
    const el = document.createElement("div");
    el.className = "panel";
    el.innerHTML = "<h3 style='font-size:11px;text-transform:uppercase;letter-spacing:0.08em;color:var(--muted-foreground);margin:0 0 8px;padding-bottom:4px;border-bottom:1px solid var(--line)'>Sensory observation (actual, local patch only)</h3><dl style='display:grid;grid-template-columns:auto 1fr;gap:2px 10px;margin:4px 0 0;font-size:12.5px'></dl>";
    const dl = el.querySelector("dl");
    for (const [k, v] of rows) { const a = document.createElement("dt"); a.textContent = k; const b = document.createElement("dd"); b.textContent = v; dl.appendChild(a); dl.appendChild(b); }
    MIND_MEMORY.parentNode.insertBefore(el, MIND_MEMORY);
    el._miSensory = true;
  }

  function renderMemory(st) {
    // Remove any previously rendered sensory block so it refreshes in place.
    const parent = MIND_MEMORY.parentNode;
    document.querySelectorAll("#mind-inspector-panel > div[class='panel'][_miSensory]").forEach((n) => n.remove());

    const mem = st.memory || {};
    const entries = mem.entries || [];
    const nTotal = mem.n_entries !== undefined ? mem.n_entries : entries.length;
    const cap = mem.capacity !== undefined ? mem.capacity : 500;

    const wrap = document.createElement("div");
    wrap.className = "panel";
    wrap.innerHTML = "<h3 style='font-size:11px;text-transform:uppercase;letter-spacing:0.08em;color:var(--muted-foreground);margin:0 0 8px;padding-bottom:4px;border-bottom:1px solid var(--line)'>Episodic memory timeline</h3>" +
      "<div class='fine' style='margin-bottom:6px'>" + entries.length + " of " + nTotal + " retained entries shown (capacity " + cap + ")" + "</div>" +
      "<div style='max-height:170px;overflow:auto;border:1px solid var(--line);border-radius:8px;background:rgba(0,0,0,.25);font-size:11.5px' id='mind-memory-list'></div>";
    MIND_MEMORY.parentNode.insertBefore(wrap, MIND_MEMORY);
    const list = $("mind-memory-list");
    if (!list) return;
    list.innerHTML = "";
    if (entries.length === 0) { list.innerHTML = "<div class='empty' style='padding:6px'>No memories yet.</div>"; return; }
    for (const e of entries.slice(-12)) {
      const row = document.createElement("div");
      row.style.padding = "4px 6px"; row.style.borderBottom = "1px solid var(--line)";
      row.innerHTML = "<div style='color:var(--muted-foreground);font-size:10px'>t=" + fmt(e.t_gyr, 3) + " Gyr</div>" +
        "<div>reward=" + fmt(e.reward, 3) + " &middot; action=" + escapeHtml((e.action && e.action.action_type) || "–") + "</div>";
      list.appendChild(row);
    }
  }

  function renderController(st) {
    const ctrl = st.controller || {};
    const rows = [];
    rows.push(["Controller type", st.controller_type || ctrl.controller_type || "–"]);
    rows.push(["Actions taken", ctrl.actions_taken !== undefined ? ctrl.actions_taken : "–"]);
    const sm = st.self_model || {};
    if (sm.calibration_error !== undefined) rows.push(["Self-model calibration error", fmt(sm.calibration_error, 4)]);
    if (sm.n_updates !== undefined) rows.push(["Self-model updates", sm.n_updates]);
    const ws = ctrl.workspace || {};
    if (ws.n_broadcasts !== undefined) rows.push(["Workspace broadcasts", ws.n_broadcasts]);
    if (ctrl.recent_prediction_error && ctrl.recent_prediction_error.length) {
      const last = ctrl.recent_prediction_error[ctrl.recent_prediction_error.length - 1];
      rows.push(["Recent prediction error", fmt(last, 4)]);
    }
    const el = document.createElement("div");
    el.className = "panel";
    el.innerHTML = "<h3 style='font-size:11px;text-transform:uppercase;letter-spacing:0.08em;color:var(--muted-foreground);margin:0 0 8px;padding-bottom:4px;border-bottom:1px solid var(--line)'>Controller / self-model / workspace</h3><dl style='display:grid;grid-template-columns:auto 1fr;gap:2px 10px;margin:4px 0 0;font-size:12.5px'></dl>";
    const dl = el.querySelector("dl");
    for (const [k, v] of rows) { const a = document.createElement("dt"); a.textContent = k; const b = document.createElement("dd"); b.textContent = v; dl.appendChild(a); dl.appendChild(b); }
    MIND_ACTIONS.parentNode.insertBefore(el, MIND_ACTIONS);
    el._miCtrl = true;
  }

  async function renderControls(st) {
    const parent = MIND_ACTIONS.parentNode;
    document.querySelectorAll("#mind-inspector-panel > div[class='panel'][_miControls]").forEach((n) => n.remove());

    const wrap = document.createElement("div");
    wrap.className = "panel";
    const oid = currentOid || "organism";
    wrap.innerHTML = "<h3 style='font-size:11px;text-transform:uppercase;letter-spacing:0.08em;color:var(--muted-foreground);margin:0 0 8px;padding-bottom:4px;border-bottom:1px solid var(--line)'>Controls</h3>" +
      "<div class='row' style='margin-bottom:6px'>" +
      "<button id='mi-step1' class='pill secondary'>Step 1</button>" +
      "<button id='mi-step10' class='pill secondary'>Step 10</button>" +
      "<button id='mi-checkpoint' class='pill secondary'>Checkpoint</button>" +
      "</div>" +
      "<div class='row' style='margin-bottom:6px'>" +
      "<input id='mi-save-id' type='text' placeholder='checkpoint id' style='width:auto;flex:1;padding:4px 8px;border-radius:8px;border:1px solid var(--line);background:rgba(0,0,0,.35);color:var(--ink)' />" +
      "<button id='mi-save' class='pill secondary'>Save</button>" +
      "</div>" +
      "<div id='mi-load-list' style='font-size:11px'></div>" +
      "<div class='fine' style='margin-top:4px'>Pause/resume: step through the organism; save checkpoints to disk and reload across restarts.</div>";
    MIND_CONTROLS.parentNode.insertBefore(wrap, MIND_CONTROLS);
    wrap._miControls = true;

    $("mi-step1").onclick = async () => { await stepN(1); await renderSaves(); };
    $("mi-step10").onclick = async () => { await stepN(10); await renderSaves(); };
    $("mi-checkpoint").onclick = async () => { try { await app.organismCheckpoint(oid); MIND_TEXT.textContent = "Checkpoint exported (see console/network)."; } catch (e) { MIND_TEXT.textContent = "Checkpoint failed: " + e.message; } };
    $("mi-save").onclick = async () => { const id = $("mi-save-id").value.trim() || ("cp-" + Date.now()); await saveCurrent(id); $("mi-save-id").value = ""; await renderSaves(); };

    await renderSaves();
  }

  async function renderSaves() {
    if (!ORG_SAVE_LIST) return;
    const saves = await listSaves();
    if (!saves.length) { ORG_SAVE_LIST.style.display = "none"; return; }
    ORG_SAVE_LIST.style.display = "block";
    ORG_SAVE_LIST.innerHTML = "<div class='fine' style='margin-bottom:4px'>Saved checkpoints:</div>";
    for (const s of saves.slice(0, 20)) {
      const el = document.createElement("div");
      el.style.padding = "3px 6px"; el.style.borderBottom = "1px solid var(--line)";
      el.innerHTML = "<span>" + escapeHtml(s.id) + "</span> <span class='fine' style='float:right'>t=" + (s.t_gyr !== undefined ? s.t_gyr.toFixed(3) : "–") + " Gyr</span>" +
        "<button style='float:right;margin-left:6px;background:none;border:none;color:var(--acc);cursor:pointer'>Restore</button>";
      el.querySelector("button").onclick = () => { loadSave(s.id, null); };
      ORG_SAVE_LIST.appendChild(el);
    }
  }

  // ---- wiring ----
  $("btn-org-refresh").onclick = async () => { renderList(); await renderSaves(); };
  $("btn-org-saves").onclick = async () => { const show = ORG_SAVE_LIST.style.display !== "block"; await renderSaves(); if (show) ORG_SAVE_LIST.style.display = "block"; else ORG_SAVE_LIST.style.display = "none"; };
  $("btn-org-create").onclick = async () => {
    const id = prompt("Organism id (e.g. org-1):");
    if (!id) return;
    const arch = prompt("Architecture (ecs_iw | reactive | recurrent | world_model):", "ecs_iw");
    if (!arch) return;
    try {
      const j = await app.organismCreate({ id, architecture: arch, seed: 0, patch_index: 0 });
      currentOid = j.id;
      currentHid = j.habitat_id;
      renderList();
      render(j.state || {});
      MIND_TEXT.textContent = "Created organism " + j.id + " in habitat " + j.habitat_id + ".";
    } catch (e) { alert("Create failed: " + e.message); }
  };

  // Expose for scripting hook / console.
  g.__mindInspector = { selectOrganism, inspect, stepN, listSaves, current: () => currentOid };

  // Deferred init: app is created by main.js after this module loads.
  function init(a) {
    app = a;
    renderList();
    renderSaves();
  }
  if (g.__app) { init(g.__app); } else {
    const poller = setInterval(() => { if (g.__app) { clearInterval(poller); init(g.__app); } }, 50);
  }

})(typeof window !== "undefined" ? window : globalThis);
