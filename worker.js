// ============================================================
// LEZARD AI — Worker : simule une tranche de la population, tick après tick.
// Reçoit les cerveaux et les gènes, avance de n ticks à la demande de la page,
// et renvoie un instantané (hauteurs, états, poses des meilleurs).
// ============================================================
importScripts("https://cdn.jsdelivr.net/npm/planck@1.0.0/dist/planck.min.js");
const VER = new URL(self.location.href).searchParams.get("v") || "0";
importScripts("physics.js?v=" + VER);

let L = [];       // lézards de ce worker
let OFF = 0;      // indice global du premier lézard

function snapshot() {
  const n = L.length;
  const gains = new Float32Array(n), states = new Uint8Array(n);
  let alive = 0;
  const cand = [];
  for (let i = 0; i < n; i++) {
    const z = L[i];
    gains[i] = z.maxY - H0;
    states[i] = z.cause;
    if (!z.finished) alive++;
    if (z.cause !== 2) cand.push(i);      // les lézards tombés ne sont pas montrés
  }
  // actifs d'abord, puis du plus haut au plus bas
  cand.sort((a, b) => (L[b].finished ? 0 : 1) - (L[a].finished ? 0 : 1) || L[b].y - L[a].y);
  const top = cand.slice(0, 8).map((i) => ({
    id: OFF + i, active: !L[i].finished, y: L[i].y, maxY: L[i].maxY,
    pose: L[i].pose(), grips: L[i].gripsArr(), genes: L[i].genesArr,
  }));
  return { type: "snap", alive: alive, gains: gains, states: states, top: top };
}

self.onmessage = (e) => {
  const m = e.data;
  try {
    if (m.type === "init") {
      if (typeof planck === "undefined") throw new Error("planck.js n'a pas été chargé dans le worker");
      OFF = m.offset;
      L = [];
      const NG = BODY_KEYS.length;
      for (let i = 0; i < m.count; i++) {
        const brain = m.brains.subarray(i * N_PARAMS, (i + 1) * N_PARAMS);
        const g = {};
        BODY_KEYS.forEach((k, j) => { g[k] = m.genes[i * NG + j]; });
        L.push(createLizard(brain, g));
      }
      postMessage({ type: "ready" });
    } else if (m.type === "tick") {
      for (let k = 0; k < m.n; k++) {
        let any = false;
        for (let i = 0; i < L.length; i++) {
          if (!L[i].finished) { L[i].step(); any = true; }
        }
        if (!any) break;
      }
      postMessage(snapshot());
    } else if (m.type === "collect") {
      postMessage({
        type: "result",
        gain: L.map((z) => z.maxY - H0),
        energy: L.map((z) => z.energyUsed),
        time: L.map((z) => z.t),
        cause: L.map((z) => z.cause),
        mass: L.map((z) => z.mass),
      });
    }
  } catch (err) {
    postMessage({ type: "error", message: String(err && err.stack ? err.stack : err) });
  }
};
