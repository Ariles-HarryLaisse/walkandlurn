// ============================================================
// LEZARD AI — Page : pilote les Workers, affiche l'arbre, les meilleurs lézards,
// l'altimètre des 1000 lézards et le graphique de progression.
// (Ce fichier suppose que physics.js est déjà chargé.)
// ============================================================

const canvas = document.getElementById("cv");
const ctx = canvas.getContext("2d");
const statsEl = document.getElementById("stats");

const ui = { paused: false, reset: false, speed: 1, turbo: false };
window.ui = ui;

function resize() { canvas.width = canvas.clientWidth; canvas.height = canvas.clientHeight; }
window.addEventListener("resize", resize);
resize();

const PX = 260;   // pixels par mètre
const view = {
  top: [], gains: new Float32Array(0), states: new Uint8Array(0),
  pop: 0, alive: 0, gen: 0, simT: 0, speedX: 0, workers: 0, camY: H0 + 0.4,
};
let perfHistory = [];

window.setStats = (txt) => { statsEl.textContent = txt; };
window.setHistory = (txt) => { try { perfHistory = JSON.parse(txt); } catch (e) { perfHistory = []; } };
window.setGen = (g) => { view.gen = g; };
window.waitFrame = () => new Promise((r) => requestAnimationFrame(r));

// Cède la main au navigateur sans attendre la prochaine image (mode turbo)
const _mc = new MessageChannel();
let _mcResolve = null;
_mc.port1.onmessage = () => { const r = _mcResolve; _mcResolve = null; if (r) r(); };
const yieldNow = () => new Promise((r) => { _mcResolve = r; _mc.port2.postMessage(0); });

// ---------- Workers ----------
let pool = [];
function ensurePool() {
  if (pool.length) return;
  const hc = navigator.hardwareConcurrency || 4;
  const n = Math.max(1, Math.min(hc - 1, 15));      // on laisse un cœur à la page
  for (let i = 0; i < n; i++) pool.push(new Worker("worker.js?v=" + (window.BUILD || 0)));
  view.workers = n;
}
const ask = (w, msg, transfer) => new Promise((resolve, reject) => {
  w.onmessage = (e) => (e.data.type === "error" ? reject(new Error(e.data.message)) : resolve(e.data));
  w.onerror = (e) => reject(new Error("Worker : " + (e.message || "erreur")));
  w.postMessage(msg, transfer || []);
});

function b64ToF32(b64) {
  const bin = atob(b64);
  const u8 = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
  return new Float32Array(u8.buffer);
}

// ---------- Une génération : tous les lézards grimpent en même temps ----------
// Se termine quand le dernier lézard est à court d'énergie (ou tombé).
window.runGeneration = async function (brainsB64, genesB64, pop) {
  ensurePool();
  const NP = SIM_INFO.nParams, NG = SIM_INFO.bodyKeys.length;
  const brains = b64ToF32(brainsB64), genes = b64ToF32(genesB64);
  const W = Math.min(pool.length, pop);
  const per = Math.ceil(pop / W);
  const jobs = [];
  for (let w = 0, off = 0; w < W && off < pop; w++) {
    const count = Math.min(per, pop - off);
    jobs.push({
      w: pool[w], off: off, count: count,
      b: brains.slice(off * NP, (off + count) * NP),
      g: genes.slice(off * NG, (off + count) * NG),
    });
    off += count;
  }
  view.pop = pop; view.alive = pop; view.simT = 0; view.top = [];
  view.gains = new Float32Array(pop); view.states = new Uint8Array(pop);
  await Promise.all(jobs.map((j) =>
    ask(j.w, { type: "init", offset: j.off, count: j.count, brains: j.b, genes: j.g }, [j.b.buffer, j.g.buffer])));

  let turboN = 4, lastTs = performance.now();
  while (true) {
    while (ui.paused && !ui.reset) await window.waitFrame();
    if (ui.reset) { ui.reset = false; return JSON.stringify({ aborted: true }); }
    const n = ui.turbo ? turboN : Math.max(1, Math.round(ui.speed));
    const t0 = performance.now();
    const snaps = await Promise.all(jobs.map((j) => ask(j.w, { type: "tick", n: n })));
    const dtMs = performance.now() - t0;

    let alive = 0, cand = [];
    snaps.forEach((s, i) => {
      view.gains.set(s.gains, jobs[i].off);
      view.states.set(s.states, jobs[i].off);
      alive += s.alive;
      cand = cand.concat(s.top);
    });
    cand.sort((a, b) => (b.active ? 1 : 0) - (a.active ? 1 : 0) || b.y - a.y);
    view.top = cand.slice(0, 12);
    view.alive = alive;
    view.simT += n / 60;

    const now = performance.now();
    const inst = (n / 60) / ((now - lastTs) / 1000);
    lastTs = now;
    view.speedX = view.speedX === 0 ? inst : view.speedX * 0.8 + inst * 0.2;

    if (alive === 0) break;
    if (ui.turbo) {
      turboN = clamp(Math.round(turboN * 80 / Math.max(dtMs, 1)), 1, 600);   // ~80 ms de calcul par envoi
      await yieldNow();
    } else {
      await window.waitFrame();
    }
  }

  const res = await Promise.all(jobs.map((j) => ask(j.w, { type: "collect" })));
  const cat = (k) => [].concat(...res.map((r) => r[k]));
  return JSON.stringify({
    aborted: false, gain: cat("gain"), energy: cat("energy"), time: cat("time"),
    cause: cat("cause"), mass: cat("mass"),
  });
};

// ============================================================
// Rendu
// ============================================================
const sx = (x) => canvas.width / 2 + x * PX;
const sy = (y) => canvas.height * 0.6 - (y - view.camY) * PX;

function drawTree() {
  const x0 = sx(-TRUNK_HALF), x1 = sx(TRUNK_HALF);
  const yBot = Math.min(sy(0), canvas.height);
  const g = ctx.createLinearGradient(x0, 0, x1, 0);
  g.addColorStop(0, "#5b3d22"); g.addColorStop(0.5, "#7a5232"); g.addColorStop(1, "#4e331c");
  ctx.fillStyle = g;
  ctx.fillRect(x0, 0, x1 - x0, yBot);          // tronc sans limite vers le haut

  // sillons d'écorce : leur forme dépend de l'altitude réelle, donc ils défilent avec la caméra
  ctx.strokeStyle = "rgba(30,18,8,0.35)";
  ctx.lineWidth = 2;
  for (let i = 0; i < 9; i++) {
    const x = x0 + ((i + 0.5) * (x1 - x0)) / 9 + Math.sin(i * 7.1) * 6;
    ctx.beginPath();
    for (let y = 0; y <= yBot; y += 24) {
      const wy = view.camY + (canvas.height * 0.6 - y) / PX;
      const wob = Math.sin(wy * 5 + i) * 4;
      if (y === 0) ctx.moveTo(x + wob, y); else ctx.lineTo(x + wob, y);
    }
    ctx.stroke();
  }

  // repères d'altitude (hauteur au-dessus du point de départ)
  ctx.font = "11px sans-serif";
  const hTop = view.camY + (canvas.height * 0.6) / PX, hBot = view.camY - (canvas.height * 0.4) / PX;
  for (let h = Math.ceil((hBot - H0) / 0.5) * 0.5; h < hTop - H0; h += 0.5) {
    const y = sy(H0 + h);
    if (y < 0 || y > canvas.height) continue;
    const major = Math.abs(h - Math.round(h)) < 1e-6;
    ctx.strokeStyle = "rgba(255,255,255,0.45)"; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(x0, y); ctx.lineTo(x0 + (major ? 24 : 12), y); ctx.stroke();
    if (major) { ctx.fillStyle = "#333"; ctx.fillText(h.toFixed(0) + " m", x0 - 42, y + 4); }
  }
}

const bulk = (s) => clamp(0.75 + 0.25 * s, 0.8, 1.5);   // un muscle plus fort épaissit le membre

function drawPose(it, alpha, detailed) {
  const P = it.pose, gi = (k) => it.genes[BODY_KEYS.indexOf(k)];
  ctx.globalAlpha = alpha;
  ctx.lineCap = "round";
  POSE_NAMES.forEach((name, i) => {
    const x = P[i * 3], y = P[i * 3 + 1], a = P[i * 3 + 2];
    const hx = DIMS[name][0], hy = DIMS[name][1];
    let k = 1;
    if (name.startsWith("up")) k = bulk(gi("sh" + name[2]));
    else if (name.startsWith("low")) k = bulk(gi("el" + name[3]));
    else if (name === "chest") k = bulk(gi("spine"));
    else if (name === "tail1") k = bulk(gi("tail"));
    const dx = Math.cos(a) * hx, dy = Math.sin(a) * hx;
    const X1 = sx(x - dx), Y1 = sy(y - dy), X2 = sx(x + dx), Y2 = sy(y + dy);
    const w = hy * 2 * PX * k;
    let fill = "#8fbf5a";
    if (detailed) fill = name === "chest" ? "#76a83d" : name.endsWith("R") ? "#527c2b" : "#6b9b37";
    if (detailed) {
      ctx.strokeStyle = "#2f4a1c"; ctx.lineWidth = w + 3;
      ctx.beginPath(); ctx.moveTo(X1, Y1); ctx.lineTo(X2, Y2); ctx.stroke();
    }
    ctx.strokeStyle = fill; ctx.lineWidth = w;
    ctx.beginPath(); ctx.moveTo(X1, Y1); ctx.lineTo(X2, Y2); ctx.stroke();
  });

  if (detailed) {
    const ci = POSE_NAMES.indexOf("chest");
    const cx = P[ci * 3], cy = P[ci * 3 + 1], ca = P[ci * 3 + 2];
    const hX = cx + Math.cos(ca) * 0.17, hY = cy + Math.sin(ca) * 0.17;
    ctx.fillStyle = "#2f4a1c";
    ctx.beginPath(); ctx.arc(sx(hX), sy(hY), 0.06 * PX + 1.5, 0, 6.283); ctx.fill();
    ctx.fillStyle = "#76a83d";
    ctx.beginPath(); ctx.arc(sx(hX), sy(hY), 0.06 * PX, 0, 6.283); ctx.fill();
    for (const s of [-1, 1]) {
      const ex = cx + Math.cos(ca) * 0.185 - Math.sin(ca) * 0.03 * s;
      const ey = cy + Math.sin(ca) * 0.185 + Math.cos(ca) * 0.03 * s;
      ctx.fillStyle = "#fff"; ctx.beginPath(); ctx.arc(sx(ex), sy(ey), 4, 0, 6.283); ctx.fill();
      ctx.fillStyle = "#111"; ctx.beginPath(); ctx.arc(sx(ex), sy(ey), 2, 0, 6.283); ctx.fill();
    }
    // griffes : dorées quand elles sont plantées dans le tronc
    LIMBS.forEach((l, li) => {
      const bi = POSE_NAMES.indexOf("low" + l);
      const bx = P[bi * 3], by = P[bi * 3 + 1], ba = P[bi * 3 + 2];
      const tx = sx(bx + Math.cos(ba) * LL / 2), ty = sy(by + Math.sin(ba) * LL / 2);
      ctx.beginPath();
      if (it.grips[li]) {
        ctx.fillStyle = "#ffcc33"; ctx.strokeStyle = "#a05a00"; ctx.lineWidth = 2;
        ctx.arc(tx, ty, 8, 0, 6.283); ctx.fill(); ctx.stroke();
      } else {
        ctx.fillStyle = "#3d5a20"; ctx.arc(tx, ty, 5, 0, 6.283); ctx.fill();
      }
    });
  }
  ctx.globalAlpha = 1;
}

// Altimètre : un point par lézard, placé à la hauteur maximale atteinte
function drawAltimeter() {
  const x = 12, w = 100, top = 150, bottom = canvas.height - 24;
  if (bottom - top < 60) return;
  ctx.fillStyle = "rgba(20,20,35,0.78)";
  ctx.fillRect(x - 4, top - 22, w + 8, bottom - top + 40);
  ctx.fillStyle = "#fff"; ctx.font = "11px sans-serif";
  ctx.fillText("Altimètre (" + view.pop + " lézards)", x, top - 8);
  let mx = 2;
  for (let i = 0; i < view.gains.length; i++) if (view.gains[i] > mx) mx = view.gains[i];
  mx *= 1.1;
  const H = bottom - top;
  const colors = ["#7CFC9A", "#f5c542", "#e06666", "#9aa0a6"];   // actif, épuisé, tombé, temps écoulé
  for (let i = 0; i < view.gains.length; i++) {
    ctx.fillStyle = colors[view.states[i]] || "#fff";
    const px = x + ((i * 0.61803) % 1) * (w - 3);
    const py = bottom - (view.gains[i] / mx) * H;
    ctx.fillRect(px, py, 3, 3);
  }
  ctx.fillStyle = "#aaa";
  ctx.fillText("0 m", x, bottom + 14);
  ctx.fillText(mx.toFixed(1) + " m", x + 60, top + 10);
}

function drawOverlay() {
  const lines = [
    "Génération " + view.gen,
    "En vie : " + view.alive + " / " + view.pop,
    "Temps simulé : " + view.simT.toFixed(1) + " s",
    "Vitesse réelle ×" + view.speedX.toFixed(0) + "  (" + view.workers + " workers)",
  ];
  ctx.fillStyle = "rgba(20,20,35,0.78)";
  ctx.fillRect(8, 8, 232, 22 + lines.length * 16);
  ctx.fillStyle = "#fff"; ctx.font = "12px sans-serif";
  lines.forEach((t, i) => ctx.fillText(t, 16, 26 + i * 16));
  ctx.fillStyle = "#7CFC9A"; ctx.fillText("● actif", 16, 26 + lines.length * 16);
  ctx.fillStyle = "#f5c542"; ctx.fillText("● épuisé", 72, 26 + lines.length * 16);
  ctx.fillStyle = "#e06666"; ctx.fillText("● tombé", 140, 26 + lines.length * 16);
}

function drawChart() {
  const w = 270, h = 120, x = canvas.width - w - 12, y = 12;
  ctx.fillStyle = "rgba(20,20,35,0.78)";
  ctx.fillRect(x, y, w, h);
  ctx.fillStyle = "#fff"; ctx.font = "11px sans-serif";
  ctx.fillText("Progrès : hauteur par génération (m)", x + 8, y + 15);
  if (perfHistory.length < 2) {
    ctx.fillStyle = "#aaa"; ctx.fillText("en attente de la 2e génération…", x + 8, y + 60);
    return;
  }
  const maxV = Math.max(1, ...perfHistory.map((p) => p[0]));
  const n = perfHistory.length;
  const px0 = x + 10, py0 = y + h - 12, pw = w - 20, ph = h - 40;
  const line = (idx, color) => {
    ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.beginPath();
    perfHistory.forEach((p, i) => {
      const X = px0 + (i / (n - 1)) * pw, Y = py0 - (p[idx] / maxV) * ph;
      if (i === 0) ctx.moveTo(X, Y); else ctx.lineTo(X, Y);
    });
    ctx.stroke();
  };
  line(1, "#6EC1FF");
  line(0, "#7CFC9A");
  ctx.fillStyle = "#7CFC9A"; ctx.fillText("meilleur", x + 150, y + 15);
  ctx.fillStyle = "#6EC1FF"; ctx.fillText("moyenne", x + 205, y + 15);
  ctx.fillStyle = "#aaa"; ctx.fillText(maxV.toFixed(1) + " m", x + 8, y + 30);
}

function draw() {
  const lead = view.top[0];
  const target = (lead ? lead.y : H0) + 0.4;
  view.camY += (target - view.camY) * 0.1;

  ctx.clearRect(0, 0, canvas.width, canvas.height);
  drawTree();
  const yg = sy(0);
  if (yg < canvas.height) {
    ctx.fillStyle = "#5a7d3a";
    ctx.fillRect(0, yg, canvas.width, canvas.height - yg);
  }
  for (let i = view.top.length - 1; i >= 1; i--) drawPose(view.top[i], 0.3, false);   // fantômes
  if (lead) drawPose(lead, 1, true);                                                   // le meilleur, en détail
  drawAltimeter();
  drawOverlay();
  drawChart();
}

function renderLoop() { draw(); requestAnimationFrame(renderLoop); }
