// ============================================================
// LEZARD AI — Physique (planck.js, portage de Box2D)
// Le plan de l'écran est le plan du tronc : le lézard est vu de face,
// collé à l'écorce comme un gecko sur un mur. La gravité le tire vers le bas.
// Chaque patte peut planter sa griffe dans le tronc (joint) ou la retirer.
// ============================================================

const canvas = document.getElementById("cv");
const ctx = canvas.getContext("2d");
const statsEl = document.getElementById("stats");

const ui = { paused: false, reset: false, speed: 1, turbo: false };
window.ui = ui;

function resize() { canvas.width = canvas.clientWidth; canvas.height = canvas.clientHeight; }
window.addEventListener("resize", resize);
resize();

// ---------- Constantes générales ----------
const PX = 260;              // pixels par mètre à l'écran
const DT = 1 / 240;          // pas de physique fixe
const SUBSTEPS = 4;          // pas physiques par action (1 action = 1/60 s)
const NO_SELF = -1;          // les membres ne se heurtent pas entre eux
const V = (x, y) => planck.Vec2(x, y);
const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
const deg = (d) => (d * Math.PI) / 180;

// ---------- Arbre ----------
const TRUNK_HALF = 0.6;      // demi-largeur du tronc (m) : on ne peut s'accrocher que dessus
const START_Y = 1.0;         // hauteur de la colonne vertébrale au départ
const H0 = START_Y + 0.13;   // hauteur de départ du centre de la poitrine
const GOAL_Y = H0 + 1.5;     // sommet à atteindre
const TREE_TOP = GOAL_Y + 0.9;
const FALL_Y = 0.75;         // sous cette hauteur, le lézard est considéré comme tombé

// ---------- Griffes ----------
const GRIP_MAX = 70;         // force (N) au-delà de laquelle la griffe glisse et lâche
const MIN_HOLD = 0.2;        // durée minimale (s) entre deux changements d'état d'une griffe

// ---------- Corps : paramètres qui évolueront avec la sélection naturelle ----------
// massScale multiplie la masse de tout le corps ; strength multiplie la force des muscles.
const BODY = { massScale: 1.0, strength: 1.0 };

// ---------- Muscles (moteurs d'articulation) ----------
// r = amplitude de la consigne (rad), tmax = couple maximal (N·m) = force du muscle,
// gain = réactivité, vmax = vitesse angulaire maximale (rad/s).
const JT = {
  spine: { r: 0.6, tmax: 4.0, gain: 6, vmax: 3 },
  tail1: { r: 0.8, tmax: 0.8, gain: 5, vmax: 3 },
  sh:    { r: 1.2, tmax: 4.0, gain: 8, vmax: 5 },
  el:    { r: 1.6, tmax: 3.0, gain: 8, vmax: 6 },
  tailp: { r: 0.0, tmax: 0.3, gain: 3, vmax: 2 },   // bout de queue : passif, souple
};
const SMOOTH_TAU = 0.08;     // lissage des commandes de l'IA (s)
const GAIT_T = 1.6;          // période de l'horloge interne fournie à l'IA (s)

// Actions de l'IA : 10 muscles puis 4 griffes (>0 = accrochée)
const CTRL = ["spine", "tail1", "shFL", "elFL", "shFR", "elFR", "shRL", "elRL", "shRR", "elRR"];
const LIMBS = ["FL", "FR", "RL", "RR"];   // avant-gauche, avant-droite, arrière-gauche, arrière-droite

// Dimensions des pattes (m)
const LU = 0.12, LL = 0.14;

const typeOf = (n) => {
  if (n.startsWith("sh")) return "sh";
  if (n.startsWith("el")) return "el";
  if (n === "tail2" || n === "tail3") return "tailp";
  return n;
};
// Les pattes de droite sont le miroir de celles de gauche : même commande = mouvement symétrique
const signOf = (n) => ((n.startsWith("sh") || n.startsWith("el")) && n.endsWith("R") ? -1 : 1);

let world, GROUND, B = {}, J = {}, DIMS = {};
let targets = {}, smoothed = {}, grips = {}, gripTimer = {};
let simT = 0, noGripT = 0, camY = null;

// ---------- Construction ----------
function part(name, cx, cy, angle, hx, hy, dens) {
  const b = world.createBody({ type: "dynamic", position: V(cx, cy), angle: angle, angularDamping: 0.1 });
  b.createFixture(planck.Box(hx, hy), {
    density: dens * BODY.massScale, friction: 0.8, restitution: 0.0, filterGroupIndex: NO_SELF,
  });
  DIMS[name] = [hx, hy];
  B[name] = b;
  return b;
}

function hinge(name, a, b, ax, ay, lo, hi) {
  const t = JT[typeOf(name)].tmax * BODY.strength;
  const def = {
    referenceAngle: b.getAngle() - a.getAngle(),
    enableLimit: true, lowerAngle: lo, upperAngle: hi,
    enableMotor: true, maxMotorTorque: t, motorSpeed: 0,
    collideConnected: false,
  };
  J[name] = world.createJoint(planck.RevoluteJoint(def, a, b, V(ax, ay)));
}

function limb(l, root, rx, ry, a1, a2) {
  const ex = rx + Math.cos(a1) * LU, ey = ry + Math.sin(a1) * LU;
  const fx = ex + Math.cos(a2) * LL, fy = ey + Math.sin(a2) * LL;
  part("up" + l, (rx + ex) / 2, (ry + ey) / 2, a1, LU / 2, 0.022, 15);
  part("low" + l, (ex + fx) / 2, (ey + fy) / 2, a2, LL / 2, 0.018, 15);
  // griffe : petit disque au bout de l'avant-bras
  B["low" + l].createFixture(planck.Circle(V(LL / 2, 0), 0.022), {
    density: 15 * BODY.massScale, friction: 0.8, restitution: 0.0, filterGroupIndex: NO_SELF,
  });
  hinge("sh" + l, root, B["up" + l], rx, ry, -1.3, 1.3);
  hinge("el" + l, B["up" + l], B["low" + l], ex, ey, -1.8, 1.8);
}

function build() {
  world = planck.World({ gravity: V(0, -9.8) });
  GROUND = world.createBody();
  GROUND.createFixture(planck.Edge(V(-50, 0), V(50, 0)), { friction: 1.0, restitution: 0.0 });
  B = {}; J = {}; DIMS = {};
  targets = {}; smoothed = {}; grips = {}; gripTimer = {};
  simT = 0; noGripT = 0; camY = null;
  const y0 = START_Y;

  // Tronc du lézard : bassin en bas, poitrine en haut (axe local x = vers le haut)
  part("chest", 0, y0 + 0.13, Math.PI / 2, 0.13, 0.055, 20);
  part("pelvis", 0, y0 - 0.13, Math.PI / 2, 0.13, 0.05, 20);
  // Tête : un disque fixé en haut de la poitrine
  B.chest.createFixture(planck.Circle(V(0.17, 0), 0.06), {
    density: 20 * BODY.massScale, friction: 0.8, restitution: 0.0, filterGroupIndex: NO_SELF,
  });
  hinge("spine", B.pelvis, B.chest, 0, y0, -0.6, 0.6);

  // Queue : 3 segments qui pendent sous le bassin
  part("tail1", 0, y0 - 0.26 - 0.07, -Math.PI / 2, 0.07, 0.03, 10);
  part("tail2", 0, y0 - 0.40 - 0.06, -Math.PI / 2, 0.06, 0.022, 10);
  part("tail3", 0, y0 - 0.52 - 0.05, -Math.PI / 2, 0.05, 0.015, 10);
  hinge("tail1", B.pelvis, B.tail1, 0, y0 - 0.26, -0.8, 0.8);
  hinge("tail2", B.tail1, B.tail2, 0, y0 - 0.40, -0.8, 0.8);
  hinge("tail3", B.tail2, B.tail3, 0, y0 - 0.52, -0.8, 0.8);

  // Pattes écartées : (angle du haut de la patte, angle du bas)
  limb("FL", B.chest, -0.04, y0 + 0.20, deg(150), deg(110));
  limb("FR", B.chest, 0.04, y0 + 0.20, deg(30), deg(70));
  limb("RL", B.pelvis, -0.04, y0 - 0.20, deg(210), deg(250));
  limb("RR", B.pelvis, 0.04, y0 - 0.20, deg(330), deg(290));

  for (const n of CTRL) smoothed[n] = 0;
  // Au départ, les 4 griffes sont plantées dans le tronc
  for (const l of LIMBS) { grips[l] = null; gripTimer[l] = 1; attach(l, footTip(l)); }
}

function initPhysics() {
  build();
  window.simInfo = { nObs: observe().length, nAct: CTRL.length + LIMBS.length, h0: H0 };
}

// ---------- Griffes ----------
function footTip(l) { return B["low" + l].getWorldPoint(V(LL / 2, 0)); }

function attach(l, p) {
  grips[l] = world.createJoint(planck.RevoluteJoint({ collideConnected: false }, B["low" + l], GROUND, p));
  gripTimer[l] = 0;
}

function release(l) {
  if (grips[l]) { world.destroyJoint(grips[l]); grips[l] = null; }
  gripTimer[l] = 0;
}

function updateGrips(cmd) {
  for (const l of LIMBS) {
    gripTimer[l] += 1 / 60;
    const want = cmd[l] > 0;
    const g = grips[l];
    if (g) {
      const f = g.getReactionForce(1 / DT);
      if (Math.hypot(f.x, f.y) > GRIP_MAX) { release(l); continue; }     // la griffe glisse
      if (!want && gripTimer[l] > MIN_HOLD) release(l);
    } else if (want && gripTimer[l] > MIN_HOLD) {
      const p = footTip(l);
      if (Math.abs(p.x) < TRUNK_HALF && p.y > 0.04 && p.y < TREE_TOP) attach(l, p);
    }
  }
}

const gripCount = () => LIMBS.filter((l) => grips[l]).length;

// ---------- Muscles ----------
function setMotors() {
  const alpha = 1 / 60 / (1 / 60 + SMOOTH_TAU);
  for (const n of CTRL) {
    smoothed[n] += ((targets[n] || 0) - smoothed[n]) * alpha;
    const m = JT[typeOf(n)];
    const tgt = signOf(n) * m.r * smoothed[n];
    const w = clamp(m.gain * (tgt - J[n].getJointAngle()), -m.vmax, m.vmax);
    J[n].setMotorSpeed(w);
  }
  for (const n of ["tail2", "tail3"]) {
    const m = JT.tailp;
    J[n].setMotorSpeed(clamp(m.gain * (0 - J[n].getJointAngle()), -m.vmax, m.vmax));
  }
}

// ---------- Capteurs (ce que l'IA observe) ----------
function observe() {
  const o = [];
  for (const n of CTRL) {
    const s = signOf(n);
    o.push((s * J[n].getJointAngle()) / Math.PI, (s * J[n].getJointSpeed()) / 10);
  }
  let a = B.chest.getAngle() - Math.PI / 2;
  a = Math.atan2(Math.sin(a), Math.cos(a));
  const v = B.chest.getLinearVelocity();
  const cp = B.chest.getPosition();
  o.push(a / Math.PI, B.chest.getAngularVelocity() / 10, v.x / 2, v.y / 2, (cp.y - H0) / 1.5);
  for (const l of LIMBS) {
    const p = footTip(l);
    o.push(grips[l] ? 1 : 0, (p.x - cp.x) / 0.4, (p.y - cp.y) / 0.4);
  }
  o.push(Math.sin((2 * Math.PI * simT) / GAIT_T), Math.cos((2 * Math.PI * simT) / GAIT_T));
  return o;
}

// ---------- Un pas de simulation (commun au mode normal et au mode turbo) ----------
function stepCore(a) {
  CTRL.forEach((n, i) => { targets[n] = clamp(a[i], -1, 1); });
  const gcmd = {};
  LIMBS.forEach((l, i) => { gcmd[l] = a[CTRL.length + i]; });
  updateGrips(gcmd);
  setMotors();
  for (let i = 0; i < SUBSTEPS; i++) world.step(DT, 8, 3);
  simT += 1 / 60;
  const y = B.chest.getPosition().y;
  noGripT = gripCount() === 0 ? noGripT + 1 / 60 : 0;
  return {
    obs: observe(),
    y: y,
    grips: LIMBS.map((l) => (grips[l] ? 1 : 0)),
    fallen: y < FALL_Y || noGripT > 0.6,
    success: y >= GOAL_Y,
  };
}

// ---------- Interface appelée depuis Python (mode normal : un pas à la fois) ----------
window.sim = {
  reset() {
    build();
    return JSON.stringify(observe());
  },
  step(json) {
    return JSON.stringify(stepCore(JSON.parse(json)));
  },
};

// ---------- Mode turbo : un essai complet simulé d'un seul bloc ----------
// Le réseau de neurones est recalculé ici (mêmes poids que Python), ce qui évite un
// aller-retour Python à chaque pas. Le travail est découpé en tranches de ~20 ms pour que
// la page reste réactive (boutons, graphique, affichage échantillonné du lézard).
const _mc = new MessageChannel();
let _mcResolve = null;
_mc.port1.onmessage = () => { const r = _mcResolve; _mcResolve = null; if (r) r(); };
const yieldNow = () => new Promise((r) => { _mcResolve = r; _mc.port2.postMessage(0); });

window.fastEpisode = async function (thetaJson, nObs, nHid, nAct, maxSteps, stagnation) {
  const th = JSON.parse(thetaJson);
  let i = 0;
  const W1 = th.slice(i, i + nObs * nHid); i += nObs * nHid;
  const b1 = th.slice(i, i + nHid); i += nHid;
  const W2 = th.slice(i, i + nHid * nAct); i += nHid * nAct;
  const b2 = th.slice(i, i + nAct);
  const hid = new Array(nHid), act = new Array(nAct);
  const policy = (obs) => {
    for (let j = 0; j < nHid; j++) {
      let s = b1[j];
      for (let k = 0; k < nObs; k++) s += obs[k] * W1[k * nHid + j];
      hid[j] = Math.tanh(s);
    }
    for (let j = 0; j < nAct; j++) {
      let s = b2[j];
      for (let k = 0; k < nHid; k++) s += hid[k] * W2[k * nAct + j];
      act[j] = Math.tanh(s);
    }
    return act;
  };

  build();
  let obs = observe();
  let steps = 0, maxY = H0, tImprove = 0, stab = 0, fell = false, success = false;
  let lastGrips = [1, 1, 1, 1];
  const t0 = performance.now();
  let sliceStart = t0;
  while (steps < maxSteps) {
    while (ui.paused && !ui.reset) await window.waitFrame();
    if (ui.reset) return JSON.stringify({ aborted: true });
    const r = stepCore(policy(obs));
    obs = r.obs;
    lastGrips = r.grips;
    steps++;
    const t = steps / 60;
    if (r.y > maxY + 0.01) { maxY = r.y; tImprove = t; }
    if (r.grips[0] + r.grips[1] + r.grips[2] + r.grips[3] >= 3) stab += 1 / 60;
    if (r.fallen) { fell = true; break; }
    if (r.success) { success = true; break; }
    if (t - tImprove > stagnation) break;
    if (performance.now() - sliceStart > 20) { await yieldNow(); sliceStart = performance.now(); }
  }
  return JSON.stringify({
    aborted: false, steps: steps, maxY: maxY, stab: stab, fell: fell, success: success,
    grips: lastGrips, ms: performance.now() - t0,
  });
};

window.waitFrame = () => new Promise((r) => requestAnimationFrame(r));
window.setStats = (txt) => { statsEl.textContent = txt; };

let perfHistory = [];
window.setHistory = (txt) => {
  try { perfHistory = JSON.parse(txt); } catch (e) { perfHistory = []; }
};

// ---------- Rendu ----------
const sx = (x) => canvas.width / 2 + x * PX;
const sy = (y) => canvas.height * 0.6 - (y - camY) * PX;

function drawTree() {
  const x0 = sx(-TRUNK_HALF), x1 = sx(TRUNK_HALF);
  const yTop = sy(TREE_TOP), yBot = sy(0);

  const g = ctx.createLinearGradient(x0, 0, x1, 0);
  g.addColorStop(0, "#5b3d22"); g.addColorStop(0.5, "#7a5232"); g.addColorStop(1, "#4e331c");
  ctx.fillStyle = g;
  ctx.fillRect(x0, yTop, x1 - x0, yBot - yTop);

  // sillons d'écorce
  ctx.strokeStyle = "rgba(30,18,8,0.35)";
  ctx.lineWidth = 2;
  const ys = Math.max(yTop, -20), ye = Math.min(yBot, canvas.height + 20);
  for (let i = 0; i < 9; i++) {
    const x = x0 + ((i + 0.5) * (x1 - x0)) / 9 + Math.sin(i * 7.1) * 6;
    ctx.beginPath();
    for (let y = ys; y <= ye; y += 30) {
      const wob = Math.sin(y * 0.02 + i) * 4;
      if (y === ys) ctx.moveTo(x + wob, y); else ctx.lineTo(x + wob, y);
    }
    ctx.stroke();
  }

  // repères d'altitude
  ctx.font = "11px sans-serif";
  for (let h = 0.5; h < TREE_TOP; h += 0.5) {
    const y = sy(h);
    if (y < -10 || y > canvas.height + 10) continue;
    ctx.strokeStyle = "rgba(255,255,255,0.4)";
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(x0, y); ctx.lineTo(x0 + 16, y); ctx.stroke();
    ctx.fillStyle = "#333";
    ctx.fillText(h.toFixed(1) + " m", x0 - 40, y + 4);
  }

  // ligne d'arrivée
  const yg = sy(GOAL_Y);
  ctx.setLineDash([8, 6]);
  ctx.strokeStyle = "#ffd54a"; ctx.lineWidth = 2;
  ctx.beginPath(); ctx.moveTo(x0 - 20, yg); ctx.lineTo(x1 + 20, yg); ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = "#7a5a00";
  ctx.fillText("SOMMET", x1 + 24, yg + 4);

  // feuillage
  ctx.fillStyle = "#2f6b2a";
  ctx.beginPath(); ctx.ellipse(canvas.width / 2, yTop, 280, 90, 0, 0, 6.283); ctx.fill();
  ctx.fillStyle = "#3f8a37";
  ctx.beginPath(); ctx.ellipse(canvas.width / 2 - 60, yTop - 30, 180, 60, 0, 0, 6.283); ctx.fill();
}

function drawCapsule(name, fill) {
  const b = B[name];
  const [hx, hy] = DIMS[name];
  const p = b.getPosition(), a = b.getAngle();
  const dx = Math.cos(a) * hx, dy = Math.sin(a) * hx;
  const x1 = sx(p.x - dx), y1 = sy(p.y - dy), x2 = sx(p.x + dx), y2 = sy(p.y + dy);
  ctx.lineCap = "round";
  ctx.strokeStyle = "#2f4a1c";                 // contour
  ctx.lineWidth = hy * 2 * PX + 3;
  ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
  ctx.strokeStyle = fill;                      // peau
  ctx.lineWidth = hy * 2 * PX;
  ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
}

function drawLizard() {
  for (const n of ["tail3", "tail2", "tail1"]) drawCapsule(n, "#5d8a30");
  // pattes de droite d'abord (plus sombres), puis celles de gauche
  for (const l of ["RR", "FR"]) { drawCapsule("up" + l, "#527c2b"); drawCapsule("low" + l, "#527c2b"); }
  for (const l of ["RL", "FL"]) { drawCapsule("up" + l, "#6b9b37"); drawCapsule("low" + l, "#6b9b37"); }
  drawCapsule("pelvis", "#6b9b37");
  drawCapsule("chest", "#76a83d");

  // tête
  const h = B.chest.getWorldPoint(V(0.17, 0));
  ctx.fillStyle = "#2f4a1c";
  ctx.beginPath(); ctx.arc(sx(h.x), sy(h.y), 0.06 * PX + 1.5, 0, 6.283); ctx.fill();
  ctx.fillStyle = "#76a83d";
  ctx.beginPath(); ctx.arc(sx(h.x), sy(h.y), 0.06 * PX, 0, 6.283); ctx.fill();
  for (const s of [-1, 1]) {
    const e = B.chest.getWorldPoint(V(0.185, s * 0.03));
    ctx.fillStyle = "#fff";
    ctx.beginPath(); ctx.arc(sx(e.x), sy(e.y), 4, 0, 6.283); ctx.fill();
    ctx.fillStyle = "#111";
    ctx.beginPath(); ctx.arc(sx(e.x), sy(e.y), 2, 0, 6.283); ctx.fill();
  }

  // griffes : dorées quand elles sont plantées dans le tronc
  for (const l of LIMBS) {
    const p = footTip(l);
    ctx.beginPath();
    if (grips[l]) {
      ctx.fillStyle = "#ffcc33"; ctx.strokeStyle = "#a05a00"; ctx.lineWidth = 2;
      ctx.arc(sx(p.x), sy(p.y), 8, 0, 6.283); ctx.fill(); ctx.stroke();
    } else {
      ctx.fillStyle = "#3d5a20";
      ctx.arc(sx(p.x), sy(p.y), 5, 0, 6.283); ctx.fill();
    }
  }
}

function drawChart() {
  const w = 270, h = 120, x = canvas.width - w - 12, y = 12;
  ctx.fillStyle = "rgba(20,20,35,0.78)";
  ctx.fillRect(x, y, w, h);
  ctx.fillStyle = "#fff";
  ctx.font = "11px sans-serif";
  ctx.fillText("Progrès : hauteur gagnée par génération (m)", x + 8, y + 15);
  if (perfHistory.length < 2) {
    ctx.fillStyle = "#aaa";
    ctx.fillText("en attente de la 2e génération…", x + 8, y + 60);
    return;
  }
  const maxV = Math.max(1.5, ...perfHistory.map((p) => p[0]));
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
  ctx.fillStyle = "#7CFC9A"; ctx.fillText("meilleur", x + 150, y + 15 + 0);
  ctx.fillStyle = "#6EC1FF"; ctx.fillText("moyenne", x + 205, y + 15 + 0);
}

function draw() {
  const target = B.chest.getPosition().y + 0.25;
  if (camY === null) camY = target;
  camY += (target - camY) * 0.12;

  ctx.clearRect(0, 0, canvas.width, canvas.height);
  drawTree();
  const yg = sy(0);
  if (yg < canvas.height) {
    ctx.fillStyle = "#5a7d3a";
    ctx.fillRect(0, yg, canvas.width, canvas.height - yg);
  }
  drawLizard();
  drawChart();
}

function renderLoop() { draw(); requestAnimationFrame(renderLoop); }
