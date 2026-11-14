// ============================================================
// WALKING AI — Physique (planck.js, portage de Box2D)
// Construit le bonhomme, applique les commandes de l'IA,
// gère le sol et expose tout à Python via window.sim.
// ============================================================

const canvas = document.getElementById("cv");
const ctx = canvas.getContext("2d");
const statsEl = document.getElementById("stats");

const ui = { paused: false, reset: false, speed: 1 };
window.ui = ui;

function resize() { canvas.width = canvas.clientWidth; canvas.height = canvas.clientHeight; }
window.addEventListener("resize", resize);
resize();

// ---------- Constantes ----------
const PX = 220;             // pixels par mètre à l'écran
const K = 0.7;              // échelle générale du bonhomme (plus petit que nature)
const THIN = 0.5;           // épaisseur des membres (plus petit = plus fin)
const DT = 1 / 240;         // pas de physique fixe
const SUBSTEPS = 4;         // pas physiques par action (1 action = 1/60 s de jeu)
const DENS = 55;            // densité des membres (kg/m²)
const NO_SELF = -1;         // groupe de filtrage : les membres ne se heurtent pas entre eux
const GROUND_PX = 90;       // hauteur du sol à l'écran (pixels)
const HEAD_R = 0.12;        // rayon de la tête (m)
const V = (x, y) => planck.Vec2(x, y);

// Ordre des 11 actions envoyées par l'IA (valeurs entre -1 et 1).
// "spine" permet au buste de se pencher pour chercher l'équilibre, comme un vrai corps.
const ACT = ["spine", "hipL", "kneeL", "ankleL", "shL", "elbL", "hipR", "kneeR", "ankleR", "shR", "elbR"];

// Articulations souples : couple = ressort (kp) vers la cible + amortissement critique (calculé
// dynamiquement à partir de l'inertie réelle), borné par max (N·m).
// c = angle central, r = amplitude de la consigne autour de c.
// damping : 1.0 = amortissement critique (aucune oscillation), <1 = un peu plus vivant/naturel.
const MOT = {
  spine: { c: 0.0,  r: 0.5, kp: 35, max: 70,  damping: 0.9 },
  hip:   { c: 0.25, r: 0.9, kp: 55, max: 130, damping: 0.9 },
  knee:  { c: -0.35,r: 1.0, kp: 45, max: 110, damping: 0.9 },
  ankle: { c: 0.0,  r: 0.5, kp: 20, max: 45,  damping: 0.9 },
  sh:    { c: 0.0,  r: 2.0, kp: 5,  max: 18,  damping: 0.85 },
  elb:   { c: -1.2, r: 1.0, kp: 4,  max: 11,  damping: 0.85 },
};
const kindOf = (n) => n.replace(/[LR]$/, "");

// Lissage des commandes reçues de l'IA : évite les à-coups d'une consigne à l'autre.
const SMOOTH_TAU = 0.1; // secondes

let world, B = {}, J = {}, JD = {}, DIMS = {}, HEAD_START = 1, X0 = 0;
let targets = {};     // consignes brutes envoyées par l'IA
let smoothed = {};    // consignes lissées, réellement utilisées

function part(name, x, y, angle, hx, hy, dens = DENS) {
  const b = world.createBody({ type: "dynamic", position: V(x * K, y * K), angle, angularDamping: 0.04, linearDamping: 0.0 });
  const th = hy * THIN;
  DIMS[name] = [hx * K, th * K];
  b.createFixture(planck.Box(hx * K, th * K), { density: dens, friction: 0.9, restitution: 0.0, filterGroupIndex: NO_SELF });
  B[name] = b;
}

function hinge(name, a, b, ax, ay, lo, hi) {
  const def = {
    referenceAngle: b.getAngle() - a.getAngle(),
    enableLimit: true, lowerAngle: lo, upperAngle: hi,
    enableMotor: false,
    collideConnected: false,
  };
  J[name] = world.createJoint(planck.RevoluteJoint(def, a, b, V(ax * K, ay * K)));
  JD[name] = { a, b };
}

// ---------- Construction du bonhomme ----------
function initPhysics() { build(); }

function build() {
  world = planck.World({ gravity: V(0, -9.8) });
  world.createBody().createFixture(planck.Edge(V(-500, 0), V(500, 0)), { friction: 1.0, restitution: 0.0 });
  B = {}; J = {}; JD = {}; DIMS = {}; targets = {}; smoothed = {};
  const h = 1.1; // hauteur de la hanche avant mise à l'échelle

  // Bassin (racine : les hanches et la colonne partent d'ici)
  part("pelvis", 0, h + 0.10, Math.PI / 2, 0.18, 0.10, 45);
  // Poitrine, reliée au bassin par la colonne : c'est elle qui porte bras, cou et tête
  part("chest",  0, h + 0.45, Math.PI / 2, 0.22, 0.11, 35);

  for (const [sd, s] of [["L", -1], ["R", 1]]) {
    part("thigh" + sd, s * 0.07, h - 0.22, -Math.PI / 2, 0.22, 0.07);
    part("shin" + sd,  s * 0.07, h - 0.66, -Math.PI / 2, 0.22, 0.06);
    part("foot" + sd,  s * 0.03, h - 0.92, 0, 0.18, 0.04);
    part("uarm" + sd,  s * 0.12, h + 0.58, -Math.PI / 2, 0.13, 0.05);
    part("farm" + sd,  s * 0.12, h + 0.32, -Math.PI / 2, 0.13, 0.045);
  }
  HEAD_START = (h + 0.92) * K;
  const head = world.createBody({ type: "dynamic", position: V(0, HEAD_START), angularDamping: 0.04 });
  head.createFixture(planck.Circle(HEAD_R * K), { density: DENS, friction: 0.9, restitution: 0.0, filterGroupIndex: NO_SELF });
  B.head = head;

  hinge("spine", B.pelvis, B.chest, 0, h + 0.28, -0.3, 0.5);
  hinge("neck", B.chest, B.head, 0, h + 0.80, -0.8, 0.8);
  for (const [sd, s] of [["L", -1], ["R", 1]]) {
    const xh = s * 0.07, xs = s * 0.12;
    hinge("hip" + sd,   B.pelvis,        B["thigh" + sd], xh, h,        -0.8, 2.2);
    hinge("knee" + sd,  B["thigh" + sd], B["shin" + sd],  xh, h - 0.44, -2.6, 0.3);
    hinge("ankle" + sd, B["shin" + sd],  B["foot" + sd],  xh, h - 0.88, -0.9, 0.9);
    hinge("sh" + sd,    B.chest,         B["uarm" + sd], xs, h + 0.70, -3.0, 3.0);
    hinge("elb" + sd,   B["uarm" + sd],  B["farm" + sd], xs, h + 0.45, -2.8, 0.4);
  }
  X0 = B.pelvis.getPosition().x;
}

// ---------- Capteurs (ce que l'IA observe à chaque pas) ----------
function observe() {
  const o = [];
  for (const n of ACT) o.push(J[n].getJointAngle() / Math.PI, J[n].getJointSpeed() / 10);
  let a = B.chest.getAngle() - Math.PI / 2;
  a = Math.atan2(Math.sin(a), Math.cos(a));
  const v = B.pelvis.getLinearVelocity();
  o.push(a / Math.PI, B.chest.getAngularVelocity() / 10, v.x / 3, v.y / 3, B.head.getPosition().y / HEAD_START);
  o.push(B.footL.getPosition().y < 0.1 * K ? 1 : 0, B.footR.getPosition().y < 0.1 * K ? 1 : 0);
  return o;
}

function isFallen() { return B.head.getPosition().y < 0.55 * HEAD_START; }

// Couple = ressort critique vers la cible lissée. L'amortissement critique est calculé à partir
// de l'inertie réelle de chaque membre, ce qui évite les oscillations et les à-coups rigides.
function applyTorques() {
  for (const n of ACT) {
    const m = MOT[kindOf(n)];
    const alpha = DT / (DT + SMOOTH_TAU);
    smoothed[n] = (smoothed[n] || 0) + ((targets[n] || 0) - (smoothed[n] || 0)) * alpha;

    const q = J[n].getJointAngle();
    const qd = J[n].getJointSpeed();
    const target = m.c + smoothed[n] * m.r;

    const I = Math.max(JD[n].b.getInertia(), 1e-4);
    const kd = m.damping * 2 * Math.sqrt(m.kp * I);
    let tau = m.kp * (target - q) - kd * qd;
    tau = Math.max(-m.max, Math.min(m.max, tau));
    JD[n].b.applyTorque(tau, true);
    JD[n].a.applyTorque(-tau, true);
  }
}

// ---------- Interface appelée depuis Python ----------
window.sim = {
  reset() {
    build();
    return JSON.stringify(observe());
  },
  step(json) {
    const a = JSON.parse(json);
    ACT.forEach((n, i) => { targets[n] = Math.max(-1, Math.min(1, a[i])); });
    for (let i = 0; i < SUBSTEPS; i++) {
      applyTorques();
      world.step(DT, 8, 3);
    }
    return JSON.stringify({ obs: observe(), dx: B.pelvis.getPosition().x - X0, fallen: isFallen() });
  },
};

window.waitFrame = () => new Promise((r) => requestAnimationFrame(r));
window.setStats = (txt) => { statsEl.textContent = txt; };

// ---------- Rendu et caméra ----------
function toScreen(x, y, camX) {
  return { x: canvas.width / 2 + (x - camX) * PX, y: canvas.height - GROUND_PX - y * PX };
}

// Dessine un membre comme une capsule arrondie avec un dégradé, pour un rendu plus organique
// qu'un simple trait.
function drawLimb(name, camX) {
  const b = B[name];
  const [hx, hy] = DIMS[name];
  const p = b.getPosition(), a = b.getAngle();
  const dx = Math.cos(a) * hx, dy = Math.sin(a) * hx;
  const A = toScreen(p.x - dx, p.y - dy, camX);
  const Z = toScreen(p.x + dx, p.y + dy, camX);
  const width = hy * 2 * PX;

  const grad = ctx.createLinearGradient(A.x, A.y - width / 2, A.x, A.y + width / 2);
  grad.addColorStop(0, "#f3c9a0");
  grad.addColorStop(0.5, "#e0a878");
  grad.addColorStop(1, "#c98a5c");

  ctx.strokeStyle = grad;
  ctx.lineWidth = width;
  ctx.lineCap = "round";
  ctx.beginPath(); ctx.moveTo(A.x, A.y); ctx.lineTo(Z.x, Z.y); ctx.stroke();

  // léger contour pour détacher le membre du fond
  ctx.strokeStyle = "rgba(90,55,30,0.35)";
  ctx.lineWidth = 1.5;
  ctx.beginPath(); ctx.moveTo(A.x, A.y); ctx.lineTo(Z.x, Z.y); ctx.stroke();
}

function draw() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const y0 = canvas.height - GROUND_PX;
  ctx.fillStyle = "#5a7d3a";
  ctx.fillRect(0, y0, canvas.width, GROUND_PX);
  const camX = B.pelvis.getPosition().x;

  // membres, tronc (bassin + poitrine) dessiné en dernier pour recouvrir les attaches
  const order = Object.keys(DIMS).filter((n) => n !== "pelvis" && n !== "chest");
  for (const n of order) drawLimb(n, camX);
  drawLimb("pelvis", camX);
  drawLimb("chest", camX);

  // articulations : petits ronds discrets de la même teinte, pas de points rouges
  ctx.fillStyle = "#e0a878";
  for (const n of Object.keys(JD)) {
    const j = J[n];
    const anchor = j.getAnchorA();
    const pt = toScreen(anchor.x, anchor.y, camX);
    ctx.beginPath(); ctx.arc(pt.x, pt.y, 3, 0, 6.283); ctx.fill();
  }

  // tête
  const hp = B.head.getPosition();
  const H = toScreen(hp.x, hp.y, camX);
  const headGrad = ctx.createRadialGradient(H.x - 4, H.y - 4, 2, H.x, H.y, HEAD_R * K * PX);
  headGrad.addColorStop(0, "#f3c9a0");
  headGrad.addColorStop(1, "#d89b6c");
  ctx.fillStyle = headGrad;
  ctx.beginPath(); ctx.arc(H.x, H.y, HEAD_R * K * PX, 0, 6.283); ctx.fill();
}

function renderLoop() { draw(); requestAnimationFrame(renderLoop); }
