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
const DENS = 60;            // densité des membres (kg/m²)
const NO_SELF = -1;         // groupe de filtrage : les membres ne se heurtent pas entre eux
const GROUND_PX = 90;       // hauteur du sol à l'écran (pixels)
const HEAD_R = 0.12;        // rayon de la tête (m)
const V = (x, y) => planck.Vec2(x, y);

// Ordre des 10 actions envoyées par l'IA (valeurs entre -1 et 1)
const ACT = ["hipL", "kneeL", "ankleL", "shL", "elbL", "hipR", "kneeR", "ankleR", "shR", "elbR"];

// Articulations souples : couple = ressort (kp) + amortissement (kd), borné par max (N·m).
// c = angle central, r = amplitude de la consigne autour de c.
const MOT = {
  hip:   { c: 0.3,  r: 0.9, kp: 120, kd: 8,   max: 150 },
  knee:  { c: -0.4, r: 1.0, kp: 90,  kd: 6,   max: 120 },
  ankle: { c: 0.0,  r: 0.5, kp: 40,  kd: 3,   max: 50 },
  sh:    { c: 0.0,  r: 2.0, kp: 6,   kd: 1.2, max: 20 },
  elb:   { c: -1.2, r: 1.0, kp: 4,   kd: 0.8, max: 12 },
};
const kindOf = (n) => n.replace(/[LR]$/, "");

let world, B = {}, J = {}, JD = {}, DIMS = {}, HEAD_START = 1, X0 = 0;
let targets = {};

function part(name, x, y, angle, hx, hy, dens = DENS) {
  const b = world.createBody({ type: "dynamic", position: V(x * K, y * K), angle, angularDamping: 0.02 });
  const th = hy * THIN;
  DIMS[name] = [hx * K, th * K];
  b.createFixture(planck.Box(hx * K, th * K), { density: dens, friction: 0.9, filterGroupIndex: NO_SELF });
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
  world.createBody().createFixture(planck.Edge(V(-500, 0), V(500, 0)), { friction: 0.9 });
  B = {}; J = {}; JD = {}; DIMS = {}; targets = {};
  const h = 1.1; // hauteur de la hanche avant mise à l'échelle

  part("torso", 0, h + 0.35, Math.PI / 2, 0.35, 0.12, 40);
  for (const [sd, s] of [["L", -1], ["R", 1]]) {
    part("thigh" + sd, s * 0.07, h - 0.22, -Math.PI / 2, 0.22, 0.07);
    part("shin" + sd,  s * 0.07, h - 0.66, -Math.PI / 2, 0.22, 0.06);
    part("foot" + sd,  s * 0.03, h - 0.92, 0, 0.18, 0.04);
    part("uarm" + sd,  s * 0.12, h + 0.47, -Math.PI / 2, 0.13, 0.05);
    part("farm" + sd,  s * 0.12, h + 0.21, -Math.PI / 2, 0.13, 0.045);
  }
  HEAD_START = (h + 0.82) * K;
  const head = world.createBody({ type: "dynamic", position: V(0, HEAD_START), angularDamping: 0.02 });
  head.createFixture(planck.Circle(HEAD_R * K), { density: DENS, friction: 0.9, filterGroupIndex: NO_SELF });
  B.head = head;

  hinge("neck", B.torso, B.head, 0, h + 0.70, -0.8, 0.8);
  for (const [sd, s] of [["L", -1], ["R", 1]]) {
    const xh = s * 0.07, xs = s * 0.12;
    hinge("hip" + sd,   B.torso,         B["thigh" + sd], xh, h,        -0.8, 2.2);
    hinge("knee" + sd,  B["thigh" + sd], B["shin" + sd],  xh, h - 0.44, -2.6, 0.3);
    hinge("ankle" + sd, B["shin" + sd],  B["foot" + sd],  xh, h - 0.88, -0.9, 0.9);
    hinge("sh" + sd,    B.torso,         B["uarm" + sd], xs, h + 0.60, -3.0, 3.0);
    hinge("elb" + sd,   B["uarm" + sd],  B["farm" + sd], xs, h + 0.34, -2.8, 0.4);
  }
  X0 = B.torso.getPosition().x;
}

// ---------- Capteurs (ce que l'IA observe à chaque pas) ----------
function observe() {
  const o = [];
  for (const n of ACT) o.push(J[n].getJointAngle() / Math.PI, J[n].getJointSpeed() / 10);
  let a = B.torso.getAngle() - Math.PI / 2;
  a = Math.atan2(Math.sin(a), Math.cos(a));
  const v = B.torso.getLinearVelocity();
  o.push(a / Math.PI, B.torso.getAngularVelocity() / 10, v.x / 3, v.y / 3, B.head.getPosition().y / HEAD_START);
  o.push(B.footL.getPosition().y < 0.1 * K ? 1 : 0, B.footR.getPosition().y < 0.1 * K ? 1 : 0);
  return o;
}

function isFallen() { return B.head.getPosition().y < 0.55 * HEAD_START; }

function applyTorques() {
  for (const n of ACT) {
    const m = MOT[kindOf(n)];
    const q = J[n].getJointAngle();
    const qd = J[n].getJointSpeed();
    const target = m.c + (targets[n] || 0) * m.r;
    let tau = m.kp * (target - q) - m.kd * qd;
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
    return JSON.stringify({ obs: observe(), dx: B.torso.getPosition().x - X0, fallen: isFallen() });
  },
};

window.waitFrame = () => new Promise((r) => requestAnimationFrame(r));
window.setStats = (txt) => { statsEl.textContent = txt; };

// ---------- Rendu et caméra ----------
function toScreen(x, y, camX) {
  return { x: canvas.width / 2 + (x - camX) * PX, y: canvas.height - GROUND_PX - y * PX };
}

function drawLimb(name, camX) {
  const b = B[name];
  const [hx, hy] = DIMS[name];
  const p = b.getPosition(), a = b.getAngle();
  const dx = Math.cos(a) * hx, dy = Math.sin(a) * hx;
  const A = toScreen(p.x - dx, p.y - dy, camX);
  const Z = toScreen(p.x + dx, p.y + dy, camX);
  ctx.lineWidth = hy * 2 * PX;
  ctx.beginPath(); ctx.moveTo(A.x, A.y); ctx.lineTo(Z.x, Z.y); ctx.stroke();
}

function draw() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const y0 = canvas.height - GROUND_PX;
  ctx.fillStyle = "#5a7d3a";
  ctx.fillRect(0, y0, canvas.width, GROUND_PX);
  const camX = B.torso.getPosition().x;
  ctx.strokeStyle = "#222";
  ctx.lineCap = "round";
  for (const n of Object.keys(DIMS)) drawLimb(n, camX);
  const hp = B.head.getPosition();
  const H = toScreen(hp.x, hp.y, camX);
  ctx.fillStyle = "#222";
  ctx.beginPath(); ctx.arc(H.x, H.y, HEAD_R * K * PX, 0, 6.283); ctx.fill();
}

function renderLoop() { draw(); requestAnimationFrame(renderLoop); }
