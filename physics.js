// ============================================================
// LEZARD AI — Noyau de simulation d'UN lézard (planck.js / Box2D)
// Aucun accès au DOM : ce fichier sert à la page (constantes, dimensions)
// et aux Workers (simulation). Le plan de l'écran est le plan du tronc :
// le lézard est vu de face, collé à l'écorce, la gravité le tire vers le bas.
//
// COHÉRENCE MUSCLES / ÉNERGIE / POIDS
//  - Chaque muscle a une force maximale (couple, N·m). Plus il est fort, plus il pèse
//    (MUSCLE_KG_PER_NM) et plus il coûte cher à entretenir (BASAL_PER_NM).
//  - La réserve d'énergie a un poids (FAT_KG_PER_UNIT par unité de gène « energy »).
//  - Le squelette a une masse propre (gène « mass »). Les griffes tiennent d'autant mieux
//    qu'il est massif, mais moins que proportionnellement (loi carré-cube : ^(2/3)).
//  - Tout se paie en énergie : tenir une position (HOLD_COST), travailler (WORK_COST),
//    freiner (BRAKE_COST), et simplement vivre (basal). Plus le lézard est lourd, plus
//    ses muscles doivent forcer, donc plus il consomme.
// ============================================================

const DT = 1 / 240;                 // pas de physique fixe
const SUBSTEPS = 4;                 // pas physiques par tick (1 tick = 1/60 s)
const V = (x, y) => planck.Vec2(x, y);
const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
const deg = (d) => (d * Math.PI) / 180;

// ---------- Arbre : AUCUNE limite de hauteur ----------
const TRUNK_HALF = 0.6;             // demi-largeur du tronc (m) : on ne s'accroche que dessus
const START_Y = 1.0;                // hauteur de la colonne vertébrale au départ
const H0 = START_Y + 0.13;          // hauteur de départ du centre de la poitrine
const FALL_Y = 0.75;                // sous cette hauteur : tombé
const T_MAX = 120;                  // filet de sécurité (s) : un lézard ne vit pas plus longtemps

// ---------- Griffes ----------
const GRIP_BASE = 70;               // force de griffe (N) pour un squelette de masse ×1
const MIN_HOLD = 0.2;               // durée minimale (s) entre deux changements d'état d'une griffe

// ---------- Gènes du corps (ordre fixe, partagé avec Python) ----------
//   mass   : masse du squelette         spine/tail : force des muscles du tronc et de la queue
//   shF/elF: épaules/coudes avant       shR/elR    : hanches/genoux arrière
//   energy : taille de la réserve d'énergie
const BODY_KEYS = ["mass", "spine", "tail", "shF", "elF", "shR", "elR", "energy"];
const BODY_LO = [0.6, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.4];
const BODY_HI = [1.8, 3, 3, 3, 3, 3, 3, 3];

// ---------- Masses (kg), pour un gène ×1 ----------
const SKEL = { chest: 0.50, pelvis: 0.40, up: 0.04, low: 0.05, tail1: 0.03, tail2: 0.02, tail3: 0.015 };
const MUSCLE_KG_PER_NM = 0.02;      // masse de muscle par N·m de force maximale
const FAT_KG_PER_UNIT = 0.25;       // masse de la réserve d'énergie pour un gène energy = 1

// ---------- Énergie (unités : joules et watts « de jeu ») ----------
const ENERGY_BASE = 200;            // capacité (J) pour un gène energy = 1
const BASAL_W0 = 0.5;               // métabolisme de base (W)
const BASAL_PER_NM = 0.08;          // entretien des muscles (W par N·m de force maximale)
const HOLD_COST = 0.3;              // coût pour tenir un couple (W par N·m), même immobile
const WORK_COST = 1.0;              // coût du travail mécanique positif (W par W produit)
const BRAKE_COST = 0.25;            // coût du freinage (W par W absorbé)

// ---------- Muscles : force de base (N·m) et réglages des moteurs ----------
// r = amplitude de la consigne (rad), tmax = force pour un gène ×1,
// gain = réactivité, vmax = vitesse angulaire maximale (rad/s).
const JT = {
  spine: { r: 0.6, tmax: 4.0, gain: 6, vmax: 3 },
  tail1: { r: 0.8, tmax: 0.8, gain: 5, vmax: 3 },
  sh:    { r: 1.2, tmax: 4.0, gain: 8, vmax: 5 },
  el:    { r: 1.6, tmax: 3.0, gain: 8, vmax: 6 },
  tailp: { r: 0.0, tmax: 0.3, gain: 3, vmax: 2 },   // bout de queue : passif, souple
};
const SMOOTH_TAU = 0.08;            // lissage des commandes de l'IA (s)
const GAIT_T = 1.6;                 // période de l'horloge interne fournie à l'IA (s)

const CTRL = ["spine", "tail1", "shFL", "elFL", "shFR", "elFR", "shRL", "elRL", "shRR", "elRR"];
const LIMBS = ["FL", "FR", "RL", "RR"];   // avant-gauche, avant-droite, arrière-gauche, arrière-droite
const LU = 0.12, LL = 0.14;               // longueurs haut / bas de patte (m)

const typeOf = (n) => {
  if (n.startsWith("sh")) return "sh";
  if (n.startsWith("el")) return "el";
  if (n === "tail2" || n === "tail3") return "tailp";
  return n;
};
// Les pattes de droite sont le miroir de celles de gauche
const signOf = (n) => ((n.startsWith("sh") || n.startsWith("el")) && n.endsWith("R") ? -1 : 1);

// Dimensions (demi-longueur, demi-épaisseur) des segments : utilisées pour le rendu
const DIMS = {
  chest: [0.13, 0.055], pelvis: [0.13, 0.05],
  tail1: [0.07, 0.03], tail2: [0.06, 0.022], tail3: [0.05, 0.015],
};
for (const l of LIMBS) { DIMS["up" + l] = [LU / 2, 0.022]; DIMS["low" + l] = [LL / 2, 0.018]; }
// Ordre de dessin (arrière-plan d'abord)
const POSE_NAMES = ["tail3", "tail2", "tail1", "upRR", "lowRR", "upFR", "lowFR",
                    "upRL", "lowRL", "upFL", "lowFL", "pelvis", "chest"];

// ---------- Réseau de neurones ----------
const NH = 20;                                        // neurones cachés
const N_ACT = CTRL.length + LIMBS.length;             // 10 muscles + 4 griffes
const N_OBS = CTRL.length * 2 + 5 + 1 + LIMBS.length * 3 + 2;   // = 40
const N_PARAMS = N_OBS * NH + NH + NH * N_ACT + N_ACT;

const SIM_INFO = {
  nObs: N_OBS, nAct: N_ACT, nHid: NH, nGrip: LIMBS.length, nParams: N_PARAMS,
  bodyKeys: BODY_KEYS, lo: BODY_LO, hi: BODY_HI, h0: H0, tMax: T_MAX,
};
if (typeof self !== "undefined") self.simInfoJson = JSON.stringify(SIM_INFO);

// ============================================================
// Un lézard = un monde physique à lui. brain : Float32Array(N_PARAMS) ; g : gènes {clé: valeur}
// ============================================================
function createLizard(brain, g) {
  const world = planck.World({ gravity: V(0, -9.8) });
  const ground = world.createBody();
  ground.createFixture(planck.Edge(V(-50, 0), V(50, 0)), { friction: 1.0, restitution: 0.0 });
  const B = {}, J = {}, grips = {}, gripTimer = {}, targets = {}, smoothed = {}, gcmd = {};
  const gripMax = GRIP_BASE * Math.pow(g.mass, 2 / 3);
  let noGripT = 0, totalMass = 0;

  // Force maximale d'un muscle = force de base × gène correspondant
  const tmaxOf = (n) => {
    const ty = typeOf(n);
    if (ty === "sh" || ty === "el") return JT[ty].tmax * g[ty + n[2]];   // shF, elF, shR, elR
    if (ty === "spine") return JT.spine.tmax * g.spine;
    if (ty === "tail1") return JT.tail1.tmax * g.tail;
    return JT.tailp.tmax;
  };

  // Un segment : la masse est fixée directement en kg (densité = masse / aire)
  function part(name, cx, cy, angle, hx, hy, mass, circ) {
    const b = world.createBody({ type: "dynamic", position: V(cx, cy), angle: angle, angularDamping: 0.1 });
    const area = 4 * hx * hy + (circ ? Math.PI * circ.r * circ.r : 0);
    const fd = { density: mass / area, friction: 0.8, restitution: 0.0, filterGroupIndex: -1 };
    b.createFixture(planck.Box(hx, hy), fd);
    if (circ) b.createFixture(planck.Circle(V(circ.x, 0), circ.r), fd);
    B[name] = b;
    totalMass += mass;
    return b;
  }

  function hinge(name, a, b, ax, ay, lo, hi) {
    J[name] = world.createJoint(planck.RevoluteJoint({
      referenceAngle: b.getAngle() - a.getAngle(),
      enableLimit: true, lowerAngle: lo, upperAngle: hi,
      enableMotor: true, maxMotorTorque: tmaxOf(name), motorSpeed: 0,
      collideConnected: false,
    }, a, b, V(ax, ay)));
  }

  function limb(l, root, rx, ry, a1, a2) {
    const ex = rx + Math.cos(a1) * LU, ey = ry + Math.sin(a1) * LU;
    const fx = ex + Math.cos(a2) * LL, fy = ey + Math.sin(a2) * LL;
    const mUp = SKEL.up * g.mass + MUSCLE_KG_PER_NM * tmaxOf("sh" + l);
    const mLow = SKEL.low * g.mass + MUSCLE_KG_PER_NM * tmaxOf("el" + l);
    part("up" + l, (rx + ex) / 2, (ry + ey) / 2, a1, LU / 2, 0.022, mUp);
    part("low" + l, (ex + fx) / 2, (ey + fy) / 2, a2, LL / 2, 0.018, mLow, { x: LL / 2, r: 0.022 });
    hinge("sh" + l, root, B["up" + l], rx, ry, -1.3, 1.3);
    hinge("el" + l, B["up" + l], B["low" + l], ex, ey, -1.8, 1.8);
  }

  // ----- construction du corps -----
  const y0 = START_Y;
  part("chest", 0, y0 + 0.13, Math.PI / 2, 0.13, 0.055,
       SKEL.chest * g.mass + MUSCLE_KG_PER_NM * tmaxOf("spine"), { x: 0.17, r: 0.06 });   // + tête
  part("pelvis", 0, y0 - 0.13, Math.PI / 2, 0.13, 0.05,
       SKEL.pelvis * g.mass + FAT_KG_PER_UNIT * g.energy);                                // + réserve d'énergie
  hinge("spine", B.pelvis, B.chest, 0, y0, -0.6, 0.6);

  part("tail1", 0, y0 - 0.33, -Math.PI / 2, 0.07, 0.03, SKEL.tail1 * g.mass + MUSCLE_KG_PER_NM * tmaxOf("tail1"));
  part("tail2", 0, y0 - 0.46, -Math.PI / 2, 0.06, 0.022, SKEL.tail2 * g.mass);
  part("tail3", 0, y0 - 0.57, -Math.PI / 2, 0.05, 0.015, SKEL.tail3 * g.mass);
  hinge("tail1", B.pelvis, B.tail1, 0, y0 - 0.26, -0.8, 0.8);
  hinge("tail2", B.tail1, B.tail2, 0, y0 - 0.40, -0.8, 0.8);
  hinge("tail3", B.tail2, B.tail3, 0, y0 - 0.52, -0.8, 0.8);

  limb("FL", B.chest, -0.04, y0 + 0.20, deg(150), deg(110));
  limb("FR", B.chest, 0.04, y0 + 0.20, deg(30), deg(70));
  limb("RL", B.pelvis, -0.04, y0 - 0.20, deg(210), deg(250));
  limb("RR", B.pelvis, 0.04, y0 - 0.20, deg(330), deg(290));

  let sumTmax = 0;
  for (const n of CTRL) { smoothed[n] = 0; sumTmax += tmaxOf(n); }
  const basal = BASAL_W0 + BASAL_PER_NM * sumTmax;
  const E0 = ENERGY_BASE * g.energy;

  // ----- griffes -----
  const footTip = (l) => B["low" + l].getWorldPoint(V(LL / 2, 0));
  function attach(l, p) {
    grips[l] = world.createJoint(planck.RevoluteJoint({ collideConnected: false }, B["low" + l], ground, p));
    gripTimer[l] = 0;
  }
  function release(l) {
    if (grips[l]) { world.destroyJoint(grips[l]); grips[l] = null; }
    gripTimer[l] = 0;
  }
  function updateGrips() {
    for (const l of LIMBS) {
      gripTimer[l] += 1 / 60;
      const want = gcmd[l] > 0;
      const gj = grips[l];
      if (gj) {
        const f = gj.getReactionForce(1 / DT);
        if (Math.hypot(f.x, f.y) > gripMax) { release(l); continue; }          // la griffe glisse
        if (!want && gripTimer[l] > MIN_HOLD) release(l);
      } else if (want && gripTimer[l] > MIN_HOLD) {
        const p = footTip(l);
        if (Math.abs(p.x) < TRUNK_HALF && p.y > 0.04) attach(l, p);            // pas de limite en hauteur
      }
    }
  }
  for (const l of LIMBS) { grips[l] = null; gripTimer[l] = 1; attach(l, footTip(l)); }
  const gripCount = () => LIMBS.filter((l) => grips[l]).length;

  // ----- muscles -----
  function setMotors() {
    const alpha = 1 / 60 / (1 / 60 + SMOOTH_TAU);
    for (const n of CTRL) {
      smoothed[n] += ((targets[n] || 0) - smoothed[n]) * alpha;
      const m = JT[typeOf(n)];
      const tgt = signOf(n) * m.r * smoothed[n];
      J[n].setMotorSpeed(clamp(m.gain * (tgt - J[n].getJointAngle()), -m.vmax, m.vmax));
    }
    for (const n of ["tail2", "tail3"]) {
      const m = JT.tailp;
      J[n].setMotorSpeed(clamp(m.gain * (0 - J[n].getJointAngle()), -m.vmax, m.vmax));
    }
  }

  // ----- cerveau : réseau de neurones à une couche cachée -----
  const W1 = brain.subarray(0, N_OBS * NH);
  const b1 = brain.subarray(N_OBS * NH, N_OBS * NH + NH);
  const W2 = brain.subarray(N_OBS * NH + NH, N_OBS * NH + NH + NH * N_ACT);
  const b2 = brain.subarray(N_OBS * NH + NH + NH * N_ACT);
  const hid = new Float64Array(NH), act = new Float64Array(N_ACT);
  function policy(obs) {
    for (let j = 0; j < NH; j++) {
      let s = b1[j];
      for (let k = 0; k < N_OBS; k++) s += obs[k] * W1[k * NH + j];
      hid[j] = Math.tanh(s);
    }
    for (let j = 0; j < N_ACT; j++) {
      let s = b2[j];
      for (let k = 0; k < NH; k++) s += hid[k] * W2[k * N_ACT + j];
      const v = Math.tanh(s);
      act[j] = Number.isFinite(v) ? v : 0;
    }
    return act;
  }

  const lz = {
    finished: false, cause: 0,          // cause : 0 actif, 1 à court d'énergie, 2 tombé, 3 temps maximal
    t: 0, y: H0, maxY: H0, energy: E0, energyUsed: 0,
    mass: 0, genesArr: BODY_KEYS.map((k) => g[k]),
  };
  lz.mass = totalMass;

  // ----- ce que le lézard perçoit -----
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
    o.push(a / Math.PI, B.chest.getAngularVelocity() / 10, v.x / 2, v.y / 2, clamp((cp.y - H0) / 1.5, -2, 2));
    o.push(clamp(lz.energy / E0, 0, 1));
    for (const l of LIMBS) {
      const p = footTip(l);
      o.push(grips[l] ? 1 : 0, (p.x - cp.x) / 0.4, (p.y - cp.y) / 0.4);
    }
    o.push(Math.sin((2 * Math.PI * lz.t) / GAIT_T), Math.cos((2 * Math.PI * lz.t) / GAIT_T));
    return o;
  }

  // ----- un tick = 1/60 s -----
  lz.step = function () {
    if (lz.finished) return;
    const a = policy(observe());
    for (let i = 0; i < CTRL.length; i++) targets[CTRL[i]] = clamp(a[i], -1, 1);
    for (let i = 0; i < LIMBS.length; i++) gcmd[LIMBS[i]] = a[CTRL.length + i];
    updateGrips();
    setMotors();
    for (let i = 0; i < SUBSTEPS; i++) world.step(DT, 8, 3);
    lz.t += 1 / 60;

    const y = B.chest.getPosition().y;
    if (!Number.isFinite(y)) { lz.finished = true; lz.cause = 2; return; }
    lz.y = y;
    if (y > lz.maxY) lz.maxY = y;
    noGripT = gripCount() === 0 ? noGripT + 1 / 60 : 0;

    // consommation d'énergie de ce tick
    let P = basal;
    for (const n of CTRL) {
      const j = J[n];
      const tq = j.getMotorTorque ? j.getMotorTorque(1 / DT) : 0;
      if (!Number.isFinite(tq)) continue;
      const pw = tq * j.getJointSpeed();
      P += HOLD_COST * Math.abs(tq) + (pw > 0 ? WORK_COST * pw : BRAKE_COST * -pw);
    }
    lz.energy -= P / 60;
    lz.energyUsed = Math.min(E0, E0 - lz.energy);

    if (y < FALL_Y || noGripT > 0.6) { lz.finished = true; lz.cause = 2; }
    else if (lz.energy <= 0) { lz.finished = true; lz.cause = 1; lz.energyUsed = E0; }
    else if (lz.t >= T_MAX) { lz.finished = true; lz.cause = 3; }
  };

  // ----- données pour l'affichage -----
  lz.pose = function () {
    const o = new Array(POSE_NAMES.length * 3);
    POSE_NAMES.forEach((n, i) => {
      const b = B[n], p = b.getPosition();
      o[i * 3] = p.x; o[i * 3 + 1] = p.y; o[i * 3 + 2] = b.getAngle();
    });
    return o;
  };
  lz.gripsArr = () => LIMBS.map((l) => (grips[l] ? 1 : 0));
  return lz;
}
