# ============================================================
# LEZARD AI — Apprentissage (Python, exécuté dans le navigateur via Pyodide)
# Un réseau de neurones pilote 10 muscles et 4 griffes. Il est entraîné par
# stratégies d'évolution : à chaque génération, 16 variantes (8 paires
# symétriques) grimpent à l'arbre ; les meilleures orientent la suivante.
# La physique est déléguée à physics.js (window.sim).
#
# Aucune séquence de pattes n'est programmée. Seule une récompense de stabilité
# (garder au moins 3 griffes plantées) pousse le lézard à ne bouger qu'une patte
# à la fois : main, jambe, autre main, autre jambe... c'est à lui de le découvrir.
# ============================================================

import js
import json
import asyncio
import time
import numpy as np

# Tailles lues dans physics.js : jamais de décalage entre corps et réseau
N_OBS = int(js.window.simInfo.nObs)
N_ACT = int(js.window.simInfo.nAct)
H0 = float(js.window.simInfo.h0)   # hauteur de départ de la poitrine
N_GRIP = 4                         # les 4 dernières sorties commandent les griffes
N_HID = 20
N_PARAMS = N_OBS * N_HID + N_HID + N_HID * N_ACT + N_ACT

POP = 16                # variantes par génération (paires symétriques)
SIGMA = 0.1             # amplitude des variations testées
LR = 0.06               # vitesse d'apprentissage
MAX_STEPS = 14 * 60     # durée maximale d'un essai : 14 s
STAGNATION = 5.0        # sans progrès en hauteur pendant 5 s : l'essai s'arrête
GOAL_GAIN = 1.5         # hauteur à gagner pour atteindre le sommet (m)

SAVE_KEY = "lizard_ai_save"


class Reset(Exception):
    """Levée quand l'utilisateur clique sur Réinitialiser."""
    pass


def unpack(theta):
    i = 0
    W1 = theta[i:i + N_OBS * N_HID].reshape(N_OBS, N_HID); i += N_OBS * N_HID
    b1 = theta[i:i + N_HID]; i += N_HID
    W2 = theta[i:i + N_HID * N_ACT].reshape(N_HID, N_ACT); i += N_HID * N_ACT
    b2 = theta[i:i + N_ACT]
    return W1, b1, W2, b2


def policy(theta, obs):
    W1, b1, W2, b2 = unpack(theta)
    h = np.tanh(obs @ W1 + b1)
    return np.tanh(h @ W2 + b2)


def fresh_theta():
    th = np.random.randn(N_PARAMS) * 0.1
    th[-N_GRIP:] = 1.5      # au départ, le lézard a tendance à s'agripper
    return th


_meas = {"t": time.perf_counter(), "n": 0, "x": 1.0, "yield": time.perf_counter()}


def count_steps(n):
    """Mesure la vitesse réellement obtenue (en multiples du temps réel)."""
    _meas["n"] += n
    now = time.perf_counter()
    dt = now - _meas["t"]
    if dt >= 1.0:
        _meas["x"] = _meas["n"] / 60.0 / dt
        _meas["t"] = now
        _meas["n"] = 0


async def wait_unpaused():
    ui = js.window.ui
    while True:
        if ui.reset:
            ui.reset = False
            raise Reset()
        if not ui.paused:
            return
        await js.window.waitFrame()


async def pace(k):
    """Pause / réinitialisation / vitesse. On cède la main au navigateur toutes les
    `speed` étapes, ou plus tôt si le calcul dure plus de 14 ms (pour garder la page fluide)."""
    await wait_unpaused()
    speed = max(1, int(js.window.ui.speed))
    now = time.perf_counter()
    if k % speed == 0 or now - _meas["yield"] > 0.014:
        await js.window.waitFrame()
        _meas["yield"] = time.perf_counter()


class Learner:
    def __init__(self):
        self.theta = fresh_theta()
        self.gen = 0
        self.attempts = 0
        self.falls = 0
        self.best_gain = 0.0
        self.best_alive = 0.0
        self.last_gain = 0.0
        self.last_grips = [1, 1, 1, 1]
        self.history = []       # [meilleur, moyenne] de hauteur gagnée par génération

    def to_dict(self):
        return {
            "theta": self.theta.tolist(),
            "gen": self.gen,
            "attempts": self.attempts,
            "falls": self.falls,
            "best_gain": self.best_gain,
            "best_alive": self.best_alive,
            "history": self.history,
        }

    @staticmethod
    def from_dict(d):
        theta = np.array(d["theta"], dtype=float)
        if theta.shape[0] != N_PARAMS:
            return Learner()    # sauvegarde d'un autre corps : on repart d'un réseau neuf
        L = Learner()
        L.theta = theta
        L.gen = d.get("gen", 0)
        L.attempts = d.get("attempts", 0)
        L.falls = d.get("falls", 0)
        L.best_gain = d.get("best_gain", 0.0)
        L.best_alive = d.get("best_alive", 0.0)
        L.history = d.get("history", [])
        return L


def level(L):
    if L.best_gain >= GOAL_GAIN:
        return "5 - atteint le sommet"
    if L.best_gain >= 0.6:
        return "4 - grimpe"
    if L.best_gain >= 0.15:
        return "3 - monte un peu"
    if L.best_alive >= 4:
        return "2 - tient accroché"
    return "1 - tombe tout de suite"


def publish(L, live=None):
    gain, grips = (live if live else (L.last_gain, L.last_grips))
    gs = "".join("●" if g else "○" for g in grips)
    js.window.setStats(
        f"Génération {L.gen} · Tentatives {L.attempts} · Hauteur {gain:.2f} m · "
        f"Chutes {L.falls} · Record {L.best_gain:.2f} m · "
        f"Prises (AG AD RG RD) {gs} · Vitesse réelle ×{_meas['x']:.0f} · Niveau {level(L)}"
    )


def push_history(L):
    js.window.setHistory(json.dumps(L.history))


def finish(L, max_y, stab, steps, fell, success, grips):
    """Calcule la récompense d'un essai (commun au mode normal et au mode turbo)."""
    gain = max(0.0, max_y - H0)
    t = steps / 60
    reward = 15.0 * gain + 0.3 * stab + (20.0 if success else 0.0) - (3.0 if fell else 0.0)
    L.last_gain = gain
    L.last_grips = grips
    return reward, gain, t, fell


async def run_episode_fast(theta, L):
    """Mode turbo : l'essai entier est simulé côté navigateur, sans aller-retour par pas."""
    await wait_unpaused()
    txt = await js.window.fastEpisode(
        json.dumps(theta.tolist()), N_OBS, N_HID, N_ACT, MAX_STEPS, STAGNATION
    )
    r = json.loads(txt)
    if r.get("aborted"):
        await wait_unpaused()       # lève Reset si c'est la cause
        return await run_episode_fast(theta, L)
    _meas["n"] += r["steps"]
    count_steps(0)
    return finish(L, r["maxY"], r["stab"], r["steps"], r["fell"], r["success"], r["grips"])


async def run_episode(theta, L):
    if js.window.ui.turbo:
        return await run_episode_fast(theta, L)
    obs = json.loads(js.window.sim.reset())
    steps = 0
    max_y = H0
    t_improve = 0.0
    stab = 0.0              # secondes passées avec au moins 3 griffes plantées
    fell = False
    success = False
    grips = [1, 1, 1, 1]
    while steps < MAX_STEPS:
        await pace(steps)
        a = policy(theta, np.nan_to_num(np.array(obs, dtype=float)))
        res = json.loads(js.window.sim.step(json.dumps(a.tolist())))
        obs = res["obs"]
        y = res["y"]
        grips = res["grips"]
        steps += 1
        count_steps(1)
        t = steps / 60
        if y > max_y + 0.01:
            max_y = y
            t_improve = t
        if sum(grips) >= 3:
            stab += 1 / 60
        if steps % 6 == 0:
            publish(L, live=(max(0.0, max_y - H0), grips))
        if res["fallen"]:
            fell = True
            break
        if res["success"]:
            success = True
            break
        if t - t_improve > STAGNATION:
            break
    return finish(L, max_y, stab, steps, fell, success, grips)


def save(L):
    js.window.localStorage.setItem(SAVE_KEY, json.dumps(L.to_dict()))


def load():
    raw = js.window.localStorage.getItem(SAVE_KEY)
    if raw is None:
        return None
    try:
        return Learner.from_dict(json.loads(raw))
    except Exception:
        return None


async def check_save_load(L):
    """Vérifie les boutons Sauvegarder / Charger entre deux générations."""
    if getattr(js.window, "requestSave", False):
        js.window.requestSave = False
        save(L)
    if getattr(js.window, "requestLoad", False):
        js.window.requestLoad = False
        loaded = load()
        if loaded is not None:
            push_history(loaded)
            return loaded
    return L


async def run_generation(L):
    half = POP // 2
    eps = np.random.randn(half, N_PARAMS)
    noise = np.concatenate([eps, -eps])      # paires symétriques : estimation plus fiable
    rewards = np.zeros(POP)
    gains = []
    for i in range(POP):
        reward, gain, t, fell = await run_episode(L.theta + SIGMA * noise[i], L)
        rewards[i] = reward
        gains.append(gain)
        L.attempts += 1
        if fell:
            L.falls += 1
        L.best_gain = max(L.best_gain, gain)
        L.best_alive = max(L.best_alive, t)
        publish(L)
    # Classement des récompenses (robuste aux valeurs extrêmes)
    ranks = rewards.argsort().argsort().astype(float)
    adv = ranks / (POP - 1) - 0.5
    L.theta = L.theta + LR * (noise.T @ adv) / (POP * SIGMA)
    L.gen += 1
    L.history.append([float(max(gains)), float(np.mean(gains))])
    L.history = L.history[-400:]
    push_history(L)
    publish(L)


async def train():
    L = load() or Learner()
    push_history(L)
    publish(L)
    while True:
        try:
            L = await check_save_load(L)
            await run_generation(L)
            L = await check_save_load(L)
        except Reset:
            L = Learner()
            push_history(L)
            publish(L)


asyncio.ensure_future(train())
