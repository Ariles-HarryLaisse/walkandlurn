# ============================================================
# WALKING AI — Apprentissage (Python, exécuté dans le navigateur via Pyodide)
# Un réseau de neurones (11 sorties, 1 couche cachée) est entraîné
# par stratégies d'évolution : à chaque génération, plusieurs variantes
# sont testées, et les meilleures orientent la génération suivante.
# La physique est déléguée à physics.js (window.sim).
# ============================================================

import js
import json
import asyncio
import numpy as np

N_OBS = 29          # taille de l'observation : 11 articulations x2 (angle, vitesse) + 7 (voir physics.js)
N_ACT = 11          # une commande par articulation, colonne vertébrale comprise
N_HID = 18          # neurones cachés
N_PARAMS = N_OBS * N_HID + N_HID + N_HID * N_ACT + N_ACT

POP = 12            # variantes testées par génération
SIGMA = 0.1         # amplitude des variations testées
LR = 0.05           # vitesse d'apprentissage
MAX_STEPS = 12 * 60 # durée maximale d'un essai (12 s à 60 pas/s)
FALL_TIME = 1.0     # rester au sol plus de 1 s = essai terminé (chute)

SAVE_KEY = "walking_ai_save"


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


async def pace(k):
    """Respecte pause / réinitialisation / vitesse choisies dans l'interface."""
    ui = js.window.ui
    while True:
        if ui.reset:
            ui.reset = False
            raise Reset()
        if not ui.paused:
            break
        await js.window.waitFrame()
    if k % max(1, int(ui.speed)) == 0:
        await js.window.waitFrame()


async def run_episode(theta):
    obs = json.loads(js.window.sim.reset())
    steps = 0
    fallen_t = 0.0
    fell = False
    dx = 0.0
    while steps < MAX_STEPS:
        await pace(steps)
        a = policy(theta, np.nan_to_num(np.array(obs, dtype=float)))
        res = json.loads(js.window.sim.step(json.dumps(a.tolist())))
        obs = res["obs"]
        dx = res["dx"]
        steps += 1
        fallen_t = fallen_t + 1 / 60 if res["fallen"] else 0.0
        if fallen_t >= FALL_TIME:
            fell = True
            break
    t = steps / 60
    reward = 3.0 * dx + 0.5 * t - (5.0 if fell else 0.0)
    return reward, dx, t, fell


class Learner:
    def __init__(self):
        self.theta = np.random.randn(N_PARAMS) * 0.1
        self.gen = 0
        self.attempts = 0
        self.falls = 0
        self.best_dist = 0.0
        self.best_alive = 0.0
        self.last_dist = 0.0

    def to_dict(self):
        return {
            "theta": self.theta.tolist(),
            "gen": self.gen,
            "attempts": self.attempts,
            "falls": self.falls,
            "best_dist": self.best_dist,
            "best_alive": self.best_alive,
        }

    @staticmethod
    def from_dict(d):
        L = Learner()
        theta = np.array(d["theta"], dtype=float)
        if theta.shape[0] != N_PARAMS:
            # Sauvegarde d'une ancienne version du corps (nombre d'articulations différent) :
            # on repart d'un réseau neuf plutôt que de charger des poids incompatibles.
            return Learner()
        L.theta = theta
        L.gen = d["gen"]
        L.attempts = d["attempts"]
        L.falls = d["falls"]
        L.best_dist = d["best_dist"]
        L.best_alive = d["best_alive"]
        return L


def level(L):
    if L.best_dist >= 15 and L.best_alive >= 10:
        return "5 - marche durablement"
    if L.best_dist >= 5:
        return "4 - marche"
    if L.best_dist >= 1:
        return "3 - quelques pas"
    if L.best_alive >= 2:
        return "2 - équilibre bref"
    return "1 - incapable de tenir debout"


def publish(L):
    js.window.setStats(
        f"Génération {L.gen} · Tentatives {L.attempts} · Distance {L.last_dist:.2f} m · "
        f"Chutes {L.falls} · Meilleur {L.best_dist:.2f} m · Niveau {level(L)}"
    )


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
    """Vérifie les boutons Sauvegarder / Charger à chaque génération."""
    if getattr(js.window, "requestSave", False):
        js.window.requestSave = False
        save(L)
    if getattr(js.window, "requestLoad", False):
        js.window.requestLoad = False
        loaded = load()
        if loaded is not None:
            return loaded
    return L


async def run_generation(L):
    noise = np.random.randn(POP, N_PARAMS)
    rewards = np.zeros(POP)
    for i in range(POP):
        reward, dx, t, fell = await run_episode(L.theta + SIGMA * noise[i])
        rewards[i] = reward
        L.attempts += 1
        if fell:
            L.falls += 1
        L.last_dist = dx
        L.best_dist = max(L.best_dist, dx)
        L.best_alive = max(L.best_alive, t)
        publish(L)
    advantage = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
    L.theta = L.theta + LR * (noise.T @ advantage) / (POP * SIGMA)
    L.gen += 1
    publish(L)


async def train():
    L = load() or Learner()
    publish(L)
    while True:
        try:
            L = await check_save_load(L)
            await run_generation(L)
            L = await check_save_load(L)
        except Reset:
            L = Learner()
            publish(L)


asyncio.ensure_future(train())
