# ============================================================
# LEZARD AI — Évolution (Python, exécuté dans le navigateur via Pyodide)
#
# Chaque génération : 1000 lézards grimpent en même temps, simulés en parallèle par des
# Workers (voir view.js / worker.js). La génération s'arrête quand le DERNIER lézard est
# à court d'énergie (ou tombé, ou arrêté après 10 s sans progrès). Ensuite :
#   1. on classe les lézards par hauteur maximale atteinte (au centimètre près) ;
#      à égalité, celui qui a consommé le moins d'énergie passe devant ;
#   2. les 500 premiers survivent et créent chacun une copie identique ;
#   3. chaque copie peut subir des mutations RARES : sur le cerveau (quelques poids),
#      sur la masse du squelette, sur la force des muscles, sur la réserve d'énergie.
# Les 500 autres disparaissent. Les 1000 lézards recommencent au pied de l'arbre.
# ============================================================

import js
import json
import base64
import asyncio
import numpy as np

INFO = json.loads(js.window.simInfoJson)       # tailles lues dans physics.js : aucun décalage possible
N_PARAMS = int(INFO["nParams"])
N_GRIP = int(INFO["nGrip"])
KEYS = INFO["bodyKeys"]
N_GENES = len(KEYS)
LO = np.array(INFO["lo"], dtype=np.float32)
HI = np.array(INFO["hi"], dtype=np.float32)

# ---------- Réglages de l'évolution ----------
POP = 1000                  # nombre de lézards simultanés
ELITE = POP // 2            # nombre de survivants (qui se copient)
P_MUT_WEIGHT = 0.01         # probabilité qu'un poids du cerveau mute dans une copie
SIGMA_WEIGHT = 0.25         # amplitude d'une mutation de poids
P_MUT_GENE = 0.05           # probabilité qu'un gène du corps (masse, muscle, énergie) mute
SIGMA_GENE = 0.15           # amplitude d'une mutation de gène (variation relative, ~15 %)
REPLAY_PARENTS = False      # False : les 500 parents ne sont PAS resimulés (leur résultat est identique,
                            # la physique étant déterministe) -> génération 2× plus rapide.
                            # True : tous les lézards rejouent à chaque génération.
SAVE_TOP = 100              # nombre de lézards conservés dans la sauvegarde
SAVE_KEY = "lizard_evo_save"


class Reset(Exception):
    """Levée quand l'utilisateur clique sur Réinitialiser."""
    pass


def b64(arr):
    return base64.b64encode(np.ascontiguousarray(arr, dtype=np.float32).tobytes()).decode("ascii")


def unb64(s, shape):
    return np.frombuffer(base64.b64decode(s), dtype=np.float32).reshape(shape).copy()


# ---------- Mutations ----------
def mutate_brains(b):
    mask = np.random.rand(*b.shape) < P_MUT_WEIGHT
    noise = np.random.randn(*b.shape).astype(np.float32) * SIGMA_WEIGHT
    return (b + mask * noise).astype(np.float32)


def mutate_genes(g):
    mask = np.random.rand(*g.shape) < P_MUT_GENE
    factor = np.exp(np.random.randn(*g.shape) * SIGMA_GENE).astype(np.float32)
    return np.clip(np.where(mask, g * factor, g), LO, HI).astype(np.float32)


class Population:
    def __init__(self):
        self.brains = (np.random.randn(POP, N_PARAMS) * 0.1).astype(np.float32)
        self.brains[:, -N_GRIP:] = 1.5          # au départ, les lézards ont tendance à s'agripper
        self.genes = np.ones((POP, N_GENES), dtype=np.float32)
        self.gen = 0
        self.best_ever = 0.0
        self.history = []                        # [meilleur, moyenne] de hauteur par génération
        self.last = None                         # statistiques de la dernière génération
        self.table = []                          # une ligne par génération, pour le panneau
        self.fit = None                          # résultats déjà connus des parents (gain, énergie, masse)


def level(best):
    if best >= 10:
        return "5 - grimpeurs experts"
    if best >= 3:
        return "4 - grimpent haut"
    if best >= 1:
        return "3 - grimpent"
    if best >= 0.15:
        return "2 - premières montées"
    return "1 - s'accrochent à peine"


def publish(S):
    if S.last is None:
        js.window.setStats(f"Génération {S.gen} · {POP} lézards prêts · Niveau {level(S.best_ever)}")
    else:
        s = S.last
        js.window.setStats(
            f"Génération {S.gen} · Record {S.best_ever:.2f} m · Meilleur de la génération {s['best']:.2f} m "
            f"({s['energy']:.0f} J) · Moyenne {s['mean']:.2f} m · "
            f"Survivants : masse ×{s['g_mass']:.2f}, muscles ×{s['g_musc']:.2f}, énergie ×{s['g_en']:.2f} "
            f"· Poids moyen {s['kg']:.2f} kg · Niveau {level(S.best_ever)}"
        )
    js.window.setGen(S.gen)
    push_panel(S)


def push_panel(S):
    js.window.setPanel(json.dumps({
        "gen": S.gen, "level": level(S.best_ever), "best_ever": S.best_ever,
        "last": S.last, "rows": S.table[-8:], "pop": POP, "elite": ELITE,
        "cached": ELITE if (S.fit is not None and not REPLAY_PARENTS) else 0,
    }))


def push_history(S):
    js.window.setHistory(json.dumps(S.history))


async def run_generation(S):
    # Les parents ont déjà été évalués : à l'identique, ils referaient exactement la même chose.
    k = ELITE if (S.fit is not None and not REPLAY_PARENTS) else 0
    txt = await js.window.runGeneration(b64(S.brains[k:]), b64(S.genes[k:]), POP - k)
    r = json.loads(txt)
    if r.get("aborted"):
        raise Reset()

    gain = np.array(r["gain"], dtype=np.float64)
    energy = np.array(r["energy"], dtype=np.float64)
    mass = np.array(r["mass"], dtype=np.float64)
    if k:
        gain = np.concatenate([S.fit[0], gain])
        energy = np.concatenate([S.fit[1], energy])
        mass = np.concatenate([S.fit[2], mass])

    # Classement : hauteur au centimètre près (décroissante), puis énergie consommée (croissante)
    order = np.lexsort((energy, -np.round(gain, 2)))
    surv = order[:ELITE]

    parents_b = S.brains[surv]
    parents_g = S.genes[surv]
    musc_cols = [i for i, k in enumerate(KEYS) if k not in ("mass", "energy")]
    S.last = {
        "best": float(gain[order[0]]),
        "energy": float(energy[order[0]]),
        "mean": float(gain.mean()),
        "kg": float(mass[surv].mean()),
        "g_mass": float(parents_g[:, KEYS.index("mass")].mean()),
        "g_musc": float(parents_g[:, musc_cols].mean()),
        "g_en": float(parents_g[:, KEYS.index("energy")].mean()),
    }
    S.best_ever = max(S.best_ever, S.last["best"])

    # Les 500 survivants restent identiques ; chacun crée une copie qui peut muter
    S.fit = (gain[surv], energy[surv], mass[surv])      # résultats des parents, réutilisés à la génération suivante
    S.brains = np.concatenate([parents_b, mutate_brains(parents_b)])
    S.genes = np.concatenate([parents_g, mutate_genes(parents_g)])

    S.gen += 1
    S.table.append([S.gen, S.last["best"], S.last["mean"], S.last["energy"], S.last["kg"],
                    S.last["g_mass"], S.last["g_musc"], S.last["g_en"]])
    S.table = S.table[-200:]
    S.history.append([S.last["best"], S.last["mean"]])
    S.history = S.history[-400:]
    push_history(S)
    publish(S)


# ---------- Sauvegarde ----------
def save(S):
    k = min(SAVE_TOP, POP)
    # après une sélection, les premiers lézards sont les meilleurs, classés
    js.window.localStorage.setItem(SAVE_KEY, json.dumps({
        "k": k, "brains": b64(S.brains[:k]), "genes": b64(S.genes[:k]),
        "gen": S.gen, "best_ever": S.best_ever, "history": S.history, "table": S.table,
    }))


def load():
    raw = js.window.localStorage.getItem(SAVE_KEY)
    if raw is None:
        return None
    try:
        d = json.loads(raw)
        k = int(d["k"])
        eb = unb64(d["brains"], (k, N_PARAMS))
        eg = unb64(d["genes"], (k, N_GENES))
        S = Population()
        S.brains[:k] = eb
        S.genes[:k] = eg
        idx = np.arange(k, POP) % k             # on complète avec des copies mutées des meilleurs
        S.brains[k:] = mutate_brains(eb[idx])
        S.genes[k:] = mutate_genes(eg[idx])
        S.gen = int(d.get("gen", 0))
        S.best_ever = float(d.get("best_ever", 0.0))
        S.history = d.get("history", [])
        S.table = d.get("table", [])
        return S
    except Exception:
        return None


async def check_save_load(S):
    if getattr(js.window, "requestSave", False):
        js.window.requestSave = False
        save(S)
    if getattr(js.window, "requestLoad", False):
        js.window.requestLoad = False
        loaded = load()
        if loaded is not None:
            push_history(loaded)
            publish(loaded)
            return loaded
    return S


async def train():
    S = load() or Population()
    push_history(S)
    publish(S)
    while True:
        try:
            S = await check_save_load(S)
            await run_generation(S)
        except Reset:
            S = Population()
            push_history(S)
            publish(S)
        except Exception as e:
            js.document.getElementById("status").textContent = "ERREUR : " + str(e)[:300]
            raise


asyncio.ensure_future(train())
