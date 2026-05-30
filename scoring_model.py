#!/usr/bin/env python3
"""
EuroMillions – Modèle de scoring avec backtest chronologique.

Le modèle apprend, sur les tirages passés, à distinguer une vraie combinaison
d'une combinaison aléatoire, à partir de features statistiques.
Score de 0 à 100 : plus il est élevé, plus la sélection ressemble
aux combinaisons historiquement sorties.

⚠  Les tirages EuroMillions sont aléatoires : ce score mesure la « typicité »
   d'une combinaison, il ne prédit pas le prochain tirage.

Usage :
  python scoring_model.py                                # entraîne + backtest (100 derniers tirages)
  python scoring_model.py --last-n 200                   # tester sur les 200 derniers tirages
  python scoring_model.py --score 5 14 23 42 49 2 8      # score 5 boules + 2 étoiles
  python scoring_model.py --n-neg 20                     # ratio négatifs/positifs
  python scoring_model.py --no-plots                     # sans graphiques
"""

import json
import random
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import entropy as scipy_entropy
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from euromillions_scraper import (
    CACHE_CSV,
    PERIODS,
    STATS_DIR,
    load_results,
)

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

BALL_COLS = ["ball_1", "ball_2", "ball_3", "ball_4", "ball_5"]
STAR_COLS = ["star_1", "star_2"]

# Noms des 29 features (même ordre que extract_features)
FEATURE_NAMES: list[str] = [
    # Fréquence historique des boules (fraction de tirages où la boule est sortie)
    "ball_freq_mean", "ball_freq_min", "ball_freq_max", "ball_freq_std",
    # Recency : nb de tirages depuis la dernière apparition de chaque boule
    "ball_rec_mean", "ball_rec_max", "ball_rec_std",
    # Fréquence des étoiles (ratio obs/espérance corrigée par période)
    "star_freq_mean", "star_freq_min",
    # Recency étoiles
    "star_rec_mean", "star_rec_max",
    # Structure de la combinaison
    "sum_balls", "range_balls", "n_odd_balls", "n_low_balls",
    # Distribution par dizaine (1-10, 11-20, …, 41-50)
    "n_dec_01_10", "n_dec_11_20", "n_dec_21_30", "n_dec_31_40", "n_dec_41_50",
    "decade_entropy",
    # Écarts entre boules consécutives (une fois triées)
    "consec_gap_mean", "consec_gap_std", "consec_gap_min", "consec_gap_max",
    "n_consec_pairs",
    # Étoiles
    "sum_stars", "star_gap", "n_stars_low_half",
]

assert len(FEATURE_NAMES) == 29


# ---------------------------------------------------------------------------
# Statistiques de référence (pré-calculées une fois sur le jeu d'entraînement)
# ---------------------------------------------------------------------------

class DrawStats:
    """
    Statistiques pré-calculées sur un ensemble de tirages.
    Toutes les lookups sont O(1) pour accélérer l'extraction de features.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        df = df.reset_index(drop=True)
        n  = len(df)
        self.n_draws = n

        # Période courante (détermine le max_star valide)
        last_date = df["date"].max()
        self.period_id = 3
        self.max_star  = 12
        for p in PERIODS:
            if p["start"] <= last_date <= p["end"]:
                self.period_id = p["id"]
                self.max_star  = p["max_star"]
                break

        # ── Fréquence des boules ─────────────────────────────────────────
        all_b = pd.concat([df[c] for c in BALL_COLS], ignore_index=True).dropna().astype(int)
        bc = Counter(all_b.tolist())
        self.ball_freq: dict[int, float] = {b: bc.get(b, 0) / n for b in range(1, 51)}

        # ── Fréquence des étoiles (normalisée par période) ───────────────
        p1_end = pd.Timestamp("2011-05-09")
        p2_end = pd.Timestamp("2016-09-26")
        n_p1 = len(df[df["date"] <= p1_end])
        n_p2 = len(df[(df["date"] > p1_end) & (df["date"] <= p2_end)])
        n_p3 = len(df[df["date"] > p2_end])
        all_s = pd.concat([df[c] for c in STAR_COLS], ignore_index=True).dropna().astype(int)
        sc = Counter(all_s.tolist())

        self.star_freq_ratio: dict[int, float] = {}
        for s in range(1, 13):
            expected = (
                (n_p1 * 2 / 9  if s <= 9  else 0.0) +
                (n_p2 * 2 / 11 if s <= 11 else 0.0) +
                (n_p3 * 2 / 12 if s <= 12 else 0.0)
            )
            self.star_freq_ratio[s] = sc.get(s, 0) / expected if expected > 0 else 0.0

        # ── Recency (tirages depuis la dernière apparition) ───────────────
        # Matrice indicatrice (n × 50) pour les boules
        ball_mat = np.zeros((n, 50), dtype=bool)
        for i, row in df[BALL_COLS].iterrows():
            for v in row.dropna().astype(int):
                if 1 <= v <= 50:
                    ball_mat[i, v - 1] = True

        self.ball_recency: dict[int, int] = {}
        for b in range(1, 51):
            idxs = np.where(ball_mat[:, b - 1])[0]
            self.ball_recency[b] = int(n - 1 - idxs[-1]) if len(idxs) else n

        # Matrice indicatrice (n × 12) pour les étoiles
        star_mat = np.zeros((n, 12), dtype=bool)
        for i, row in df[STAR_COLS].iterrows():
            for v in row.dropna().astype(int):
                if 1 <= v <= 12:
                    star_mat[i, v - 1] = True

        self.star_recency: dict[int, int] = {}
        for s in range(1, 13):
            idxs = np.where(star_mat[:, s - 1])[0]
            self.star_recency[s] = int(n - 1 - idxs[-1]) if len(idxs) else n

        # ── Distribution de référence (mean/std des features sur les vrais tirages)
        # Calculée après le premier build_dataset dans ScoringModel.fit()
        self.ref_mean: np.ndarray | None = None
        self.ref_std:  np.ndarray | None = None


# ---------------------------------------------------------------------------
# Extraction de features
# ---------------------------------------------------------------------------

def extract_features(
    balls: list[int],
    stars: list[int],
    stats: DrawStats,
) -> np.ndarray:
    """Retourne un vecteur de 29 features pour une combinaison donnée."""
    balls = sorted(balls)
    stars = sorted(stars)

    # ── Boules – fréquence ───────────────────────────────────────────────
    bf = np.array([stats.ball_freq.get(b, 0.0) for b in balls])
    bf_mean, bf_min, bf_max, bf_std = bf.mean(), bf.min(), bf.max(), bf.std()

    # ── Boules – recency ─────────────────────────────────────────────────
    br = np.array([float(stats.ball_recency.get(b, stats.n_draws)) for b in balls])
    br_mean, br_max, br_std = br.mean(), br.max(), br.std()

    # ── Étoiles – fréquence normalisée ───────────────────────────────────
    sf = np.array([stats.star_freq_ratio.get(s, 0.0) for s in stars])
    sf_mean, sf_min = sf.mean(), sf.min()

    # ── Étoiles – recency ────────────────────────────────────────────────
    sr = np.array([float(stats.star_recency.get(s, stats.n_draws)) for s in stars])
    sr_mean, sr_max = sr.mean(), sr.max()

    # ── Structure ────────────────────────────────────────────────────────
    sum_b   = float(sum(balls))           # théorique si uniforme : 127.5
    range_b = float(balls[-1] - balls[0])
    n_odd   = float(sum(b % 2 for b in balls))
    n_low   = float(sum(1 for b in balls if b <= 25))

    # ── Répartition par dizaine ──────────────────────────────────────────
    dec = [(b - 1) // 10 for b in balls]          # 0–4
    dc  = [float(dec.count(d)) for d in range(5)]
    dec_ent = float(scipy_entropy(np.array(dc) + 1e-9))

    # ── Écarts entre boules consécutives (triées) ────────────────────────
    gaps  = [balls[i + 1] - balls[i] for i in range(4)]
    g_arr = np.array(gaps, dtype=float)
    g_mean, g_std, g_min, g_max = g_arr.mean(), g_arr.std(), g_arr.min(), g_arr.max()
    n_consec = float(sum(1 for g in gaps if g == 1))

    # ── Étoiles – combinaison ────────────────────────────────────────────
    sum_s   = float(sum(stars))
    star_gap = float(stars[1] - stars[0]) if len(stars) == 2 else 0.0
    n_s_low = float(sum(1 for s in stars if s <= stats.max_star // 2))

    return np.array([
        bf_mean, bf_min, bf_max, bf_std,
        br_mean, br_max, br_std,
        sf_mean, sf_min,
        sr_mean, sr_max,
        sum_b, range_b, n_odd, n_low,
        *dc, dec_ent,
        g_mean, g_std, g_min, g_max, n_consec,
        sum_s, star_gap, n_s_low,
    ], dtype=float)


# ---------------------------------------------------------------------------
# Génération de combinaisons aléatoires (exemples négatifs)
# ---------------------------------------------------------------------------

def random_combination(max_star: int = 12) -> tuple[list[int], list[int]]:
    balls = sorted(random.sample(range(1, 51), 5))
    stars = sorted(random.sample(range(1, max_star + 1), 2))
    return balls, stars


def build_dataset(
    draws_df: pd.DataFrame,
    stats: DrawStats,
    n_neg_ratio: int = 20,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Positifs (1) : vrais tirages.
    Négatifs (0) : n_neg_ratio combinaisons aléatoires par vrai tirage.
    """
    random.seed(seed)
    pos_X, neg_X = [], []

    for _, row in draws_df.iterrows():
        balls = sorted([int(row[c]) for c in BALL_COLS if pd.notna(row[c])])
        stars = sorted([int(row[c]) for c in STAR_COLS if pd.notna(row[c])])
        if len(balls) != 5 or len(stars) != 2:
            continue
        pos_X.append(extract_features(balls, stars, stats))

    for _ in range(len(pos_X) * n_neg_ratio):
        b, s = random_combination(stats.max_star)
        neg_X.append(extract_features(b, s, stats))

    X = np.vstack(pos_X + neg_X)
    y = np.array([1] * len(pos_X) + [0] * len(neg_X))
    return X, y


# ---------------------------------------------------------------------------
# Modèle
# ---------------------------------------------------------------------------

class ScoringModel:
    """
    Gradient Boosting entraîné à distinguer vrais tirages vs combinaisons aléatoires.
    Utilisé pour scorer et expliquer n'importe quelle sélection.
    """

    def __init__(self, n_neg_ratio: int = 20, seed: int = 42) -> None:
        self.n_neg_ratio = n_neg_ratio
        self.seed        = seed
        self.stats:  DrawStats | None = None
        self.scaler  = StandardScaler()
        self.clf     = GradientBoostingClassifier(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            min_samples_leaf=10,
            random_state=seed,
        )
        # Baseline linéaire pour comparaison
        self.lr      = LogisticRegression(max_iter=1000, random_state=seed)
        self.importances: np.ndarray | None = None

    # ── Entraînement ─────────────────────────────────────────────────────

    def fit(self, train_df: pd.DataFrame) -> "ScoringModel":
        print(f"  Statistiques de référence ({len(train_df)} tirages)…")
        self.stats = DrawStats(train_df)

        print("  Construction du jeu d'entraînement…")
        X, y = build_dataset(train_df, self.stats, self.n_neg_ratio, self.seed)
        print(f"    {int(y.sum()):,} positifs · {int((y==0).sum()):,} négatifs · {X.shape[1]} features")

        Xs = self.scaler.fit_transform(X)
        self.clf.fit(Xs, y)
        self.lr.fit(Xs, y)
        self.importances = self.clf.feature_importances_

        # Distribution de référence des vrais tirages (pour les Z-scores)
        self.stats.ref_mean = X[y == 1].mean(axis=0)
        self.stats.ref_std  = X[y == 1].std(axis=0) + 1e-9
        return self

    # ── Score d'une combinaison ───────────────────────────────────────────

    def score(
        self,
        balls: list[int],
        stars: list[int],
    ) -> float:
        """Score 0–1 (GBT). Plus élevé = plus similaire aux vrais tirages."""
        f  = extract_features(sorted(balls), sorted(stars), self.stats)
        fs = self.scaler.transform(f.reshape(1, -1))
        return float(self.clf.predict_proba(fs)[0, 1])

    def score_batch(self, combinations: list[tuple]) -> np.ndarray:
        """Score plusieurs (balls, stars) d'un coup (plus rapide)."""
        X = np.vstack([
            extract_features(sorted(b), sorted(s), self.stats)
            for b, s in combinations
        ])
        Xs = self.scaler.transform(X)
        return self.clf.predict_proba(Xs)[:, 1]

    # ── Explication ───────────────────────────────────────────────────────

    def explain(
        self,
        balls: list[int],
        stars: list[int],
    ) -> dict:
        """Retourne le score et le détail de chaque feature (valeur, z-score, importance)."""
        balls = sorted(balls)
        stars = sorted(stars)
        f     = extract_features(balls, stars, self.stats)
        z     = (f - self.stats.ref_mean) / self.stats.ref_std
        score = float(self.clf.predict_proba(
            self.scaler.transform(f.reshape(1, -1))
        )[0, 1])

        return {
            "balls":  balls,
            "stars":  stars,
            "score":  round(score * 100, 2),
            "features": {
                name: {
                    "value":      round(float(f[i]), 4),
                    "ref_mean":   round(float(self.stats.ref_mean[i]), 4),
                    "z_score":    round(float(z[i]), 3),
                    "importance": round(float(self.importances[i]), 5),
                }
                for i, name in enumerate(FEATURE_NAMES)
            },
        }

    # ── Classement marginal des numéros ──────────────────────────────────

    def rank_numbers(
        self,
        n_samples: int = 300,
        seed: int = 0,
    ) -> tuple[list[int], list[int], dict, dict]:
        """
        Pour chaque boule (1-50) et chaque étoile (1-max_star), calcule un
        score marginal en moyennant le score du modèle sur n_samples combinaisons
        aléatoires contenant ce numéro.
        Retourne (ranked_balls, ranked_stars, ball_scores, star_scores).
        """
        random.seed(seed)
        max_s = self.stats.max_star

        # ── Score marginal des boules ─────────────────────────────────────
        all_ball_combos: list[tuple] = []
        for b in range(1, 51):
            pool = [x for x in range(1, 51) if x != b]
            for _ in range(n_samples):
                others = random.sample(pool, 4)
                balls  = sorted([b] + others)
                stars  = sorted(random.sample(range(1, max_s + 1), 2))
                all_ball_combos.append((balls, stars))

        ball_sc_flat = self.score_batch(all_ball_combos)
        ball_scores: dict[int, float] = {}
        for i, b in enumerate(range(1, 51)):
            ball_scores[b] = float(np.mean(ball_sc_flat[i * n_samples:(i + 1) * n_samples]))

        # ── Score marginal des étoiles ────────────────────────────────────
        all_star_combos: list[tuple] = []
        for s in range(1, max_s + 1):
            pool = [x for x in range(1, max_s + 1) if x != s]
            for _ in range(n_samples):
                balls = sorted(random.sample(range(1, 51), 5))
                other = random.choice(pool)
                stars = sorted([s, other])
                all_star_combos.append((balls, stars))

        star_sc_flat = self.score_batch(all_star_combos)
        star_scores: dict[int, float] = {}
        for i, s in enumerate(range(1, max_s + 1)):
            star_scores[s] = float(np.mean(star_sc_flat[i * n_samples:(i + 1) * n_samples]))

        ranked_balls = sorted(ball_scores.keys(), key=lambda b: -ball_scores[b])
        ranked_stars = sorted(star_scores.keys(), key=lambda s: -star_scores[s])
        return ranked_balls, ranked_stars, ball_scores, star_scores

    # ── Backtest chronologique ────────────────────────────────────────────

    def backtest(
        self,
        test_df: pd.DataFrame,
        n_samples_ranking: int = 300,
        seed: int = 99,
    ) -> dict:
        """
        Classe tous les numéros 1-50 et les étoiles par score marginal,
        puis mesure sur chaque tirage de test combien de boules réelles
        figurent dans le top K du classement.
        Retourne le détail par seuil K.
        """
        random.seed(seed)

        ranked_balls, ranked_stars, ball_scores, star_scores = self.rank_numbers(
            n_samples=n_samples_ranking, seed=seed
        )
        max_s = self.stats.max_star

        THRESH_B = [5, 10, 15, 20, 25, 30]
        THRESH_S = [2, 3, 4, 5, 6]

        per_draw: list[dict] = []
        for _, row in test_df.iterrows():
            actual_balls = {int(row[c]) for c in BALL_COLS if pd.notna(row[c])}
            actual_stars = {int(row[c]) for c in STAR_COLS if pd.notna(row[c])}
            if len(actual_balls) != 5 or len(actual_stars) != 2:
                continue
            draw: dict = {}
            for k in THRESH_B:
                draw[f"balls_top{k}"] = len(actual_balls & set(ranked_balls[:k]))
            for k in THRESH_S:
                ks = min(k, max_s)
                draw[f"stars_top{ks}"] = len(actual_stars & set(ranked_stars[:ks]))
            per_draw.append(draw)

        n = len(per_draw)
        by_threshold: dict = {}

        for k in THRESH_B:
            key      = f"balls_top{k}"
            vals     = [r[key] for r in per_draw]
            expected = 5 * k / 50          # espérance hypergéométrique
            counts   = Counter(vals)
            by_threshold[key] = {
                "k":               k,
                "mean_found":      round(float(np.mean(vals)), 3),
                "expected_random": round(expected, 3),
                "dist":            {str(i): counts.get(i, 0) for i in range(6)},
                "pct_ge2":         round(float(np.mean([v >= 2 for v in vals])) * 100, 1),
                "pct_ge3":         round(float(np.mean([v >= 3 for v in vals])) * 100, 1),
                "pct_ge4":         round(float(np.mean([v >= 4 for v in vals])) * 100, 1),
            }

        for k in THRESH_S:
            ks       = min(k, max_s)
            key      = f"stars_top{ks}"
            vals     = [r[key] for r in per_draw]
            expected = 2 * ks / max_s
            counts   = Counter(vals)
            by_threshold[key] = {
                "k":               ks,
                "mean_found":      round(float(np.mean(vals)), 3),
                "expected_random": round(expected, 3),
                "dist":            {str(i): counts.get(i, 0) for i in range(3)},
            }

        return {
            "n_test_draws":  n,
            "ranked_balls":  ranked_balls,
            "ranked_stars":  ranked_stars,
            "ball_scores":   {str(b): round(ball_scores[b], 6) for b in range(1, 51)},
            "star_scores":   {str(s): round(star_scores[s], 6) for s in range(1, max_s + 1)},
            "by_threshold":  by_threshold,
            "per_draw":      per_draw,
        }


# ---------------------------------------------------------------------------
# Graphiques
# ---------------------------------------------------------------------------

def plot_backtest(result: dict, out: Path) -> None:
    bt      = result["by_threshold"]
    n_draws = result["n_test_draws"]
    THRESH_B = [5, 10, 15, 20, 25, 30]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f"Backtest – {n_draws} derniers tirages · boules trouvées dans le top K classement modèle",
        fontsize=11,
    )

    # 1. Boules trouvées en moyenne vs aléatoire
    ax = axes[0]
    ks_b   = [bt[f"balls_top{k}"]["k"]               for k in THRESH_B]
    found  = [bt[f"balls_top{k}"]["mean_found"]       for k in THRESH_B]
    exp    = [bt[f"balls_top{k}"]["expected_random"]  for k in THRESH_B]
    x      = np.arange(len(THRESH_B))
    w      = 0.35
    ax.bar(x - w/2, found, w, label="Modèle (moyenne réelle)", color="steelblue", alpha=0.85)
    ax.bar(x + w/2, exp,   w, label="Espérance aléatoire",    color="salmon",    alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Top {k}" for k in THRESH_B])
    ax.set_ylabel("Boules trouvées (sur 5)")
    ax.set_title("Boules trouvées par seuil")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 5.5)

    # 2. % de tirages avec ≥2 / ≥3 boules trouvées
    ax = axes[1]
    pct_ge2 = [bt[f"balls_top{k}"]["pct_ge2"] for k in THRESH_B]
    pct_ge3 = [bt[f"balls_top{k}"]["pct_ge3"] for k in THRESH_B]
    pct_ge4 = [bt[f"balls_top{k}"]["pct_ge4"] for k in THRESH_B]
    ax.plot(THRESH_B, pct_ge2, "o-", color="steelblue",  label="≥ 2 boules trouvées")
    ax.plot(THRESH_B, pct_ge3, "s-", color="darkorange",  label="≥ 3 boules trouvées")
    ax.plot(THRESH_B, pct_ge4, "^-", color="seagreen",   label="≥ 4 boules trouvées")
    ax.set_xlabel("Nombre de boules retenues (top K)")
    ax.set_ylabel("% des tirages de test")
    ax.set_title("% de tirages avec au moins N boules trouvées")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = out / "backtest.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  → {path}")


def plot_feature_importance(model: ScoringModel, out: Path) -> None:
    fi   = model.importances
    idx  = np.argsort(fi)
    top  = min(20, len(fi))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(
        [FEATURE_NAMES[i] for i in idx[-top:]],
        [fi[i] for i in idx[-top:]],
        color="steelblue", edgecolor="white",
    )
    ax.set_xlabel("Importance (GradientBoosting)")
    ax.set_title(f"Importance des features – Top {top}")
    plt.tight_layout()
    path = out / "feature_importance.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  → {path}")


def plot_score_explanation(explain_result: dict, out: Path) -> None:
    """Graphique waterfall des features qui poussent le score en haut/bas."""
    feats = explain_result["features"]
    # Sort by |z_score| × importance
    items = sorted(
        feats.items(),
        key=lambda x: abs(x[1]["z_score"]) * x[1]["importance"],
        reverse=True,
    )[:15]

    names    = [k for k, _ in items]
    z_scores = [v["z_score"] for _, v in items]
    imps     = [v["importance"] for _, v in items]
    contrib  = [z * i for z, i in zip(z_scores, imps)]  # signed contribution

    colors = ["seagreen" if c >= 0 else "tomato" for c in contrib]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    balls = explain_result["balls"]
    stars = explain_result["stars"]
    score = explain_result["score"]
    fig.suptitle(
        f"Explication du score  —  boules {balls}  étoiles {stars}\n"
        f"Score : {score:.1f}/100",
        fontsize=11,
    )

    # Panel 1: Z-scores
    ax = axes[0]
    ax.barh(names[::-1], [z_scores[i] for i in range(len(names) - 1, -1, -1)],
            color=colors[::-1], edgecolor="white")
    ax.axvline(0, color="black", lw=0.8)
    ax.axvspan(-1.5, 1.5, alpha=0.06, color="grey")
    ax.set_xlabel("Z-score vs tirages historiques")
    ax.set_title("Écart à la moyenne des tirages réels")

    # Panel 2: Valeurs des features
    ax = axes[1]
    ref_means = [feats[n]["ref_mean"] for n in names]
    vals      = [feats[n]["value"]    for n in names]
    y_pos     = np.arange(len(names))
    ax.barh(y_pos, vals, color=colors, edgecolor="white", alpha=0.7, label="Votre sélection")
    ax.plot(ref_means, y_pos, "D", ms=5, color="black", label="Moy. tirages réels", zorder=5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.set_xlabel("Valeur")
    ax.set_title("Valeurs vs moyenne historique")
    ax.legend(fontsize=8)

    plt.tight_layout()
    path = out / "score_explanation.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  → {path}")


# ---------------------------------------------------------------------------
# Validation de la combinaison
# ---------------------------------------------------------------------------

def validate_combination(
    balls: list[int],
    stars: list[int],
    max_star: int = 12,
) -> str | None:
    """Retourne un message d'erreur ou None si la combinaison est valide."""
    if len(balls) != 5:
        return f"Attendu 5 boules, reçu {len(balls)}"
    if len(stars) != 2:
        return f"Attendu 2 étoiles, reçu {len(stars)}"
    if len(set(balls)) != 5:
        return "Les 5 boules doivent être différentes"
    if len(set(stars)) != 2:
        return "Les 2 étoiles doivent être différentes"
    for b in balls:
        if not (1 <= b <= 50):
            return f"Boule invalide : {b} (doit être 1–50)"
    for s in stars:
        if not (1 <= s <= max_star):
            return f"Étoile invalide : {s} (doit être 1–{max_star} en période actuelle)"
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = sys.argv[1:]

    # ── Options ───────────────────────────────────────────────────────────
    user_combo: tuple[list[int], list[int]] | None = None
    if "--score" in args:
        idx  = args.index("--score")
        nums = [int(x) for x in args[idx + 1: idx + 8]]
        if len(nums) != 7:
            print("--score attend exactement 7 nombres : 5 boules + 2 étoiles")
            sys.exit(1)
        user_combo = (sorted(nums[:5]), sorted(nums[5:]))

    last_n = int(next(
        (args[i + 1] for i, a in enumerate(args) if a == "--last-n"), "100"
    ))
    n_neg_ratio = int(next(
        (args[i + 1] for i, a in enumerate(args) if a == "--n-neg"), "20"
    ))
    no_plots = "--no-plots" in args

    # ── Chargement ────────────────────────────────────────────────────────
    if not CACHE_CSV.exists():
        print("Données introuvables. Lancez d'abord : python euromillions_scraper.py")
        sys.exit(1)

    df = load_results()
    print(f"{len(df)} tirages  [{df['date'].min().date()} → {df['date'].max().date()}]")

    # ── Split : tout sauf les last_n derniers tirages ─────────────────────
    last_n = min(last_n, len(df) - 50)   # laisser au moins 50 tirages en train
    train_df = df.iloc[:-last_n].reset_index(drop=True)
    test_df  = df.iloc[-last_n:].reset_index(drop=True)
    print(
        f"Train : {len(train_df)} tirages "
        f"[{train_df['date'].min().date()} → {train_df['date'].max().date()}]"
    )
    print(
        f"Test  : {len(test_df)} tirages "
        f"[{test_df['date'].min().date()} → {test_df['date'].max().date()}]"
    )

    # ── Entraînement ──────────────────────────────────────────────────────
    print("\nEntraînement…")
    model = ScoringModel(n_neg_ratio=n_neg_ratio)
    model.fit(train_df)

    fi_sorted = sorted(zip(FEATURE_NAMES, model.importances), key=lambda x: -x[1])
    print("\n  Top 10 features :")
    for name, imp in fi_sorted[:10]:
        bar = "█" * int(imp * 400)
        print(f"    {name:<28}  {imp:.5f}  {bar}")

    # ── Backtest ──────────────────────────────────────────────────────────
    print(f"\nBacktest sur {len(test_df)} tirages…")
    bt = model.backtest(test_df)

    ranked_b = bt["ranked_balls"]
    ranked_s = bt["ranked_stars"]
    THRESH_B = [5, 10, 15, 20, 25, 30]
    bth      = bt["by_threshold"]

    S = "=" * 64
    print(f"\n{S}")
    print(f"  BACKTEST – {bt['n_test_draws']} derniers tirages")
    print(S)

    # Classement des boules
    print("\n  Classement des boules par score (meilleur → moins bon) :")
    for start in range(0, 50, 10):
        chunk = ranked_b[start:start + 10]
        nums  = "  ".join(f"{b:2d}" for b in chunk)
        print(f"    Rang {start+1:2d}-{start+10:2d} : {nums}")

    # Classement des étoiles
    max_s = model.stats.max_star
    print(f"\n  Classement des étoiles (1–{max_s}) :")
    print("   ", "  ".join(f"{s:2d}" for s in ranked_s))

    # Table boules trouvées
    print(f"\n  {'Top K':<8}  {'Trouvées':<10}  {'Aléatoire':<12}  "
          f"{'≥2 trouvées':<13}  {'≥3 trouvées':<13}  {'≥4 trouvées'}")
    print(f"  {'-'*75}")
    for k in THRESH_B:
        d   = bth[f"balls_top{k}"]
        mf  = d["mean_found"]
        exp = d["expected_random"]
        g2  = d["pct_ge2"]
        g3  = d["pct_ge3"]
        g4  = d["pct_ge4"]
        diff = mf - exp
        flag = f" ({diff:+.2f})"
        print(f"  Top {k:<4}  {mf:<10.3f}  {exp:<12.3f}  "
              f"{g2:<13.1f}  {g3:<13.1f}  {g4:.1f}%")

    # Table étoiles trouvées
    THRESH_S = [2, 3, 4, 5, 6]
    print(f"\n  {'Top K':<8}  {'Étoiles/2':<12}  {'Aléatoire'}")
    print(f"  {'-'*35}")
    for k in THRESH_S:
        ks = min(k, max_s)
        d  = bth.get(f"stars_top{ks}")
        if d is None:
            continue
        print(f"  Top {ks:<4}  {d['mean_found']:<12.3f}  {d['expected_random']:.3f}")

    # Verdict
    best_found = bth["balls_top30"]["mean_found"]
    best_exp   = bth["balls_top30"]["expected_random"]
    diff_pct   = (best_found - best_exp) / best_exp * 100
    if abs(diff_pct) < 3:
        verdict = "Le modèle ne trouve pas plus de chiffres que l'aléatoire (attendu : loterie)."
    elif diff_pct > 0:
        verdict = (f"Le modèle trouve légèrement plus de boules que l'aléatoire "
                   f"(+{diff_pct:.1f}% dans le top 30).")
    else:
        verdict = f"Le modèle est légèrement sous l'aléatoire ({diff_pct:.1f}%)."
    print(f"\n  → {verdict}")

    # Sauvegarde
    STATS_DIR.mkdir(parents=True, exist_ok=True)
    bt_save = {
        k: v for k, v in bt.items()
        if k != "per_draw"
    }
    bt_save["feature_importances"] = {n: round(float(v), 6) for n, v in fi_sorted}
    (STATS_DIR / "backtest.json").write_text(
        json.dumps(bt_save, indent=2, ensure_ascii=False)
    )

    if not no_plots:
        print(f"\nGraphiques → {STATS_DIR}/")
        plot_backtest(bt, STATS_DIR)
        plot_feature_importance(model, STATS_DIR)

    # ── Score d'une sélection ─────────────────────────────────────────────
    if user_combo:
        balls, stars = user_combo
        err = validate_combination(balls, stars, model.stats.max_star)
        if err:
            print(f"\nCombinaisom invalide : {err}")
            sys.exit(1)

        print(f"\n{S}")
        print(f"  SCORE  boules {balls}  étoiles {stars}")
        print(S)

        exp   = model.explain(balls, stars)
        score = exp["score"]

        # Percentile vs 9 999 combinaisons aléatoires
        rand_scores = [model.score(*random_combination()) for _ in range(9_999)]
        pct_vs_rand = float(np.mean(np.array(rand_scores) < score / 100)) * 100

        print(f"  Score : {score:.1f} / 100")
        print(f"  Percentile vs 9 999 aléatoires : {pct_vs_rand:.1f}%")
        print()

        # Détail des features (triées par impact absolu)
        feats   = exp["features"]
        ordered = sorted(feats.items(),
                         key=lambda x: abs(x[1]["z_score"]) * x[1]["importance"],
                         reverse=True)

        print(f"  {'Feature':<28}  {'Valeur':>8}  {'Réf.':>8}  "
              f"{'Z-score':>8}  {'Import.':>8}")
        print(f"  {'-'*66}")
        for name, d in ordered:
            z    = d["z_score"]
            flag = " ★" if z > 1.3 else " ▼" if z < -1.3 else ""
            print(
                f"  {name:<28}  {d['value']:>8.3f}  {d['ref_mean']:>8.3f}  "
                f"  {z:>+7.2f}  {d['importance']:>8.5f}{flag}"
            )

        if not no_plots:
            plot_score_explanation(exp, STATS_DIR)

        # Sauvegarde JSON
        (STATS_DIR / "last_score.json").write_text(
            json.dumps(exp, indent=2, ensure_ascii=False)
        )
        print(f"\n  Résultat complet → {STATS_DIR}/last_score.json")


if __name__ == "__main__":
    main()
