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
  python scoring_model.py                                # backtest (50 derniers tirages) + modèle final
  python scoring_model.py --last-n 100                   # backtest sur 100 derniers tirages
  python scoring_model.py --recommend 100000             # top 20 sur 100 000 tirages aléatoires
  python scoring_model.py --score 5 14 23 42 49 2 8      # score une sélection
  python scoring_model.py --ablation                     # étude d'ablation feature par feature + groupes
  python scoring_model.py --select-features              # exclut cooc_pairs + deficit (surapprentissage)
  python scoring_model.py --compare-features             # compare 52 features vs 44 sélectionnées
  python scoring_model.py --n-neg 20                     # ratio négatifs/positifs (défaut 20)
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

PRIMES:  frozenset[int] = frozenset({2,3,5,7,11,13,17,19,23,29,31,37,41,43,47})
SQUARES: frozenset[int] = frozenset({1,4,9,16,25,36,49})

# Noms des 52 features (même ordre que extract_features)
FEATURE_NAMES: list[str] = [
    # ── Fréquence historique des boules ───────────────────────────────────
    "ball_freq_mean", "ball_freq_min", "ball_freq_max", "ball_freq_std",
    "ball_freq_log_sum",
    # ── Déficit de fréquence (boules sous-représentées vs espérance) ──────
    "ball_freq_deficit_mean", "ball_freq_deficit_max",
    "ball_freq_deficit_std", "n_deficit_balls",
    # ── Recency boules ─────────────────────────────────────────────────────
    "ball_rec_mean", "ball_rec_max", "ball_rec_std", "ball_rec_min",
    # ── Fréquence étoiles ──────────────────────────────────────────────────
    "star_freq_mean", "star_freq_min", "star_freq_std",
    # ── Recency étoiles ────────────────────────────────────────────────────
    "star_rec_mean", "star_rec_max", "star_rec_min",
    # ── Structure ──────────────────────────────────────────────────────────
    "sum_balls", "sum_zscore",
    "range_balls", "n_odd_balls", "n_low_balls",
    # ── Distribution par dizaine ───────────────────────────────────────────
    "n_dec_01_10", "n_dec_11_20", "n_dec_21_30", "n_dec_31_40", "n_dec_41_50",
    "decade_entropy", "max_same_decade",
    # ── Écarts entre boules consécutives ───────────────────────────────────
    "consec_gap_mean", "consec_gap_std", "consec_gap_min", "consec_gap_max",
    "n_consec_pairs",
    # ── Étoiles ────────────────────────────────────────────────────────────
    "sum_stars", "star_gap", "n_stars_low_half",
    # ── Hot zones (50 et 10 derniers tirages) ──────────────────────────────
    "ball_hot50_mean", "ball_hot50_n",
    "ball_hot10_mean", "ball_hot10_n",
    "ball_freq_momentum",
    # ── Co-occurrence paires ───────────────────────────────────────────────
    "pair_cooc_mean", "pair_cooc_max", "pair_cooc_log_sum", "pair_cooc_min",
    # ── Rang + co-occurrence étoiles ───────────────────────────────────────
    "ball_rank_mean",
    "star_cooc",
    # ── Numérologie (biais comportemental des joueurs) ─────────────────────
    "n_prime_balls", "n_square_balls",
]

assert len(FEATURE_NAMES) == 52

# Groupes de features pour l'étude d'ablation
FEATURE_GROUPS: dict[str, list[str]] = {
    "freq_hist":    ["ball_freq_mean", "ball_freq_min", "ball_freq_max",
                     "ball_freq_std", "ball_freq_log_sum"],
    "deficit":      ["ball_freq_deficit_mean", "ball_freq_deficit_max",
                     "ball_freq_deficit_std", "n_deficit_balls"],
    "recency_ball": ["ball_rec_mean", "ball_rec_max", "ball_rec_std", "ball_rec_min"],
    "star_freq":    ["star_freq_mean", "star_freq_min", "star_freq_std"],
    "star_rec":     ["star_rec_mean", "star_rec_max", "star_rec_min"],
    "structure":    ["sum_balls", "sum_zscore", "range_balls", "n_odd_balls", "n_low_balls"],
    "decades":      ["n_dec_01_10", "n_dec_11_20", "n_dec_21_30", "n_dec_31_40",
                     "n_dec_41_50", "decade_entropy", "max_same_decade"],
    "gaps":         ["consec_gap_mean", "consec_gap_std", "consec_gap_min",
                     "consec_gap_max", "n_consec_pairs"],
    "stars_comb":   ["sum_stars", "star_gap", "n_stars_low_half"],
    "hot_recent":   ["ball_hot50_mean", "ball_hot50_n", "ball_hot10_mean",
                     "ball_hot10_n", "ball_freq_momentum"],
    "cooc_pairs":   ["pair_cooc_mean", "pair_cooc_max",
                     "pair_cooc_log_sum", "pair_cooc_min"],
    "rank":         ["ball_rank_mean"],
    "cooc_stars":   ["star_cooc"],
    "numerology":   ["n_prime_balls", "n_square_balls"],
}

# Sélection de features basée sur l'ablation :
# cooc_pairs (surapprentissage, -0.250) + deficit (neutre, -0.250) sont exclus.
# Les groupes conservés battent l'aléatoire en isolation (Seul T30 ≥ 3.000).
_ABLATION_EXCLUDE = {"cooc_pairs", "deficit"}
SELECTED_INDICES: list[int] = [
    i for i, name in enumerate(FEATURE_NAMES)
    if all(name not in FEATURE_GROUPS.get(grp, []) for grp in _ABLATION_EXCLUDE)
]
# 52 - 4 (cooc_pairs) - 4 (deficit) = 44 features
assert len(SELECTED_INDICES) == 44


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

        # ── Recency + matrices indicatrices ──────────────────────────────
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

        # ── Co-occurrence des paires de boules ───────────────────────────
        bm = ball_mat.astype(np.float32)
        self.pair_cooc: np.ndarray = (bm.T @ bm) / n   # 50×50 : frac. tirages en commun
        np.fill_diagonal(self.pair_cooc, 0.0)

        # ── Co-occurrence des paires d'étoiles ───────────────────────────
        sm = star_mat.astype(np.float32)
        star_cooc_mat = (sm.T @ sm) / n   # 12×12
        np.fill_diagonal(star_cooc_mat, 0.0)
        self.star_cooc_mat: np.ndarray = star_cooc_mat

        # ── Fréquence récente (50 et 10 derniers tirages) ───────────────
        w50 = min(50, n)
        b50 = pd.concat([df.iloc[-w50:][c] for c in BALL_COLS]).dropna().astype(int)
        bc50 = Counter(b50.tolist())
        self.ball_freq_50: dict[int, float] = {b: bc50.get(b, 0) / w50 for b in range(1, 51)}

        w10 = min(10, n)
        b10 = pd.concat([df.iloc[-w10:][c] for c in BALL_COLS]).dropna().astype(int)
        bc10 = Counter(b10.tolist())
        self.ball_freq_10: dict[int, float] = {b: bc10.get(b, 0) / w10 for b in range(1, 51)}

        # ── Rang de fréquence globale (normalisé 0–1, plus petit = plus fréquent) ──
        sorted_b = sorted(range(1, 51), key=lambda b: -self.ball_freq[b])
        self.ball_rank: dict[int, float] = {
            b: (sorted_b.index(b) + 1) / 50.0 for b in range(1, 51)
        }

        # ── Déficit de fréquence (boules sous-représentées vs espérance) ──
        expected_freq = 5.0 / 50.0  # = 0.10
        self.ball_freq_deficit: dict[int, float] = {
            b: max(0.0, expected_freq - self.ball_freq[b]) / expected_freq
            for b in range(1, 51)
        }

        # ── Distribution théorique de la somme (loi hypergéométrique approx.) ──
        # E[sum] = 5 × 51/2 = 127.5,  Var[sum] = 5 × (50²-1)/12 × (50-5)/(50-1) ≈ 218.75
        self.sum_mean = 127.5
        self.sum_std  = float(np.sqrt(218.75))

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
    """Retourne un vecteur de 41 features pour une combinaison donnée."""
    balls = sorted(balls)
    stars = sorted(stars)

    # ── Boules – fréquence ───────────────────────────────────────────────
    bf = np.array([stats.ball_freq.get(b, 0.0) for b in balls])
    bf_mean, bf_min, bf_max, bf_std = bf.mean(), bf.min(), bf.max(), bf.std()

    # ── Boules – recency ─────────────────────────────────────────────────
    br = np.array([float(stats.ball_recency.get(b, stats.n_draws)) for b in balls])
    br_mean, br_max, br_std, br_min = br.mean(), br.max(), br.std(), br.min()

    # ── Étoiles – fréquence normalisée ───────────────────────────────────
    sf = np.array([stats.star_freq_ratio.get(s, 0.0) for s in stars])
    sf_mean, sf_min, sf_std = sf.mean(), sf.min(), sf.std()

    # ── Étoiles – recency ────────────────────────────────────────────────
    sr = np.array([float(stats.star_recency.get(s, stats.n_draws)) for s in stars])
    sr_mean, sr_max, sr_min = sr.mean(), sr.max(), sr.min()

    # ── Structure ────────────────────────────────────────────────────────
    sum_b   = float(sum(balls))           # théorique si uniforme : 127.5
    range_b = float(balls[-1] - balls[0])
    n_odd   = float(sum(b % 2 for b in balls))
    n_low   = float(sum(1 for b in balls if b <= 25))

    # ── Répartition par dizaine ──────────────────────────────────────────
    dec = [(b - 1) // 10 for b in balls]          # 0–4
    dc  = [float(dec.count(d)) for d in range(5)]
    dec_ent = float(scipy_entropy(np.array(dc) + 1e-9))
    max_dec = float(max(dc))

    # ── Écarts entre boules consécutives (triées) ────────────────────────
    gaps  = [balls[i + 1] - balls[i] for i in range(4)]
    g_arr = np.array(gaps, dtype=float)
    g_mean, g_std, g_min, g_max = g_arr.mean(), g_arr.std(), g_arr.min(), g_arr.max()
    n_consec = float(sum(1 for g in gaps if g == 1))

    # ── Étoiles – combinaison ────────────────────────────────────────────
    sum_s   = float(sum(stars))
    star_gap = float(stars[1] - stars[0]) if len(stars) == 2 else 0.0
    n_s_low = float(sum(1 for s in stars if s <= stats.max_star // 2))

    # ── Log-produit des fréquences ───────────────────────────────────────
    freq_log_sum = float(np.sum(np.log(bf + 1e-9)))

    # ── Déficit de fréquence (boules "en retard" sur leur quota) ─────────
    deficit      = np.array([stats.ball_freq_deficit.get(b, 0.0) for b in balls])
    deficit_mean = float(deficit.mean())
    deficit_max  = float(deficit.max())
    deficit_std  = float(deficit.std())
    n_deficit    = float((deficit > 0).sum())

    # ── Z-score de la somme vs distribution théorique ────────────────────
    sum_zscore = float((sum_b - stats.sum_mean) / stats.sum_std)

    # ── Fréquence récente (50 et 10 derniers tirages) ────────────────────
    bf50  = np.array([stats.ball_freq_50.get(b, 0.0) for b in balls])
    hot50_mean = float(bf50.mean())
    hot50_n    = float((bf50 > 0.0).sum())

    bf10  = np.array([stats.ball_freq_10.get(b, 0.0) for b in balls])
    hot10_mean = float(bf10.mean())
    hot10_n    = float((bf10 > 0.0).sum())

    # ── Momentum : fréquence récente vs historique ────────────────────────
    ball_freq_momentum = float(hot50_mean - bf_mean)

    # ── Co-occurrence historique des paires de boules ────────────────────
    cooc_vals = np.array([
        stats.pair_cooc[balls[i] - 1, balls[j] - 1]
        for i in range(5) for j in range(i + 1, 5)
    ])
    cooc_mean    = float(cooc_vals.mean())
    cooc_max     = float(cooc_vals.max())
    cooc_min     = float(cooc_vals.min())
    cooc_log_sum = float(np.sum(np.log(cooc_vals + 1e-6)))

    # ── Numérologie ───────────────────────────────────────────────────────
    n_prime  = float(sum(1 for b in balls if b in PRIMES))
    n_square = float(sum(1 for b in balls if b in SQUARES))

    # ── Co-occurrence des étoiles ─────────────────────────────────────────
    star_cooc_val = float(stats.star_cooc_mat[stars[0] - 1, stars[1] - 1]) if len(stars) == 2 else 0.0

    # ── Rang de fréquence globale ─────────────────────────────────────────
    ranks     = np.array([stats.ball_rank.get(b, 0.5) for b in balls])
    rank_mean = float(ranks.mean())

    return np.array([
        # freq_hist (5)
        bf_mean, bf_min, bf_max, bf_std,
        freq_log_sum,
        # deficit (4)
        deficit_mean, deficit_max, deficit_std, n_deficit,
        # recency_ball (4)
        br_mean, br_max, br_std, br_min,
        # star_freq (3)
        sf_mean, sf_min, sf_std,
        # star_rec (3)
        sr_mean, sr_max, sr_min,
        # structure (5)
        sum_b, sum_zscore,
        range_b, n_odd, n_low,
        # decades (7)
        *dc, dec_ent, max_dec,
        # gaps (5)
        g_mean, g_std, g_min, g_max, n_consec,
        # stars_comb (3)
        sum_s, star_gap, n_s_low,
        # hot_recent (5)
        hot50_mean, hot50_n, hot10_mean, hot10_n,
        ball_freq_momentum,
        # cooc_pairs (4)
        cooc_mean, cooc_max, cooc_log_sum, cooc_min,
        # rank + star_cooc (2)
        rank_mean,
        star_cooc_val,
        # numerology (2)
        n_prime, n_square,
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

    def __init__(
        self,
        n_neg_ratio:  int = 20,
        seed:         int = 42,
        fast:         bool = False,
        feature_mask: list[int] | None = None,
    ) -> None:
        self.n_neg_ratio  = n_neg_ratio
        self.seed         = seed
        self.feature_mask = feature_mask   # None = toutes les features
        self.stats:  DrawStats | None = None
        self.scaler  = StandardScaler()
        if fast:
            self.clf = GradientBoostingClassifier(
                n_estimators=200,
                max_depth=3,
                learning_rate=0.1,
                subsample=0.8,
                min_samples_leaf=10,
                random_state=seed,
            )
        else:
            self.clf = GradientBoostingClassifier(
                n_estimators=600,
                max_depth=4,
                learning_rate=0.03,
                subsample=0.8,
                min_samples_leaf=5,
                random_state=seed,
            )
        self.lr      = LogisticRegression(max_iter=1000, random_state=seed)
        self.importances: np.ndarray | None = None

    def _mask(self, X: np.ndarray) -> np.ndarray:
        return X[:, self.feature_mask] if self.feature_mask is not None else X

    def _active_names(self) -> list[str]:
        if self.feature_mask is None:
            return FEATURE_NAMES
        return [FEATURE_NAMES[i] for i in self.feature_mask]

    # ── Entraînement ─────────────────────────────────────────────────────

    def fit(self, train_df: pd.DataFrame) -> "ScoringModel":
        print(f"  Statistiques de référence ({len(train_df)} tirages)…")
        self.stats = DrawStats(train_df)

        print("  Construction du jeu d'entraînement…")
        X, y = build_dataset(train_df, self.stats, self.n_neg_ratio, self.seed)
        Xm   = self._mask(X)
        print(f"    {int(y.sum()):,} positifs · {int((y==0).sum()):,} négatifs · {Xm.shape[1]} features")

        Xs = self.scaler.fit_transform(Xm)
        self.clf.fit(Xs, y)
        self.lr.fit(Xs, y)
        self.importances = self.clf.feature_importances_

        # Distribution de référence (sur les vrais tirages, dans l'espace masqué)
        self.stats.ref_mean = Xm[y == 1].mean(axis=0)
        self.stats.ref_std  = Xm[y == 1].std(axis=0) + 1e-9
        return self

    # ── Score d'une combinaison ───────────────────────────────────────────

    def score(self, balls: list[int], stars: list[int]) -> float:
        """Score 0–1 (GBT). Plus élevé = plus similaire aux vrais tirages."""
        f  = extract_features(sorted(balls), sorted(stars), self.stats)
        fs = self.scaler.transform(self._mask(f.reshape(1, -1)))
        return float(self.clf.predict_proba(fs)[0, 1])

    def score_batch(self, combinations: list[tuple]) -> np.ndarray:
        """Score plusieurs (balls, stars) d'un coup (plus rapide)."""
        X  = np.vstack([
            extract_features(sorted(b), sorted(s), self.stats)
            for b, s in combinations
        ])
        Xs = self.scaler.transform(self._mask(X))
        return self.clf.predict_proba(Xs)[:, 1]

    # ── Explication ───────────────────────────────────────────────────────

    def explain(
        self,
        balls: list[int],
        stars: list[int],
    ) -> dict:
        """Retourne le score et le détail de chaque feature (valeur, z-score, importance)."""
        balls  = sorted(balls)
        stars  = sorted(stars)
        f_full = extract_features(balls, stars, self.stats)
        fm     = self._mask(f_full.reshape(1, -1))[0]
        z      = (fm - self.stats.ref_mean) / self.stats.ref_std
        score  = float(self.clf.predict_proba(
            self.scaler.transform(fm.reshape(1, -1))
        )[0, 1])

        return {
            "balls":  balls,
            "stars":  stars,
            "score":  round(score * 100, 2),
            "features": {
                name: {
                    "value":      round(float(fm[i]), 4),
                    "ref_mean":   round(float(self.stats.ref_mean[i]), 4),
                    "z_score":    round(float(z[i]), 3),
                    "importance": round(float(self.importances[i]), 5),
                }
                for i, name in enumerate(self._active_names())
            },
        }

    # ── Classement marginal des numéros ──────────────────────────────────

    def rank_numbers(
        self,
        n_samples: int = 500,
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



# ---------------------------------------------------------------------------
# Walk-forward backtest (vrai backtest chronologique)
# ---------------------------------------------------------------------------

def walk_forward_backtest(
    df: pd.DataFrame,
    last_n: int = 20,
    n_neg_ratio: int = 10,
    n_samples_ranking: int = 200,
    seed: int = 42,
    feature_mask: list[int] | None = None,
    label: str = "",
) -> dict:
    """
    Pour chaque tirage t parmi les last_n derniers :
      1. Entraîne un modèle sur tous les tirages 0..t-1
      2. Classe les boules 1-50 par score marginal
      3. Mesure combien de boules du tirage t sont dans le top K
    Puis recommence pour t-1 (entraîne sur 0..t-2, teste t-1), etc.
    Résultat : vrai backtest sans fuite d'information.
    """
    THRESH_B = [5, 10, 15, 20, 25, 30]
    THRESH_S = [2, 3, 4, 5, 6]

    per_draw: list[dict] = []

    hdr = (f"  {'Tirage':<12}  {'Boules tirées':<18}"
           f"  {'T5':>3}  {'T10':>3}  {'T15':>3}  {'T20':>3}  {'T25':>3}  {'T30':>3}")
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))

    for i in range(last_n):
        # t = index (0-based) du tirage à prédire
        t = len(df) - last_n + i
        if t < 100:
            continue  # besoin d'au moins 100 tirages pour entraîner

        train_df = df.iloc[:t].reset_index(drop=True)
        test_row = df.iloc[t]

        actual_balls = {int(test_row[c]) for c in BALL_COLS if pd.notna(test_row[c])}
        actual_stars = {int(test_row[c]) for c in STAR_COLS if pd.notna(test_row[c])}
        if len(actual_balls) != 5 or len(actual_stars) != 2:
            continue

        # Entraînement sur les t premiers tirages
        model = ScoringModel(n_neg_ratio=n_neg_ratio, seed=seed, fast=True,
                             feature_mask=feature_mask)
        model.fit(train_df)

        # Classement des boules avec ce modèle
        ranked_b, ranked_s, _, _ = model.rank_numbers(n_samples=n_samples_ranking, seed=seed)
        max_s = model.stats.max_star

        draw: dict = {"date": str(test_row["date"].date())}
        for k in THRESH_B:
            draw[f"balls_top{k}"] = len(actual_balls & set(ranked_b[:k]))
        for k in THRESH_S:
            ks = min(k, max_s)
            draw[f"stars_top{ks}"] = len(actual_stars & set(ranked_s[:ks]))
        per_draw.append(draw)

        balls_str = " ".join(f"{b:2d}" for b in sorted(actual_balls))
        print(
            f"  {draw['date']:<12}  {balls_str:<18}"
            f"  {draw['balls_top5']:>3}"
            f"  {draw['balls_top10']:>3}"
            f"  {draw['balls_top15']:>3}"
            f"  {draw['balls_top20']:>3}"
            f"  {draw['balls_top25']:>3}"
            f"  {draw['balls_top30']:>3}"
        )

    n = len(per_draw)
    by_threshold: dict = {}

    for k in THRESH_B:
        key    = f"balls_top{k}"
        vals   = [r[key] for r in per_draw]
        exp    = 5 * k / 50
        counts = Counter(vals)
        by_threshold[key] = {
            "k":               k,
            "mean_found":      round(float(np.mean(vals)), 3),
            "expected_random": round(exp, 3),
            "dist":            {str(i): counts.get(i, 0) for i in range(6)},
            "pct_ge2":         round(float(np.mean([v >= 2 for v in vals])) * 100, 1),
            "pct_ge3":         round(float(np.mean([v >= 3 for v in vals])) * 100, 1),
            "pct_ge4":         round(float(np.mean([v >= 4 for v in vals])) * 100, 1),
        }

    for k in THRESH_S:
        key    = f"stars_top{k}"
        vals   = [r.get(key, 0) for r in per_draw]
        exp    = 2 * k / 12
        counts = Counter(vals)
        by_threshold[key] = {
            "k":               k,
            "mean_found":      round(float(np.mean(vals)), 3),
            "expected_random": round(exp, 3),
            "dist":            {str(i): counts.get(i, 0) for i in range(3)},
        }

    return {
        "n_test_draws":  n,
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
# Ablation study
# ---------------------------------------------------------------------------

def _fast_ablation_eval(
    precomputed: list,
    feat_indices: list[int],
    n_samp: int,
    seed: int,
) -> float:
    """Un passage walk-forward avec un sous-ensemble de features pré-calculé."""
    per_draw_top30 = []
    for item in precomputed:
        if item is None:
            continue
        X, y, X_rank, actual_balls = item
        Xs  = X[:, feat_indices]
        Xrs = X_rank[:, feat_indices]

        scaler = StandardScaler()
        Xs_fit  = scaler.fit_transform(Xs)
        Xrs_fit = scaler.transform(Xrs)

        clf = GradientBoostingClassifier(
            n_estimators=100, max_depth=2, learning_rate=0.15,
            subsample=0.8, min_samples_leaf=10, random_state=seed,
        )
        clf.fit(Xs_fit, y)

        scores = clf.predict_proba(Xrs_fit)[:, 1]
        ball_sc = {b: float(scores[i * n_samp:(i + 1) * n_samp].mean())
                   for i, b in enumerate(range(1, 51))}
        ranked  = sorted(range(1, 51), key=lambda b: -ball_sc[b])
        per_draw_top30.append(len(actual_balls & set(ranked[:30])))

    return round(float(np.mean(per_draw_top30)), 3) if per_draw_top30 else 0.0


def ablation_study(
    df: pd.DataFrame,
    last_n: int = 20,
    seed:   int = 42,
) -> None:
    """
    Étude d'ablation par groupe de features.
    Pour chaque groupe G :
      - Leave-one-out  : entraîne sans G,  mesure la dégradation (impact)
      - Group-only     : entraîne avec G seul, mesure la prédictivité isolée
    Affiche aussi les importances GBT par groupe sur un modèle unique final.
    """
    N_SAMP  = 50   # combos par boule pour le ranking (vitesse > précision)
    N_NEG   = 5    # ratio négatifs/positifs (réduit pour rapidité)
    S       = "=" * 72

    print(f"\n{S}")
    print("  ÉTUDE D'ABLATION  –  52 features  ·  14 groupes")
    print(S)
    print(f"\n  Pré-calcul des {last_n} datasets walk-forward (n_neg={N_NEG})…")

    # ── 1. Pré-calcul des matrices (une fois par position) ───────────────
    precomputed: list = []
    for i in range(last_n):
        t = len(df) - last_n + i
        if t < 100:
            precomputed.append(None)
            continue

        train_df     = df.iloc[:t].reset_index(drop=True)
        test_row     = df.iloc[t]
        actual_balls = {int(test_row[c]) for c in BALL_COLS if pd.notna(test_row[c])}
        if len(actual_balls) != 5:
            precomputed.append(None)
            continue

        stats    = DrawStats(train_df)
        X, y     = build_dataset(train_df, stats, n_neg_ratio=N_NEG, seed=seed)

        random.seed(seed)
        rank_combos = []
        for b in range(1, 51):
            pool = [x for x in range(1, 51) if x != b]
            for _ in range(N_SAMP):
                others = random.sample(pool, 4)
                balls  = sorted([b] + others)
                stars  = sorted(random.sample(range(1, stats.max_star + 1), 2))
                rank_combos.append((balls, stars))

        X_rank = np.vstack([extract_features(b, s, stats) for b, s in rank_combos])
        precomputed.append((X, y, X_rank, actual_balls))

    # ── 2. Importances GBT sur un modèle unique (toutes les features) ────
    print("  Entraînement modèle de référence pour importances GBT…")
    ref_stats = DrawStats(df.iloc[:-last_n].reset_index(drop=True))
    X_ref, y_ref = build_dataset(df.iloc[:-last_n].reset_index(drop=True),
                                  ref_stats, n_neg_ratio=N_NEG, seed=seed)
    ref_clf = GradientBoostingClassifier(
        n_estimators=300, max_depth=3, learning_rate=0.08,
        subsample=0.8, min_samples_leaf=5, random_state=seed,
    )
    sc_ref = StandardScaler()
    ref_clf.fit(sc_ref.fit_transform(X_ref), y_ref)
    gbt_imp: np.ndarray = ref_clf.feature_importances_

    # ── 3. Baseline (toutes les features) ────────────────────────────────
    all_idx = list(range(len(FEATURE_NAMES)))
    print("  Baseline (toutes features)…")
    baseline = _fast_ablation_eval(precomputed, all_idx, N_SAMP, seed)

    # ── 4. Leave-one-out + group-only ────────────────────────────────────
    n_groups = len(FEATURE_GROUPS)
    results  = []

    for gi, (group_name, feat_names) in enumerate(FEATURE_GROUPS.items(), 1):
        feat_idx = [FEATURE_NAMES.index(f) for f in feat_names]
        keep_idx = [i for i in all_idx if i not in feat_idx]
        group_gbt_imp = float(gbt_imp[feat_idx].sum())

        print(f"  [{gi:02d}/{n_groups}] {group_name:<16}  "
              f"leave-one-out…", end="", flush=True)
        without = _fast_ablation_eval(precomputed, keep_idx, N_SAMP, seed)

        print("  group-only…", end="", flush=True)
        only    = _fast_ablation_eval(precomputed, feat_idx,  N_SAMP, seed)
        print()

        results.append({
            "group":      group_name,
            "n_feat":     len(feat_names),
            "gbt_imp":    group_gbt_imp,
            "without":    without,
            "impact":     round(baseline - without, 3),
            "only":       only,
            "only_delta": round(only - 3.0, 3),
        })

    # ── 5. Feature par feature (importances GBT triées) ──────────────────
    fi_sorted = sorted(enumerate(FEATURE_NAMES),
                       key=lambda x: -gbt_imp[x[0]])
    group_of: dict[str, str] = {}
    for gname, fnames in FEATURE_GROUPS.items():
        for fn in fnames:
            group_of[fn] = gname

    print(f"\n{S}")
    print("  FEATURE PAR FEATURE  –  importance GBT (modèle de référence)")
    print(S)
    print(f"\n  {'#':>3}  {'Feature':<26}  {'Groupe':<16}  {'GBT Imp.':>9}")
    print(f"  {'-'*62}")
    for rank_i, (fi, fname) in enumerate(fi_sorted, 1):
        bar  = "█" * int(gbt_imp[fi] * 200)
        grp  = group_of.get(fname, "?")
        print(f"  {rank_i:>3}  {fname:<26}  {grp:<16}  {gbt_imp[fi]*100:>8.2f}%  {bar}")

    # ── 6. Résultat ablation par groupe ──────────────────────────────────
    results.sort(key=lambda r: -r["impact"])

    print(f"\n{S}")
    print(f"  ABLATION PAR GROUPE  ·  baseline Top30 = {baseline:.3f}"
          f"  (Δ vs aléatoire : {baseline - 3.0:+.3f})")
    print(S)
    print(f"\n  {'Groupe':<16}  {'N':>3}  {'GBT%':>6}  "
          f"{'Sans T30':>9}  {'Impact':>8}  "
          f"{'Seul T30':>9}  {'Seul Δ':>8}")
    print(f"  {'-'*72}")

    for r in results:
        imp_flag  = " ★" if r["impact"] > 0.05 else (" ▲" if r["impact"] > 0 else "")
        only_flag = " ★" if r["only_delta"] > 0 else ""
        print(
            f"  {r['group']:<16}  {r['n_feat']:>3}  "
            f"{r['gbt_imp']*100:>5.1f}%  "
            f"  {r['without']:>7.3f}  {r['impact']:>+8.3f}{imp_flag}  "
            f"  {r['only']:>7.3f}  {r['only_delta']:>+8.3f}{only_flag}"
        )

    print(f"\n  Légende :")
    print(f"    Impact  = Baseline − Sans  (positif → ce groupe améliore le score)")
    print(f"    Seul Δ  = performance isolée vs aléatoire (3.000 attendu)")
    print(f"    GBT%    = part des importances GBT pour ce groupe")


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
        (args[i + 1] for i, a in enumerate(args) if a == "--last-n"), "50"
    ))
    n_recommend = int(next(
        (args[i + 1] for i, a in enumerate(args) if a == "--recommend"), "0"
    ))
    n_neg_ratio = int(next(
        (args[i + 1] for i, a in enumerate(args) if a == "--n-neg"), "20"
    ))
    no_plots         = "--no-plots" in args
    run_ablation     = "--ablation" in args
    select_features  = "--select-features" in args   # exclut cooc_pairs + deficit
    compare_features = "--compare-features" in args  # montre les deux modèles

    # ── Chargement ────────────────────────────────────────────────────────
    if not CACHE_CSV.exists():
        print("Données introuvables. Lancez d'abord : python euromillions_scraper.py")
        sys.exit(1)

    df = load_results()
    last_n = min(last_n, len(df) - 100)
    print(f"{len(df)} tirages  [{df['date'].min().date()} → {df['date'].max().date()}]")
    print(
        f"Walk-forward backtest sur les {last_n} derniers tirages\n"
        f"  (chaque tirage est prédit par un modèle entraîné sur tous les tirages précédents)\n"
    )

    # ── Walk-forward backtest ─────────────────────────────────────────────
    active_mask  = SELECTED_INDICES if select_features else None
    mask_label   = f"44 features sélectionnées" if select_features else "52 features"

    bt = walk_forward_backtest(df, last_n=last_n, n_neg_ratio=10,
                               n_samples_ranking=200, feature_mask=active_mask)

    THRESH_B = [5, 10, 15, 20, 25, 30]
    THRESH_S = [2, 3, 4, 5, 6]
    S = "=" * 68

    def _print_bt_summary(bt_res: dict, title: str) -> None:
        bth = bt_res["by_threshold"]
        print(f"\n{S}")
        print(f"  {title}")
        print(S)
        print(f"\n  {'Top K':<8}  {'Trouvées':<10}  {'Aléatoire':<12}  "
              f"{'≥2':>5}  {'≥3':>5}  {'≥4':>5}  {'Δ vs aléatoire':>15}")
        print(f"  {'-'*72}")
        for k in THRESH_B:
            d    = bth[f"balls_top{k}"]
            mf, exp  = d["mean_found"], d["expected_random"]
            diff = mf - exp
            flag = " ▲" if diff > 0.05 else (" ▼" if diff < -0.05 else "")
            print(f"  Top {k:<4}  {mf:<10.3f}  {exp:<12.3f}  "
                  f"{d['pct_ge2']:>5.1f}  {d['pct_ge3']:>5.1f}  {d['pct_ge4']:>5.1f}"
                  f"  {diff:>+8.3f}{flag}")
        print(f"\n  {'Étoiles Top K':<14}  {'Trouvées':<10}  {'Aléatoire'}")
        print(f"  {'-'*40}")
        for k in THRESH_S:
            d = bth.get(f"stars_top{k}")
            if d is None:
                continue
            diff = d["mean_found"] - d["expected_random"]
            print(f"  Top {k:<10}  {d['mean_found']:<10.3f}  {d['expected_random']:.3f}"
                  f"  ({diff:+.3f})")
        best_found = bth["balls_top30"]["mean_found"]
        best_exp   = bth["balls_top30"]["expected_random"]
        diff_pct   = (best_found - best_exp) / best_exp * 100
        if abs(diff_pct) < 3:
            verdict = "≈ aléatoire."
        elif diff_pct > 0:
            verdict = f"+{diff_pct:.1f}% au-dessus de l'aléatoire dans le top 30."
        else:
            verdict = f"{diff_pct:.1f}% sous l'aléatoire dans le top 30."
        print(f"\n  → {verdict}")

    _print_bt_summary(bt, f"RÉSUMÉ – {bt['n_test_draws']} tirages  [{mask_label}]")

    # ── Comparaison (--compare-features) ─────────────────────────────────
    if compare_features:
        other_mask  = None if select_features else SELECTED_INDICES
        other_label = "52 features (toutes)" if select_features else "44 features sélectionnées"
        print(f"\n  Comparaison : backtest avec {other_label}…")
        bt_other = walk_forward_backtest(df, last_n=last_n, n_neg_ratio=10,
                                         n_samples_ranking=200, feature_mask=other_mask)
        _print_bt_summary(bt_other, f"COMPARAISON – {bt_other['n_test_draws']} tirages  [{other_label}]")

        # Tableau côte à côte
        print(f"\n{S}")
        print(f"  COMPARAISON DIRECTE  –  Top K boules trouvées")
        print(S)
        print(f"\n  Top K   {mask_label:>26}  {other_label:>26}  Δ")
        print(f"  {'-'*70}")
        bth_a = bt["by_threshold"]
        bth_b = bt_other["by_threshold"]
        for k in THRESH_B:
            a = bth_a[f"balls_top{k}"]["mean_found"]
            b = bth_b[f"balls_top{k}"]["mean_found"]
            delta = a - b
            flag  = " ▲" if delta > 0.05 else (" ▼" if delta < -0.05 else "")
            print(f"  Top {k:<3}  {a:>26.3f}  {b:>26.3f}  {delta:>+.3f}{flag}")

    # Sauvegarde backtest
    STATS_DIR.mkdir(parents=True, exist_ok=True)
    bt_save = {k: v for k, v in bt.items() if k != "per_draw"}
    (STATS_DIR / "backtest.json").write_text(
        json.dumps(bt_save, indent=2, ensure_ascii=False)
    )

    # ── Modèle final (entraîné sur tous les tirages sauf le dernier) ──────
    print(f"\n{S}")
    print("  MODÈLE FINAL  (entraîné sur la totalité des tirages)")
    print(S)
    print("\nEntraînement…")
    model = ScoringModel(n_neg_ratio=n_neg_ratio, feature_mask=active_mask)
    model.fit(df.reset_index(drop=True))

    fi_sorted = sorted(zip(model._active_names(), model.importances), key=lambda x: -x[1])
    print("\n  Top 10 features :")
    for name, imp in fi_sorted[:10]:
        bar = "█" * int(imp * 400)
        print(f"    {name:<28}  {imp:.5f}  {bar}")

    # Classement actuel des boules
    print("\nClassement des boules (score marginal sur données actuelles)…")
    ranked_b, ranked_s, _, _ = model.rank_numbers(n_samples=500)
    max_s = model.stats.max_star

    print("\n  Classement des boules (meilleur → moins bon) :")
    for start in range(0, 50, 10):
        chunk = ranked_b[start:start + 10]
        nums  = "  ".join(f"{b:2d}" for b in chunk)
        print(f"    Rang {start+1:2d}-{start+10:2d} : {nums}")
    print(f"\n  Classement des étoiles (1–{max_s}) :")
    print("   ", "  ".join(f"{s:2d}" for s in ranked_s))

    if not no_plots:
        print(f"\nGraphiques → {STATS_DIR}/")
        plot_backtest(bt, STATS_DIR)
        plot_feature_importance(model, STATS_DIR)

    # ── Étude d'ablation ──────────────────────────────────────────────────
    if run_ablation:
        ablation_study(df, last_n=min(last_n, 20))

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

    # ── Mode recommandation : tirage aléatoire + top scores ───────────────
    if n_recommend > 0:
        print(f"\n{S}")
        print(f"  RECOMMANDATIONS  –  top 20 / {n_recommend:,} tirages aléatoires")
        print(S)
        print(f"\n  Génération de {n_recommend:,} combinaisons…")

        random.seed(0)
        combos = [random_combination(model.stats.max_star) for _ in range(n_recommend)]
        scores = model.score_batch(combos)

        # Scores de référence pour percentile
        ref_scores = scores  # on se compare à la distribution générée

        top_idx = np.argsort(scores)[::-1][:20]

        print(f"\n  {'Rang':>5}  {'Boules':>18}  {'Étoiles':>8}  {'Score':>7}  {'Pct':>7}")
        print(f"  {'-'*55}")
        for rank, i in enumerate(top_idx, 1):
            b, s   = combos[i]
            sc     = scores[i]
            pct    = float(np.mean(ref_scores <= sc)) * 100
            b_str  = " ".join(f"{x:2d}" for x in sorted(b))
            s_str  = " ".join(f"{x:2d}" for x in sorted(s))
            print(f"  {rank:>5}  {b_str}  {s_str}  {sc*100:>6.1f}%  {pct:>6.1f}%")

        # Sauvegarde
        reco = [
            {
                "rank": i + 1,
                "balls": sorted(combos[top_idx[i]][0]),
                "stars": sorted(combos[top_idx[i]][1]),
                "score": round(float(scores[top_idx[i]]) * 100, 2),
                "percentile": round(float(np.mean(ref_scores <= scores[top_idx[i]])) * 100, 1),
            }
            for i in range(len(top_idx))
        ]
        (STATS_DIR / "recommendations.json").write_text(
            json.dumps(reco, indent=2, ensure_ascii=False)
        )
        print(f"\n  → {STATS_DIR}/recommendations.json")


if __name__ == "__main__":
    main()
