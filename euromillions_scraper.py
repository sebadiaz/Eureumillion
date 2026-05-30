#!/usr/bin/env python3
"""
EuroMillions – téléchargement des résultats FDJ et statistiques.

Source officielle : API FDJ (sto.api.fdj.fr) — données depuis le 20/02/2004.
  Page FDJ : https://www.fdj.fr/jeux-de-tirage/euromillions-my-million/historique

Source de secours : UK National Lottery XML API (derniers ~52 tirages).

Usage :
  python euromillions_scraper.py                    # télécharge si pas de cache
  python euromillions_scraper.py --refresh          # force le re-téléchargement
  python euromillions_scraper.py --data-file my.csv # import CSV local
  python euromillions_scraper.py --no-plots         # stats texte uniquement

Format CSV local accepté :
  date,ball_1,ball_2,ball_3,ball_4,ball_5,star_1,star_2
  2024-01-12,3,21,34,42,49,5,8
  (virgule ou point-virgule ; date ISO ou DD/MM/YYYY)
"""

import io
import json
import sys
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from xml.etree import ElementTree as ET

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import requests
import seaborn as sns

try:
    from scipy import stats as scipy_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_DIR  = Path("data")
STATS_DIR = Path("stats")
CACHE_CSV = DATA_DIR / "euromillions_results.csv"

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; euromillions-stats/1.0)"}

# FDJ – API officielle (sto.api.fdj.fr)
# Chaque UUID correspond à un fichier ZIP contenant un CSV historique.
FDJ_BASE = (
    "https://www.sto.api.fdj.fr/anonymous/service-draw-info/v3/documentations/{uuid}"
)
FDJ_ZIPS = [
    # (uuid, description)
    ("1a2b3c4d-9876-4562-b3fc-2c963f66afa8", "2004-2011"),
    ("1a2b3c4d-9876-4562-b3fc-2c963f66afa9", "2011-2014"),
    ("1a2b3c4d-9876-4562-b3fc-2c963f66afb6", "2014-2016"),
    ("1a2b3c4d-9876-4562-b3fc-2c963f66afc6", "2016-2019"),
    ("1a2b3c4d-9876-4562-b3fc-2c963f66afd6", "2019-2020"),
    ("1a2b3c4d-9876-4562-b3fc-2c963f66afe6", "2020-today"),
]

# UK National Lottery – source de secours
UK_XML_LATEST = (
    "https://www.national-lottery.co.uk/results/euromillions/draw-history/xml"
)
UK_XML_BY_NUM = (
    "https://www.national-lottery.co.uk/results/euromillions/draw-history/{n}/xml"
)


# ---------------------------------------------------------------------------
# FDJ source
# ---------------------------------------------------------------------------

def _parse_fdj_zip(data: bytes) -> pd.DataFrame:
    """Extract and normalise the CSV inside a FDJ ZIP file."""
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        name = next(n for n in z.namelist() if n.endswith(".csv"))
        # FDJ files are latin-1 encoded
        raw = z.read(name).decode("latin-1", errors="replace")

    sep = ";" if raw.count(";") > raw.count(",") else ","
    # index_col=False: some files have a trailing ";" in data rows which
    # causes pandas to treat the first column as an index if omitted.
    df = pd.read_csv(io.StringIO(raw), sep=sep, dtype=str, index_col=False)
    # Normalise column names
    df.columns = [
        c.strip()
         .lower()
         .replace("é", "e")
         .replace("è", "e")
         .replace("ê", "e")
         .replace("û", "u")
         .replace(" ", "_")
        for c in df.columns
    ]

    # Keep only the columns we need
    keep = {}
    for col in df.columns:
        if col == "date_de_tirage":
            keep[col] = "date"
        elif col == "boule_1":
            keep[col] = "ball_1"
        elif col == "boule_2":
            keep[col] = "ball_2"
        elif col == "boule_3":
            keep[col] = "ball_3"
        elif col == "boule_4":
            keep[col] = "ball_4"
        elif col == "boule_5":
            keep[col] = "ball_5"
        elif col == "etoile_1":
            keep[col] = "star_1"
        elif col == "etoile_2":
            keep[col] = "star_2"

    df = df.rename(columns=keep)[list(keep.values())]

    # Parse dates: oldest file uses YYYYMMDD, others DD/MM/YYYY or DD/MM/YY
    dates = df["date"].str.strip()
    parsed = pd.to_datetime(dates, format="%Y%m%d", errors="coerce")
    # Rows that failed the YYYYMMDD format use DD/MM/YYYY or DD/MM/YY
    mask = parsed.isna()
    if mask.any():
        parsed[mask] = pd.to_datetime(
            dates[mask], format="%d/%m/%Y", errors="coerce"
        )
    mask = parsed.isna()
    if mask.any():
        parsed[mask] = pd.to_datetime(
            dates[mask], format="%d/%m/%y", errors="coerce"
        )
    df["date"] = parsed

    df = df.dropna(subset=["date"])
    for col in ["ball_1", "ball_2", "ball_3", "ball_4", "ball_5", "star_1", "star_2"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.dropna(subset=["ball_1"])


def fetch_from_fdj(dest: Path = CACHE_CSV) -> Path:
    """Download and merge all FDJ historical ZIP files."""
    print("Source : FDJ API officielle (sto.api.fdj.fr)")
    frames: list[pd.DataFrame] = []

    for uuid, label in FDJ_ZIPS:
        url = FDJ_BASE.format(uuid=uuid)
        try:
            resp = requests.get(url, timeout=20, headers=HEADERS)
            resp.raise_for_status()
            df = _parse_fdj_zip(resp.content)
            print(f"  [{label}]  {len(df):>4} tirages  "
                  f"({df['date'].min().date()} → {df['date'].max().date()})")
            frames.append(df)
        except Exception as e:
            print(f"  [{label}]  ⚠️  {e}")

    if not frames:
        raise RuntimeError("Aucun fichier FDJ récupéré.")

    combined = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(subset=["date"])
        .sort_values("date")
        .reset_index(drop=True)
    )
    combined["date"] = combined["date"].dt.strftime("%Y-%m-%d")

    dest.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(dest, index=False)
    print(f"\n  Total : {len(combined)} tirages → {dest}  "
          f"({dest.stat().st_size:,} octets)")
    return dest


# ---------------------------------------------------------------------------
# UK National Lottery – source de secours
# ---------------------------------------------------------------------------

def _parse_uk_xml(xml_text: str) -> dict | None:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return None
    draw_el = root.find(".//draw")
    if draw_el is None:
        return None
    balls = [b.text for b in root.findall(".//ball") if b.text]
    stars = [b.text for b in root.findall(".//bonus-ball") if b.text]
    if len(balls) < 5:
        return None
    return {
        "date":   draw_el.findtext("draw-date"),
        "ball_1": int(balls[0]), "ball_2": int(balls[1]),
        "ball_3": int(balls[2]), "ball_4": int(balls[3]),
        "ball_5": int(balls[4]),
        "star_1": int(stars[0]) if stars else None,
        "star_2": int(stars[1]) if len(stars) > 1 else None,
    }


def fetch_from_uk_lottery(dest: Path = CACHE_CSV) -> Path:
    """Download available draws from UK National Lottery XML API (fallback)."""
    print("Source de secours : UK National Lottery XML API")

    with requests.Session() as sess:
        resp = sess.get(UK_XML_LATEST, timeout=10, headers=HEADERS)
        resp.raise_for_status()
        latest = _parse_uk_xml(resp.text)

    if not latest:
        raise RuntimeError("Impossible de lire le dernier tirage UK")

    latest_num = int(
        ET.fromstring(
            requests.get(UK_XML_LATEST, timeout=10, headers=HEADERS).text
        ).findtext(".//draw-number") or "0"
    )

    # Find first available draw (binary search)
    def available(n: int, sess: requests.Session) -> bool:
        r = sess.get(UK_XML_BY_NUM.format(n=n), timeout=8, headers=HEADERS)
        return r.status_code == 200 and "draw-number" in r.text

    with requests.Session() as sess:
        lo_hard = max(1, latest_num - 200)
        hi_hard = latest_num
        while lo_hard < hi_hard:
            mid = (lo_hard + hi_hard) // 2
            if available(mid, sess):
                hi_hard = mid
            else:
                lo_hard = mid + 1
        first_num = lo_hard

    total = latest_num - first_num + 1
    print(f"  {total} tirages disponibles (#{first_num} → #{latest_num})")

    rows: list[dict] = []
    with requests.Session() as sess:
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = {
                pool.submit(
                    lambda n=n: _parse_uk_xml(
                        sess.get(UK_XML_BY_NUM.format(n=n),
                                 timeout=8, headers=HEADERS).text
                    )
                ): n
                for n in range(first_num, latest_num + 1)
            }
            for fut in as_completed(futures):
                r = fut.result()
                if r:
                    rows.append(r)

    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest, index=False)
    print(f"  {len(df)} tirages → {dest}")
    return dest


# ---------------------------------------------------------------------------
# Import CSV local
# ---------------------------------------------------------------------------

def import_local_csv(src: Path, dest: Path = CACHE_CSV) -> Path:
    """
    Import a user-provided CSV file.
    Accepts comma or semicolon separator, French or English column names,
    dates in ISO (YYYY-MM-DD) or European (DD/MM/YYYY) format.
    """
    raw = src.read_text(encoding="utf-8", errors="replace")
    sep = ";" if raw.count(";") > raw.count(",") else ","
    df = pd.read_csv(io.StringIO(raw), sep=sep, dtype=str)
    df.columns = [c.strip().lower() for c in df.columns]

    rename: dict[str, str] = {}
    for col in df.columns:
        c = col.replace("é", "e").replace("è", "e")
        mapping = {
            "date_de_tirage": "date", "date": "date",
            "boule_1": "ball_1", "ball_1": "ball_1", "num1": "ball_1",
            "boule_2": "ball_2", "ball_2": "ball_2", "num2": "ball_2",
            "boule_3": "ball_3", "ball_3": "ball_3", "num3": "ball_3",
            "boule_4": "ball_4", "ball_4": "ball_4", "num4": "ball_4",
            "boule_5": "ball_5", "ball_5": "ball_5", "num5": "ball_5",
            "etoile_1": "star_1", "star_1": "star_1",
            "etoile_2": "star_2", "star_2": "star_2",
        }
        if c in mapping:
            rename[col] = mapping[c]

    df = df.rename(columns=rename)
    df["date"] = pd.to_datetime(
        df.get("date", pd.Series(dtype=str)), dayfirst=True, errors="coerce"
    )
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")

    dest.parent.mkdir(parents=True, exist_ok=True)
    cols = [c for c in ["date","ball_1","ball_2","ball_3","ball_4","ball_5",
                         "star_1","star_2"] if c in df.columns]
    df[cols].to_csv(dest, index=False)
    print(f"  {len(df)} tirages importés → {dest}")
    return dest


# ---------------------------------------------------------------------------
# Orchestrateur
# ---------------------------------------------------------------------------

def ensure_data(
    dest: Path = CACHE_CSV,
    refresh: bool = False,
    data_file: Path | None = None,
) -> Path:
    if data_file:
        print(f"Utilisation du fichier local : {data_file}")
        return import_local_csv(data_file, dest)

    if not refresh and dest.exists() and dest.stat().st_size > 500:
        print(f"Cache trouvé : {dest}  (--refresh pour re-télécharger)")
        return dest

    for fetcher, name in [
        (fetch_from_fdj, "FDJ"),
        (fetch_from_uk_lottery, "UK National Lottery"),
    ]:
        try:
            return fetcher(dest)
        except Exception as e:
            print(f"  [{name}] indisponible : {e}")

    raise RuntimeError(
        "Aucune source disponible. Fournissez un CSV local avec --data-file"
    )


# ---------------------------------------------------------------------------
# Périodes réglementaires EuroMillions
# ---------------------------------------------------------------------------

PERIODS: list[dict] = [
    {
        "id":           1,
        "label":        "P1 · 2004–2011",
        "full_label":   "Période 1 : 2004 → 9 mai 2011  (5/50 + 2/9)",
        "start":        pd.Timestamp("2004-01-01"),
        "end":          pd.Timestamp("2011-05-09"),
        "max_star":     9,
        "jackpot_odds": 76_275_360,
        "color":        "#4C72B0",
    },
    {
        "id":           2,
        "label":        "P2 · 2011–2016",
        "full_label":   "Période 2 : 10 mai 2011 → sept. 2016  (5/50 + 2/11)",
        "start":        pd.Timestamp("2011-05-10"),
        "end":          pd.Timestamp("2016-09-26"),
        "max_star":     11,
        "jackpot_odds": 116_531_800,
        "color":        "#DD8452",
    },
    {
        "id":           3,
        "label":        "P3 · 2016–auj.",
        "full_label":   "Période 3 : depuis sept. 2016  (5/50 + 2/12)",
        "start":        pd.Timestamp("2016-09-27"),
        "end":          pd.Timestamp("2099-12-31"),
        "max_star":     12,
        "jackpot_odds": 139_838_160,
        "color":        "#55A868",
    },
]


def assign_period(df: pd.DataFrame) -> pd.DataFrame:
    """Add a 'period' column (1, 2 or 3) based on draw date."""
    df = df.copy()
    df["period"] = 0
    for p in PERIODS:
        mask = (df["date"] >= p["start"]) & (df["date"] <= p["end"])
        df.loc[mask, "period"] = p["id"]
    return df


# ---------------------------------------------------------------------------
# Chargement
# ---------------------------------------------------------------------------

def load_results(path: Path = CACHE_CSV) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    df.columns = [c.strip().lower() for c in df.columns]
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    for col in ["ball_1","ball_2","ball_3","ball_4","ball_5","star_1","star_2"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = assign_period(df)
    return df


# ---------------------------------------------------------------------------
# Statistiques
# ---------------------------------------------------------------------------

def _chi2(observed: list[int], expected_per_cell: float) -> tuple[float, float] | tuple[None, None]:
    if not HAS_SCIPY:
        return None, None
    chi2, p = scipy_stats.chisquare(observed)
    return round(float(chi2), 3), round(float(p), 6)


def _ball_freq(sub: pd.DataFrame) -> Counter:
    ball_cols = [c for c in sub.columns if c.startswith("ball_")]
    return Counter(
        pd.concat([sub[c] for c in ball_cols], ignore_index=True).dropna().astype(int).tolist()
    )


def _star_freq(sub: pd.DataFrame, max_star: int) -> Counter:
    star_cols = [c for c in sub.columns if c.startswith("star_")]
    raw = pd.concat([sub[c] for c in star_cols], ignore_index=True).dropna().astype(int)
    # discard stars outside valid range (data anomalies in transition draws)
    return Counter(v for v in raw.tolist() if 1 <= v <= max_star)


def compute_period_stats(df: pd.DataFrame) -> list[dict]:
    """Per-period ball and star frequency with chi-square uniformity test."""
    results = []
    for p in PERIODS:
        sub = df[df["period"] == p["id"]]
        if sub.empty:
            continue
        total     = len(sub)
        max_star  = p["max_star"]
        bf        = _ball_freq(sub)
        sf        = _star_freq(sub, max_star)

        obs_balls = [bf.get(n, 0) for n in range(1, 51)]
        obs_stars = [sf.get(n, 0) for n in range(1, max_star + 1)]
        chi2_b, p_b = _chi2(obs_balls, total * 5 / 50)
        chi2_s, p_s = _chi2(obs_stars, total * 2 / max_star)

        results.append({
            "period_id":          p["id"],
            "label":              p["full_label"],
            "draws":              total,
            "max_star":           max_star,
            "jackpot_odds":       p["jackpot_odds"],
            "ball_frequency":     {str(k): v for k, v in sorted(bf.items())},
            "star_frequency":     {str(k): v for k, v in sorted(sf.items())},
            "ball_expected":      round(total * 5 / 50, 1),
            "star_expected":      round(total * 2 / max_star, 1),
            "ball_chi2":          chi2_b,
            "ball_chi2_pvalue":   p_b,
            "star_chi2":          chi2_s,
            "star_chi2_pvalue":   p_s,
        })
    return results


def _star_normalized_expected(df: pd.DataFrame) -> dict[int, float]:
    """
    For each star 1–12, compute the expected count summed across all periods
    where that star was in play.
    Expected(star n) = sum_p [ draws_p * 2 / max_star_p  if n <= max_star_p ]
    """
    expected: dict[int, float] = {}
    for n in range(1, 13):
        e = 0.0
        for p in PERIODS:
            if n <= p["max_star"]:
                sub = df[df["period"] == p["id"]]
                e += len(sub) * 2 / p["max_star"]
        expected[n] = e
    return expected


def compute_stats(df: pd.DataFrame) -> dict:
    ball_cols = [c for c in df.columns if c.startswith("ball_")]
    star_cols = [c for c in df.columns if c.startswith("star_")]
    if not ball_cols:
        raise ValueError("Aucune colonne 'ball_*' trouvée.")

    total    = len(df)
    bf_all   = _ball_freq(df)
    sf_all   = Counter(
        pd.concat([df[c] for c in star_cols], ignore_index=True).dropna().astype(int).tolist()
    ) if star_cols else Counter()

    # Global chi-square (balls)
    obs_balls_all = [bf_all.get(n, 0) for n in range(1, 51)]
    chi2_b, p_b   = _chi2(obs_balls_all, total * 5 / 50)

    # Fréquence des paires
    pair_counter: Counter = Counter()
    for _, row in df[ball_cols].iterrows():
        nums = sorted(int(v) for v in row.dropna())
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                pair_counter[(nums[i], nums[j])] += 1

    # Écart moyen entre apparitions consécutives
    avg_gap: dict[int, float] = {}
    for num in range(1, 51):
        idx = df.index[
            df[ball_cols].apply(lambda r: num in r.dropna().astype(int).values, axis=1)
        ].tolist()
        if len(idx) >= 2:
            gaps = [b - a for a, b in zip(idx, idx[1:])]
            avg_gap[num] = round(sum(gaps) / len(gaps), 1)

    # Fréquence normalisée des étoiles (expected per period)
    star_expected = _star_normalized_expected(df)
    star_norm = {
        n: round(sf_all.get(n, 0) / e, 4) if e > 0 else 0.0
        for n, e in star_expected.items()
    }

    return {
        "total_draws": total,
        "date_range": {
            "first": str(df["date"].min().date()) if "date" in df.columns else "N/A",
            "last":  str(df["date"].max().date()) if "date" in df.columns else "N/A",
        },
        "ball_frequency":          {str(k): v for k, v in sorted(bf_all.items())},
        "ball_expected_per_number": round(total * 5 / 50, 1),
        "ball_chi2_global":         chi2_b,
        "ball_chi2_pvalue_global":  p_b,
        "star_frequency":           {str(k): v for k, v in sorted(sf_all.items())},
        "star_normalized_ratio":    {str(k): v for k, v in star_norm.items()},
        "top_10_balls":             dict(bf_all.most_common(10)),
        "bottom_10_balls":          dict(bf_all.most_common()[:-11:-1]),
        "top_5_stars_normalized":   dict(
            sorted(star_norm.items(), key=lambda x: -x[1])[:5]
        ),
        "top_10_pairs":             {str(k): v for k, v in pair_counter.most_common(10)},
        "avg_gap_per_number":       {str(k): v for k, v in avg_gap.items()},
        "periods":                  compute_period_stats(df),
    }


# ---------------------------------------------------------------------------
# Graphiques
# ---------------------------------------------------------------------------

def _ball_counts_from(sub: pd.DataFrame) -> tuple[list[int], list[int]]:
    freq = _ball_freq(sub)
    nums = list(range(1, 51))
    return nums, [freq.get(n, 0) for n in nums]


# ── 1. Fréquence globale des boules ────────────────────────────────────────

def plot_ball_frequency(df: pd.DataFrame, out: Path) -> None:
    nums, counts = _ball_counts_from(df)
    srt  = sorted(zip(nums, counts), key=lambda x: x[1], reverse=True)
    top5 = {n for n, _ in srt[:5]}
    bot5 = {n for n, _ in srt[-5:]}

    fig, ax = plt.subplots(figsize=(16, 5))
    bars = ax.bar(nums, counts, color="royalblue", edgecolor="white", linewidth=0.4)
    for bar, n in zip(bars, nums):
        if n in top5:   bar.set_color("seagreen")
        elif n in bot5: bar.set_color("tomato")

    expected = len(df) * 5 / 50
    ax.axhline(expected, color="grey", linestyle="--", linewidth=0.8,
               label=f"Espérance ({expected:.0f})")
    ax.set_xlabel("Numéro")
    ax.set_ylabel("Tirages")
    ax.set_title(
        f"Fréquence des numéros (1–50) · {len(df)} tirages  [2004–auj.]\n"
        "■ vert = top 5  ·  ■ rouge = bottom 5"
    )
    ax.set_xticks(nums)
    ax.tick_params(axis="x", labelsize=7)
    ax.legend()
    plt.tight_layout()
    p = out / "ball_frequency.png"
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"  → {p}")


# ── 2. Fréquence des boules par période (normalisée) ───────────────────────

def plot_ball_by_period(df: pd.DataFrame, out: Path) -> None:
    """
    For each period, plot the relative frequency of each ball:
    relative = observed / expected  (expected = draws_p * 5 / 50).
    Values > 1 mean the number appeared more than expected.
    """
    nums = list(range(1, 51))
    fig, axes = plt.subplots(3, 1, figsize=(16, 11), sharex=True, sharey=True)
    fig.suptitle(
        "Fréquence relative des boules par période\n"
        "(valeur = observé / espérance  —  1.0 = parfaitement uniforme)",
        fontsize=11,
    )

    for ax, p in zip(axes, PERIODS):
        sub  = df[df["period"] == p["id"]]
        if sub.empty:
            ax.set_title(f"{p['label']}  (aucun tirage)")
            continue
        total    = len(sub)
        expected = total * 5 / 50
        freq     = _ball_freq(sub)
        ratios   = [freq.get(n, 0) / expected for n in nums]

        colors = ["seagreen" if r >= 1.10 else "tomato" if r <= 0.90 else p["color"]
                  for r in ratios]
        ax.bar(nums, ratios, color=colors, edgecolor="white", linewidth=0.3)
        ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
        ax.axhspan(0.90, 1.10, alpha=0.06, color="grey")
        ax.set_ylabel("Obs./Esp.")
        ax.set_title(
            f"{p['label']}  ({total} tirages · espérance = {expected:.0f}/boule)"
        )
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    axes[-1].set_xlabel("Numéro")
    axes[-1].set_xticks(nums)
    axes[-1].tick_params(axis="x", labelsize=7)
    plt.tight_layout()
    path = out / "ball_by_period.png"
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  → {path}")


# ── 3. Fréquence des étoiles par période ───────────────────────────────────

def plot_star_by_period(df: pd.DataFrame, out: Path) -> None:
    """One panel per period, stars 1–max_star, with expected line."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        "Fréquence des étoiles chanceuses par période\n"
        "(les règles changent : 9 → 11 → 12 étoiles disponibles)",
        fontsize=11,
    )

    for ax, p in zip(axes, PERIODS):
        sub      = df[df["period"] == p["id"]]
        total    = len(sub)
        max_star = p["max_star"]
        if sub.empty:
            ax.set_title(p["label"])
            continue
        sf       = _star_freq(sub, max_star)
        star_nums = list(range(1, max_star + 1))
        counts    = [sf.get(n, 0) for n in star_nums]
        expected  = total * 2 / max_star

        colors = [
            "darkorange" if c >= expected * 1.10 else
            "steelblue"  if c <= expected * 0.90 else
            p["color"]
            for c in counts
        ]
        ax.bar(star_nums, counts, color=colors, edgecolor="white", linewidth=0.4)
        ax.axhline(expected, color="grey", linestyle="--", linewidth=0.9,
                   label=f"Esp. ({expected:.0f})")
        ax.set_title(
            f"{p['label']}\n{total} tirages · étoiles 1–{max_star}"
        )
        ax.set_xlabel("Étoile")
        ax.set_ylabel("Tirages" if ax is axes[0] else "")
        ax.set_xticks(star_nums)
        ax.legend(fontsize=8)

    plt.tight_layout()
    path = out / "star_by_period.png"
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  → {path}")


# ── 4. Étoiles normalisées sur toute la période ────────────────────────────

def plot_star_normalized(df: pd.DataFrame, out: Path, stats: dict) -> None:
    """
    For each star 1–12, show observed / expected where expected accounts for the
    fact that stars 10/11/12 were not available during earlier periods.
    """
    star_expected = _star_normalized_expected(df)
    star_obs      = {k: v for k, v in stats["star_frequency"].items()}
    star_obs_int  = {int(k): int(v) for k, v in star_obs.items()}

    nums   = list(range(1, 13))
    ratios = [star_obs_int.get(n, 0) / star_expected[n] if star_expected[n] > 0 else 0
              for n in nums]

    # Vertical annotations: which period each star was introduced
    period_intro = {n: (1 if n <= 9 else 2 if n <= 11 else 3) for n in nums}

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["seagreen" if r >= 1.05 else "tomato" if r <= 0.95 else "gold"
              for r in ratios]
    bars = ax.bar(nums, ratios, color=colors, edgecolor="white", linewidth=0.5)
    ax.axhline(1.0, color="black", linewidth=0.9, linestyle="--",
               label="Référence (uniforme)")
    ax.axhspan(0.95, 1.05, alpha=0.07, color="grey", label="±5 %")

    for bar, n, r in zip(bars, nums, ratios):
        ax.text(bar.get_x() + bar.get_width() / 2, r + 0.005,
                f"{r:.3f}", ha="center", va="bottom", fontsize=7.5)

    # Period intro labels
    for n in [10, 12]:
        pi = period_intro[n]
        ax.annotate(
            f"← intro. P{pi}",
            xy=(n, 0), xytext=(n, -0.07),
            fontsize=7, ha="center", color="dimgrey",
            arrowprops=None,
        )

    ax.set_xlabel("Étoile chanceuse")
    ax.set_ylabel("Ratio  observé / espérance théorique")
    ax.set_title(
        "Fréquence normalisée des étoiles (1–12) sur toute l'histoire\n"
        "Espérance corrigée par période (étoiles 10–12 non disponibles en P1/P2)"
    )
    ax.set_xticks(nums)
    ax.set_ylim(bottom=max(0, min(ratios) - 0.15))
    ax.legend()
    plt.tight_layout()
    path = out / "star_normalized.png"
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  → {path}")


# ── 5. Carte de chaleur (déviation par rapport à l'espérance) ──────────────

def plot_heatmap(df: pd.DataFrame, out: Path) -> None:
    """Heatmap showing % deviation from expected frequency (positive = hot)."""
    nums, counts = _ball_counts_from(df)
    expected     = len(df) * 5 / 50
    deviations   = [(c - expected) / expected * 100 for c in counts]

    grid_dev    = [[deviations[r * 10 + c] for c in range(10)] for r in range(5)]
    grid_annots = [[f"{r*10+c+1}\n{deviations[r*10+c]:+.1f}%" for c in range(10)]
                   for r in range(5)]

    fig, ax = plt.subplots(figsize=(13, 5))
    sns.heatmap(
        grid_dev, annot=grid_annots, fmt="s", cmap="RdYlGn",
        center=0, linewidths=0.5, ax=ax,
        cbar_kws={"label": "Écart à l'espérance (%)"},
        xticklabels=False, yticklabels=False,
    )
    ax.set_title(
        f"Carte de chaleur – écart à l'espérance par numéro  ·  {len(df)} tirages\n"
        "Vert = plus souvent que prévu  ·  Rouge = moins souvent"
    )
    plt.tight_layout()
    p = out / "heatmap.png"
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"  → {p}")


# ── 6. Tirages par année ────────────────────────────────────────────────────

def plot_draws_per_year(df: pd.DataFrame, out: Path) -> None:
    if "date" not in df.columns:
        return
    by_year = df.groupby(df["date"].dt.year).size()
    if by_year.empty:
        return

    period_colors = {}
    for year in by_year.index:
        ts = pd.Timestamp(f"{year}-07-01")
        for p in PERIODS:
            if p["start"] <= ts <= p["end"]:
                period_colors[year] = p["color"]
                break
        else:
            period_colors[year] = "steelblue"

    colors = [period_colors.get(y, "steelblue") for y in by_year.index]
    fig, ax = plt.subplots(figsize=(max(8, len(by_year) * 0.7), 4))
    ax.bar(by_year.index, by_year.values, color=colors, edgecolor="white")
    ax.set_xlabel("Année")
    ax.set_ylabel("Tirages")
    ax.set_title("Tirages EuroMillions par année (couleur = période réglementaire)")
    ax.set_xticks(by_year.index)
    ax.tick_params(axis="x", rotation=45)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=p["color"], label=p["label"]) for p in PERIODS
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=8)
    plt.tight_layout()
    p = out / "draws_per_year.png"
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"  → {p}")


# ── 7. Écart moyen entre apparitions ───────────────────────────────────────

def plot_gap_analysis(df: pd.DataFrame, out: Path, stats: dict) -> None:
    gaps = {int(k): v for k, v in stats.get("avg_gap_per_number", {}).items()}
    if not gaps:
        return
    nums = sorted(gaps)
    vals = [gaps[n] for n in nums]
    expected_gap = len(df) / (len(df) * 5 / 50)  # ≈ 10 draws between appearances

    fig, ax = plt.subplots(figsize=(16, 4))
    colors = ["tomato" if v > expected_gap * 1.15 else
              "seagreen" if v < expected_gap * 0.85 else
              "mediumpurple"
              for v in vals]
    ax.bar(nums, vals, color=colors, edgecolor="white", linewidth=0.4)
    ax.axhline(expected_gap, color="black", linewidth=0.8, linestyle="--",
               label=f"Espérance ({expected_gap:.1f})")
    ax.set_xlabel("Numéro")
    ax.set_ylabel("Écart moyen (tirages)")
    ax.set_title("Écart moyen entre deux apparitions consécutives de chaque numéro")
    ax.set_xticks(nums)
    ax.tick_params(axis="x", labelsize=7)
    ax.legend()
    plt.tight_layout()
    p = out / "avg_gap.png"
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"  → {p}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args      = sys.argv[1:]
    refresh   = "--refresh" in args
    no_plots  = "--no-plots" in args
    data_file = None
    if "--data-file" in args:
        idx = args.index("--data-file")
        if idx + 1 < len(args):
            data_file = Path(args[idx + 1])

    ensure_data(CACHE_CSV, refresh=refresh, data_file=data_file)

    df = load_results()
    if df.empty:
        print("Aucune donnée. Vérifiez data/euromillions_results.csv")
        sys.exit(1)

    date_range = (
        f"{df['date'].min().date()} → {df['date'].max().date()}"
        if "date" in df.columns else "date inconnue"
    )
    print(f"\n{len(df)} tirages chargés  [{date_range}]")

    # ── Statistiques ──────────────────────────────────────────────────────
    stats = compute_stats(df)
    STATS_DIR.mkdir(parents=True, exist_ok=True)
    (STATS_DIR / "stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False)
    )
    print(f"Statistiques → {STATS_DIR}/stats.json")

    S = "=" * 56

    # Résumé par période
    print(f"\n{S}")
    print("  RÉPARTITION PAR PÉRIODE RÉGLEMENTAIRE")
    print(S)
    for ps in stats["periods"]:
        draws  = ps["draws"]
        ms     = ps["max_star"]
        chi2b  = f"χ²={ps['ball_chi2']:.1f} p={ps['ball_chi2_pvalue']:.3f}" if ps["ball_chi2"] else ""
        chi2s  = f"χ²={ps['star_chi2']:.1f} p={ps['star_chi2_pvalue']:.3f}" if ps["star_chi2"] else ""
        print(f"\n  {ps['label']}")
        print(f"    {draws} tirages  |  étoiles 1–{ms}  |  1 chance sur {ps['jackpot_odds']:,}")
        print(f"    Boules : esp.={ps['ball_expected']:.0f}/numéro  {chi2b}")
        print(f"    Étoiles: esp.={ps['star_expected']:.0f}/étoile  {chi2s}")

    # Chi-carré global boules
    if stats["ball_chi2_global"] is not None:
        print(f"\n  Test d'uniformité global (boules 1–50) :")
        print(f"    χ² = {stats['ball_chi2_global']:.2f}  |  p = {stats['ball_chi2_pvalue_global']:.4f}")
        sig = "pas de biais significatif détecté" if stats["ball_chi2_pvalue_global"] > 0.05 \
              else "BIAIS SIGNIFICATIF (p < 0.05)"
        print(f"    → {sig}  (ddl=49, α=0.05)")

    # Top 10 boules
    print(f"\n{S}")
    print("  TOP 10 BOULES (toutes périodes confondues)")
    print(S)
    for num, cnt in sorted(stats["top_10_balls"].items(), key=lambda x: -x[1]):
        exp  = stats["ball_expected_per_number"]
        dev  = (cnt - exp) / exp * 100
        bar  = "█" * int(abs(dev) / 0.5)
        print(f"  {int(num):>2}  {cnt:>5} fois  ({dev:+.1f}%)  {bar}")

    # Étoiles normalisées
    print(f"\n{S}")
    print("  ÉTOILES — fréquence normalisée (obs/espérance corrigée par période)")
    print(S)
    sf    = {int(k): int(v) for k, v in stats["star_frequency"].items()}
    sr    = {int(k): float(v) for k, v in stats["star_normalized_ratio"].items()}
    se    = _star_normalized_expected(df)
    for n in range(1, 13):
        obs  = sf.get(n, 0)
        exp  = se.get(n, 0.0)
        rat  = sr.get(n, 0.0)
        avail = "P1-P2-P3" if n <= 9 else ("P2-P3" if n <= 11 else "P3")
        flag = " ★" if rat >= 1.05 else " ▼" if rat <= 0.95 else ""
        print(f"  étoile {n:>2}  {obs:>5} obs  esp.={exp:>6.0f}  ratio={rat:.3f}{flag}  [{avail}]")

    # Top paires
    print(f"\n{S}")
    print("  TOP 10 PAIRES DE BOULES")
    print(S)
    for pair_str, cnt in list(stats["top_10_pairs"].items())[:10]:
        print(f"  {pair_str:<16}  {cnt:>4} fois")

    # ── Graphiques ────────────────────────────────────────────────────────
    if not no_plots:
        print(f"\nGénération des graphiques → {STATS_DIR}/")
        plot_ball_frequency(df, STATS_DIR)
        plot_ball_by_period(df, STATS_DIR)
        plot_star_by_period(df, STATS_DIR)
        plot_star_normalized(df, STATS_DIR, stats)
        plot_heatmap(df, STATS_DIR)
        plot_draws_per_year(df, STATS_DIR)
        plot_gap_analysis(df, STATS_DIR, stats)

    print(f"\nTerminé. Résultats dans {STATS_DIR}/")


if __name__ == "__main__":
    main()
