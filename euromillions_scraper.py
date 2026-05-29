#!/usr/bin/env python3
"""
EuroMillions – téléchargement des résultats et statistiques.

Sources supportées (par ordre de priorité) :
  1. UK National Lottery XML API  (tirages récents, ~52 disponibles)
  2. FDJ / data.gouv.fr CSV       (historique complet, utilisé si disponible)
  3. Fichier CSV local            (import manuel via --data-file)

Usage :
  python euromillions_scraper.py                    # télécharge depuis les sources dispo
  python euromillions_scraper.py --refresh          # force le re-téléchargement
  python euromillions_scraper.py --data-file my.csv # utilise un fichier local
  python euromillions_scraper.py --no-plots         # stats texte uniquement

Format CSV local attendu :
  date,ball_1,ball_2,ball_3,ball_4,ball_5,star_1,star_2
  2024-01-12,3,21,34,42,49,5,8
  (séparateur virgule ou point-virgule, date ISO ou DD/MM/YYYY)
"""

import csv
import json
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import StringIO
from pathlib import Path
from xml.etree import ElementTree as ET

import matplotlib.pyplot as plt
import pandas as pd
import requests
import seaborn as sns

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_DIR   = Path("data")
STATS_DIR  = Path("stats")
CACHE_CSV  = DATA_DIR / "euromillions_results.csv"

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; euromillions-stats/1.0)"}

UK_XML_LATEST  = "https://www.national-lottery.co.uk/results/euromillions/draw-history/xml"
UK_XML_BY_NUM  = "https://www.national-lottery.co.uk/results/euromillions/draw-history/{n}/xml"

# FDJ open data (data.gouv.fr) – indisponible si le serveur est en 503
FDJ_DATAGOUV_API = "https://www.data.gouv.fr/api/1/datasets/?q=euromillions+fdj&page_size=5"


# ---------------------------------------------------------------------------
# UK National Lottery – source XML
# ---------------------------------------------------------------------------

def _parse_uk_xml(xml_text: str) -> dict | None:
    """Parse a single draw from UK National Lottery XML. Returns None on failure."""
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
        "draw_number": draw_el.findtext("draw-number"),
        "date": draw_el.findtext("draw-date"),
        "ball_1": int(balls[0]),
        "ball_2": int(balls[1]),
        "ball_3": int(balls[2]),
        "ball_4": int(balls[3]),
        "ball_5": int(balls[4]),
        "star_1": int(stars[0]) if len(stars) > 0 else None,
        "star_2": int(stars[1]) if len(stars) > 1 else None,
    }


def _fetch_uk_draw(n: int, session: requests.Session) -> dict | None:
    """Fetch one draw by number from UK National Lottery XML API."""
    try:
        resp = session.get(
            UK_XML_BY_NUM.format(n=n),
            timeout=10,
            headers=HEADERS,
        )
        if resp.status_code != 200:
            return None
        return _parse_uk_xml(resp.text)
    except Exception:
        return None


def fetch_from_uk_lottery(dest: Path = CACHE_CSV) -> Path:
    """
    Download all available draws from UK National Lottery XML API.
    Uses parallel requests; respects the server with a small concurrency cap.
    """
    print("Source : UK National Lottery XML API")

    # 1. Get latest draw number
    with requests.Session() as sess:
        resp = sess.get(UK_XML_LATEST, timeout=10, headers=HEADERS)
        resp.raise_for_status()
        latest = _parse_uk_xml(resp.text)

    if latest is None:
        raise RuntimeError("Impossible de lire le dernier tirage UK")

    latest_num = int(latest["draw_number"])
    print(f"  Dernier tirage : #{latest_num} ({latest['date']})")

    # 2. Find oldest available draw by binary search
    def is_available(n: int, sess: requests.Session) -> bool:
        r = _fetch_uk_draw(n, sess)
        return r is not None

    with requests.Session() as sess:
        lo, hi = max(1, latest_num - 500), latest_num
        # Quick scan backwards in steps of 50
        lo = latest_num
        for step in [1, 10, 25, 50, 100]:
            candidate = latest_num - step
            if candidate >= 1 and is_available(candidate, sess):
                lo = candidate
        # Refine with binary search around lo
        lo_hard = max(1, lo - 100)
        hi_hard = lo
        while lo_hard < hi_hard:
            mid = (lo_hard + hi_hard) // 2
            if is_available(mid, sess):
                hi_hard = mid
            else:
                lo_hard = mid + 1
        first_num = lo_hard

    total = latest_num - first_num + 1
    print(f"  Premier tirage disponible : #{first_num}  →  {total} tirages à télécharger")

    # 3. Fetch all draws in parallel
    rows: list[dict] = []
    with requests.Session() as sess:
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = {
                pool.submit(_fetch_uk_draw, n, sess): n
                for n in range(first_num, latest_num + 1)
            }
            done = 0
            for fut in as_completed(futures):
                result = fut.result()
                if result:
                    rows.append(result)
                done += 1
                if done % 10 == 0:
                    print(f"  {done}/{total} téléchargés…", end="\r", flush=True)

    print(f"  {len(rows)}/{total} tirages récupérés                    ")

    if not rows:
        raise RuntimeError("Aucun tirage récupéré depuis la source UK")

    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest, index=False)
    print(f"  Sauvegardé : {dest}  ({dest.stat().st_size:,} octets)")
    return dest


# ---------------------------------------------------------------------------
# FDJ / data.gouv.fr – source CSV secondaire
# ---------------------------------------------------------------------------

def fetch_from_fdj(dest: Path = CACHE_CSV) -> Path:
    """
    Télécharge les résultats historiques FDJ depuis data.gouv.fr.
    Nécessite que le serveur data.gouv.fr soit accessible (peut être en 503).
    """
    print("Source : FDJ / data.gouv.fr")
    resp = requests.get(FDJ_DATAGOUV_API, timeout=15, headers=HEADERS)
    resp.raise_for_status()

    csv_url = None
    for ds in resp.json().get("data", []):
        title = ds.get("title", "").lower()
        if "euromillion" not in title:
            continue
        for resource in ds.get("resources", []):
            fmt = resource.get("format", "").upper()
            url = resource.get("url", "")
            if fmt == "CSV" or url.lower().endswith(".csv"):
                csv_url = url
                print(f"  Dataset : {ds['title']}")
                break
        if csv_url:
            break

    if not csv_url:
        raise RuntimeError("Aucun dataset EuroMillions CSV trouvé sur data.gouv.fr")

    print(f"  Téléchargement : {csv_url}")
    resp = requests.get(csv_url, timeout=60, headers=HEADERS)
    resp.raise_for_status()

    # FDJ CSV uses semicolons and French column names
    df = _parse_fdj_csv(resp.text)

    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest, index=False)
    print(f"  Sauvegardé : {dest}  ({dest.stat().st_size:,} octets)")
    return dest


def _parse_fdj_csv(text: str) -> pd.DataFrame:
    """Parse FDJ CSV (semicolon-separated, French column names)."""
    sep = ";" if text.count(";") > text.count(",") else ","
    df = pd.read_csv(StringIO(text), sep=sep, dtype=str)
    df.columns = [c.strip().lower() for c in df.columns]

    rename: dict[str, str] = {}
    for col in df.columns:
        c = col.replace(" ", "_")
        if "date" in c:
            rename[col] = "date"
        elif c in ("boule_1", "ball_1", "num1", "n1"):
            rename[col] = "ball_1"
        elif c in ("boule_2", "ball_2", "num2", "n2"):
            rename[col] = "ball_2"
        elif c in ("boule_3", "ball_3", "num3", "n3"):
            rename[col] = "ball_3"
        elif c in ("boule_4", "ball_4", "num4", "n4"):
            rename[col] = "ball_4"
        elif c in ("boule_5", "ball_5", "num5", "n5"):
            rename[col] = "ball_5"
        elif c in ("etoile_1", "star_1", "lucky_star_1", "ls1"):
            rename[col] = "star_1"
        elif c in ("etoile_2", "star_2", "lucky_star_2", "ls2"):
            rename[col] = "star_2"

    df = df.rename(columns=rename)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
        df = df.dropna(subset=["date"])
        df["date"] = df["date"].dt.strftime("%Y-%m-%d")
        df = df.sort_values("date").reset_index(drop=True)

    return df[
        [c for c in ["date", "ball_1", "ball_2", "ball_3", "ball_4", "ball_5",
                      "star_1", "star_2"] if c in df.columns]
    ]


# ---------------------------------------------------------------------------
# Load from local CSV
# ---------------------------------------------------------------------------

def load_local_csv(path: Path) -> pd.DataFrame:
    """
    Load a manually provided CSV file.
    Supports comma or semicolon separators, French or English column names,
    date in ISO (YYYY-MM-DD) or European (DD/MM/YYYY) format.
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    sep = ";" if text.count(";") > text.count(",") else ","
    df = pd.read_csv(StringIO(text), sep=sep, dtype=str)
    df.columns = [c.strip().lower() for c in df.columns]
    return _parse_fdj_csv(text)


# ---------------------------------------------------------------------------
# Download orchestrator
# ---------------------------------------------------------------------------

def ensure_data(
    dest: Path = CACHE_CSV,
    refresh: bool = False,
    data_file: Path | None = None,
) -> Path:
    if data_file:
        print(f"Utilisation du fichier local : {data_file}")
        df = load_local_csv(data_file)
        dest.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(dest, index=False)
        print(f"  {len(df)} tirages importés → {dest}")
        return dest

    if not refresh and dest.exists() and dest.stat().st_size > 500:
        print(f"Cache trouvé : {dest}  (--refresh pour re-télécharger)")
        return dest

    # Try FDJ first (more complete), fall back to UK National Lottery
    errors = []
    for fetcher, name in [
        (fetch_from_fdj, "FDJ / data.gouv.fr"),
        (fetch_from_uk_lottery, "UK National Lottery"),
    ]:
        try:
            return fetcher(dest)
        except Exception as e:
            print(f"  [{name}] indisponible : {e}")
            errors.append(f"{name}: {e}")

    raise RuntimeError(
        "Aucune source disponible.\n"
        + "\n".join(errors)
        + "\n\nAstuce : fournissez un fichier CSV local avec --data-file"
    )


# ---------------------------------------------------------------------------
# Load & validate
# ---------------------------------------------------------------------------

def load_results(path: Path = CACHE_CSV) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    df.columns = [c.strip().lower() for c in df.columns]

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    for col in ["ball_1", "ball_2", "ball_3", "ball_4", "ball_5", "star_1", "star_2"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def compute_stats(df: pd.DataFrame) -> dict:
    ball_cols = [c for c in df.columns if c.startswith("ball_")]
    star_cols = [c for c in df.columns if c.startswith("star_")]

    if not ball_cols:
        raise ValueError("Aucune colonne 'ball_*' trouvée dans les données.")

    all_balls = (
        pd.concat([df[c] for c in ball_cols], ignore_index=True).dropna().astype(int)
    )
    all_stars = (
        pd.concat([df[c] for c in star_cols], ignore_index=True).dropna().astype(int)
        if star_cols else pd.Series([], dtype=int)
    )

    ball_freq = Counter(all_balls.tolist())
    star_freq  = Counter(all_stars.tolist()) if len(all_stars) else Counter()

    total = len(df)

    # Pair frequency
    pair_counter: Counter = Counter()
    for _, row in df[ball_cols].iterrows():
        nums = sorted(int(v) for v in row.dropna())
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                pair_counter[(nums[i], nums[j])] += 1

    # Gap analysis: how many draws between consecutive appearances of each number
    avg_gap: dict[int, float] = {}
    for num in range(1, 51):
        appearances = df.index[
            df[ball_cols].apply(
                lambda r: num in r.dropna().astype(int).values, axis=1
            )
        ].tolist()
        if len(appearances) >= 2:
            gaps = [b - a for a, b in zip(appearances, appearances[1:])]
            avg_gap[num] = sum(gaps) / len(gaps)

    return {
        "total_draws": total,
        "date_range": {
            "first": str(df["date"].min().date()) if "date" in df.columns else "N/A",
            "last":  str(df["date"].max().date()) if "date" in df.columns else "N/A",
        },
        "ball_frequency":    {str(k): v for k, v in sorted(ball_freq.items())},
        "star_frequency":    {str(k): v for k, v in sorted(star_freq.items())},
        "top_10_balls":      dict(ball_freq.most_common(10)),
        "bottom_10_balls":   dict(ball_freq.most_common()[:-11:-1]),
        "top_5_stars":       dict(star_freq.most_common(5)) if star_freq else {},
        "top_10_pairs":      {str(k): v for k, v in pair_counter.most_common(10)},
        "avg_gap_per_number": {str(k): round(v, 1) for k, v in avg_gap.items()},
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _ball_freq_counts(df: pd.DataFrame) -> tuple[list[int], list[int]]:
    ball_cols = [c for c in df.columns if c.startswith("ball_")]
    all_balls = (
        pd.concat([df[c] for c in ball_cols], ignore_index=True).dropna().astype(int)
    )
    freq = Counter(all_balls.tolist())
    numbers = list(range(1, 51))
    counts  = [freq.get(n, 0) for n in numbers]
    return numbers, counts


def plot_ball_frequency(df: pd.DataFrame, out_dir: Path) -> None:
    numbers, counts = _ball_freq_counts(df)
    sorted_nc = sorted(zip(numbers, counts), key=lambda x: x[1], reverse=True)
    top5 = {n for n, _ in sorted_nc[:5]}
    bot5 = {n for n, _ in sorted_nc[-5:]}

    fig, ax = plt.subplots(figsize=(16, 5))
    bars = ax.bar(numbers, counts, color="royalblue", edgecolor="white", linewidth=0.4)
    for bar, n in zip(bars, numbers):
        if n in top5:
            bar.set_color("seagreen")
        elif n in bot5:
            bar.set_color("tomato")

    total = len(df)
    ax.axhline(total * 5 / 50, color="grey", linestyle="--", linewidth=0.8,
               label=f"Espérance ({total*5//50} fois)")
    ax.set_xlabel("Numéro")
    ax.set_ylabel("Nombre de tirages")
    ax.set_title(
        f"Fréquence des numéros EuroMillions (1–50)  –  {total} tirages\n"
        "■ vert = top 5  |  ■ rouge = bottom 5"
    )
    ax.set_xticks(numbers)
    ax.tick_params(axis="x", labelsize=7)
    ax.legend()
    plt.tight_layout()
    path = out_dir / "ball_frequency.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  → {path}")


def plot_star_frequency(df: pd.DataFrame, out_dir: Path) -> None:
    star_cols = [c for c in df.columns if c.startswith("star_")]
    if not star_cols:
        return
    all_stars = (
        pd.concat([df[c] for c in star_cols], ignore_index=True).dropna().astype(int)
    )
    freq = Counter(all_stars.tolist())
    numbers = list(range(1, 13))
    counts  = [freq.get(n, 0) for n in numbers]

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(numbers, counts, color="gold", edgecolor="white", linewidth=0.4)
    top3 = {n for n, _ in sorted(zip(numbers, counts), key=lambda x: x[1], reverse=True)[:3]}
    for bar, n in zip(bars, numbers):
        if n in top3:
            bar.set_color("darkorange")

    total = len(df)
    ax.axhline(total * 2 / 12, color="grey", linestyle="--", linewidth=0.8,
               label=f"Espérance ({total*2//12} fois)")
    ax.set_xlabel("Étoile chanceuse")
    ax.set_ylabel("Nombre de tirages")
    ax.set_title(f"Fréquence des étoiles chanceuses (1–12)  –  {total} tirages")
    ax.set_xticks(numbers)
    ax.legend()
    plt.tight_layout()
    path = out_dir / "star_frequency.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  → {path}")


def plot_heatmap(df: pd.DataFrame, out_dir: Path) -> None:
    numbers, counts = _ball_freq_counts(df)
    grid   = [[counts[r * 10 + c] for c in range(10)] for r in range(5)]
    annots = [[str(r * 10 + c + 1) for c in range(10)] for r in range(5)]

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.heatmap(
        grid, annot=annots, fmt="s", cmap="YlOrRd",
        linewidths=0.5, ax=ax,
        cbar_kws={"label": "Fréquence"},
        xticklabels=False, yticklabels=False,
    )
    ax.set_title(f"Carte de chaleur – numéros EuroMillions  ({len(df)} tirages)")
    plt.tight_layout()
    path = out_dir / "heatmap.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  → {path}")


def plot_draws_per_year(df: pd.DataFrame, out_dir: Path) -> None:
    if "date" not in df.columns:
        return
    by_year = df.groupby(df["date"].dt.year).size()
    if by_year.empty:
        return

    fig, ax = plt.subplots(figsize=(max(8, len(by_year) * 0.7), 4))
    ax.bar(by_year.index, by_year.values, color="steelblue", edgecolor="white")
    ax.set_xlabel("Année")
    ax.set_ylabel("Nombre de tirages")
    ax.set_title("Tirages EuroMillions par année")
    ax.set_xticks(by_year.index)
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    path = out_dir / "draws_per_year.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  → {path}")


def plot_gap_analysis(df: pd.DataFrame, out_dir: Path, stats: dict) -> None:
    """Bar chart: average gap (in draws) between appearances of each number."""
    gaps = {int(k): v for k, v in stats.get("avg_gap_per_number", {}).items()}
    if not gaps:
        return

    numbers = sorted(gaps.keys())
    values  = [gaps[n] for n in numbers]

    fig, ax = plt.subplots(figsize=(16, 4))
    ax.bar(numbers, values, color="mediumpurple", edgecolor="white", linewidth=0.4)
    ax.set_xlabel("Numéro")
    ax.set_ylabel("Écart moyen (en tirages)")
    ax.set_title("Écart moyen entre apparitions de chaque numéro")
    ax.set_xticks(numbers)
    ax.tick_params(axis="x", labelsize=7)
    plt.tight_layout()
    path = out_dir / "avg_gap.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = sys.argv[1:]

    refresh   = "--refresh" in args
    no_plots  = "--no-plots" in args
    data_file = None
    if "--data-file" in args:
        idx = args.index("--data-file")
        if idx + 1 < len(args):
            data_file = Path(args[idx + 1])

    # --- Téléchargement ---
    ensure_data(CACHE_CSV, refresh=refresh, data_file=data_file)

    # --- Chargement ---
    df = load_results()
    if df.empty:
        print("Aucune donnée chargée. Vérifiez le fichier data/euromillions_results.csv")
        sys.exit(1)

    date_range = (
        f"{df['date'].min().date()} → {df['date'].max().date()}"
        if "date" in df.columns
        else "date inconnue"
    )
    print(f"\n{len(df)} tirages chargés  [{date_range}]")

    if len(df) < 50:
        print(
            f"\n  ⚠️  Seulement {len(df)} tirages disponibles."
            "  Les statistiques seront peu significatives."
            "\n  Pour un historique complet, importez un CSV via --data-file"
            "\n  ou relancez quand data.gouv.fr (FDJ) sera de nouveau accessible."
        )

    # --- Statistiques ---
    stats = compute_stats(df)
    STATS_DIR.mkdir(parents=True, exist_ok=True)
    stats_path = STATS_DIR / "stats.json"
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    print(f"\nStatistiques exportées : {stats_path}")

    print(f"\n{'='*50}")
    print(f"{'=== TOP 10 numéros les plus tirés ==='}")
    print(f"{'='*50}")
    for num, cnt in sorted(stats["top_10_balls"].items(), key=lambda x: -x[1]):
        pct = cnt / (stats["total_draws"] * 5) * 100
        bar = "█" * int(pct * 3)
        print(f"  {int(num):>2}  {bar:<18}  {cnt:>4} fois  ({pct:.1f} %)")

    if stats["top_5_stars"]:
        print(f"\n{'='*50}")
        print(f"{'=== TOP 5 étoiles chanceuses ==='}")
        print(f"{'='*50}")
        for num, cnt in sorted(stats["top_5_stars"].items(), key=lambda x: -x[1]):
            pct = cnt / (stats["total_draws"] * 2) * 100
            bar = "█" * int(pct * 3)
            print(f"  {int(num):>2}  {bar:<18}  {cnt:>4} fois  ({pct:.1f} %)")

    print(f"\n{'='*50}")
    print("=== TOP 10 paires les plus fréquentes ===")
    print(f"{'='*50}")
    for pair_str, cnt in list(stats["top_10_pairs"].items())[:10]:
        print(f"  {pair_str:<14}  {cnt:>4} fois")

    # --- Graphiques ---
    if not no_plots:
        print("\nGénération des graphiques dans stats/ ...")
        plot_ball_frequency(df, STATS_DIR)
        plot_star_frequency(df, STATS_DIR)
        plot_heatmap(df, STATS_DIR)
        plot_draws_per_year(df, STATS_DIR)
        plot_gap_analysis(df, STATS_DIR, stats)

    print(f"\nTerminé. Résultats dans {STATS_DIR}/")


if __name__ == "__main__":
    main()
