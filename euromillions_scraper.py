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
import pandas as pd
import requests
import seaborn as sns

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
    return df


# ---------------------------------------------------------------------------
# Statistiques
# ---------------------------------------------------------------------------

def compute_stats(df: pd.DataFrame) -> dict:
    ball_cols = [c for c in df.columns if c.startswith("ball_")]
    star_cols = [c for c in df.columns if c.startswith("star_")]

    if not ball_cols:
        raise ValueError("Aucune colonne 'ball_*' trouvée.")

    all_balls = pd.concat([df[c] for c in ball_cols], ignore_index=True).dropna().astype(int)
    all_stars = (
        pd.concat([df[c] for c in star_cols], ignore_index=True).dropna().astype(int)
        if star_cols else pd.Series([], dtype=int)
    )

    ball_freq = Counter(all_balls.tolist())
    star_freq  = Counter(all_stars.tolist()) if len(all_stars) else Counter()
    total      = len(df)

    # Fréquence des paires
    pair_counter: Counter = Counter()
    for _, row in df[ball_cols].iterrows():
        nums = sorted(int(v) for v in row.dropna())
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                pair_counter[(nums[i], nums[j])] += 1

    # Écart moyen entre deux apparitions consécutives
    avg_gap: dict[int, float] = {}
    for num in range(1, 51):
        idx = df.index[
            df[ball_cols].apply(lambda r: num in r.dropna().astype(int).values, axis=1)
        ].tolist()
        if len(idx) >= 2:
            gaps = [b - a for a, b in zip(idx, idx[1:])]
            avg_gap[num] = round(sum(gaps) / len(gaps), 1)

    return {
        "total_draws": total,
        "date_range": {
            "first": str(df["date"].min().date()) if "date" in df.columns else "N/A",
            "last":  str(df["date"].max().date()) if "date" in df.columns else "N/A",
        },
        "ball_frequency":     {str(k): v for k, v in sorted(ball_freq.items())},
        "star_frequency":     {str(k): v for k, v in sorted(star_freq.items())},
        "top_10_balls":       dict(ball_freq.most_common(10)),
        "bottom_10_balls":    dict(ball_freq.most_common()[:-11:-1]),
        "top_5_stars":        dict(star_freq.most_common(5)) if star_freq else {},
        "top_10_pairs":       {str(k): v for k, v in pair_counter.most_common(10)},
        "avg_gap_per_number": {str(k): v for k, v in avg_gap.items()},
    }


# ---------------------------------------------------------------------------
# Graphiques
# ---------------------------------------------------------------------------

def _ball_counts(df: pd.DataFrame) -> tuple[list[int], list[int]]:
    ball_cols = [c for c in df.columns if c.startswith("ball_")]
    freq = Counter(
        pd.concat([df[c] for c in ball_cols], ignore_index=True)
        .dropna().astype(int).tolist()
    )
    nums = list(range(1, 51))
    return nums, [freq.get(n, 0) for n in nums]


def plot_ball_frequency(df: pd.DataFrame, out: Path) -> None:
    nums, counts = _ball_counts(df)
    srt = sorted(zip(nums, counts), key=lambda x: x[1], reverse=True)
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
        f"Fréquence des numéros EuroMillions (1–50)  ·  {len(df)} tirages\n"
        "■ vert = top 5  ·  ■ rouge = bottom 5"
    )
    ax.set_xticks(nums)
    ax.tick_params(axis="x", labelsize=7)
    ax.legend()
    plt.tight_layout()
    p = out / "ball_frequency.png"
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"  → {p}")


def plot_star_frequency(df: pd.DataFrame, out: Path) -> None:
    star_cols = [c for c in df.columns if c.startswith("star_")]
    if not star_cols:
        return
    freq = Counter(
        pd.concat([df[c] for c in star_cols], ignore_index=True)
        .dropna().astype(int).tolist()
    )
    nums = list(range(1, 13))
    counts = [freq.get(n, 0) for n in nums]
    top3 = {n for n, _ in sorted(zip(nums, counts), key=lambda x: x[1], reverse=True)[:3]}

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(nums, counts, color="gold", edgecolor="white", linewidth=0.4)
    for bar, n in zip(bars, nums):
        if n in top3: bar.set_color("darkorange")

    expected = len(df) * 2 / 12
    ax.axhline(expected, color="grey", linestyle="--", linewidth=0.8,
               label=f"Espérance ({expected:.0f})")
    ax.set_xlabel("Étoile chanceuse")
    ax.set_ylabel("Tirages")
    ax.set_title(f"Fréquence des étoiles chanceuses (1–12)  ·  {len(df)} tirages")
    ax.set_xticks(nums)
    ax.legend()
    plt.tight_layout()
    p = out / "star_frequency.png"
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"  → {p}")


def plot_heatmap(df: pd.DataFrame, out: Path) -> None:
    nums, counts = _ball_counts(df)
    grid   = [[counts[r * 10 + c] for c in range(10)] for r in range(5)]
    annots = [[str(r * 10 + c + 1) for c in range(10)] for r in range(5)]

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.heatmap(grid, annot=annots, fmt="s", cmap="YlOrRd",
                linewidths=0.5, ax=ax, cbar_kws={"label": "Fréquence"},
                xticklabels=False, yticklabels=False)
    ax.set_title(f"Carte de chaleur – numéros EuroMillions  ·  {len(df)} tirages")
    plt.tight_layout()
    p = out / "heatmap.png"
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"  → {p}")


def plot_draws_per_year(df: pd.DataFrame, out: Path) -> None:
    if "date" not in df.columns:
        return
    by_year = df.groupby(df["date"].dt.year).size()
    if by_year.empty:
        return
    fig, ax = plt.subplots(figsize=(max(8, len(by_year) * 0.6), 4))
    ax.bar(by_year.index, by_year.values, color="steelblue", edgecolor="white")
    ax.set_xlabel("Année")
    ax.set_ylabel("Tirages")
    ax.set_title("Tirages EuroMillions par année")
    ax.set_xticks(by_year.index)
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    p = out / "draws_per_year.png"
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"  → {p}")


def plot_gap_analysis(df: pd.DataFrame, out: Path, stats: dict) -> None:
    gaps = {int(k): v for k, v in stats.get("avg_gap_per_number", {}).items()}
    if not gaps:
        return
    nums = sorted(gaps)
    vals = [gaps[n] for n in nums]
    fig, ax = plt.subplots(figsize=(16, 4))
    ax.bar(nums, vals, color="mediumpurple", edgecolor="white", linewidth=0.4)
    ax.set_xlabel("Numéro")
    ax.set_ylabel("Écart moyen (tirages)")
    ax.set_title("Écart moyen entre deux apparitions consécutives de chaque numéro")
    ax.set_xticks(nums)
    ax.tick_params(axis="x", labelsize=7)
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

    # Stats
    stats = compute_stats(df)
    STATS_DIR.mkdir(parents=True, exist_ok=True)
    stats_path = STATS_DIR / "stats.json"
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    print(f"\nStatistiques → {stats_path}")

    sep = "=" * 52

    print(f"\n{sep}")
    print("  TOP 10 numéros les plus tirés")
    print(sep)
    for num, cnt in sorted(stats["top_10_balls"].items(), key=lambda x: -x[1]):
        pct = cnt / (stats["total_draws"] * 5) * 100
        bar = "█" * int(pct * 2.5)
        print(f"  {int(num):>2}  {bar:<20}  {cnt:>5} fois  ({pct:.2f} %)")

    if stats["top_5_stars"]:
        print(f"\n{sep}")
        print("  TOP 5 étoiles chanceuses")
        print(sep)
        for num, cnt in sorted(stats["top_5_stars"].items(), key=lambda x: -x[1]):
            pct = cnt / (stats["total_draws"] * 2) * 100
            bar = "█" * int(pct * 2)
            print(f"  {int(num):>2}  {bar:<20}  {cnt:>5} fois  ({pct:.2f} %)")

    print(f"\n{sep}")
    print("  TOP 10 paires les plus fréquentes")
    print(sep)
    for pair_str, cnt in list(stats["top_10_pairs"].items())[:10]:
        print(f"  {pair_str:<16}  {cnt:>4} fois")

    if not no_plots:
        print(f"\nGraphiques → {STATS_DIR}/")
        STATS_DIR.mkdir(parents=True, exist_ok=True)
        plot_ball_frequency(df, STATS_DIR)
        plot_star_frequency(df, STATS_DIR)
        plot_heatmap(df, STATS_DIR)
        plot_draws_per_year(df, STATS_DIR)
        plot_gap_analysis(df, STATS_DIR, stats)

    print(f"\nTerminé.")


if __name__ == "__main__":
    main()
