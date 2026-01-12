#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
from dataclasses import dataclass
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


@dataclass
class SeriesData:
    name: str
    df_all: pd.DataFrame
    df_ok: pd.DataFrame
    duration_col: str
    status_col: Optional[str]
    ts_col: Optional[str]


def _find_col(df: pd.DataFrame, key: str) -> Optional[str]:
    key = key.lower()
    for c in df.columns:
        if key in c.lower():
            return c
    return None


def load_csv(path: str, name: str, only_2xx: bool) -> SeriesData:
    df = pd.read_csv(path)
    df = df.rename(columns={c: c.strip() for c in df.columns})

    duration_col = _find_col(df, "duration")
    status_col = _find_col(df, "status")
    ts_col = _find_col(df, "timestamp")

    if duration_col is None:
        raise ValueError(f"[{name}] Keine 'duration'-Spalte gefunden. Spalten: {list(df.columns)}")

    df[duration_col] = pd.to_numeric(df[duration_col], errors="coerce")
    df_all = df[df[duration_col].notna()].copy()

    if only_2xx and status_col is not None:
        df_ok = df_all[df_all[status_col].between(200, 299, inclusive="both")].copy()
    else:
        df_ok = df_all.copy()

    if ts_col is not None:
        df_all[ts_col] = pd.to_numeric(df_all[ts_col], errors="coerce")
        df_ok[ts_col] = pd.to_numeric(df_ok[ts_col], errors="coerce")

    return SeriesData(name, df_all, df_ok, duration_col, status_col, ts_col)


def ensure_outdir(outdir: str) -> None:
    os.makedirs(outdir, exist_ok=True)


def savefig(path: str) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def durations(series: SeriesData) -> np.ndarray:
    return series.df_ok[series.duration_col].to_numpy()


# ---------- Plot 1: cumulative ----------
def plot_cumulative(series_list: List[SeriesData], outdir: str, prefix: str) -> None:
    plt.figure(figsize=(10, 6))
    for s in series_list:
        d = durations(s)
        y_s = np.cumsum(d) / 1000.0
        x = np.arange(1, len(y_s) + 1)
        plt.plot(x, y_s, label=s.name)

    plt.xlabel("Anzahl der Einträge (kumulativ)")
    plt.ylabel("Aufsummierte Duration (s)")
    plt.title("Kumulative Duration vs. Anzahl der Einträge")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.legend()
    savefig(os.path.join(outdir, f"{prefix}01_kumulativ.png"))


# ---------- ECDF helpers ----------
def ecdf_xy(d: np.ndarray):
    d = np.sort(d)
    y = np.arange(1, len(d) + 1) / len(d)
    return d, y


def compute_tail_metrics(series_list: List[SeriesData], p_zoom: float) -> pd.DataFrame:
    rows = []
    for s in series_list:
        total = len(s.df_all)
        ok = len(s.df_ok)
        err_rate = (1 - ok / total) if total else np.nan

        d = durations(s)
        if len(d) == 0:
            rows.append({
                "Comparator": s.name,
                "n_total": total,
                "n_used": ok,
                "err_rate": err_rate,
                "p50": np.nan, "p90": np.nan, "p95": np.nan, "p99": np.nan, "p999": np.nan,
                f"p{p_zoom:g}": np.nan,
                "max": np.nan,
                "count_gt_p99": np.nan,
            })
            continue

        p50, p90, p95, p99, p999 = np.percentile(d, [50, 90, 95, 99, 99.9])
        pz = np.percentile(d, p_zoom)
        mx = float(np.max(d))
        count_gt_p99 = int(np.sum(d > p99))

        rows.append({
            "Comparator": s.name,
            "n_total": total,
            "n_used": ok,
            "err_rate": err_rate,
            "p50": float(p50),
            "p90": float(p90),
            "p95": float(p95),
            "p99": float(p99),
            "p999": float(p999),
            f"p{p_zoom:g}": float(pz),
            "max": mx,
            "count_gt_p99": count_gt_p99,
        })

    return pd.DataFrame(rows)


def plot_ecdf_zoom(series_list: List[SeriesData], outdir: str, prefix: str, p_zoom: float) -> float:
    # global xmax = max(P_zoom) across series
    pvals = []
    for s in series_list:
        d = durations(s)
        if len(d) > 0:
            pvals.append(np.percentile(d, p_zoom))
    xmax = float(max(pvals)) if pvals else 1.0

    plt.figure(figsize=(10, 6))
    for s in series_list:
        d = durations(s)
        if len(d) == 0:
            continue
        x, y = ecdf_xy(d)
        plt.plot(x, y, label=s.name)

    plt.xlim(0, xmax)
    plt.xlabel("Duration (ms)")
    plt.ylabel("Anteil ≤ x")
    plt.title(f"ECDF der Duration (Zoom bis P{p_zoom:g})")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.legend()
    savefig(os.path.join(outdir, f"{prefix}02_ecdf_zoom.png"))

    return xmax


def plot_tail_ccdf(series_list: List[SeriesData], outdir: str, prefix: str, x_from: float) -> None:
    """
    CCDF = 1 - ECDF, zeigt direkt wieviel Anteil > x ist (Tail).
    x_from z.B. globales P99 aus allen Serien oder der ECDF-Zoom-Grenzwert.
    """
    plt.figure(figsize=(10, 6))
    for s in series_list:
        d = durations(s)
        if len(d) == 0:
            continue
        x, y = ecdf_xy(d)
        ccdf = 1.0 - y

        # nur Tail-Bereich plotten
        mask = x >= x_from
        if np.any(mask):
            plt.plot(x[mask], ccdf[mask], label=s.name)

    plt.xlabel("Duration (ms)")
    plt.ylabel("Anteil > x  (CCDF)")
    plt.title(f"Tail-Plot: Anteil langsamer als x (ab {x_from:.2f} ms)")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.legend()
    savefig(os.path.join(outdir, f"{prefix}02b_tail_ccdf.png"))


# ---------- Plot 3: percentiles ----------
def plot_percentiles(series_list: List[SeriesData], outdir: str, prefix: str,
                     percentiles=(50, 90, 95, 99)) -> None:
    rows = []
    for s in series_list:
        d = durations(s)
        vals = np.percentile(d, percentiles) if len(d) else [np.nan] * len(percentiles)
        rows.append({"Comparator": s.name, **{f"P{p}": float(v) for p, v in zip(percentiles, vals)}})

    dfp = pd.DataFrame(rows)

    plt.figure(figsize=(10, 6))
    x = np.arange(len(dfp))
    for p in percentiles:
        plt.plot(x, dfp[f"P{p}"], marker="o", label=f"P{p}")

    plt.xticks(x, dfp["Comparator"])
    plt.xlabel("Comparator")
    plt.ylabel("Duration (ms)")
    plt.title("Perzentile der Duration (P50/P90/P95/P99)")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.legend()
    savefig(os.path.join(outdir, f"{prefix}03_perzentile.png"))

    dfp.to_csv(os.path.join(outdir, f"{prefix}perzentile.csv"), index=False)


# ---------- Plot 4: time series ----------
def rolling_quantile(arr: np.ndarray, window: int, q: float) -> np.ndarray:
    s = pd.Series(arr)
    return s.rolling(window=window, min_periods=max(10, window // 10)).quantile(q).to_numpy()


def plot_duration_trend_binned(series_list, outdir, prefix,
                               bin_size=500,
                               show_raw=False,
                               raw_downsample=200,
                               y_limit_mode="p99",   # "p99" oder "none"
                               y_margin=0.10):       # 10% Luft nach oben
    """
    Pro Comparator ein Subplot:
      - Median (P50), P95, P99 pro Bin
      - Band: Median..P95
      - KEINE Spike-Schwelle / KEINE Spike-Bins
      - y-Limit so, dass P99 vollständig sichtbar ist (optional)
    """

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import os

    n = len(series_list)
    fig, axes = plt.subplots(n, 1, figsize=(14, 4*n), sharex=True)
    if n == 1:
        axes = [axes]

    # Wir bestimmen (optional) ein gemeinsames y-Maximum basierend auf dem maximalen P99 über alle Serien/Bins
    global_p99_max = None
    if y_limit_mode == "p99":
        p99_max_list = []
        for s in series_list:
            df = s.df_ok.copy()
            if len(df) == 0:
                continue
            if "#" in df.columns:
                df = df.sort_values("#")

            d = df[s.duration_col].to_numpy()
            bin_id = (np.arange(len(d)) // bin_size)
            g = pd.DataFrame({"d": d, "bin": bin_id}).groupby("bin")["d"]
            p99 = g.quantile(0.99).to_numpy()
            if len(p99):
                p99_max_list.append(np.nanmax(p99))

        if p99_max_list:
            global_p99_max = float(np.nanmax(p99_max_list)) * (1.0 + y_margin)

    for ax, s in zip(axes, series_list):
        df = s.df_ok.copy()
        if len(df) == 0:
            ax.set_title(f"{s.name} (keine Daten)")
            continue

        if "#" in df.columns:
            df = df.sort_values("#")

        d = df[s.duration_col].to_numpy()
        x = np.arange(1, len(d) + 1)

        # optional: Rohpunkte (stark ausgedünnt)
        if show_raw:
            idx = np.arange(0, len(d), raw_downsample)
            ax.plot(x[idx], d[idx], linestyle="None", marker=".", markersize=2, alpha=0.12)

        # Binning
        bin_id = (np.arange(len(d)) // bin_size)
        tmp = pd.DataFrame({"x": x, "d": d, "bin": bin_id})
        g = tmp.groupby("bin")["d"]

        med = g.median().to_numpy()
        p95 = g.quantile(0.95).to_numpy()
        p99 = g.quantile(0.99).to_numpy()

        bin_index = g.count().index.to_numpy()
        x_bin = (bin_index * bin_size) + (bin_size / 2)

        # Plot: Band + Linien
        ax.fill_between(x_bin, med, p95, alpha=0.2, label="Band: Median..P95")
        ax.plot(x_bin, med, linewidth=2, label="Median (P50)")
        ax.plot(x_bin, p95, linewidth=2, label="P95")
        ax.plot(x_bin, p99, linewidth=2, label="P99")

        ax.set_title(f"{s.name} – Trend (Bin={bin_size} Requests)")
        ax.set_ylabel("Duration (ms)")
        ax.grid(True, linestyle="--", linewidth=0.5)
        ax.legend()

        # y-Limit so setzen, dass P99 voll sichtbar ist
        if y_limit_mode == "p99" and global_p99_max is not None:
            ax.set_ylim(0, global_p99_max)
        # sonst: keine Begrenzung (zeigt ggf. Ausreißer und macht alles kleiner)

    axes[-1].set_xlabel("Request #")
    plt.tight_layout()

    out_path = os.path.join(outdir, f"{prefix}04_trend_binned_p99.png")
    plt.savefig(out_path, dpi=200)
    plt.close()





def main():
    ap = argparse.ArgumentParser(
        description="Visualisierungen (kumulativ, ECDF-zoom, optional Tail-CCDF, Perzentile, Zeitverlauf) aus Comparator-CSV-Logs."
    )
    ap.add_argument("--damerau", default="./results/DamerauLevenshteinComparator_result.csv")
    ap.add_argument("--jaro", default="./results/JaroWinklerComparator_result.csv")
    ap.add_argument("--qgram", default="./results/q-gram_result.csv")

    ap.add_argument("--outdir", default="plots")
    ap.add_argument("--prefix", default="")

    ap.add_argument("--include-non2xx", action="store_true",
                    help="Wenn gesetzt: Non-2xx werden NICHT herausgefiltert (falls Status-Spalte existiert).")

    ap.add_argument("--ecdf-zoom-p", type=float, default=99.0,
                    help="ECDF Zoom-Grenze als Perzentil (z.B. 99, 99.5, 99.9).")

    ap.add_argument("--tail-ccdf", action="store_true",
                    help="Erzeugt zusätzlich einen Tail-Plot (CCDF) ab der ECDF-Zoom-Grenze.")

    ap.add_argument("--window", type=int, default=1000,
                    help="Rolling window size (Anzahl Requests) für Zeitverlauf-Median/P95.")
    ap.add_argument("--downsample", type=int, default=5,
                    help="Downsampling-Faktor für Scatter im Zeitverlauf (1 = kein Downsampling).")

    args = ap.parse_args()
    only_2xx = not args.include_non2xx

    ensure_outdir(args.outdir)

    series_list = [
        load_csv(args.damerau, "Damerau-Levenshtein", only_2xx=only_2xx),
        load_csv(args.jaro, "Jaro-Winkler", only_2xx=only_2xx),
        load_csv(args.qgram, "q-gram", only_2xx=only_2xx),
    ]

    # Plots
    plot_cumulative(series_list, args.outdir, args.prefix)
    xmax_zoom = plot_ecdf_zoom(series_list, args.outdir, args.prefix, p_zoom=args.ecdf_zoom_p)
    plot_percentiles(series_list, args.outdir, args.prefix)
    plot_duration_trend_binned(series_list, args.outdir, args.prefix)

    # Tail-Metriken speichern
    tail_df = compute_tail_metrics(series_list, p_zoom=args.ecdf_zoom_p)
    tail_csv = os.path.join(args.outdir, f"{args.prefix}tail_metrics.csv")
    tail_df.to_csv(tail_csv, index=False)

    # Optional Tail-Plot (CCDF)
    if args.tail_ccdf:
        plot_tail_ccdf(series_list, args.outdir, args.prefix, x_from=xmax_zoom)

    print(f"Fertig. Output in: {os.path.abspath(args.outdir)}")
    print(f"- ECDF Zoom: {os.path.join(args.outdir, f'{args.prefix}02_ecdf_zoom.png')}")
    if args.tail_ccdf:
        print(f"- Tail CCDF: {os.path.join(args.outdir, f'{args.prefix}02b_tail_ccdf.png')}")
    print(f"- Tail Metrics CSV: {tail_csv}")


if __name__ == "__main__":
    main()
