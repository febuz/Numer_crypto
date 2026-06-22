#!/usr/bin/env python3
"""
Symbol Selection Proof — Issue #17: Deel de keuze voor goede en slechte symbolen

Tests 5 new feature families on top of the 22 Numerai Spectra features,
using CPCV-lite (n_blocks walk-forward folds) with a per-era shuffled control.
Also accepts optional Yiedl metamodel data (--yiedl) as additional signal.

Feature families under test
  1. size_rank     : within-era market-cap percentile rank (proxy for "source rank"
                     / number of platforms tracking the coin — Edwin's 2nd-most-
                     important feature in the WhatsApp analysis)
  2. btc_relative  : each momentum/volume feature minus the BTC value on that date
                     (proxy for Edwin's "price_vs_btc")
  3. momentum_cross: feature_momentum * feature_volume_ewa interaction
                     (Edwin's 3rd most important)
  4. trend_confirm : momentum × (RSI − 0.5) × 2  — positive when both indicators
                     agree on direction
  5. vol_to_cap    : volume_avg / market_cap_avg  — liquidity / interest ratio

Proof gate (NW Newey-West HAC, lag=12):
  A family contributes if:
    • aug_RMSE < shuffled_RMSE  (real > shuffled control in rank-RMSE)
    • |NW_t| ≥ 1.5  (borderline acceptable, same gate as research stage)
  Features with importance > 95th-pct of shuffled importances are also flagged.

Usage:
  python3 scripts/features/symbol_selection_proof.py
  python3 scripts/features/symbol_selection_proof.py --live         # + live ranking
  python3 scripts/features/symbol_selection_proof.py --yiedl        # + Yiedl meta
  python3 scripts/features/symbol_selection_proof.py --live --yiedl

Output: console + reports/symbol_selection_proof_result.json
"""
from __future__ import annotations

import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "12")
os.environ.setdefault("MKL_NUM_THREADS", "12")
os.environ.setdefault("POLARS_MAX_THREADS", "12")

import argparse
import json
import time
import urllib.request
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

try:
    from scipy import stats as scipy_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

EDS2_TRAIN = Path("/media/knight2/EDS2/projects/numerai-signals/data/live_current/crypto_v20_train.parquet")
EDS2_LIVE  = Path("/media/knight2/EDS2/projects/numerai-signals/data/live_current/crypto_v20_live.parquet")

TARGET_20 = "target_binned_return_20"
TARGET_60 = "target_binned_return_60"

SPECTRA = [
    "feature_bollinger_20d", "feature_bollinger_60d",
    "feature_close_avg_20d",  "feature_close_avg_60d",
    "feature_close_ewa_20d",  "feature_close_ewa_60d",
    "feature_market_cap_avg_20d", "feature_market_cap_avg_60d",
    "feature_market_cap_ewa_20d", "feature_market_cap_ewa_60d",
    "feature_momentum_20d",   "feature_momentum_60d",
    "feature_relative_strength_index_20d", "feature_relative_strength_index_60d",
    "feature_sharpe_ratio_20d",   "feature_sharpe_ratio_60d",
    "feature_volatility_20d", "feature_volatility_60d",
    "feature_volume_avg_20d", "feature_volume_avg_60d",
    "feature_volume_ewa_20d", "feature_volume_ewa_60d",
]

LGB_PARAMS = dict(
    objective="regression", num_leaves=31, max_depth=5,
    learning_rate=0.05, feature_fraction=0.5, min_data_in_leaf=200,
    lambda_l1=1.0, lambda_l2=1.0, bagging_fraction=0.7, bagging_freq=1,
    verbose=-1, num_threads=12, seed=42,
)
N_ROUNDS = 250
# Use only last N_TRAIN_DAYS days to keep each fold manageable (3h budget guard)
N_TRAIN_DAYS = 1095  # ~3 years


def _load_parquet(local: Path, eds2: Path, url: str | None = None) -> pl.DataFrame:
    if local.exists():
        return pl.read_parquet(local)
    if eds2.exists():
        local.parent.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy(eds2, local)
        print(f"  [data] copied {local.name} from EDS2")
        return pl.read_parquet(local)
    if url:
        try:
            import numerapi
            local.parent.mkdir(parents=True, exist_ok=True)
            capi = numerapi.CryptoAPI()
            # derive dataset key from url (e.g. crypto/v2.0/train.parquet)
            key = "/".join(url.split("/")[-3:])
            print(f"  [data] downloading {key} ...")
            capi.download_dataset(key, str(local))
            return pl.read_parquet(local)
        except Exception as exc:
            raise RuntimeError(f"Cannot obtain {local.name}: {exc}") from exc
    raise FileNotFoundError(f"{local} not found and no EDS2/URL fallback available")


def add_new_features(df: pl.DataFrame) -> tuple[pl.DataFrame, dict[str, list[str]]]:
    """
    Derive 5 new feature families from the Spectra columns.
    Returns (augmented_df, {family_name: [column_names]}).
    """
    families: dict[str, list[str]] = {}
    cols_present = set(df.columns)

    # --- 1. size_rank: within-era market-cap percentile rank ---
    # Edwin finding: source_rank (listing prominence) was the 2nd most important
    # feature (47% accuracy model). Market-cap rank is the PIT-safe proxy for
    # "how many platforms track this coin".
    sr_cols: list[str] = []
    for raw in ["feature_market_cap_avg_20d", "feature_market_cap_ewa_20d", "feature_market_cap_avg_60d"]:
        if raw in cols_present:
            nc = f"derived_size_rank_{raw[len('feature_market_cap_'):] }"
            df = df.with_columns(
                (pl.col(raw).rank("average").over("date") /
                 pl.col(raw).count().over("date").cast(pl.Float64)).alias(nc)
            )
            sr_cols.append(nc)
    families["size_rank"] = sr_cols

    # --- 2. btc_relative: each feature minus BTC's value on same date ---
    # Proxy for Edwin's "price_vs_btc" — isolates alpha from BTC macro move.
    btc_cols: list[str] = []
    syms = df["symbol"].unique().to_list() if "symbol" in df.columns else []
    if "BTC" in syms:
        btc_pivot = (
            df.filter(pl.col("symbol") == "BTC")
            .select(["date"] + [c for c in SPECTRA if c in cols_present])
            .rename({c: f"_btc_{c}" for c in SPECTRA if c in cols_present})
        )
        df = df.join(btc_pivot, on="date", how="left")
        for raw in ["feature_momentum_20d", "feature_momentum_60d",
                    "feature_volume_ewa_20d", "feature_close_ewa_20d"]:
            if raw in cols_present and f"_btc_{raw}" in df.columns:
                suffix = raw[len("feature_"):]
                nc = f"derived_btc_rel_{suffix}"
                df = df.with_columns((pl.col(raw) - pl.col(f"_btc_{raw}")).alias(nc))
                btc_cols.append(nc)
        df = df.drop([c for c in df.columns if c.startswith("_btc_")])
    families["btc_relative"] = btc_cols

    # --- 3. momentum_cross: momentum × volume interaction ---
    # Edwin's 3rd most important feature: momentum_x_volume.
    mx_cols: list[str] = []
    for t in ["20d", "60d"]:
        mom, vol = f"feature_momentum_{t}", f"feature_volume_ewa_{t}"
        if mom in cols_present and vol in cols_present:
            nc = f"derived_mom_x_vol_{t}"
            df = df.with_columns((pl.col(mom) * pl.col(vol)).alias(nc))
            mx_cols.append(nc)
    families["momentum_cross"] = mx_cols

    # --- 4. trend_confirm: momentum × RSI alignment ---
    # Strong positive when both momentum > 0 and RSI > 0.5 (overbought in uptrend).
    tc_cols: list[str] = []
    for t in ["20d", "60d"]:
        mom, rsi = f"feature_momentum_{t}", f"feature_relative_strength_index_{t}"
        if mom in cols_present and rsi in cols_present:
            nc = f"derived_trend_confirm_{t}"
            df = df.with_columns(
                (pl.col(mom) * (pl.col(rsi) - 0.5) * 2.0).alias(nc)
            )
            tc_cols.append(nc)
    families["trend_confirm"] = tc_cols

    # --- 5. vol_to_cap: volume / market-cap (liquidity / interest proxy) ---
    vc_cols: list[str] = []
    for t in ["20d", "60d"]:
        vol, cap = f"feature_volume_avg_{t}", f"feature_market_cap_avg_{t}"
        if vol in cols_present and cap in cols_present:
            nc = f"derived_vol_to_cap_{t}"
            df = df.with_columns((pl.col(vol) / (pl.col(cap) + 1e-9)).alias(nc))
            vc_cols.append(nc)
    families["vol_to_cap"] = vc_cols

    return df, families


def _era_shuffle(df: pl.DataFrame, cols: list[str], seed: int = 42) -> pl.DataFrame:
    """
    Jointly shuffle `cols` within each era (date) using a random sort key.
    Preserves cross-column covariance within the family while breaking
    the family-target relationship.
    """
    rng = np.random.default_rng(seed)
    shuffle_key = pl.Series("_shuf_key", rng.uniform(size=df.height))
    df = df.with_columns(shuffle_key)
    for c in cols:
        if c in df.columns:
            df = df.with_columns(
                pl.col(c).sort_by("_shuf_key").over("date").alias(c)
            )
    return df.drop("_shuf_key")


def _rank_rmse_per_era(df: pl.DataFrame, target: str, pred: np.ndarray) -> list[float]:
    """Compute per-era rank-RMSE and return as a list."""
    dates = df["date"].to_list()
    t_vals = df[target].to_numpy()
    rmse_list: list[float] = []
    unique_dates = sorted(set(dates))
    for d in unique_dates:
        mask = np.array([x == d for x in dates])
        yt, yp = t_vals[mask], pred[mask]
        if len(yt) < 5:
            continue
        yr = np.argsort(np.argsort(yt)) / len(yt)
        pr = np.argsort(np.argsort(yp)) / len(yp)
        rmse_list.append(float(np.sqrt(np.mean((yr - pr) ** 2))))
    return rmse_list


def _train_predict(X_tr, y_tr, X_te, seed: int = 42) -> np.ndarray:
    p = LGB_PARAMS.copy()
    p["seed"] = seed
    model = lgb.train(p, lgb.Dataset(X_tr, label=y_tr), num_boost_round=N_ROUNDS)
    return model.predict(X_te)


def cpcv_lite(df: pl.DataFrame, feats: list[str], target: str,
              n_blocks: int = 6) -> list[float]:
    """
    Walk-forward CPCV-lite: split dates into n_blocks, hold out each in turn.
    Returns flat list of per-era rank-RMSE values from OOS predictions.
    """
    dates = df["date"].unique().sort().to_list()
    edges = [dates[i * len(dates) // n_blocks] for i in range(n_blocks)] + [dates[-1]]
    all_rmse: list[float] = []
    for k in range(n_blocks):
        lo, hi = edges[k], edges[k + 1]
        is_test = (pl.col("date") >= lo) & (pl.col("date") <= hi)
        tr = df.filter(~is_test).filter(pl.col(target).is_not_null())
        te = df.filter(is_test).filter(pl.col(target).is_not_null())
        if tr.height < 2000 or te.height < 50:
            continue
        X_tr = tr.select(feats).fill_null(0.0).to_numpy().astype(np.float32)
        y_tr = tr[target].to_numpy().astype(np.float32)
        X_te = te.select(feats).fill_null(0.0).to_numpy().astype(np.float32)
        pred = _train_predict(X_tr, y_tr, X_te)
        all_rmse.extend(_rank_rmse_per_era(te, target, pred))
    return all_rmse


def newey_west(deltas: list[float], lag: int = 12) -> tuple[float, float]:
    """Newey-West HAC t-statistic and two-sided p-value."""
    d = np.array(deltas)
    n = len(d)
    if n < lag + 3:
        return 0.0, 1.0
    mu = d.mean()
    gamma0 = np.var(d, ddof=1)
    nw_var = gamma0
    for h in range(1, lag + 1):
        cov_h = float(np.cov(d[h:], d[:-h])[0, 1]) if len(d[h:]) > 1 else 0.0
        nw_var += 2.0 * (1.0 - h / (lag + 1)) * cov_h
    se = np.sqrt(max(nw_var, 1e-14) / n)
    t = mu / se
    if HAS_SCIPY:
        p = float(2 * scipy_stats.t.sf(abs(t), df=n - 1))
    else:
        p = float(np.exp(-0.717 * abs(t) - 0.416 * t ** 2))  # rough approx
    return float(t), p


def importance_vs_random(df: pl.DataFrame, feats: list[str], target: str,
                          n_shuffle: int = 3, seed: int = 42) -> dict[str, tuple[float, float, bool]]:
    """
    Train LGB on real data and on globally-shuffled target (n_shuffle times).
    Returns {feat: (real_gain, random_p95_threshold, is_above_random)}.
    """
    tr = df.filter(pl.col(target).is_not_null())
    X = tr.select(feats).fill_null(0.0).to_numpy().astype(np.float32)
    y = tr[target].to_numpy().astype(np.float32)
    real_model = lgb.train(LGB_PARAMS, lgb.Dataset(X, label=y), num_boost_round=N_ROUNDS)
    real_imp = dict(zip(feats, real_model.feature_importance(importance_type="gain")))

    rng = np.random.default_rng(seed)
    shuf_gains: list[np.ndarray] = []
    for _ in range(n_shuffle):
        ys = rng.permutation(y)
        m = lgb.train(LGB_PARAMS, lgb.Dataset(X, label=ys), num_boost_round=N_ROUNDS)
        shuf_gains.append(m.feature_importance(importance_type="gain"))
    threshold = float(np.percentile(np.concatenate(shuf_gains), 95))

    result: dict[str, tuple[float, float, bool]] = {}
    for f in feats:
        g = real_imp.get(f, 0.0)
        result[f] = (float(g), threshold, g > threshold)
    return result


def load_yiedl(df: pl.DataFrame) -> pl.DataFrame:
    """
    Optionally merge Yiedl macro crypto features if available from the
    EDS2 feature store (yagg_pvm_* columns). Falls back to no-op if absent.
    """
    try:
        yiedl_path = Path("/media/knight2/EDS2/projects/numerai-signals/data/feature_store/live_matrix.parquet")
        if not yiedl_path.exists():
            print("  [yiedl] live_matrix.parquet not found — skipping")
            return df, []
        ym = pl.read_parquet(yiedl_path)
        yiedl_feats = [c for c in ym.columns if c.startswith("yagg_pvm")]
        if not yiedl_feats:
            print("  [yiedl] no yagg_pvm columns — skipping")
            return df, []
        # Only keep date + yiedl features (these are era-level, not symbol-level)
        ym = ym.select(["date"] + yiedl_feats).unique("date")
        df = df.join(ym, on="date", how="left")
        avail = [c for c in yiedl_feats if c in df.columns]
        print(f"  [yiedl] joined {len(avail)} macro features")
        return df, avail
    except Exception as e:
        print(f"  [yiedl] error: {e} — skipping")
        return df, []


def main() -> None:
    ap = argparse.ArgumentParser(description="Symbol selection proof — Issue #17")
    ap.add_argument("--live",      action="store_true", help="classify live universe")
    ap.add_argument("--yiedl",     action="store_true", help="add Yiedl metamodel features")
    ap.add_argument("--n-blocks",  type=int, default=6, help="CPCV walk-forward blocks")
    ap.add_argument("--target",    default=TARGET_20,   help="target column")
    ap.add_argument("--nw-lag",    type=int, default=12, help="Newey-West lag")
    ap.add_argument("--out-json",  default=str(REPORTS / "symbol_selection_proof_result.json"))
    args = ap.parse_args()
    t0 = time.time()

    print("=" * 70)
    print("SYMBOL SELECTION PROOF  —  Issue #17")
    print("Deel de keuze voor goede en slechte symbolen")
    print("=" * 70)

    # ------------------------------------------------------------------ data
    train_local = DATA / "train" / "crypto_v20_train.parquet"
    df = _load_parquet(train_local, EDS2_TRAIN, "crypto/v2.0/train.parquet")
    feats_spectra = [c for c in SPECTRA if c in df.columns]
    # Trim to recent history for speed (3h budget guard: 3y window fits in ~30min)
    max_date = df["date"].max()
    from datetime import timedelta
    cutoff = max_date - timedelta(days=N_TRAIN_DAYS)
    df = df.filter(pl.col("date") >= cutoff)
    print(f"[data] rows={df.height:,} symbols={df['symbol'].n_unique()} "
          f"eras={df['date'].n_unique()} feats={len(feats_spectra)} "
          f"target={args.target}  window={N_TRAIN_DAYS}d")

    # optional Yiedl metamodel
    yiedl_feats: list[str] = []
    if args.yiedl:
        df, yiedl_feats = load_yiedl(df)

    # --------------------------------------------------------- new features
    print("\n[features] building new families ...")
    df, families = add_new_features(df)
    for name, cols in families.items():
        print(f"  {name}: {len(cols)} columns  {cols}")
    if yiedl_feats:
        families["yiedl_meta"] = yiedl_feats

    # ------------------------------------------------------ baseline CPCV
    print(f"\n[baseline] CPCV-lite n_blocks={args.n_blocks} on {len(feats_spectra)} Spectra feats ...")
    base_rmse = cpcv_lite(df, feats_spectra, args.target, n_blocks=args.n_blocks)
    mean_base = float(np.mean(base_rmse))
    print(f"[baseline] mean rank-RMSE={mean_base:.5f}  n_eras={len(base_rmse)}")

    # ------------------------------------------------- per-family proof
    print("\n[proof] testing feature families (augmented vs shuffled control) ...")
    proof_rows: list[dict] = []

    for family_name, family_cols in families.items():
        if not family_cols:
            print(f"  [{family_name}] SKIP (no columns)")
            continue

        feats_aug = feats_spectra + family_cols

        # augmented OOS
        aug_rmse = cpcv_lite(df, feats_aug, args.target, n_blocks=args.n_blocks)
        mean_aug = float(np.mean(aug_rmse))

        # shuffled control: jointly shuffle family within each era
        df_shuf = _era_shuffle(df.clone(), family_cols, seed=42)
        shuf_rmse = cpcv_lite(df_shuf, feats_aug, args.target, n_blocks=args.n_blocks)
        mean_shuf = float(np.mean(shuf_rmse))

        # Real signal = how much better is aug vs shuf?
        # Positive delta → real features outperform shuffled (good)
        n = min(len(aug_rmse), len(shuf_rmse))
        delta_signal = [shuf_rmse[i] - aug_rmse[i] for i in range(n)]
        t_stat, p_val = newey_west(delta_signal, lag=args.nw_lag)
        delta_vs_base = mean_aug - mean_base
        delta_vs_shuf = mean_aug - mean_shuf  # negative = real beats shuffled

        significant = (delta_vs_shuf < 0) and (t_stat >= 1.5)
        sig_str = "YES ✓" if significant else "  no"
        print(f"  [{family_name:<15}] aug={mean_aug:.5f}  shuf={mean_shuf:.5f}  "
              f"Δbase={delta_vs_base:+.5f}  Δshuf={delta_vs_shuf:+.5f}  "
              f"NW_t={t_stat:.2f}  p={p_val:.3f}  signal={sig_str}")

        proof_rows.append({
            "family":         family_name,
            "n_cols":         len(family_cols),
            "cols":           family_cols,
            "baseline_rmse":  mean_base,
            "aug_rmse":       mean_aug,
            "shuffled_rmse":  mean_shuf,
            "delta_vs_base":  delta_vs_base,
            "delta_vs_shuf":  delta_vs_shuf,
            "nw_t":           t_stat,
            "p_val":          p_val,
            "significant":    significant,
        })

    # ------------------------------------------- feature importance report
    sig_cols = [c for r in proof_rows if r["significant"] for c in r["cols"]]
    imp_report: dict = {}
    if sig_cols:
        all_feats_for_imp = feats_spectra + sig_cols
        print(f"\n[importance] evaluating {len(all_feats_for_imp)} features "
              f"(22 Spectra + {len(sig_cols)} proven new) vs random ...")
        imp_result = importance_vs_random(
            df.filter(pl.col(args.target).is_not_null()),
            all_feats_for_imp,
            args.target,
            n_shuffle=3,
        )
        above = {f: v for f, v in imp_result.items() if v[2]}
        threshold = list(imp_result.values())[0][1] if imp_result else 0.0
        base_above = [f for f in above if f in feats_spectra]
        new_above  = [f for f in above if f in sig_cols]
        print(f"[importance] random p95 threshold = {threshold:.0f}")
        print(f"[importance] Spectra above random : {len(base_above)}/{len(feats_spectra)}")
        print(f"[importance] New feats above random: {len(new_above)}/{len(sig_cols)}")
        if new_above:
            print(f"\n  New features with importance > random (top 10):")
            sorted_new = sorted(
                [(f, imp_result[f][0]) for f in new_above], key=lambda x: -x[1]
            )[:10]
            for f, gain in sorted_new:
                print(f"    {f:<45}  gain={gain:>10.0f}")
        imp_report = {
            "threshold_p95": threshold,
            "spectra_above_random": base_above,
            "new_above_random":     new_above,
            "all": {f: {"gain": v[0], "above_random": v[2]} for f, v in imp_result.items()},
        }
    else:
        print("\n[importance] no significant families — skipping importance test")

    # ----------------------------------------------- live classification
    live_output: dict = {}
    if args.live:
        print("\n[live] classifying current round symbols ...")
        live_local = DATA / "live" / "crypto_v20_live.parquet"
        live_df = _load_parquet(live_local, EDS2_LIVE, "crypto/v2.0/live.parquet")
        live_df, _ = add_new_features(live_df)

        # Train on full history with proven features only
        feats_live = feats_spectra + (sig_cols if sig_cols else [])
        tr = df.filter(pl.col(args.target).is_not_null())
        X_tr = tr.select(feats_live).fill_null(0.0).to_numpy().astype(np.float32)
        y_tr = tr[args.target].to_numpy().astype(np.float32)
        lf = [c for c in feats_live if c in live_df.columns]
        X_live = live_df.select(lf).fill_null(0.0).to_numpy().astype(np.float32)

        live_model = lgb.train(LGB_PARAMS, lgb.Dataset(X_tr, label=y_tr),
                               num_boost_round=N_ROUNDS)
        pred = live_model.predict(X_live)
        rank = (np.argsort(np.argsort(pred)) + 0.5) / len(pred)

        live_df = live_df.select(["symbol"]).with_columns(pl.Series("signal", rank))
        live_sorted = live_df.sort("signal", descending=True)

        top_good = live_sorted.head(25)
        top_bad  = live_sorted.tail(25).sort("signal")

        print("\n  GOEDE symbolen (predicted outperformers, top 25):")
        for row in top_good.iter_rows(named=True):
            bar = "+" * int(row["signal"] * 20)
            print(f"    {row['symbol']:<12}  signal={row['signal']:.4f}  {bar}")

        print("\n  SLECHTE symbolen (predicted underperformers, bottom 25):")
        for row in top_bad.iter_rows(named=True):
            bar = "-" * int((1 - row["signal"]) * 20)
            print(f"    {row['symbol']:<12}  signal={row['signal']:.4f}  {bar}")

        live_output = {
            "good_symbols":  top_good.to_dicts(),
            "bad_symbols":   top_bad.to_dicts(),
            "all_symbols":   live_sorted.to_dicts(),
        }

    # ------------------------------------------------------ final summary
    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print("SAMENVATTING / FINAL REPORT")
    print(f"{'='*70}")
    print(f"Baseline rank-RMSE (22 Spectra): {mean_base:.5f}")
    print(f"Target: {args.target}  |  CPCV blocks: {args.n_blocks}  "
          f"|  NW lag: {args.nw_lag}  |  Eras in OOS: {len(base_rmse)}")
    print()
    hdr = f"  {'Family':<18} {'#cols':>5} {'aug_RMSE':>10} {'Δbaseline':>10} {'Δshuffled':>10} {'NW_t':>7} {'p':>7}  signal?"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for r in proof_rows:
        sig = "YES ✓" if r["significant"] else "  no "
        print(f"  {r['family']:<18} {r['n_cols']:>5} "
              f"{r['aug_rmse']:>10.5f} {r['delta_vs_base']:>+10.5f} "
              f"{r['delta_vs_shuf']:>+10.5f} {r['nw_t']:>7.2f} {r['p_val']:>7.3f}  {sig}")

    print(f"\nElapsed: {elapsed:.0f}s")

    # ------------------------------------------------------ save JSON
    summary = {
        "target":                args.target,
        "n_blocks":              args.n_blocks,
        "nw_lag":                args.nw_lag,
        "baseline_rmse":         mean_base,
        "n_eras_oos":            len(base_rmse),
        "proof_results":         proof_rows,
        "importance":            imp_report,
        "live":                  live_output,
        "elapsed_s":             round(elapsed, 1),
    }
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2, default=str))
    print(f"[saved] {out_json}")


if __name__ == "__main__":
    main()
