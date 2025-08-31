#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Phase 2 — Interactive runner (Phase-1 methodology, pairwise masking only)
- You choose:
    • algorithm  : masking / ahe / fe / all
    • scenario   : small / moderate / full
    • repeats    : # trials (default 5)
- Scenario behavior (automatic K):
    small    -> use 10% of rows, clients(K) = number of rows in that 10%
    moderate -> use 50% of rows, clients(K) = number of rows in that 50%
    full     -> use 100% of rows, clients(K) = number of rows (all)
"""

import sys, os, time, hashlib
import numpy as np
import pandas as pd

# ---------- Datasets ----------
DATASETS = [
    ("/Users/sabinanto/Downloads/heart_disease_clean.csv", "HeartDisease"),
    ("/Users/sabinanto/Downloads/harmergedd.csv",          "HAR"),
    ("/Users/sabinanto/Downloads/faostattt_clean.csv",     "FAOSTATT"),
]

# Fractions per scenario
SCENARIO_FRACS = {"small": 0.10, "moderate": 0.50, "full": 1.00}

# Defaults
DEFAULT_REPEATS = 5
MASKING_DTYPE_BYTES = 4
AHE_FE_KEY_BITS = 2048
AGG_EPS = 1e-8

# ---------- Utilities ----------
def load_numeric_df(path: str, dropna=True) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    num = df.select_dtypes(include=[np.number]).copy()
    if dropna:
        num = num.dropna(axis=0, how="all")
    return num

def subset(df: pd.DataFrame, frac: float, rs: int) -> pd.DataFrame:
    return df if frac >= 1.0 else df.sample(frac=frac, random_state=rs)

def split_indices(n: int, k: int):
    sizes = [n // k + (1 if r < n % k else 0) for r in range(k)]
    out, start = [], 0
    for sz in sizes:
        out.append(np.arange(start, start + sz))
        start += sz
    return out

def pair_seed(ns: str, i: int, j: int) -> int:
    h = hashlib.sha256(f"{ns}|{i}|{j}".encode()).digest()
    return int.from_bytes(h[:8], "big") % (2**32 - 1)

def mean_std(arr):
    if not arr: return 0.0, 0.0
    if len(arr) == 1: return float(arr[0]), 0.0
    return float(np.mean(arr)), float(np.std(arr, ddof=1))

def fmt_ms(mean, std): return f"{mean:.2f} ± {std:.2f}"
def kb(x): return round(x / 1024.0, 3)

# ---------- Masking ----------
def client_local_sums(X: np.ndarray, idxs):
    return [X[idx, :].sum(axis=0) if len(idx) > 0 else np.zeros(X.shape[1]) for idx in idxs]

def apply_pairwise_masks(update_vecs, namespace: str):
    K = len(update_vecs)
    d = update_vecs[0].shape[0] if K else 0
    masks = [np.zeros(d, dtype=np.float64) for _ in range(K)]
    t0 = time.perf_counter()
    for i in range(K):
        for j in range(i+1, K):
            r = np.random.default_rng(pair_seed(namespace, i, j)).standard_normal(d)
            masks[i] += r
            masks[j] -= r
    masked = [update_vecs[i] + masks[i] for i in range(K)]
    return masked, (time.perf_counter() - t0) * 1000.0

def run_masking_round(df: pd.DataFrame, K: int, ns: str, dtype_bytes: int = 4):
    X = df.to_numpy(float, copy=False)
    n, d = X.shape
    if n == 0 or d == 0:
        return {"n": n, "d": d, "K": 0, "e2e": 0.0, "bpc": 0, "tot": 0, "ok": True}

    K_eff = min(K, n)
    idxs = split_indices(n, K_eff)

    t0 = time.perf_counter()
    sums = client_local_sums(X, idxs)
    masked, _ = apply_pairwise_masks(sums, ns)
    agg_sum = np.sum(masked, axis=0)
    e2e = (time.perf_counter() - t0) * 1000.0

    plain_mean = X.mean(axis=0)
    agg_mean = agg_sum / n
    ok = bool(np.allclose(agg_mean, plain_mean, atol=AGG_EPS))

    bpc = d * dtype_bytes
    tot = bpc * K_eff
    return {"n": n, "d": d, "K": K_eff, "e2e": e2e, "bpc": int(bpc), "tot": int(tot), "ok": ok}

# ---------- AHE / FE (simulated) ----------
def _simulate_encrypt_vector(vec: np.ndarray, key_bits: int, times=1):
    scaled = (vec * 1e6).astype(np.int64, copy=False)
    mod = (1 << key_bits) - 159
    base = 65537
    t0 = time.perf_counter()
    for x in scaled:
        for _ in range(times):
            _ = pow((x % mod) + 2, base, mod)
    return (time.perf_counter() - t0) * 1000.0

def _simulate_decrypt_vector(vec: np.ndarray, key_bits: int, times=1):
    mod = (1 << key_bits) - 211
    lam = 131071
    t0 = time.perf_counter()
    for _ in vec:
        for _ in range(times):
            _ = pow(3, lam, mod)
    return (time.perf_counter() - t0) * 1000.0

def run_ahe_round(df: pd.DataFrame, K: int, ns: str, key_bits: int = 2048):
    X = df.to_numpy(float, copy=False)
    n, d = X.shape
    if n == 0 or d == 0:
        return {"n": n, "d": d, "K": 0, "e2e": 0.0, "bpc": 0, "tot": 0, "ok": True}

    K_eff = min(K, n)
    idxs = split_indices(n, K_eff)
    sums = client_local_sums(X, idxs)

    t_enc = sum(_simulate_encrypt_vector(s, key_bits, times=1) for s in sums)
    agg_sum = np.sum(sums, axis=0)
    t_dec = _simulate_decrypt_vector(agg_sum, key_bits, times=1)

    plain_mean = X.mean(axis=0)
    agg_mean = agg_sum / n
    ok = bool(np.allclose(agg_mean, plain_mean, atol=AGG_EPS))

    e2e = t_enc + t_dec
    ct_bytes = key_bits // 8
    bpc = d * ct_bytes
    tot = bpc * K_eff
    return {"n": n, "d": d, "K": K_eff, "e2e": e2e, "bpc": int(bpc), "tot": int(tot), "ok": ok}

def run_fe_round(df: pd.DataFrame, K: int, ns: str, key_bits: int = 2048):
    X = df.to_numpy(float, copy=False)
    n, d = X.shape
    if n == 0 or d == 0:
        return {"n": n, "d": d, "K": 0, "e2e": 0.0, "bpc": 0, "tot": 0, "ok": True}

    K_eff = min(K, n)
    idxs = split_indices(n, K_eff)
    sums = client_local_sums(X, idxs)

    t_enc = sum(_simulate_encrypt_vector(s, key_bits, times=2) for s in sums)
    agg_sum = np.sum(sums, axis=0)
    t_dec = _simulate_decrypt_vector(agg_sum, key_bits, times=1)

    plain_mean = X.mean(axis=0)
    agg_mean = agg_sum / n
    ok = bool(np.allclose(agg_mean, plain_mean, atol=AGG_EPS))

    e2e = t_enc + t_dec
    ct_bytes = key_bits // 8
    bpc = d * ct_bytes
    tot = bpc * K_eff
    return {"n": n, "d": d, "K": K_eff, "e2e": e2e, "bpc": int(bpc), "tot": int(tot), "ok": ok}

# ---------- Interactive prompts ----------
def ask_algorithm():
    while True:
        s = input("Choose algorithm [masking / ahe / fe / all]: ").strip().lower()
        if s in {"masking","ahe","fe","all"}:
            return s
        print("Please type: masking, ahe, fe, or all.")

def ask_scenario():
    while True:
        s = input("Choose scenario [small / moderate / full]: ").strip().lower()
        if s in {"small","moderate","full"}:
            return s
        print("Please type: small, moderate, or full.")

def ask_repeats(default=DEFAULT_REPEATS):
    s = input(f"How many repeats? [default {default}]: ").strip()
    if not s:
        return default
    try:
        r = int(s)
        return max(1, r)
    except ValueError:
        print("Invalid, using default.")
        return default

# ---------- Main ----------
def main():
    print("Using Python:", sys.executable)
    algo_choice = ask_algorithm()
    scenario = ask_scenario()
    repeats = ask_repeats()
    frac = SCENARIO_FRACS[scenario]
    algos = ["masking", "ahe", "fe"] if algo_choice == "all" else [algo_choice]

    print("\n[DATASET SUMMARY]")
    loaded = []
    for path, name in DATASETS:
        try:
            df_num = load_numeric_df(path, dropna=True)
            print(f"  {name:<10} -> rows={df_num.shape[0]}, features(d)={df_num.shape[1]} | {path}")
            loaded.append((df_num, name))
        except Exception as e:
            print(f"  {name:<10} -> [ERROR] {e}")
    if not loaded:
        print("No datasets loaded. Check paths.")
        sys.exit(1)

    results = []
    for algo in algos:
        for df_all, name in loaded:
            # Subset rows per scenario
            df_use = subset(df_all, frac, rs=123)
            n_use = df_use.shape[0]
            K = n_use  # <-- number of rows = clients

            print(f"[INFO] {name}: scenario={scenario} ({int(frac*100)}% rows -> {n_use}), K={K}")

            e2es = []
            ok_all = True
            bpc = tot = 0
            K_eff = 0

            for r in range(repeats):
                sub = subset(df_all, frac, 42 + r) if frac < 1.0 else df_all
                n_rep = sub.shape[0]
                K_rep = n_rep  # always rows = clients
                ns = f"{name}:{scenario}:{r}"

                if algo == "masking":
                    res = run_masking_round(sub, K_rep, ns, MASKING_DTYPE_BYTES)
                elif algo == "ahe":
                    res = run_ahe_round(sub, K_rep, ns, AHE_FE_KEY_BITS)
                else:
                    res = run_fe_round(sub, K_rep, ns, AHE_FE_KEY_BITS)

                e2es.append(res["e2e"])
                ok_all = ok_all and bool(res["ok"])
                bpc, tot = res["bpc"], res["tot"]
                K_eff = res["K"]

            m, s = mean_std(e2es)
            results.append({
                "algorithm": algo.upper(),
                "dataset": name,
                "scenario": scenario,
                "clients(K)": K_eff,
                "end_to_end_ms": fmt_ms(m, s),
                "bytes_per_client": bpc,
                "per_client_kB": kb(bpc),
                "total_bytes": tot,
                "total_kB": kb(tot),
                "aggregation_ok": ok_all
            })

    df = pd.DataFrame(results, columns=[
        "algorithm","dataset","scenario","clients(K)",
        "end_to_end_ms","bytes_per_client","per_client_kB","total_bytes","total_kB",
        "aggregation_ok"
    ])

    # Sort results
    algo_order = {"MASKING": 0, "AHE": 1, "FE": 2}
    ds_order = {name:i for i,(_,name) in enumerate(DATASETS)}
    df["__a"] = df["algorithm"].map(algo_order).fillna(99)
    df["__d"] = df["dataset"].map(ds_order)
    df = df.sort_values(["__a","__d"]).drop(columns=["__a","__d"])

    print("\n=== Results (Timing, Communication, Aggregation) ===")
    print(df.to_string(index=False))

    csv_algo = "all" if algo_choice == "all" else algos[0]
    out_csv = f"phase2_{csv_algo}_{scenario}_rep{repeats}.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved CSV: {os.path.abspath(out_csv)}")

if __name__ == "__main__":
    main()
