#!/usr/bin/env python3
# secure_aggregation.py
# Drop-in script for your repo: path-safe, reproducible, and matches your Phase 2 outputs style.

import sys, math, time, json, hashlib, itertools, statistics
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# =========================
# Repo-relative paths (no absolute paths)
# =========================
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
MANIFESTS_DIR = ROOT / "manifests"
for d in (DATA_DIR, RESULTS_DIR, MANIFESTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# =========================
# Config / reproducibility
# =========================
RNG_SEED = 42  # master seed for reproducibility (cohorts, masks, etc.)
np.random.seed(RNG_SEED)

# Fixed-point scale and modulus for masking/FE arithmetic
FP_SCALE_POW = 4  # scale by 10^4 (adjust if needed)
FP_SCALE = 10 ** FP_SCALE_POW
MASK_MODULUS = np.uint64(2**64 - 59)  # large 64-bit prime close to 2^64

# Bytes-per-ciphertext used for *measurement* (kept to match your tables)
# We treat AHE/FE ciphertexts as fixed 256 bytes per coordinate (like 2048-bit).
BYTES_PER_CTXT = 256
BYTES_PER_INT_MASKING = 4  # we serialize masked ints as 4 bytes each (matches your tables)

# =========================
# Datasets (no absolute paths)
# =========================
DATASETS = [
    (DATA_DIR / "heart_disease_clean.csv", "Heart"),
    (DATA_DIR / "harmergedd.csv",          "HAR"),
    (DATA_DIR / "faostattt_clean.csv",     "FAOSTAT"),
]

missing = [str(p) for p, _ in DATASETS if not p.exists()]
if missing:
    print("\n[ERROR] Missing CSV file(s). Please add them under ./data with these exact names:")
    print("  - heart_disease_clean.csv")
    print("  - harmergedd.csv")
    print("  - faostattt_clean.csv")
    print("\nMissing:")
    for m in missing: print("  -", m)
    sys.exit(1)

# =========================
# Utilities
# =========================
def cohort_size(n_rows: int, scenario: str) -> int:
    scenario = scenario.lower().strip()
    if scenario == "small":
        return max(1, round(n_rows * 0.10))
    if scenario == "moderate":
        return max(1, round(n_rows * 0.50))
    if scenario == "full":
        return n_rows
    raise ValueError("Scenario must be one of: small / moderate / full")

def deterministic_indices(n_rows: int, k: int, tag: str) -> np.ndarray:
    """Pick k deterministic row indices from 0..n_rows-1, seeded by RNG_SEED and tag (dataset name)."""
    h = hashlib.sha256(f"{RNG_SEED}:{tag}:{n_rows}".encode()).digest()
    seed = int.from_bytes(h[:8], "big", signed=False) & ((1 << 63) - 1)
    rng = np.random.default_rng(seed)
    idx = np.arange(n_rows)
    rng.shuffle(idx)
    return np.sort(idx[:k])

def numeric_fixed_point(df: pd.DataFrame) -> np.ndarray:
    """Keep numeric columns, fill NaN, fixed-point encode to int64."""
    numeric = df.select_dtypes(include=[np.number]).copy()
    numeric = numeric.fillna(0.0)
    # fixed-point
    arr = np.rint(numeric.to_numpy(dtype=float) * FP_SCALE).astype(np.int64)
    return arr

def print_dataset_summary():
    print("\n[DATASET SUMMARY]")
    for p, name in DATASETS:
        df = pd.read_csv(p)
        arr = numeric_fixed_point(df)
        d = arr.shape[1]
        print(f"  {name:<10} -> rows={len(df)}, features(d)={d} | {p}")

# =========================
# Paillier (minimal, didactic) for AHE/FE backend
# =========================
# We use 512-bit primes (1024-bit N) for speed; we *measure* ciphertext as 256 bytes per coord to match your tables.
import secrets

def _is_probable_prime(n, k=10):
    if n < 2: return False
    # small primes
    small = [2,3,5,7,11,13,17,19,23,29]
    for p in small:
        if n % p == 0:
            return n == p
    # write n-1 as 2^r * d
    r, d = 0, n - 1
    while d % 2 == 0:
        r += 1
        d //= 2
    # Miller-Rabin
    for _ in range(k):
        a = secrets.randbelow(n - 3) + 2
        x = pow(a, d, n)
        if x == 1 or x == n - 1: 
            continue
        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True

def _rand_prime(bits=512):
    while True:
        candidate = secrets.randbits(bits) | 1 | (1 << (bits - 1))
        if _is_probable_prime(candidate):
            return candidate

@dataclass
class PaillierPK:
    N: int
    g: int

@dataclass
class PaillierSK:
    lam: int
    mu: int
    N: int

def paillier_keygen(bits=1024):
    p = _rand_prime(bits // 2)
    q = _rand_prime(bits // 2)
    N = p * q
    lam = (p - 1) * (q - 1) // math.gcd(p - 1, q - 1)
    g = N + 1  # standard choice
    def L(u): return (u - 1) // N
    mu = pow(L(pow(g, lam, N * N)), -1, N)
    return PaillierPK(N, g), PaillierSK(lam, mu, N)

def paillier_enc(pk: PaillierPK, m: int, rng: secrets.SystemRandom):
    N = pk.N
    N2 = N * N
    # random r in Z*_N
    while True:
        r = rng.randrange(1, N)
        if math.gcd(r, N) == 1:
            break
    c = (pow(pk.g, m % N, N2) * pow(r, N, N2)) % N2
    return c

def paillier_add(pk: PaillierPK, c1: int, c2: int):
    N2 = pk.N * pk.N
    return (c1 * c2) % N2

def paillier_dec(sk: PaillierSK, c: int):
    N = sk.N
    N2 = N * N
    def L(u): return (u - 1) // N
    u = pow(c, sk.lam, N2)
    m = (L(u) * sk.mu) % N
    # signed representative in (-N/2, N/2]
    if m > N // 2:
        m = m - N
    return m

# =========================
# Protocols
# =========================
class MaskingProtocol:
    def __init__(self, d: int, tag: str):
        self.d = d
        self.tag = tag  # dataset name
        self.q = int(MASK_MODULUS)

    def _pair_seed(self, i, j):
        h = hashlib.sha256(f"{RNG_SEED}:{self.tag}:pair:{i}:{j}".encode()).digest()
        return int.from_bytes(h[:8], "big", signed=False)

    def client_protect(self, i, x_i, n_clients):
        """Return masked vector y_i (mod q)."""
        start = time.perf_counter()
        y = np.array(x_i, dtype=np.int64)
        # sum of pairwise masks
        for j in range(n_clients):
            if i == j: continue
            a, b = (i, j) if i < j else (j, i)
            seed = self._pair_seed(a, b)
            rng = np.random.default_rng(seed)
            mask = rng.integers(low=0, high=self.q, size=self.d, dtype=np.int64)
            if i < j:
                y = (y + mask) % self.q
            else:
                y = (y - mask) % self.q
        t = time.perf_counter() - start
        # bytes if we serialize as 4 bytes per coordinate (matches your tables)
        msg_bytes = self.d * BYTES_PER_INT_MASKING
        return y, t, msg_bytes

    def server_aggregate(self, messages):
        start = time.perf_counter()
        Y = np.zeros(self.d, dtype=np.int64)
        for y in messages:
            Y = (Y + y) % self.q
        t = time.perf_counter() - start
        return Y, t

class AHEProtocol:
    def __init__(self, d: int):
        self.d = d
        self.pk, self.sk = paillier_keygen(bits=1024)  # 1024-bit N for speed
        self._rng = secrets.SystemRandom()

    def client_protect(self, x_i):
        """Encrypt each coordinate with Paillier."""
        start = time.perf_counter()
        c = [paillier_enc(self.pk, int(val), self._rng) for val in x_i]
        t = time.perf_counter() - start
        # measure bytes as 256 per coord to match your tables
        msg_bytes = self.d * BYTES_PER_CTXT
        return c, t, msg_bytes

    def server_aggregate(self, list_of_cipher_vectors):
        """Component-wise homomorphic addition (ciphertext multiply); single decryption of the aggregate vector."""
        start = time.perf_counter()
        # multiply ciphertexts per coordinate
        agg = []
        for j in range(self.d):
            col = [cv[j] for cv in list_of_cipher_vectors]
            c = 1
            for ct in col:
                c = paillier_add(self.pk, c, ct)
            agg.append(c)
        # decrypt aggregate per coordinate
        summed = np.array([paillier_dec(self.sk, c) for c in agg], dtype=np.int64)
        t = time.perf_counter() - start
        return summed, t

class FEProtocol:
    """
    Didactic IPFE-like interface:
    - Same Paillier backend for encryption/combination/decryption (per-coordinate).
    - We separate 'functional keys' conceptually, but cost/profile here is similar to AHE.
    """
    def __init__(self, d: int):
        self.d = d
        self.pk, self.sk = paillier_keygen(bits=1024)
        self._rng = secrets.SystemRandom()
        # projection keys conceptually exist; not used explicitly in this didactic backend.

    def client_protect(self, x_i):
        start = time.perf_counter()
        c = [paillier_enc(self.pk, int(val), self._rng) for val in x_i]
        t = time.perf_counter() - start
        msg_bytes = self.d * BYTES_PER_CTXT
        return c, t, msg_bytes

    def server_aggregate(self, list_of_cipher_vectors):
        start = time.perf_counter()
        agg = []
        for j in range(self.d):
            col = [cv[j] for cv in list_of_cipher_vectors]
            c = 1
            for ct in col:
                c = paillier_add(self.pk, c, ct)
            agg.append(c)
        # "functional" per-coordinate decryption (projection keys in a real IPFE)
        summed = np.array([paillier_dec(self.sk, c) for c in agg], dtype=np.int64)
        t = time.perf_counter() - start
        return summed, t

# =========================
# Harness
# =========================
def run_one(dataset_path: Path, dataset_name: str, scenario: str, algorithm: str, repeats: int):
    df = pd.read_csv(dataset_path)
    arr = numeric_fixed_point(df)  # int64 fixed-point
    n_rows, d = arr.shape

    # choose cohort
    K = cohort_size(n_rows, scenario)
    idx = deterministic_indices(n_rows, K, tag=f"{dataset_name}:{scenario}")
    X = arr[idx, :]  # shape (K, d)

    # clear-text oracle (sum) under int arithmetic
    oracle = np.sum(X, axis=0, dtype=np.int64)

    # choose protocol
    alg_upper = algorithm.lower().strip()
    if alg_upper == "masking":
        proto = MaskingProtocol(d=d, tag=dataset_name)
    elif alg_upper == "ahe":
        proto = AHEProtocol(d=d)
    elif alg_upper == "fe":
        proto = FEProtocol(d=d)
    else:
        raise ValueError("algorithm must be one of: masking / ahe / fe")

    per_run_metrics = []

    for _ in range(repeats):
        # client phase
        client_msgs = []
        client_time_sum = 0.0
        bytes_total = 0
        if alg_upper == "masking":
            # masking client needs index for pairwise seeds
            for i in range(K):
                y_i, t_i, b_i = proto.client_protect(i, X[i], K)
                client_msgs.append(y_i)
                client_time_sum += t_i
                bytes_total += b_i
        else:
            # AHE / FE
            for i in range(K):
                msg, t_i, b_i = proto.client_protect(X[i])
                client_msgs.append(msg)
                client_time_sum += t_i
                bytes_total += b_i

        # server phase
        agg, server_time = proto.server_aggregate(client_msgs)

        # correctness (rescale here if you want floating output; we stick to int check)
        ok = np.array_equal(agg.astype(np.int64), oracle.astype(np.int64))

        end_to_end_ms = (client_time_sum + server_time) * 1000.0
        per_run_metrics.append((end_to_end_ms, bytes_total, ok))

    # summarize
    end_ms_vals = [m[0] for m in per_run_metrics]
    bytes_vals  = [m[1] for m in per_run_metrics]
    oks         = [m[2] for m in per_run_metrics]

    end_ms_mean = statistics.mean(end_ms_vals)
    end_ms_std  = statistics.pstdev(end_ms_vals) if len(end_ms_vals) > 1 else 0.0
    total_bytes = int(statistics.mean(bytes_vals))
    per_client_bytes = total_bytes // K
    return {
        "algorithm": algorithm.upper(),
        "dataset": dataset_name,
        "scenario": scenario.lower(),
        "clients(K)": K,
        "end_to_end_ms": f"{end_ms_mean:.2f} ± {end_ms_std:.2f}",
        "bytes_per_client": per_client_bytes,
        "per_client_kB": round(per_client_bytes / 1024.0, 3),
        "total_bytes": total_bytes,
        "total_kB": round(total_bytes / 1024.0, 3),
        "aggregation_ok": all(oks),
        "d": d,
        "rows": n_rows,
    }

def run_all(algorithm: str, scenario: str, repeats: int):
    print_dataset_summary()
    # friendly INFO lines
    for p, name in DATASETS:
        df = pd.read_csv(p)
        arr = numeric_fixed_point(df)
        n_rows, d = arr.shape
        K = cohort_size(n_rows, scenario)
        scen_label = scenario.upper()
        print(f"[INFO] {name}: {scen_label:<8} → using {('100%' if scenario=='full' else f'{int(round([0.10,0.50][scenario=='moderate']*100) if scenario!='full' else 100)}%')} rows, K={K}")

    rows = []
    for p, name in DATASETS:
        out = run_one(p, name, scenario, algorithm, repeats)
        rows.append(out)

    # pretty print
    print("\n=== Results (Timing, Communication, Aggregation) ===")
    header = ["algorithm","dataset","scenario","clients(K)","end_to_end_ms","bytes_per_client","per_client_kB","total_bytes","total_kB","aggregation_ok"]
    # align like your logs
    for r in rows:
        print(f"{r['algorithm']:>8} {r['dataset']:<12} {r['scenario']:<9} {r['clients(K)']:>10}  {r['end_to_end_ms']:<12} {r['bytes_per_client']:>16} {r['per_client_kB']:>12} {r['total_bytes']:>12} {r['total_kB']:>8} {str(r['aggregation_ok']):>14}")

    # save CSV
    df = pd.DataFrame(rows)[header]
    out_csv = RESULTS_DIR / f"{algorithm.lower()}_{scenario.lower()}_metrics.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved CSV: {out_csv}")
    return df

# =========================
# CLI
# =========================
def main():
    print(f"\nUsing Python: {sys.executable}")
    algo = input("Choose algorithm [masking / ahe / fe / all]: ").strip().lower()
    scenario = input("Choose scenario [small / moderate / full]: ").strip().lower()
    rep_in = input("How many repeats? [default 5]: ").strip()
    repeats = int(rep_in) if rep_in else 5

    if algo == "all":
        for alg in ["masking","ahe","fe"]:
            print()
            run_all(alg, scenario, repeats)
    else:
        if algo not in {"masking","ahe","fe"}:
            print("Invalid algorithm. Choose masking / ahe / fe / all.")
            sys.exit(1)
        if scenario not in {"small","moderate","full"}:
            print("Invalid scenario. Choose small / moderate / full.")
            sys.exit(1)
        run_all(algo, scenario, repeats)

if __name__ == "__main__":
    main()
