"""Stage 1b: EHR -> physiology params -> simulated RAW device streams (data/<profile>/raw_wearables/<pid>.parquet)."""
import sys, time, numpy as np, pandas as pd
from multiprocessing import Pool
from importlib import resources
from config import *
from features import ehr_features
from wearables.derive_params import derive_params
from wearables.simulate import simulate_patient

def load_tables(profile):
    d = data_dir(profile) / "ehr"
    return {k: pd.read_csv(d / f"{k}.csv") for k in ("patients", "labs", "conditions", "medications", "genomics")}

def _templates():
    import simglucose
    p = resources.files("simglucose") / "params" / "vpatient_params.csv"
    df = pd.read_csv(str(p)); q = pd.read_csv(str(resources.files("simglucose") / "params" / "Quest.csv"))
    df = df.merge(q[["Name", "CR"]], on="Name"); return df[df.Name.str.startswith("adult")].reset_index(drop=True)

def _one(args):
    profile, pid, days = args
    t = load_tables(profile); pat = t["patients"].set_index("patient_id", drop=False).loc[pid]
    f = ehr_features(pat, *(t[k][t[k].patient_id == pid] for k in ("labs", "conditions", "medications", "genomics")))
    seed = int(pid.replace("-", "")[:8], 16)
    p = derive_params(pat, f, seed)
    tm = _templates(); row = tm.iloc[seed % len(tm)]
    # one-parameter calibration: shift the controller set-point so mean glucose matches the HbA1c-implied eAG
    target = 28.7 * f["hba1c_latest"] - 46.7
    for _ in range(3):
        m = simulate_patient(pid, p, 2, seed + 1, row, calib=True)
        if abs(m - target) < 4: break
        p["gref"] = float(np.clip(p["gref"] * (target / m) ** 1.2, 55, 320))
    p["calibrated_gref"] = p["gref"]
    raw, truth = simulate_patient(pid, p, days, seed, row)
    out = data_dir(profile) / "raw_wearables"; out.mkdir(exist_ok=True)
    raw.to_parquet(out / f"{pid}.parquet", index=False)
    tr = data_dir(profile) / "truth"; tr.mkdir(exist_ok=True); truth.assign(patient_id=pid).to_parquet(tr / f"{pid}.parquet", index=False)
    return pid, p

def run(profile="sample", workers=8, limit=None):
    days = PROFILES[profile]["days"]; t = load_tables(profile)
    ids = list(t["patients"].patient_id)[:limit]; t0 = time.time()
    with Pool(workers) as pool:
        res = []
        for i, r in enumerate(pool.imap_unordered(_one, [(profile, i, days) for i in ids])):
            res.append(r)
            if (i + 1) % 10 == 0: print(f"[wearables] {i+1}/{len(ids)}  {time.time()-t0:.0f}s", flush=True)
    pd.DataFrame([{**p, "patient_id": pid} for pid, p in res]).to_csv(data_dir(profile) / "sim_params.csv", index=False)
    print(f"[wearables] {profile}: {len(ids)} patients in {time.time()-t0:.0f}s")

if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "sample", limit=int(sys.argv[2]) if len(sys.argv) > 2 else None)
