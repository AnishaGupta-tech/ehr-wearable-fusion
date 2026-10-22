"""End-to-end pipeline.

  python run_pipeline.py --profile sample            # ~50 patients, 7 days (minutes)
  python run_pipeline.py --profile full              # ~800 patients, 14 days (about an hour)
  python run_pipeline.py --profile sample --stages ingest,models

Stages:  synthea -> ehr -> wearables -> ingest -> models
  synthea    Synthea (Apache-2.0) population -> data/synthea_raw/csv   (skipped if already present; needs Java 17 + jar)
  ehr        multi-table EHR + SIMULATED genomics           -> data/<profile>/ehr/*.csv
  wearables  EHR-conditioned RAW device streams             -> data/<profile>/raw_wearables/<pid>.parquet
  ingest     Raw Device Streams -> Ingestion & Alignment -> Fused Patient Timeline
                                                            -> data/<profile>/timeseries.parquet (+ fused_timeline.parquet)
  models     3-model ablation vs persistence                -> data/<profile>/results/
"""
import argparse, os, subprocess, time
from pathlib import Path
from config import ROOT, SYNTHEA_CSV

def stage_synthea():
    if (SYNTHEA_CSV / "patients.csv").exists(): print("[synthea] already generated, skipping"); return
    jar = ROOT / "synthea" / "synthea-with-dependencies.jar"; jar.parent.mkdir(exist_ok=True)
    if not jar.exists():
        subprocess.check_call(["curl", "-sL", "-o", str(jar), "https://github.com/synthetichealth/synthea/releases/download/master-branch-latest/synthea-with-dependencies.jar"])
    subprocess.check_call(["java", "-jar", str(jar), "-p", "3000", "-s", "42", "-a", "35-80", "--exporter.csv.export=true",
                           "--exporter.fhir.export=false", f"--exporter.baseDirectory={ROOT / 'data' / 'synthea_raw'}"], cwd=ROOT / "synthea")

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--profile", default="sample", choices=["sample", "full"])
    ap.add_argument("--stages", default="synthea,ehr,wearables,ingest,models,events"); ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(); stages = a.stages.split(","); t0 = time.time()
    if "synthea" in stages: stage_synthea()
    if "ehr" in stages:
        from ehr.build_ehr import build; build(a.profile)
    if "wearables" in stages:
        from wearables.generate import run; run(a.profile, workers=a.workers)
    if "ingest" in stages:
        from ingestion.align import build_timeseries; build_timeseries(a.profile)
    if "models" in stages:
        from models.ablation import run as ablate; ablate(a.profile, stride=3 if a.profile == "sample" else 6)
    if "events" in stages:
        from models.events import run as events; events(a.profile, eval_stride=1 if a.profile == "sample" else 2, train_stride=3 if a.profile == "sample" else 6)
    print(f"[pipeline] done in {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
