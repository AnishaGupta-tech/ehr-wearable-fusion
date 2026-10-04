"""Shared configuration. All synthetic; nothing here is real patient data."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).parent
SYNTHEA_CSV = ROOT / "data" / "synthea_raw" / "csv"
ANCHOR = pd.Timestamp("2026-09-30")          # "today" for the synthetic EHR / end of wearable window
LAB_WINDOW_DAYS = 730

PROFILES = {
    # n_patients, wearable days
    "sample": dict(n_patients=50, days=7, mix=(0.40, 0.30, 0.30)),
    "full":   dict(n_patients=800, days=14, mix=(0.38, 0.31, 0.31)),
}

def data_dir(profile: str) -> Path:
    p = ROOT / "data" / profile
    p.mkdir(parents=True, exist_ok=True)
    return p

SIM_LABEL = "SIMULATED"
SNPS = [  # snp, gene, ref/risk alleles, other allele, risk allele freq, odds ratio per allele
    dict(snp_id="rs7903146",  gene="TCF7L2",  risk="T", other="C", freq=0.30, OR=1.40),
    dict(snp_id="rs5219",     gene="KCNJ11",  risk="T", other="C", freq=0.36, OR=1.14),
    dict(snp_id="rs1801282",  gene="PPARG",   risk="C", other="G", freq=0.85, OR=1.14),
    dict(snp_id="rs9939609",  gene="FTO",     risk="A", other="T", freq=0.40, OR=1.20),
    dict(snp_id="rs13266634", gene="SLC30A8", risk="C", other="T", freq=0.65, OR=1.12),
]
