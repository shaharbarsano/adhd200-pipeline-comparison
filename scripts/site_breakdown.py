"""
site_breakdown.py
===========================================================================
Step 4 of 4 (supplementary; can be run anytime after build_dataset.py).
Report the final matched cohort's composition by site (Chapter 5.2, Table 1).

Determines each matched subject's site by independently scanning BOTH
pipelines' raw site folders, then cross-checks that the two agree -- rather
than assuming site membership is pipeline-independent without checking.
Site is a property of the subject, not the pipeline, so full agreement is
expected; the cross-check confirms this rather than assuming it.

Requires output/multisite_data_final.npz (see build_dataset.py) and the
raw data/Athena and data/NIAK folders.
"""

import re
from pathlib import Path
from collections import defaultdict

import numpy as np

DATA_ROOT = Path("data")
OUTPUT_ROOT = Path("output")

ATHENA_ROOT = DATA_ROOT / "Athena"
NIAK_ROOT = DATA_ROOT / "NIAK"
NPZ_PATH = OUTPUT_ROOT / "multisite_data_final.npz"

ATHENA_SITE_DIRS = {
    "KKI": "KKI", "NYU": "NYU", "OHSU": "OHSU",
    "Peking_1": "Beijing", "Peking_2": "Beijing", "Peking_3": "Beijing",
    "Pittsburgh": "Pittsburgh", "WashU": "WashU", "NeuroIMAGE": "NeuroIMAGE",
}
NIAK_SITE_DIRS = {
    "KKI": "KKI", "Beijing": "Beijing", "NYU": "NYU", "OHSU": "OHSU",
    "Pittsburgh": "Pittsburgh", "WashU": "WashU", "NeuroIMAGE": "NeuroIMAGE",
}


def athena_extract_subject_id(fp: Path) -> str:
    name = fp.stem.replace("sfnwmrda", "")
    return name.split("_")[0].zfill(7)


def niak_extract_subject_id(fp: Path) -> str:
    m = re.search(r"tseries_rois_X_(\d+)_run1\.mat", fp.name)
    return m.group(1).zfill(7)


def build_athena_site_lookup():
    lookup = {}
    for folder_name, logical_site in ATHENA_SITE_DIRS.items():
        files = (ATHENA_ROOT / folder_name).glob(
            "*/sfnwmrda*_session_1_rest_1_cc200_TCs.1D")
        for fp in files:
            sid = athena_extract_subject_id(fp)
            lookup[sid] = logical_site
    return lookup


def build_niak_site_lookup():
    lookup = {}
    for folder_name, logical_site in NIAK_SITE_DIRS.items():
        files = (NIAK_ROOT / folder_name / "rois").glob("tseries_rois_X_*_run1.mat")
        for fp in files:
            sid = niak_extract_subject_id(fp)
            lookup[sid] = logical_site
    return lookup


def main():
    athena_sites = build_athena_site_lookup()
    niak_sites = build_niak_site_lookup()
    print(f"Built site lookups: {len(athena_sites)} Athena subjects, "
          f"{len(niak_sites)} NIAK subjects.")

    # Load the final matched cohort
    data = np.load(NPZ_PATH)
    subject_ids = [str(s) for s in data["subject_ids"]]
    y = data["y_athena"]  # original DX values (0/1/2/3), same for both pipelines here
    y_bin = (y != 0).astype(int)  # 0 = control, 1 = ADHD (any subtype)

    print(f"Loaded {len(subject_ids)} matched subjects from {NPZ_PATH}\n")

    # Cross-check: does every matched subject have a site in BOTH lookups,
    # and do the two pipelines agree on what it is?
    disagreements = []
    missing_athena, missing_niak = [], []
    for sid in subject_ids:
        a_site = athena_sites.get(sid)
        n_site = niak_sites.get(sid)
        if a_site is None:
            missing_athena.append(sid)
        if n_site is None:
            missing_niak.append(sid)
        if a_site is not None and n_site is not None and a_site != n_site:
            disagreements.append((sid, a_site, n_site))

    print("=" * 60)
    print("CROSS-VALIDATION")
    print("=" * 60)
    print(f"Matched subjects missing from Athena site lookup: {len(missing_athena)}"
          f" {missing_athena if missing_athena else ''}")
    print(f"Matched subjects missing from NIAK site lookup:   {len(missing_niak)}"
          f" {missing_niak if missing_niak else ''}")
    print(f"Subjects where Athena and NIAK disagree on site:  {len(disagreements)}")
    for sid, a, n in disagreements:
        print(f"  {sid}: Athena says {a}, NIAK says {n}")

    if not disagreements and not missing_athena and not missing_niak:
        print("\nAll matched subjects found in both lookups, with full agreement.")

    # Build the table using NIAK's lookup (validated above to agree with Athena's)
    counts = defaultdict(lambda: {"total": 0, "control": 0, "adhd": 0})
    for sid, dx in zip(subject_ids, y_bin):
        site = niak_sites.get(sid) or athena_sites.get(sid)
        if site is None:
            continue
        counts[site]["total"] += 1
        if dx == 0:
            counts[site]["control"] += 1
        else:
            counts[site]["adhd"] += 1

    print("\n" + "=" * 60)
    print("SITE BREAKDOWN OF FINAL MATCHED COHORT")
    print("=" * 60)
    print(f"{'Site':<14}{'N':>6}{'Control':>10}{'ADHD':>8}")
    print("-" * 38)
    total_n, total_c, total_a = 0, 0, 0
    for site in sorted(counts):
        c = counts[site]
        print(f"{site:<14}{c['total']:>6}{c['control']:>10}{c['adhd']:>8}")
        total_n += c["total"]
        total_c += c["control"]
        total_a += c["adhd"]
    print("-" * 38)
    print(f"{'Total':<14}{total_n:>6}{total_c:>10}{total_a:>8}")


if __name__ == "__main__":
    main()