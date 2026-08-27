# ADHD-200 Pipeline Comparison: Athena vs. NIAK

Code for the experimental component (Chapter 5) of the B.Sc. Computer
Science seminar *"Machine Learning on rs-fMRI for ADHD Classification: A
Computational Perspective on Representation and Methodology."*

Compares two independently preprocessed pipelines (Athena/CC200 and
NIAK/ROI1000) across three classifiers (SVM, Random Forest, MLP) on a
matched subject cohort from the ADHD-200 dataset.

**Author:** Shahar Barsano<br> **Supervisor:** Dr. Shalom Mandel<br> **Date:** September 2026

## Requirements

- Python 3.10+
- numpy, pandas, scipy, scikit-learn

```
pip install numpy pandas scipy scikit-learn
```

## Data

Not included in this repository. Requires the ADHD-200 Preprocessed
repository (Athena and NIAK pipelines) and the global phenotypic file,
available from the Neuro Bureau via NITRC:
http://www.nitrc.org/frs/?group_id=383

Expected folder structure:

```
data/
├── Athena/<site>/...          (CC200 time series, per site)
├── NIAK/<site>/rois/...       (ROI1000 time series, per site)
└── adhd200_preprocessed_phenotypics.tsv
```

Site coverage, subject inclusion criteria, and quality-control policy are
described in Chapter 5.2 of the seminar.

## Scripts

Run in order:

1. **`build_dataset.py`** — loads both pipelines, applies QC-based subject
   filtering (Chapter 5.2), matches subjects across pipelines, and saves
   the final dataset to `output/multisite_data_final.npz`.
2. **`run_evaluation.py`** — nested cross-validation evaluation of all
   three classifiers on both pipelines (Chapter 5.4). Set `QUICK_TEST =
   True` for a fast sanity check before running the full version. Saves
   results incrementally to `output/evaluation_results.csv`.
3. **`site_breakdown.py`** — reports the final matched cohort's
   composition by site (Chapter 5.2, Table 1).

## Output

`output/evaluation_results.csv` contains one row per (pipeline, classifier,
fold), with accuracy, sensitivity, specificity, AUC-ROC, and the
majority-class baseline for that fold. See Chapter 5.5 of the seminar for
the aggregated results and discussion.