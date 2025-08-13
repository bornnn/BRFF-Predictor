
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
predict_two_files.py — Merge two CSVs (Tool1/Tool2), select final features, standardize (if scaler provided),
and run predictions with a saved scikit-learn model (.joblib).

Expected model save styles:
1) Bundle dict: {"model": <estimator_or_pipeline>, "meta": {"features": [...] }, "scaler": <fitted StandardScaler or compatible> (optional)}
2) Plain estimator or Pipeline object (if Pipeline already contains scaler/processing).

Usage:
  python predict_two_files.py --model best_model.joblib --tool1 tool1-fullfeature.csv --tool2 Tool2-fullfeature_final.csv --output result.csv --proba

Notes:
- By default, this script WILL NOT fit a scaler on your input (to avoid leakage).
  It will use a scaler found inside the bundle (key 'scaler') or rely on the model Pipeline's scaler.
- If you really need to fit-on-the-fly (not recommended), pass --allow-fit-scaler.
"""

import argparse
import json
import sys
import difflib
from typing import List, Optional, Tuple

import joblib
import pandas as pd
import numpy as np

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
except Exception:
    StandardScaler = object
    Pipeline = tuple  # placeholder

FINAL_FEATURES = ['Right-Cerebellum-White-Matter_Tool1', 'lh_superiorfrontal_thickness_Tool2', 'lh_rostralanteriorcingulate_thickness_Tool2', 'rh_transversetemporal_area_Tool1', 'Right-Pallidum_Tool2', 'rh_postcentral_area_Tool2', 'Left-Cerebellum-Cortex_Tool1', 'rh_fusiform_area_Tool1', 'lh_superiorparietal_volume_Tool1', 'lh_isthmuscingulate_volume_Tool2', 'rh_parstriangularis_area_Tool1', 'lh_rostralmiddlefrontal_volume_Tool2', 'CC_Anterior_Tool2', 'lh_temporalpole_area', 'rh_isthmuscingulate_area_Tool1', 'rh_superiortemporal_volume_Tool1', 'lh_supramarginal_volume_Tool2', 'rh_caudalmiddlefrontal_thickness_Tool2', 'lh_entorhinal_volume_Tool1', 'Left-Caudate_Tool1', 'rh_entorhinal_thickness_Tool2', 'rh_isthmuscingulate_thickness_Tool2', 'Right-Amygdala_Tool2', 'lh_isthmuscingulate_thickness_Tool2', 'lhSurfaceHoles_Tool1', 'rh_frontalpole_area', 'lh_middletemporal_volume_Tool1', 'rh_lingual_thickness_Tool2', 'rh_inferiortemporal_area_Tool1', 'rh_bankssts_thickness', 'rh_lingual_volume_Tool2', 'lh_fusiform_volume_Tool2', 'lh_lingual_thickness_Tool1', 'rh_parsorbitalis_thickness_Tool1', 'lh_parahippocampal_area_Tool1', 'lh_pericalcarine_thickness_Tool2', 'lh_fusiform_area_Tool2', 'lh_superiortemporal_volume_Tool2', 'lh_lateralorbitofrontal_thickness_Tool2', 'rh_caudalanteriorcingulate_thickness_Tool2', 'lh_parahippocampal_thickness_Tool2', 'lh_frontalpole_volume', 'lh_caudalmiddlefrontal_thickness_Tool2', 'Right-Putamen_Tool1', 'rh_rostralmiddlefrontal_thickness_Tool1', 'Right-Pallidum_Tool1', 'rh_rostralmiddlefrontal_volume_Tool2', 'lh_parstriangularis_area_Tool2', 'rh_middletemporal_thickness_Tool2', 'lh_caudalanteriorcingulate_thickness_Tool2', 'lh_lateraloccipital_area_Tool2', 'rh_caudalanteriorcingulate_volume_Tool2', 'Right-vessel']

DEFAULT_ID_COL = "ID"
DEFAULT_DROPS_TOOL1 = ["Sum"]           # dropped before merge if present
DEFAULT_DROPS_TOOL2 = ["Sum", "label_y"]

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Predict using two source CSVs (Tool1/Tool2) merged on ID.")
    ap.add_argument("--model", required=True, help="Path to .joblib model file.")
    ap.add_argument("--tool1", "--freesurfer", dest="tool1", required=True, help="Path to Tool1 / FreeSurfer CSV.")
    ap.add_argument("--tool2", "--fastsurfer", dest="tool2", required=True, help="Path to Tool2 / FastSurfer CSV.")
    ap.add_argument("--output", default="brain_resilience_predictions.csv", help="Output CSV filename.")
    ap.add_argument("--id-col", default=DEFAULT_ID_COL, help="ID column name (default: ID).")
    ap.add_argument("--proba", action="store_true", help="Also output class-1 probability if available.")
    ap.add_argument("--encoding", default="utf-8-sig", help="CSV encoding for input/output (default: utf-8-sig).")
    ap.add_argument("--allow-fit-scaler", action="store_true",
                    help="Fit a StandardScaler on incoming data if no scaler is provided and the model is not a Pipeline (NOT RECOMMENDED).")
    return ap.parse_args()

def load_csv(path: str, drops: List[str], encoding: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, encoding=encoding)
    except Exception as e:
        print(f"[ERROR] Failed to read CSV '{{path}}': {{e}}", file=sys.stderr)
        sys.exit(1)
    for col in drops:
        if col in df.columns:
            df = df.drop(columns=[col])
    return df

def merge_sources(df1: pd.DataFrame, df2: pd.DataFrame, id_col: str) -> pd.DataFrame:
    if id_col not in df1.columns or id_col not in df2.columns:
        print(f"[ERROR] ID column '{{id_col}}' must exist in both CSVs.", file=sys.stderr)
        sys.exit(2)
    merged = pd.merge(
        df1, df2, on=[id_col], suffixes=["_Tool1", "_Tool2"], how="inner"
    )
    if merged.empty:
        print("[ERROR] Merge resulted in 0 rows. Check your ID values.", file=sys.stderr)
        sys.exit(3)
    return merged

def load_model(path: str):
    try:
        obj = joblib.load(path)
    except Exception as e:
        print(f"[ERROR] Failed to load model from {{path}}: {{e}}", file=sys.stderr)
        sys.exit(4)
    model = obj
    meta = None
    scaler = None
    if isinstance(obj, dict):
        model = obj.get("model", None)
        meta = obj.get("meta", None)
        scaler = obj.get("scaler", None)
        if model is None:
            print("[ERROR] Bundle dict must contain a 'model' key.", file=sys.stderr)
            sys.exit(4)
    return model, meta, scaler

def ensure_features(merged: pd.DataFrame, features: List[str]):
    missing = [f for f in features if f not in merged.columns]
    if missing:
        print("[ERROR] Missing required feature columns after merge:", file=sys.stderr)
        for m in missing:
            # Suggest similar columns
            sims = difflib.get_close_matches(m, merged.columns, n=3, cutoff=0.6)
            hint = f"  - {{m}}  (similar: {{sims}})" if sims else f"  - {{m}}"
            print(hint, file=sys.stderr)
        print("\n[HINT] Check your source CSVs and the suffix rules (_Tool1/_Tool2).", file=sys.stderr)
        sys.exit(5)

def maybe_scale(X: pd.DataFrame, model, scaler, allow_fit: bool) -> np.ndarray:
    # Case 1: Provided scaler object from the bundle
    if scaler is not None:
        try:
            return scaler.transform(X)
        except Exception as e:
            print(f"[ERROR] Failed to use provided scaler.transform(): {{e}}", file=sys.stderr)
            sys.exit(6)
    # Case 2: Model is a Pipeline (assume it handles scaling internally)
    try:
        if isinstance(model, Pipeline):
            return X.values  # Pipeline will handle scaling inside predict()
    except Exception:
        pass
    # Case 3: No scaler, not a Pipeline
    if allow_fit:
        print("[WARN] No scaler provided and model is not a Pipeline. Fitting a StandardScaler on incoming data (NOT RECOMMENDED).", file=sys.stderr)
        sc = StandardScaler()
        return sc.fit_transform(X)
    else:
        # Return raw features; hope the model expects raw
        return X.values

def predict(model, X: np.ndarray, want_proba: bool) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    proba = None
    if want_proba and hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[:, 1]
    yhat = model.predict(X)
    return yhat, proba

def main():
    args = parse_args()

    df1 = load_csv(args.tool1, DEFAULT_DROPS_TOOL1, args.encoding)
    df2 = load_csv(args.tool2, DEFAULT_DROPS_TOOL2, args.encoding)
    merged = merge_sources(df1, df2, args.id_col)

    # Keep a copy of ID for output if present
    id_series = merged[args.id_col] if args.id_col in merged.columns else None

    # Ensure final features exist
    ensure_features(merged, FINAL_FEATURES)

    # Build feature matrix
    X_df = merged[FINAL_FEATURES]

    # Load model/bundle
    model, meta, scaler = load_model(args.model)

    # Maybe scale
    X = maybe_scale(X_df, model, scaler, allow_fit=args.allow_fit_scaler)

    # Predict
    yhat, proba = predict(model, X, want_proba=args.proba)

    # Build output
    out = pd.DataFrame({})
    if id_series is not None:
        out[args.id_col] = id_series
    out["pred"] = yhat
    if args.proba and proba is not None:
        out["prob_1"] = proba

    # Save
    try:
        out.to_csv(args.output, index=False, encoding=args.encoding)
    except Exception as e:
        print(f"[ERROR] Failed to write output CSV '{{args.output}}': {{e}}", file=sys.stderr)
        sys.exit(7)

    summary = {
        "model_path": args.model,
        "tool1_path": args.tool1,
        "tool2_path": args.tool2,
        "output_path": args.output,
        "n_rows": int(len(out)),
        "n_features_used": int(len(FINAL_FEATURES)),
        "features": FINAL_FEATURES,
        "probability_included": bool(args.proba and (proba is not None)),
        "id_col": args.id_col,
        "bundle_meta_present": bool(isinstance(meta, dict)),
        "scaler_in_bundle": bool(scaler is not None),
    }
    print("PREDICTION_SUMMARY:\n" + json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
