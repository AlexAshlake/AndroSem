import os
import json
import random
from pathlib import Path
from typing import List, Dict, Any, Optional


# ================= Configuration =================

# Project root directory = directory of this file
BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
SPLIT_DIR = DATA_DIR / "dataset_splits"
MAIN_OUTPUT_DIR = DATA_DIR / "main_output"

# === Input: dataset definition and main-output directory ===
# By default we evaluate on the test set produced by Split_dataset.py
# and on one specific experiment directory under data/main_output.
# You can override both via command-line arguments:
#   python Analyse_result.py [dataset_json_name] [experiment_dir_name]
#
# Example:
#   python Analyse_result.py test_set.json cicandmal2017_full
DATASET_JSON_NAME = "test_set.json"
EXPERIMENT_DIR_NAME = "cicandmal2017_full"

DATASET_DEF_PATH = SPLIT_DIR / DATASET_JSON_NAME
OUTPUT_ROOT = MAIN_OUTPUT_DIR / EXPERIMENT_DIR_NAME

# === Output: trace & statistics paths (all under OUTPUT_ROOT) ===
TRACE_ALL_PATH = OUTPUT_ROOT / "llm_trace_all_samples.json"
TRACE_MIS_PATH = OUTPUT_ROOT / "llm_trace_full_misclassified.json"
TRACE_BINARY_MIS_PATH = OUTPUT_ROOT / "llm_trace_binary_misclassified.json"
STATS_RESULT_PATH = OUTPUT_ROOT / "llm_classification_stats.json"

# === Sampling settings ===
# - SAMPLE_COUNT_ALL:
#       max number of samples for the "all samples" trace (-1 = use all)
# - SAMPLE_COUNT_MIS:
#       max number of samples for the "full misclassified" (multiclass) trace (-1 = use all)
# - SAMPLE_COUNT_BINARY_MIS:
#       max number of samples for the "binary misclassified" trace (-1 = use all)
SAMPLE_COUNT_ALL = 50
SAMPLE_COUNT_MIS = -1
SAMPLE_COUNT_BINARY_MIS = -1

# Random seed for reproducible sampling
RANDOM_SEED = 2024

# Directory names (must match MainLogic output layout)
DIR_FINAL_IR = "7_final_ir"
DIR_CLASSIFICATION = "8_classification"

# Trace switches
ENABLE_TRACE_ALL_SAMPLES = False
ENABLE_TRACE_MISCLASSIFIED_FULL = False
ENABLE_TRACE_MISCLASSIFIED_BINARY = False


# ================= Helper functions =================


def fix_long_path(path: Path) -> str:
    """Return a string path that is safe for long paths on Windows."""
    abs_path = path.resolve()
    if os.name == "nt":
        return "\\\\?\\" + str(abs_path)
    return str(abs_path)


def read_json(path: Path) -> Optional[Dict[str, Any]]:
    """Safely read a JSON file. Return None if missing or malformed."""
    if not path.exists():
        return None
    try:
        with open(fix_long_path(path), "r", encoding="utf-8", errors="ignore") as f:
            return json.load(f)
    except Exception as e:
        print(f"[Err] Failed to read {path}: {e}")
        return None


def load_dataset_hashes(dataset_path: Path) -> List[str]:
    """Load the list of sample hashes from a dataset definition JSON file."""
    data = read_json(dataset_path)
    if not data:
        print(f"[Err] Failed to load dataset definition: {dataset_path}")
        return []

    samples = data.get("samples", [])
    hashes: List[str] = []
    for item in samples:
        h = item.get("hash")
        if h:
            hashes.append(str(h))
    # remove duplicates while preserving order
    seen = set()
    unique_hashes: List[str] = []
    for h in hashes:
        if h not in seen:
            seen.add(h)
            unique_hashes.append(h)
    return unique_hashes


def get_all_classified_hashes(root_dir: Path) -> List[str]:
    """
    Scan the classification directory and return all sample hashes
    that have a *_class.json file.
    """
    class_dir = root_dir / DIR_CLASSIFICATION
    if not class_dir.exists():
        print(f"[Err] Classification directory not found: {class_dir}")
        return []

    files = list(class_dir.glob("*_class.json"))
    hashes = [f.name.replace("_class.json", "") for f in files]
    return hashes


def build_trace_record(root_dir: Path, sample_hash: str) -> Dict[str, Any]:
    """
    Build a detailed trace record for a single sample:
      - final IR (Step 7)
      - full / E-Meta / E-Summary classification predictions
    """
    path_ir = root_dir / DIR_FINAL_IR / f"{sample_hash}_ir.json"
    path_cls = root_dir / DIR_CLASSIFICATION / f"{sample_hash}_class.json"

    ir_data = read_json(path_ir)
    cls_data = read_json(path_cls)

    variants = cls_data.get("variants", {}) if cls_data else {}

    record: Dict[str, Any] = {
        "hash": sample_hash,
        "status": "complete" if (ir_data and cls_data) else "partial",
        "ground_truth": cls_data.get("ground_truth", "Unknown") if cls_data else "Unknown",
        "prediction_full": variants.get("full", {}).get("prediction", "N/A"),
        "prediction_no_meta": variants.get("E-Meta", {}).get("prediction", "N/A"),
        "prediction_no_summary": variants.get("E-Summary", {}).get("prediction", "N/A"),
        "ir": ir_data if ir_data else {},
        "classification": {
            "full": variants.get("full", {}).get("prediction", "N/A"),
            "E-Meta": variants.get("E-Meta", {}).get("prediction", "N/A"),
            "E-Summary": variants.get("E-Summary", {}).get("prediction", "N/A"),
        },
    }
    return record


def build_classification_record(root_dir: Path, sample_hash: str) -> Optional[Dict[str, Any]]:
    """Build a lightweight record used for global statistics (no IR loaded)."""
    path_cls = root_dir / DIR_CLASSIFICATION / f"{sample_hash}_class.json"
    cls_data = read_json(path_cls)
    if not cls_data:
        return None

    variants = cls_data.get("variants", {}) or {}
    preds = {
        "full": variants.get("full", {}).get("prediction", "N/A"),
        "E-Meta": variants.get("E-Meta", {}).get("prediction", "N/A"),
        "E-Summary": variants.get("E-Summary", {}).get("prediction", "N/A"),
    }

    return {
        "hash": sample_hash,
        "ground_truth": cls_data.get("ground_truth", "Unknown"),
        "predictions": preds,
    }


def _normalize_label(label: Any) -> Optional[str]:
    """
    Normalize a label into upper-case text.
    Returns None if the label is empty or Unknown / N/A-like.
    """
    if label is None:
        return None
    s = str(label).strip()
    if not s:
        return None
    if s.lower() in {"unknown", "n/a"}:
        return None
    return s.upper()


def compute_multiclass_metrics(class_records: List[Dict[str, Any]], variant_key: str) -> Dict[str, Any]:
    """
    Compute multi-class metrics for a given prediction variant:
      - overall accuracy
      - per-class precision / recall / f1 / support
      - macro-F1
    Only samples with valid (non-Unknown / non-N/A) ground truth and prediction are used.
    """
    total_samples = len(class_records)
    valid_pairs = []

    for rec in class_records:
        gt_raw = rec.get("ground_truth")
        pred_raw = rec.get("predictions", {}).get(variant_key)

        gt = _normalize_label(gt_raw)
        pred = _normalize_label(pred_raw)

        if gt is None or pred is None:
            continue

        valid_pairs.append((gt, pred))

    valid_count = len(valid_pairs)

    if valid_count == 0:
        return {
            "total_samples": total_samples,
            "valid_samples": 0,
            "accuracy": None,
            "macro_f1": None,
            "labels": {},
        }

    # collect label set
    label_set = sorted({gt for gt, _ in valid_pairs} | {pred for _, pred in valid_pairs})

    # initialise per-label stats
    per_label_stats = {
        label: {"tp": 0, "fp": 0, "fn": 0, "support": 0}
        for label in label_set
    }

    correct = 0
    for gt, pred in valid_pairs:
        if gt == pred:
            correct += 1
        per_label_stats[gt]["support"] += 1

        # update tp/fp/fn for each label
        for label in label_set:
            if pred == label and gt == label:
                per_label_stats[label]["tp"] += 1
            elif pred == label and gt != label:
                per_label_stats[label]["fp"] += 1
            elif pred != label and gt == label:
                per_label_stats[label]["fn"] += 1

    accuracy = correct / valid_count if valid_count > 0 else None

    # compute per-class precision/recall/f1
    macro_f1_sum = 0.0
    macro_f1_cnt = 0
    label_metrics: Dict[str, Any] = {}

    for label, st in per_label_stats.items():
        tp = st["tp"]
        fp = st["fp"]
        fn = st["fn"]
        support = st["support"]

        precision = tp / (tp + fp) if (tp + fp) > 0 else None
        recall = tp / (tp + fn) if (tp + fn) > 0 else None
        f1 = None
        if precision is not None and recall is not None and (precision + recall) > 0:
            f1 = 2 * precision * recall / (precision + recall)

        if f1 is not None:
            macro_f1_sum += f1
            macro_f1_cnt += 1

        label_metrics[label] = {
            "support": support,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    macro_f1 = macro_f1_sum / macro_f1_cnt if macro_f1_cnt > 0 else None

    return {
        "total_samples": total_samples,
        "valid_samples": valid_count,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "labels": label_metrics,
    }


def _label_to_binary(label: Any) -> Optional[str]:
    """
    Map a label into a binary label:
      - "BENIGN"  for benign
      - "MALICIOUS" for all other valid labels
    """
    if label is None:
        return None
    s = str(label).strip()
    if not s:
        return None
    # We treat any label whose lowercase string equals "benign" as benign.
    if s.lower() == "benign":
        return "BENIGN"
    # All other non-empty labels are treated as malicious.
    return "MALICIOUS"


def is_full_misclassified(record: Dict[str, Any]) -> bool:
    """
    Check whether a sample is misclassified in the multi-class sense
    for the "full" variant.
    """
    if record.get("status") != "complete":
        return False
    gt = _normalize_label(record.get("ground_truth"))
    pred = _normalize_label(record.get("prediction_full"))
    if gt is None or pred is None:
        return False
    return gt != pred


def is_binary_misclassified(record: Dict[str, Any]) -> bool:
    """
    Check whether a sample is misclassified in the binary sense
    (benign vs malicious) for the "full" variant.
    """
    if record.get("status") != "complete":
        return False

    gt_bin = _label_to_binary(record.get("ground_truth"))
    pred_bin = _label_to_binary(record.get("prediction_full"))

    if gt_bin is None or pred_bin is None:
        return False

    return gt_bin != pred_bin


def compute_binary_metrics(class_records: List[Dict[str, Any]], variant_key: str) -> Dict[str, Any]:
    """
    Compute binary metrics (benign vs malicious) for a given variant:
      - TP: real malicious & predicted malicious
      - FP: real benign   & predicted malicious
      - FN: real malicious & predicted benign
      - TN: real benign   & predicted benign
    Only samples whose labels can be mapped to BENIGN/MALICIOUS are used.
    """
    total_samples = len(class_records)
    tp = fp = fn = tn = 0
    valid_count = 0

    for rec in class_records:
        gt_raw = rec.get("ground_truth")
        pred_raw = rec.get("predictions", {}).get(variant_key)

        gt_bin = _label_to_binary(gt_raw)
        pred_bin = _label_to_binary(pred_raw)

        if gt_bin is None or pred_bin is None:
            continue

        valid_count += 1

        if gt_bin == "MALICIOUS" and pred_bin == "MALICIOUS":
            tp += 1
        elif gt_bin == "BENIGN" and pred_bin == "MALICIOUS":
            fp += 1
        elif gt_bin == "MALICIOUS" and pred_bin == "BENIGN":
            fn += 1
        elif gt_bin == "BENIGN" and pred_bin == "BENIGN":
            tn += 1

    if valid_count == 0:
        return {
            "total_samples": total_samples,
            "valid_samples": 0,
            "confusion": {"TP": 0, "FP": 0, "FN": 0, "TN": 0},
            "metrics": {
                "accuracy": None,
                "precision_malicious": None,
                "recall_malicious": None,
                "f1_malicious": None,
            },
            "benign_label_rule": "label.lower() == 'benign'",
        }

    accuracy = (tp + tn) / valid_count if valid_count > 0 else None
    precision_m = tp / (tp + fp) if (tp + fp) > 0 else None
    recall_m = tp / (tp + fn) if (tp + fn) > 0 else None
    f1_m = None
    if precision_m is not None and recall_m is not None and (precision_m + recall_m) > 0:
        f1_m = 2 * precision_m * recall_m / (precision_m + recall_m)

    return {
        "total_samples": total_samples,
        "valid_samples": valid_count,
        "confusion": {"TP": tp, "FP": fp, "FN": fn, "TN": tn},
        "metrics": {
            "accuracy": accuracy,
            "precision_malicious": precision_m,
            "recall_malicious": recall_m,
            "f1_malicious": f1_m,
        },
        "benign_label_rule": "label.lower() == 'benign'",
    }


def fmt_metric(v: Any) -> str:
    """Format a metric for console printing. None -> 'N/A'."""
    if v is None:
        return "N/A"
    try:
        return f"{float(v):.4f}"
    except (TypeError, ValueError):
        return str(v)


# ================= Main entry =================


def main():
    global DATASET_DEF_PATH, OUTPUT_ROOT

    random.seed(RANDOM_SEED)

    # Allow overriding dataset JSON name and experiment directory name via CLI:
    #   python Analyse_result.py [dataset_json_name] [experiment_dir_name]
    import sys

    if len(sys.argv) >= 2:
        DATASET_DEF_PATH = SPLIT_DIR / sys.argv[1]
    if len(sys.argv) >= 3:
        OUTPUT_ROOT = MAIN_OUTPUT_DIR / sys.argv[2]

    print(f"Dataset definition file : {DATASET_DEF_PATH}")
    print(f"Main output directory   : {OUTPUT_ROOT}")

    if not DATASET_DEF_PATH.exists():
        print(f"[Err] Dataset definition file does not exist: {DATASET_DEF_PATH}")
        return

    if not OUTPUT_ROOT.exists():
        print(f"[Err] Main output directory does not exist: {OUTPUT_ROOT}")
        return

    dataset_hashes = load_dataset_hashes(DATASET_DEF_PATH)
    if not dataset_hashes:
        print("[Err] No hashes loaded from dataset definition.")
        return

    all_classified_hashes = set(get_all_classified_hashes(OUTPUT_ROOT))

    eval_hashes = [h for h in dataset_hashes if h in all_classified_hashes]

    total_in_dataset = len(dataset_hashes)
    total_classified = len(eval_hashes)

    print(f"Total samples in dataset definition : {total_in_dataset}")
    print(f"Samples with classification outputs : {total_classified}")

    if total_classified == 0:
        return

    # ---------- Build classification records for statistics ----------
    class_records: List[Dict[str, Any]] = []
    for h in eval_hashes:
        rec = build_classification_record(OUTPUT_ROOT, h)
        if rec is not None:
            class_records.append(rec)

    print(f"Loaded classification records for stats: {len(class_records)}")

    variants = ["full", "E-Meta", "E-Summary"]
    stats_variants: Dict[str, Any] = {}

    for key in variants:
        m_multi = compute_multiclass_metrics(class_records, key)
        m_bin = compute_binary_metrics(class_records, key)
        stats_variants[key] = {
            "multiclass": m_multi,
            "binary": m_bin,
        }

        mc_valid = m_multi.get("valid_samples")
        mc_total = m_multi.get("total_samples")
        mc_acc = fmt_metric(m_multi.get("accuracy"))
        mc_f1 = fmt_metric(m_multi.get("macro_f1"))

        bi_valid = m_bin.get("valid_samples")
        bi_total = m_bin.get("total_samples")
        bi_acc = fmt_metric(m_bin.get("metrics", {}).get("accuracy"))
        bi_f1 = fmt_metric(m_bin.get("metrics", {}).get("f1_malicious"))

        print(f"\n=== Variant: {key} ===")
        print(f"[Multiclass] acc={mc_acc}, macro-F1={mc_f1}  (valid={mc_valid}/{mc_total})")
        print(f"[Binary]     acc={bi_acc}, F1(malicious)={bi_f1}  (valid={bi_valid}/{bi_total})")

    stats_meta = {
        "dataset_json": str(DATASET_DEF_PATH),
        "experiment_dir": str(OUTPUT_ROOT),
        "total_samples_in_dataset": total_in_dataset,
        "samples_with_classification": total_classified,
        "random_seed": RANDOM_SEED,
        "variants": variants,
        "note": "All metrics are computed over samples with valid (non-Unknown/non-N/A) labels.",
    }

    try:
        with open(fix_long_path(STATS_RESULT_PATH), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "meta": stats_meta,
                    "variants": stats_variants,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        print(f"[OK] Classification stats saved to: {STATS_RESULT_PATH}")
    except Exception as e:
        print(f"[Err] Failed to write stats file: {e}")

    # ---------- Generate trace files according to switches ----------

    # 1) Trace for all samples (subset of dataset, possibly sampled)
    if ENABLE_TRACE_ALL_SAMPLES:
        target_count = SAMPLE_COUNT_ALL
        if target_count == -1 or target_count > total_classified:
            target_count = total_classified

        selected_hashes = random.sample(eval_hashes, target_count)
        print(f"[Trace-All] Selected {len(selected_hashes)} samples (Seed: {RANDOM_SEED}).")

        trace_records_all: List[Dict[str, Any]] = []
        complete_count_all = 0
        miscls_count_all = 0

        for i, h in enumerate(selected_hashes):
            rec = build_trace_record(OUTPUT_ROOT, h)
            trace_records_all.append(rec)
            if rec.get("status") == "complete":
                complete_count_all += 1
                if is_full_misclassified(rec):
                    miscls_count_all += 1

            if (i + 1) % 10 == 0 or (i + 1) == len(selected_hashes):
                print(
                    f"[Trace-All] Processed {i + 1}/{len(selected_hashes)} "
                    f"(complete={complete_count_all}, miscls={miscls_count_all})"
                )

        meta_all = {
            "mode": "all_samples",
            "dataset_json": str(DATASET_DEF_PATH),
            "experiment_dir": str(OUTPUT_ROOT),
            "sample_count": len(selected_hashes),
            "random_seed": RANDOM_SEED,
            "total_samples_in_dataset": total_in_dataset,
            "samples_with_classification": total_classified,
            "complete_traces": complete_count_all,
            "full_misclassified": miscls_count_all,
            "fields": [
                "hash",
                "status",
                "ground_truth",
                "prediction_full",
                "prediction_no_meta",
                "prediction_no_summary",
                "ir",
                "classification",
            ],
        }

        try:
            with open(fix_long_path(TRACE_ALL_PATH), "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "meta": meta_all,
                        "traces": trace_records_all,
                    },
                    f,
                    indent=2,
                    ensure_ascii=False,
                )
            print(f"[OK] Trace (all samples) saved to: {TRACE_ALL_PATH}")
        except Exception as e:
            print(f"[Err] Failed to write trace-all file: {e}")

    # 2) Trace for full misclassified samples (multiclass view)
    if ENABLE_TRACE_MISCLASSIFIED_FULL:
        print("[Trace-Mis] Collecting full-misclassified samples over evaluation hashes.")

        misclassified_records: List[Dict[str, Any]] = []
        complete_count_mis = 0

        for i, h in enumerate(eval_hashes):
            rec = build_trace_record(OUTPUT_ROOT, h)
            if rec.get("status") == "complete":
                complete_count_mis += 1
                if is_full_misclassified(rec):
                    misclassified_records.append(rec)

            if (i + 1) % 50 == 0 or (i + 1) == total_classified:
                print(
                    f"[Trace-Mis] Scanned {i + 1}/{total_classified} "
                    f"(complete={complete_count_mis}, miscls={len(misclassified_records)})"
                )

        total_mis = len(misclassified_records)
        print(f"[Trace-Mis] Total full-misclassified samples: {total_mis}")

        # Down-sample if needed
        if SAMPLE_COUNT_MIS != -1 and total_mis > SAMPLE_COUNT_MIS:
            print(f"[Trace-Mis] Sampling {SAMPLE_COUNT_MIS} from {total_mis} misclassified samples.")
            misclassified_records = random.sample(misclassified_records, SAMPLE_COUNT_MIS)

        meta_mis = {
            "mode": "full_misclassified_only",
            "dataset_json": str(DATASET_DEF_PATH),
            "experiment_dir": str(OUTPUT_ROOT),
            "sample_count": len(misclassified_records),
            "random_seed": RANDOM_SEED,
            "total_samples_in_dataset": total_in_dataset,
            "samples_with_classification": total_classified,
            "complete_traces": complete_count_mis,
            "full_misclassified_total": total_mis,
            "fields": [
                "hash",
                "status",
                "ground_truth",
                "prediction_full",
                "prediction_no_meta",
                "prediction_no_summary",
                "ir",
                "classification",
            ],
        }

        try:
            with open(fix_long_path(TRACE_MIS_PATH), "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "meta": meta_mis,
                        "traces": misclassified_records,
                    },
                    f,
                    indent=2,
                    ensure_ascii=False,
                )
            print(f"[OK] Trace (full misclassified only) saved to: {TRACE_MIS_PATH}")
        except Exception as e:
            print(f"[Err] Failed to write trace-mis file: {e}")

    # 3) Trace for binary misclassified samples (benign vs malicious)
    if ENABLE_TRACE_MISCLASSIFIED_BINARY:
        print("[Trace-BinMis] Collecting binary-misclassified samples over evaluation hashes.")

        bin_mis_records: List[Dict[str, Any]] = []
        complete_count_bin = 0

        for i, h in enumerate(eval_hashes):
            rec = build_trace_record(OUTPUT_ROOT, h)
            if rec.get("status") == "complete":
                complete_count_bin += 1
                if is_binary_misclassified(rec):
                    bin_mis_records.append(rec)

            if (i + 1) % 50 == 0 or (i + 1) == total_classified:
                print(
                    f"[Trace-BinMis] Scanned {i + 1}/{total_classified} "
                    f"(complete={complete_count_bin}, bin_mis={len(bin_mis_records)})"
                )

        total_bin_mis = len(bin_mis_records)
        print(f"[Trace-BinMis] Total binary-misclassified samples: {total_bin_mis}")

        # Down-sample if needed
        if SAMPLE_COUNT_BINARY_MIS != -1 and total_bin_mis > SAMPLE_COUNT_BINARY_MIS:
            print(f"[Trace-BinMis] Sampling {SAMPLE_COUNT_BINARY_MIS} from {total_bin_mis} binary misclassified samples.")
            bin_mis_records = random.sample(bin_mis_records, SAMPLE_COUNT_BINARY_MIS)

        meta_bin = {
            "mode": "binary_misclassified_only",
            "dataset_json": str(DATASET_DEF_PATH),
            "experiment_dir": str(OUTPUT_ROOT),
            "sample_count": len(bin_mis_records),
            "random_seed": RANDOM_SEED,
            "total_samples_in_dataset": total_in_dataset,
            "samples_with_classification": total_classified,
            "complete_traces": complete_count_bin,
            "binary_misclassified_total": total_bin_mis,
            "fields": [
                "hash",
                "status",
                "ground_truth",
                "prediction_full",
                "prediction_no_meta",
                "prediction_no_summary",
                "ir",
                "classification",
            ],
        }

        try:
            with open(fix_long_path(TRACE_BINARY_MIS_PATH), "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "meta": meta_bin,
                        "traces": bin_mis_records,
                    },
                    f,
                    indent=2,
                    ensure_ascii=False,
                )
            print(f"[OK] Trace (binary misclassified only) saved to: {TRACE_BINARY_MIS_PATH}")
        except Exception as e:
            print(f"[Err] Failed to write binary-mis trace file: {e}")


if __name__ == "__main__":
    main()
