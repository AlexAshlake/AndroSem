import json
import random
from pathlib import Path
from collections import defaultdict, Counter


# ================= Configuration =================

# Base project directory = directory of this file
BASE_DIR = Path(__file__).resolve().parent

# 1. Preprocessed APK root (created by Batch_preprocess_apks.py)
PREPROC_ROOT = BASE_DIR / "data" / "preproc"

# 2. Output directory for dataset split definitions
OUTPUT_DIR = BASE_DIR / "data" / "dataset_splits"

# 3. Random seed for reproducible splits
RANDOM_SEED = 42

# 4. Minimum total size of Java source code per sample
MIN_SIZE_BYTES = 1024  # 1 KB


# =================================================


def parse_label_from_path(apk_path_str: str):
    """
    Parse high-level category and family from the original APK path.

    Expected layout (inside the dataset root):
        ... / APKs/<category_dir>[/<family_dir>]/<something>.apk

    Examples:
        data/APKs/Benign_2017/sample.apk
        data/APKs/Ransomware/FamilyA/sample.apk
    """
    try:
        path = Path(apk_path_str)
        parts = path.parts

        if "APKs" not in parts:
            return "Unknown", None

        apks_index = parts.index("APKs")
        if apks_index + 1 >= len(parts):
            return "Unknown", None

        raw_category = parts[apks_index + 1]
        category = "Unknown"
        family = None

        lower_cat = raw_category.lower()
        if "benign" in lower_cat:
            category = "Benign"
            family = raw_category
        elif "adware" in lower_cat:
            category = "Adware"
        elif "ransomware" in lower_cat:
            category = "Ransomware"
        elif "smsmalware" in lower_cat:
            category = "SMSMalware"
        elif "scareware" in lower_cat:
            category = "Scareware"
        else:
            # default: use directory name as category
            category = raw_category

        # For malware categories we also try to extract the family
        if category not in ("Benign", "Unknown"):
            if apks_index + 2 < len(parts) - 1:
                # Something like APKs/<category>/<family>/file.apk
                family = parts[apks_index + 2]
            else:
                family = "Generic"

        return category, family

    except Exception:
        return "Unknown", None


def stratified_split(samples, train_ratio: float = 0.7):
    """
    Split a list of samples (single category) into train and test parts.

    We ensure:
      - if len(samples) == 0 -> both empty
      - if len(samples) == 1 -> goes to train
      - otherwise both train and test have at least 1 sample
    """
    samples = list(samples)
    random.shuffle(samples)
    n = len(samples)

    if n == 0:
        return [], []
    if n == 1:
        return samples, []

    n_train = int(n * train_ratio)

    # ensure at least 1 sample in test
    if n_train == n:
        n_train = n - 1
    # ensure at least 1 sample in train
    if n_train == 0:
        n_train = 1

    return samples[:n_train], samples[n_train:]


def stratified_sample_subset(source_samples, target_ratio: float):
    """
    Stratified sampling of a subset from a given list of samples.

    target_ratio is the sampling rate *within* source_samples.

    In this project we usually want:
      - Train set  = 70% of all valid samples
      - Experiment = 5% of all valid samples
      - Mini       = 0.5% of all valid samples

    Since the experiment / mini sets are sampled from the train set,
    the sampling rates passed here are:
      rate_exp  = 0.05 / 0.7
      rate_mini = 0.005 / 0.7
    """
    cat_map = defaultdict(list)
    for s in source_samples:
        cat_map[s["category"]].append(s)

    subset = []
    for cat, items in cat_map.items():
        n_source = len(items)
        k = max(1, int(n_source * target_ratio))
        if n_source <= k:
            subset.extend(items)
        else:
            subset.extend(random.sample(items, k))

    return subset


def compute_java_size(java_root: Path) -> int:
    """
    Compute the total size (in bytes) of all *.java files under java_root.

    java_root is typically:
        <PREPROC_ROOT>/<hash>/java_src
    or:
        <PREPROC_ROOT>/<hash>/java_src/sources
    """
    total = 0
    if not java_root.exists():
        return 0

    for java_file in java_root.rglob("*.java"):
        try:
            total += java_file.stat().st_size
        except OSError:
            continue
    return total


def main():
    random.seed(RANDOM_SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not PREPROC_ROOT.exists():
        raise SystemExit(f"Preprocessed root directory does not exist: {PREPROC_ROOT}")

    print(f"Using preprocessed root: {PREPROC_ROOT}")
    print(f"Dataset definition output directory: {OUTPUT_DIR}")
    print(f"Minimum total Java size per sample: {MIN_SIZE_BYTES} bytes")
    print("=" * 50)

    # 1. Scan preprocessed samples and build per-category lists
    grouped_samples = defaultdict(list)
    stats = Counter()

    # Each subdirectory in PREPROC_ROOT is expected to be a hash (one APK)
    hash_dirs = [p for p in PREPROC_ROOT.iterdir() if p.is_dir()]
    print(f"Found {len(hash_dirs)} preprocessed sample directories.\n")

    for sample_dir in hash_dirs:
        file_hash = sample_dir.name
        status_path = sample_dir / "status.json"

        if not status_path.exists():
            stats["skipped_no_meta"] += 1
            continue

        try:
            with status_path.open("r", encoding="utf-8") as f:
                meta = json.load(f)

            apk_path = meta.get("apk_path", "")
            category, family = parse_label_from_path(apk_path)

            if category == "Unknown":
                stats["skipped_unknown_category"] += 1
                continue

            # Prefer java_src/sources if it exists, otherwise java_src root
            java_src_root = sample_dir / "java_src"
            probe_dir = java_src_root / "sources"
            if not probe_dir.exists():
                probe_dir = java_src_root

            total_java_size = compute_java_size(probe_dir)

            if total_java_size < MIN_SIZE_BYTES:
                stats["skipped_small"] += 1
                continue

            sample_obj = {
                "hash": file_hash,
                "category": category,
                "family": family,
                "size_bytes": total_java_size,
            }
            # Keep apk_path as optional extra information (useful for debugging)
            if apk_path:
                sample_obj["apk_path"] = apk_path

            grouped_samples[category].append(sample_obj)
            stats["valid"] += 1

        except Exception:
            stats["error"] += 1

    # 2. Print purification stats
    print("\n=== Purification Stats ===")
    print(f"Valid samples           : {stats.get('valid', 0)}")
    print(f"Skipped: no status.json : {stats.get('skipped_no_meta', 0)}")
    print(f"Skipped: unknown label  : {stats.get('skipped_unknown_category', 0)}")
    print(f"Skipped: too small      : {stats.get('skipped_small', 0)}")
    print(f"Errors while processing : {stats.get('error', 0)}")

    print("\nCategory distribution (valid samples):")
    for cat, lst in sorted(grouped_samples.items()):
        print(f"  {cat:<12}: {len(lst)}")
    print("=" * 50)

    # 3. Stratified 70/30 train/test split per category
    train_set = []
    test_set = []

    print("\nPerforming stratified split (70% train / 30% test)...")
    for cat, samples in grouped_samples.items():
        cat_train, cat_test = stratified_split(samples, train_ratio=0.7)
        train_set.extend(cat_train)
        test_set.extend(cat_test)

    print(f"Train set size: {len(train_set)}")
    print(f"Test set size : {len(test_set)}")

    # 4. Build small experiment subsets from the train set
    # Train = 70% of total; experiment / mini are defined as 5% / 0.5% of total
    rate_exp = 0.05 / 0.7
    rate_mini = 0.005 / 0.7

    exp_set = stratified_sample_subset(train_set, rate_exp)
    mini_set = stratified_sample_subset(train_set, rate_mini)

    print(f"Experiment set (≈5% of total) size: {len(exp_set)}")
    print(f"Mini set (≈0.5% of total) size    : {len(mini_set)}")

    # 5. Save all datasets to JSON
    datasets = {
        "train_set": train_set,
        "test_set": test_set,
        "experiment_set_5pct": exp_set,
        "mini_set_0.5pct": mini_set,
    }

    print("\nWriting dataset definition files...")
    for name, data in datasets.items():
        out_file = OUTPUT_DIR / f"{name}.json"
        dist = dict(Counter(d["category"] for d in data))

        with out_file.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "dataset_name": name,
                    "total_count": len(data),
                    "distribution": dist,
                    "samples": data,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

        print(f"{name:<20}: {len(data)} samples -> {out_file}")

    print("\nDone.")


if __name__ == "__main__":
    main()
