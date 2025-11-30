import os
import sys
import json
import hashlib
import shutil
import time
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple, Optional


# ========== Configuration (can be overridden by CLI arguments) ==========

BASE_DIR = Path(__file__).resolve().parent

# Root directory that contains all APKs (recursively)
DATASET_ROOT = BASE_DIR / "data" / "APKs"

# Root directory for preprocessed outputs
PREPROC_ROOT = BASE_DIR / "data" / "preproc"

# External tools. If they are in PATH, just use the command name.
APKTOOL_PATH = "apktool"   # e.g., "apktool" or "apktool.bat"
JADX_PATH = "jadx"         # e.g., "jadx" or "jadx.bat"

# Maximum number of parallel processes
MAX_WORKERS = 4

# Timeouts (seconds)
APKTOOL_TIMEOUT = 600      # 10 minutes
JADX_TIMEOUT = 1800        # 30 minutes

# Minimum file size (bytes) to consider for hashing (0 = no filter)
MIN_APK_SIZE = 0


# ========== Utility Functions ==========

def sha256_file(path: Path) -> str:
    """Compute SHA256 hash for a file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def run_cmd(cmd: List[str], timeout: int = 600, cwd: Optional[Path] = None) -> Tuple[bool, str]:
    """
    Run an external command and return (ok, output_text).
    The command must be provided as a list to avoid shell differences.
    """
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd is not None else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            text=True,
        )
        ok = proc.returncode == 0
        return ok, proc.stdout
    except subprocess.TimeoutExpired as e:
        return False, f"[TIMEOUT] {e}"
    except FileNotFoundError as e:
        return False, f"[NOT FOUND] {e}"
    except Exception as e:
        return False, f"[EXCEPTION] {e}"


def extract_strings(apk_path: Path, out_path: Path, min_len: int = 4) -> bool:
    """
    Extract printable strings from an APK file using a simple built-in implementation.
    This avoids depending on external 'strings' tools.
    """
    try:
        data = apk_path.read_bytes()
    except OSError as e:
        out_path.write_text(
            f"[ERROR] Failed to read APK for strings: {e}",
            encoding="utf-8",
            errors="ignore",
        )
        return False

    result: List[str] = []
    buf: List[str] = []

    def flush_buf():
        if len(buf) >= min_len:
            result.append("".join(buf))
        buf.clear()

    for b in data:
        ch = chr(b)
        # basic printable ASCII range
        if 32 <= b <= 126:
            buf.append(ch)
        else:
            flush_buf()
    flush_buf()

    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8", errors="ignore") as f:
            for line in result:
                f.write(line)
                f.write("\n")
        return True
    except OSError as e:
        out_path.write_text(
            f"[ERROR] Failed to write strings: {e}",
            encoding="utf-8",
            errors="ignore",
        )
        return False


def find_all_apks(dataset_root: Path) -> List[Path]:
    """Recursively find all .apk files under dataset_root."""
    apk_paths: List[Path] = []
    for root, _, files in os.walk(dataset_root):
        for name in files:
            if name.lower().endswith(".apk"):
                apk_paths.append(Path(root) / name)
    return apk_paths


def process_single_apk(apk_path: Path) -> Tuple[str, str]:
    """
    Preprocess a single APK:
      * compute SHA256
      * copy raw APK
      * run apktool
      * run jadx
      * extract simple strings
      * write status.json

    Returns (apk_str_path, status_flag) for logging.
    """
    apk_str = str(apk_path)
    try:
        if apk_path.stat().st_size < MIN_APK_SIZE:
            return apk_str, "SKIPPED_SMALL"

        sha = sha256_file(apk_path)
        out_dir = PREPROC_ROOT / sha
        out_dir.mkdir(parents=True, exist_ok=True)

        raw_apk = out_dir / "raw.apk"
        if not raw_apk.exists():
            shutil.copy2(apk_path, raw_apk)

        status = {
            "sha256": sha,
            # store original APK path as string (absolute or relative)
            "apk_path": str(apk_path),
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "apktool_ok": False,
            "jadx_ok": False,
            "strings_ok": False,
        }

        # Step 1: apktool decode (resources + manifest + smali)
        apktool_out = out_dir / "apktool_out"
        apktool_out.mkdir(parents=True, exist_ok=True)
        apktool_cmd = [
            APKTOOL_PATH,
            "d",
            "-f",
            "-o",
            str(apktool_out),
            str(raw_apk),
        ]
        apktool_ok, apktool_log = run_cmd(apktool_cmd, timeout=APKTOOL_TIMEOUT)
        (out_dir / "apktool.log").write_text(
            apktool_log,
            encoding="utf-8",
            errors="ignore",
        )
        status["apktool_ok"] = apktool_ok

        # Step 2: JADX to Java source
        java_src = out_dir / "java_src"
        java_src.mkdir(parents=True, exist_ok=True)
        jadx_cmd = [
            JADX_PATH,
            "-d",
            str(java_src),
            str(raw_apk),
        ]
        jadx_ok, jadx_log = run_cmd(jadx_cmd, timeout=JADX_TIMEOUT)
        (out_dir / "jadx.log").write_text(
            jadx_log,
            encoding="utf-8",
            errors="ignore",
        )
        status["jadx_ok"] = jadx_ok

        # Step 3: extract strings
        strings_path = out_dir / "strings.txt"
        strings_ok = extract_strings(raw_apk, strings_path)
        status["strings_ok"] = strings_ok

        # Write status.json
        status_path = out_dir / "status.json"
        status_path.write_text(
            json.dumps(status, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        if not apktool_ok and not jadx_ok:
            return apk_str, "FAILED_ALL"
        if not apktool_ok:
            return apk_str, "APKTOOL_FAIL"
        if not jadx_ok:
            return apk_str, "JADX_FAIL"
        if not strings_ok:
            return apk_str, "STRINGS_FAIL"
        return apk_str, "OK"

    except Exception as e:
        return apk_str, f"EXCEPTION: {e}"


def main():
    global DATASET_ROOT, PREPROC_ROOT

    print("=== Batch preprocessing of APKs ===")
    print(f"APK root            : {DATASET_ROOT}")
    print(f"Preprocessed root   : {PREPROC_ROOT}")
    print(f"Max workers         : {MAX_WORKERS}")
    print(f"APKTool path        : {APKTOOL_PATH}")
    print(f"JADX path           : {JADX_PATH}")
    print("=" * 60)

    DATASET_ROOT = Path(DATASET_ROOT).resolve()
    PREPROC_ROOT = Path(PREPROC_ROOT).resolve()

    if not DATASET_ROOT.exists():
        raise SystemExit(f"APK root directory does not exist: {DATASET_ROOT}")

    PREPROC_ROOT.mkdir(parents=True, exist_ok=True)

    apk_paths = find_all_apks(DATASET_ROOT)
    if not apk_paths:
        print("No APK files found. Nothing to do.")
        return

    print(f"Found {len(apk_paths)} APK files.")
    print("Starting preprocessing...\n")

    total = len(apk_paths)
    done = 0

    # Use process pool for parallelism
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_apk = {executor.submit(process_single_apk, apk): apk for apk in apk_paths}

        for future in as_completed(future_to_apk):
            apk = future_to_apk[future]
            try:
                apk_str, status = future.result()
            except Exception as e:
                done += 1
                print(f"[EXCEPTION] {done:5d}/{total:5d}  {apk} -> {e}")
                continue

            done += 1
            print(f"[{status:14}] {done:5d}/{total:5d}  {apk_str}")

    print("\nPreprocessing finished.")


if __name__ == "__main__":
    # Allow overriding DATASET_ROOT and PREPROC_ROOT via CLI arguments:
    #   python Batch_preprocess_apks.py [APK_ROOT] [PREPROC_ROOT]
    if len(sys.argv) >= 2:
        DATASET_ROOT = Path(sys.argv[1])
    if len(sys.argv) >= 3:
        PREPROC_ROOT = Path(sys.argv[2])

    main()
