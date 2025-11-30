# -*- coding: utf-8 -*-

import os
import re
import json
import time
import math
import logging
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, Set
from collections import defaultdict
import string

# Project root directory (assumed to be the directory containing this file)
BASE_DIR = Path(__file__).resolve().parent

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x

import networkx as nx

from transformers import AutoTokenizer

try:
    from remote_model_client import RemoteLLMClient
except ImportError:
    class RemoteLLMClient:
        def __init__(self, base_url, use_lora=False): pass
        def generate(self, prompt): return "Mock LLM Response"
        def unload(self): pass

# =========================
# Config
# =========================
class ExperimentConfig:
    """
    Central configuration for APK-Sleuth (Android Malware Analysis).
    Directory layout is aligned with the project structure:
        - data/preproc:        preprocessed APKs
        - data/dataset_splits: dataset definition JSON files
        - data/main_output:    experiment outputs (this script)
    """

    # --- I/O Directories ---

    # Preprocessed APK root (contains status.json, apktool_out, java_src, etc.)
    PREPROC_DIR: Path = BASE_DIR / "data" / "preproc"

    # Dataset definition file (produced by Split_dataset.py)
    # You can switch to train_set.json / experiment_set_5pct.json / mini_set_0.5pct.json
    DATASET_JSON: Path = BASE_DIR / "data" / "dataset_splits" / "test_set.json"

    # Root directory for this experiment's outputs
    # Under data/main_output/<experiment_name>, where <experiment_name> is arbitrary
    OUTPUT_DIR: Path = BASE_DIR / "data" / "main_output" / "cicandmal2017_full"

    # --- Local LLM / Tokenizer Settings ---
    # Local HuggingFace model path used only for token counting
    LOCAL_TOKENIZER_PATH: str = str(BASE_DIR / "models" / "qwen3-14b")

    # --- Remote LLM Server ---
    REMOTE_LLM_BASE_URL: str = "http://127.0.0.1:8000"
    USE_LORA_FOR_CLASSIFICATION: bool = True  # Use LoRA weights for steps 7 & 8

    # --- Sub-directory Names (relative to OUTPUT_DIR) ---
    DIR_METADATA = "1_metadata"
    DIR_RAW_CODE = "2_selected_code"      # intermediate: selected code files and scores
    DIR_CHUNKS = "3_semantic_chunks"      # final merged semantic chunks
    DIR_VISUALIZATIONS = "4_visualizations"
    DIR_CHUNK_SUMMARIES = "5_chunk_summaries"
    # DIR_PROGRAM_SUMMARIES = "6_program_summaries"
    DIR_FINAL_IR = "7_final_ir"
    DIR_CLASSIFICATION = "8_classification"
    DIR_RATIONALE = "9_rationale"

    # --- Step Execution Switches ---
    RUN_STEP_1_METADATA: bool = True
    RUN_STEP_2_CODE_SELECTION: bool = True
    RUN_STEP_3_SEMANTIC_CHUNKING: bool = True
    RUN_STEP_4_VISUALIZATION: bool = True
    RUN_STEP_5_CHUNK_SUMMARIZATION: bool = True
    # RUN_STEP_6_PROGRAM_SUMMARIZATION: bool = True
    RUN_STEP_7_FINAL_IR: bool = True
    RUN_STEP_8_CLASSIFICATION: bool = True
    RUN_STEP_9_RATIONALE: bool = True

    # --- Overwrite behavior (checkpointing) ---
    OVERWRITE_EXISTING: bool = False
    SAVE_LLM_INPUTS: bool = False  # Save prompts sent to LLM for debugging

    # --- Ablation Switches ---
    RUN_ABLATION_E_META: bool = True      # Run classification without metadata
    RUN_ABLATION_E_SUMMARY: bool = True   # Run classification without program summary

    # --- Token Budgets ---
    # Maximum total tokens allowed for a single app (after filtering)
    MAX_APP_TOKENS: int = 150 * 1000

    # Maximum number of tokens per semantic block (in a single context fed to the LLM)
    MAX_CHUNK_TOKENS: int = 30 * 1000

    MAX_IR_BYTES = 16 * 1024  # Upper limit of soft size for IR (metadata + code_behaviors)

    # --- Heuristics ---
    # Blacklist of well-known SDKs (relative path prefix)
    SDK_BLACKLIST = (
        "android/", "androidx/", "android/support/",
        "com/google/", "com/android/",
        "com/facebook/", "com/twitter/", "com/instagram/", "com/linkedin/",
        "com/urbanairship/", "com/appsflyer/", "com/flurry/", "com/crashlytics/",
        "com/mixpanel/", "com/segment/", "io/branch/", "com/adjust/",
        "io/fabric/", "org/apache/", "org/json/", "org/xml/", "org/w3c/",
        "okhttp3/", "okio/", "retrofit2/", "com/squareup/", "com/gson/",
        "com/mopub/", "com/inmobi/", "com/admob/", "com/unity3d/",
        "com/amazon/", "com/microsoft/", "com/adobe/",
        "com/tencent/", "com/alibaba/", "com/baidu/",
        "org/osmdroid/", "org/chromium/", "net/hockeyapp/", "com/leanplum/", "org/acra/"
    )

    IGNORE_FILES = {"R.java", "BuildConfig.java", "Manifest.java"}

    # --- Prompts (Optimized for Balance & Brevity) ---

    # [Step 5] Chunk Summary
    PROMPT_CHUNK_SUMMARY = (
        "Analyze the following Android Java code fragment and briefly describe its key behaviors.\n"
        "Keep the answer very concise.\n\n"
        "--- CODE ---\n{code_chunk}"
    )

    PROMPT_CLASSIFICATION = (
        "You are an expert Android malware analyst. Your job is to classify an app based on its Metadata and Code Behaviors.\n\n"

        "### CLASSIFICATION LABELS\n"
        "Choose exactly one of the following based on the app's **primary intent** and **behavior**:\n\n"

        "1. **[RANSOMWARE]**\n"
        "   - **Core Behavior:** Maliciously locks the device or encrypts files to demand a ransom.\n"
        "   - **Key Signals:** Threatening language ('police', 'fbi', 'pay', 'bitcoin'), persistent locking mechanisms that prevent user exit, or file encryption without user consent.\n"
        "   - **Nuance:** Do NOT classify legitimate security/anti-theft apps (e.g., 'AntiVirus', 'Find My Phone') as Ransomware, even if they can lock the screen.\n\n"

        "2. **[SMSMALWARE]**\n"
        "   - **Core Behavior:** Abuses SMS permissions to cause financial loss or steal data.\n"
        "   - **Key Signals:** Sending SMS in the background (especially to shortcodes), intercepting incoming SMS (to hide billing notifications), or executing SMS commands from a C&C server.\n"
        "   - **Nuance:** If an app is a 'Game' or 'Flashlight' but requests `SEND_SMS`, it is likely SMSMalware. Legitimate SMS apps (e.g., 'Messenger') are [BENIGN].\n\n"

        "3. **[SCAREWARE]**\n"
        "   - **Core Behavior:** Uses social engineering to frighten the user into paying or downloading other software.\n"
        "   - **Key Signals:** Fake alerts about 'Viruses', 'Battery Issues', or 'System Errors'. Often mimics system dialogs or security tools but lacks actual functionality.\n"
        "   - **Nuance:** Often disguised as 'Cleaner', 'Battery Saver', or 'Booster' apps.\n\n"

        "4. **[ADWARE]**\n"
        "   - **Core Behavior:** Displays **intrusive** or **out-of-context** advertisements that disrupt the user experience.\n"
        "   - **Key Signals:** Ads appearing outside the app (notification bar ads, home screen shortcuts), aggressive ad libraries (e.g., Airpush, Leadbolt), or services dedicated solely to pushing ads.\n"
        "   - **Nuance:** Standard banner/interstitial ads displayed *inside* the app are acceptable commercial behavior and should be classified as [BENIGN].\n\n"

        "5. **[BENIGN]**\n"
        "   - **Core Behavior:** Safe applications, even if they are low quality or use aggressive monetization (as long as it stays within the app).\n"
        "   - **Key Signals:** Permissions match the app's described functionality. Uses standard analytics/ad SDKs (Google, Facebook, Unity) appropriately.\n\n"

        "### ANALYTICAL GUIDELINES\n"
        "- **Context is King:** Compare the `package` name and permissions. Does a 'Solitaire' game need to send SMS? No -> [SMSMALWARE]. Does a 'Security' app need to lock the screen? Yes -> [BENIGN].\n"
        "- **Avoid Over-flagging:** Do not classify an app as malware just because it has 'suspicious' permissions (like SYSTEM_ALERT_WINDOW) if the app type justifies it (e.g., a 'Tools' app).\n"
        "- **Prioritize Severity:** If an app exhibits multiple behaviors, choose the most severe category (Ransomware > SMSMalware > Scareware > Adware).\n\n"

        "### OUTPUT FORMAT\n"
        "Return ONLY the label in brackets, e.g., [BENIGN]."
        "\n\n"
        "--- IR ---\n{ir_json_block}"
    )

    # [Step 9] Rationale
    PROMPT_RATIONALE = (
        "You have classified this app as {label}.\n"
        "Provide a technical justification (max 150 words) citing specific chunks if possible.\n"
        "- **Evidence**: Which specific permissions or chunk behaviors led to this verdict?\n"
        "- **Differentiation**: Why is it {label} and not something else?\n\n"
        "--- IR ---\n{ir_json_block}"
    )

    # --- Logging ---
    LOG_FILE_NAME: str = "apk_sleuth.log"
    LOG_LEVEL = logging.INFO


# =========================
# Utilities
# =========================
def setup_logging():
    ExperimentConfig.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = ExperimentConfig.OUTPUT_DIR / ExperimentConfig.LOG_FILE_NAME
    logging.basicConfig(
        level=ExperimentConfig.LOG_LEVEL,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler()]
    )

def fix_long_path(path: Path) -> str:
    abs_path = path.resolve()
    return "\\\\?\\" + str(abs_path) if os.name == 'nt' else str(abs_path)

def read_text_safe(p: Path) -> str:
    try:
        with open(fix_long_path(p), "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception:
        return ""

def write_json(p: Path, obj: Dict[str, Any]):
    try:
        with open(fix_long_path(p), "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logging.error(f"Failed to write JSON {p}: {e}")

def read_json(p: Path) -> Dict[str, Any]:
    with open(fix_long_path(p), "r", encoding="utf-8", errors="ignore") as f:
        return json.load(f)

def out_dir(name: str) -> Path:
    p = ExperimentConfig.OUTPUT_DIR / name
    if not os.path.exists(fix_long_path(p)):
        os.makedirs(fix_long_path(p))
    return p

def get_out_paths(sample_hash: str) -> Dict[str, Path]:
    return {
        "metadata": out_dir(ExperimentConfig.DIR_METADATA) / f"{sample_hash}_meta.json",
        "raw_code": out_dir(ExperimentConfig.DIR_RAW_CODE) / f"{sample_hash}_code.json",
        "chunks": out_dir(ExperimentConfig.DIR_CHUNKS) / f"{sample_hash}_chunks.json",
        "chunk_summaries": out_dir(ExperimentConfig.DIR_CHUNK_SUMMARIES) / f"{sample_hash}_summary.json",
        # "program_summary": out_dir(ExperimentConfig.DIR_PROGRAM_SUMMARIES) / f"{sample_hash}_prog_sum.json",
        "ir": out_dir(ExperimentConfig.DIR_FINAL_IR) / f"{sample_hash}_ir.json",
        "classification": out_dir(ExperimentConfig.DIR_CLASSIFICATION) / f"{sample_hash}_class.json",
        "rationale": out_dir(ExperimentConfig.DIR_RATIONALE) / f"{sample_hash}_reason.json",
    }

def get_llm_input_path(json_path: Path) -> Path:
    return json_path.with_suffix('.prompt.txt')


_PRINTABLE_SET = set(string.printable)

def _is_probably_human_string(s: str) -> bool:
    """
    This function roughly determines whether a string is "readable text," and is used for:
    - Filtering out most obvious gibberish/random strings;
    - Only counting keyword_hits and suspicious_count on readable strings;
    - Selecting interesting strings for fallback.
    """
    if len(s) < 4:
        return False

    printable = sum((ch in _PRINTABLE_SET) for ch in s)
    if printable / len(s) < 0.95:
        return False

    if not any(ch.isalpha() for ch in s):
        return False

    good = sum(
        ch.isalnum() or ch in " .,:;_-/\\[]()"
        for ch in s
    )
    if good / len(s) < 0.6:
        return False

    return True


def _extract_top_strings(strings_path: Path, limit_bytes: int = 2048) -> Tuple[List[str], Dict[str, Any]]:
    """
    Read all strings from strings.txt, first calculate string_stats based on the "readable string set",
    then select a batch of representative interesting strings within the limit_bytes (default 2KB) limit.
    Returns: (interesting_strings, string_stats)
    """
    if not strings_path.exists():
        return [], {}

    try:
        with open(fix_long_path(strings_path), "r", encoding="utf-8", errors="ignore") as f:
            raw_lines = [line.strip() for line in f]
    except Exception:
        return [], {}

    unique_strings: List[str] = []
    seen = set()
    for s in raw_lines:
        if len(s) < 4:
            continue
        if s in seen:
            continue
        seen.add(s)
        unique_strings.append(s)

    human_strings: List[str] = [s for s in unique_strings if _is_probably_human_string(s)]
    stats_source = human_strings or unique_strings

    url_re = re.compile(r"https?://|www\.[A-Za-z0-9\-]+\.[A-Za-z]{2,}", re.IGNORECASE)
    ip_re = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
    base64_re = re.compile(r"^[A-Za-z0-9+/]{40,}={0,2}$")

    suspicious_keywords = [
        "sms", "mms", "text message",
        "pin", "password", "passwd", "bank", "card", "visa", "mastercard",
        "payment", "pay now", "subscribe", "subscription", "premium",
        "bitcoin", "btc", "wallet",
        "encrypt", "decryption", "decrypt", "cipher", "aes", "rsa", "key",
        "ransom", "locker",
        "device admin", "device_admin", "administrator",
        "virus", "infected", "malware", "trojan", "risk", "warning", "alert",
        "cleaner", "booster", "optimize", "optimizer", "junk", "battery",
    ]

    ad_sdk_keywords = [
        "airpush", "startapp", "leadbolt", "admob", "mopub",
        "inmobi", "applovin", "chartboost", "mobclix", "umeng",
    ]

    total_strings = len(raw_lines)
    unique_count = len(unique_strings)
    url_count = 0
    ip_count = 0
    has_long_base64 = False

    for s in stats_source:
        if url_re.search(s):
            url_count += 1
        if ip_re.search(s):
            ip_count += 1
        if len(s) >= 40 and base64_re.match(s):
            has_long_base64 = True

    suspicious_count = 0
    keyword_hits: Dict[str, int] = {kw: 0 for kw in suspicious_keywords}
    ad_sdk_hits: Dict[str, bool] = {sdk: False for sdk in ad_sdk_keywords}

    for s in human_strings:
        lower = s.lower()
        local_hit = False
        for kw in suspicious_keywords:
            if kw in lower:
                keyword_hits[kw] += 1
                local_hit = True
        if local_hit:
            suspicious_count += 1

        for sdk in ad_sdk_keywords:
            if sdk in lower:
                ad_sdk_hits[sdk] = True

    string_stats: Dict[str, Any] = {
        "total_strings": total_strings,
        "unique_strings": unique_count,
        "url_count": url_count,
        "ip_count": ip_count,
        "suspicious_count": suspicious_count,
        "keyword_hits": {k: v for k, v in keyword_hits.items() if v > 0},
        "ad_sdk_hits": ad_sdk_hits,
        "has_long_base64": has_long_base64,
    }

    candidates: List[str] = human_strings or unique_strings

    scored_strings: List[Tuple[int, str]] = []
    for s in candidates:
        if len(s) > 200:
            continue

        lower = s.lower()
        score = 0

        if url_re.search(s):
            score += 8
        if ip_re.search(s):
            score += 6

        if "sms" in lower or "mms" in lower:
            score += 5
        if any(kw in lower for kw in ["ransom", "encrypt", "decrypt", "locker"]):
            score += 5
        if any(w in lower for w in ["virus", "infected", "malware", "warning", "risk", "cleaner"]):
            score += 3

        if any(sdk in lower for sdk in ad_sdk_keywords):
            score += 4

        if "/" in s or "\\" in s:
            score += 2
        if any(s.endswith(ext) for ext in [".xml", ".XML", ".apk", ".APK", ".MF", ".SF", ".RSA"]):
            score += 2
        if "META-INF" in s:
            score += 3

        if len(s) > 40:
            score += 1

        if "android.com/apk" in s or "www.w3.org" in s or "adobe.com" in s:
            score -= 5

        if score > 0:
            scored_strings.append((score, s))

    scored_strings.sort(key=lambda x: (-x[0], x[1]))

    final_list: List[str] = []
    current_bytes = 0
    for _, s in scored_strings:
        if current_bytes + len(s) + 4 > limit_bytes:
            break
        final_list.append(s)
        current_bytes += len(s) + 4

    if not final_list:
        fallback_source = human_strings or unique_strings
        fallback_source_sorted = sorted(fallback_source, key=lambda x: (len(x), x))

        final_list = []
        current_bytes = 0
        for s in fallback_source_sorted:
            if current_bytes + len(s) + 4 > limit_bytes:
                break
            final_list.append(s)
            current_bytes += len(s) + 4

    return final_list, string_stats



def _clean_permission_name(perm: str) -> str:
    """Remove common prefixes to save tokens."""
    perm = perm.strip()
    if perm.startswith("android.permission."):
        return perm.replace("android.permission.", "")
    if perm.startswith("com.android.launcher.permission."):
        return perm.replace("com.android.launcher.permission.", "launcher.")
    if perm.startswith("com.google.android."):
        return perm.replace("com.google.android.", "google.")
    return perm

class ModelManager:
    def __init__(self):
        self.client: Optional[RemoteLLMClient] = None
        self.current_mode: Optional[str] = None

    def ensure_model(self, mode: str):
        """
        Ensure the correct model (Base or LoRA) is loaded.
        mode: 'base' or 'lora'
        """
        if self.client and self.current_mode == mode:
            return self.client

        if self.client:
            logging.info(f"Unloading previous model ({self.current_mode})...")
            self.client.unload()
            self.client = None
            self.current_mode = None
            time.sleep(2)  

        use_lora = (mode == 'lora')
        logging.info(f"Loading LLM (LoRA={use_lora})...")
        try:
            self.client = RemoteLLMClient(ExperimentConfig.REMOTE_LLM_BASE_URL, use_lora=use_lora)
            self.current_mode = mode
            logging.info("Model loaded successfully.")
        except Exception as e:
            logging.critical(f"Failed to load model: {e}")
            raise e

        return self.client

    def generate(self, prompt: str) -> str:
        if not self.client:
            raise RuntimeError("Model not initialized.")
        for i in range(3):
            try:
                return self.client.generate(prompt=prompt)
            except Exception as e:
                logging.warning(f"LLM gen failed ({i + 1}/3): {e}")
                time.sleep(2)
        return ""


# =========================
# Token Counter & LLM Helper
# =========================
class LocalTokenCounter:
    def __init__(self, model_path: str):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        except Exception:
            self.tokenizer = None
    def count(self, text: str) -> int:
        if self.tokenizer: return len(self.tokenizer.encode(text, add_special_tokens=False))
        return len(text) // 4

def ask_llm(llm: RemoteLLMClient, prompt: str, max_retries: int = 3) -> str:
    for i in range(max_retries):
        try:
            return llm.generate(prompt=prompt)
        except Exception as e:
            logging.warning(f"LLM request failed (attempt {i+1}): {e}")
            time.sleep(2 * (i + 1))
    return ""


# =========================
# Step 1: Metadata Extraction (Manifest)
# =========================
def step_1_extract_metadata(sample_hash: str, preproc_path: Path) -> Dict[str, Any]:
    """
    Parse AndroidManifest.xml and strings.txt to generate metadata.json.
    """
    manifest_path = preproc_path / "apktool_out" / "AndroidManifest.xml"
    if not manifest_path.exists():
        manifest_path = preproc_path / "AndroidManifest.xml"  # fallback

    strings_path = preproc_path / "strings.txt"

    ns = "{http://schemas.android.com/apk/res/android}"

    meta: Dict[str, Any] = {
        "hash": sample_hash,
        "package_name": "",
        "permissions": [],
        "components": {
            "activities": [],
            "services": [],
            "receivers": [],
            "providers": [],
        },
        "interesting_strings": [],
        "string_stats": {},
        "intent_actions": {
            "receiver_actions": [],
            "service_actions": [],
            "activity_actions": [],
        },
        "manifest_signals": {
            "has_launcher_activity": False,
            "has_boot_receiver": False,
            "has_sms_receiver": False,
            "has_phone_state_listener": False,
            "exported_services_count": 0,
            "exported_receivers_count": 0,
            "has_device_admin": False,
        },
        "valid_manifest": False,
    }

    if manifest_path.exists():
        try:
            tree = ET.parse(fix_long_path(manifest_path))
            root = tree.getroot()

            meta["package_name"] = root.attrib.get("package", "")
            meta["valid_manifest"] = True

            perms: List[str] = []
            for perm in root.findall("uses-permission"):
                name = perm.attrib.get(f"{ns}name")
                if name:
                    perms.append(name)

            meta["permissions"] = sorted(set(perms))

            app_node = root.find("application")
            if app_node is not None:
                ia = meta["intent_actions"]
                signals = meta["manifest_signals"]

                def _normalize_name(name: str) -> str:
                    if name.startswith("."):
                        return meta["package_name"] + name
                    if "." not in name:
                        return meta["package_name"] + "." + name
                    return name

                for tag, key in [
                    ("activity", "activities"),
                    ("service", "services"),
                    ("receiver", "receivers"),
                    ("provider", "providers"),
                ]:
                    for item in app_node.findall(tag):
                        name = item.attrib.get(f"{ns}name")
                        if not name:
                            continue
                        full_name = _normalize_name(name)
                        meta["components"][key].append(full_name)

                        exported = item.attrib.get(f"{ns}exported")
                        if exported and exported.lower() == "true":
                            if key == "services":
                                signals["exported_services_count"] += 1
                            elif key == "receivers":
                                signals["exported_receivers_count"] += 1

                        for if_node in item.findall("intent-filter"):
                            actions: List[str] = []
                            for act in if_node.findall("action"):
                                aname = act.attrib.get(f"{ns}name")
                                if not aname:
                                    continue
                                actions.append(aname)

                                if tag == "receiver":
                                    ia["receiver_actions"].append(aname)
                                    if aname == "android.provider.Telephony.SMS_RECEIVED":
                                        signals["has_sms_receiver"] = True
                                    elif aname == "android.intent.action.BOOT_COMPLETED":
                                        signals["has_boot_receiver"] = True
                                    elif aname == "android.intent.action.PHONE_STATE":
                                        signals["has_phone_state_listener"] = True
                                elif tag == "service":
                                    ia["service_actions"].append(aname)
                                elif tag == "activity":
                                    ia["activity_actions"].append(aname)

                            cats = [
                                c.attrib.get(f"{ns}name")
                                for c in if_node.findall("category")
                                if c.attrib.get(f"{ns}name")
                            ]
                            if (
                                tag == "activity"
                                and "android.intent.action.MAIN" in actions
                                and "android.intent.category.LAUNCHER" in cats
                            ):
                                signals["has_launcher_activity"] = True

                for key in ["activities", "services", "receivers", "providers"]:
                    meta["components"][key] = sorted(meta["components"][key])


        except Exception as e:
            logging.error(f"[{sample_hash}] Manifest parse error: {e}")

    if any(p.endswith("BIND_DEVICE_ADMIN") for p in meta["permissions"]):
        meta["manifest_signals"]["has_device_admin"] = True

    if strings_path.exists():
        top_strings, string_stats = _extract_top_strings(strings_path, limit_bytes=2048)
        meta["interesting_strings"] = top_strings
        meta["string_stats"] = string_stats

    return meta



# =========================
# Step 2: Code Selection (Filtering & Prioritization)
# =========================
def calculate_priority(file_info: Dict, meta: Dict) -> int:
    """
    Assign a priority score to a Java file. Higher is better.
    Tier 1 (Score 3): Component Implementation (Service/Receiver).
    Tier 2 (Score 2): Sensitive logic keywords (Crypt, DB, Net).
    Tier 3 (Score 1): Standard Logic / Activity.
    Tier 4 (Score 0): UI / Layout / View / Adapter.
    """
    rel_path = file_info['rel_path']
    content = file_info['content']
    name_lower = Path(rel_path).name.lower()

    # 1. Component Check (Highest Priority for Background Tasks)
    # Convert path to class-like string: com/example/Foo.java -> com.example.Foo
    class_guess = rel_path.replace('/', '.').replace('\\', '.')[:-5]

    # Direct match with Manifest Services/Receivers
    for svc in meta['components']['services']:
        if svc in class_guess: return 30
    for rcv in meta['components']['receivers']:
        if rcv in class_guess: return 30

    # 2. Keyword Heuristics
    sensitive_keywords = ['crypt', 'aes', 'cipher', 'http', 'socket', 'sms', 'telephony', 'database', 'sql', 'root',
                          'su']
    if any(k in name_lower for k in sensitive_keywords):
        return 20

    # 3. Activity (Medium)
    for act in meta['components']['activities']:
        if act in class_guess: return 10
    if "activity" in name_lower: return 10

    # 4. UI Noise (Lowest)
    ui_keywords = ['adapter', 'view', 'layout', 'fragment', 'ui', 'animation', 'drawer', 'widget']
    if any(k in name_lower for k in ui_keywords):
        return 0

    return 5  # Default logic


def step_2_select_files(sample_hash: str, preproc_path: Path, meta: Dict, tc: LocalTokenCounter) -> Dict[str, Any]:
    java_root = preproc_path / "java_src" / "sources"
    if not java_root.exists():
        java_root = preproc_path / "java_src"  # fallback

    if not java_root.exists():
        return {"files": [], "truncated": False}

    candidates = []
    seen_paths = set()

    # 1. Scan and Filter
    # To optimize, we only scan directories that 'might' contain relevant code
    # But for now, scanning all and filtering by blacklist is safer.
    for f in java_root.rglob("*.java"):
        if f.name in ExperimentConfig.IGNORE_FILES: continue

        try:
            rel_path = f.relative_to(java_root).as_posix()  # Use forward slash
        except ValueError:
            continue

        # Blacklist Filter
        if rel_path.startswith(ExperimentConfig.SDK_BLACKLIST):
            continue

        abs_path = f.resolve()
        if abs_path in seen_paths: continue
        seen_paths.add(abs_path)

        content = read_text_safe(f)
        if len(content) < 50: continue  # Skip empty/tiny files

        # Basic stats
        tokens = tc.count(content)

        item = {
            "rel_path": rel_path,
            "content": content,
            "tokens": tokens
        }
        # Calculate priority
        item["score"] = calculate_priority(item, meta)
        candidates.append(item)

    # 2. Sort by Priority (High Score -> Low Score), then by Path length (shorter path usually more root)
    candidates.sort(key=lambda x: (-x["score"], len(x["rel_path"])))

    # 3. Budget Selection
    selected = []
    current_tokens = 0
    truncated = False

    for item in candidates:
        if current_tokens + item["tokens"] > ExperimentConfig.MAX_APP_TOKENS:
            truncated = True
            # If it's a high priority file, maybe we try to squeeze it?
            # For now, strict cut-off to save time.
            continue

        selected.append(item)
        current_tokens += item["tokens"]

    logging.info(
        f"[{sample_hash}] Selected {len(selected)} files. Total Tokens: {current_tokens}. Truncated: {truncated}")

    return {
        "files": selected,
        "total_tokens": current_tokens,
        "truncated": truncated,
        "total_scanned": len(candidates)
    }


# =========================
# Step 3: Semantic Chunking (Dependency Graph)
# =========================
def extract_class_name(rel_path: str) -> str:
    # com/example/A.java -> com.example.A
    return rel_path.replace('/', '.').replace('\\', '.')[:-5]


def parse_dependencies(files: List[Dict]) -> nx.Graph:
    """
    Build a graph where nodes are files (indices) and edges represent dependencies.
    """
    G = nx.Graph()

    # Map: ClassName -> FileIndex
    class_to_idx = {}
    for idx, f in enumerate(files):
        cname = extract_class_name(f['rel_path'])
        class_to_idx[cname] = idx
        G.add_node(idx, tokens=f['tokens'])

    # Regex for imports and simple new Class() usage
    # import com.example.Utils;
    import_re = re.compile(r'import\s+([\w\.]+);')

    for idx, f in enumerate(files):
        content = f['content']

        # Find imports
        for m in import_re.finditer(content):
            imported_class = m.group(1)
            if imported_class in class_to_idx:
                target_idx = class_to_idx[imported_class]
                if target_idx != idx:
                    G.add_edge(idx, target_idx)

        # (Optional) Scan for same-package references if needed,
        # but import scan is usually sufficient for Java cross-file deps.

    return G


# [Optimized] Step 3: Semantic Chunking with Graph Partitioning
# [Optimized v2] Step 3: Global Semantic Bin Packing
# [Fixed] Step 3: Topological Semantic Chunking

def get_subgraph_tokens(G, nodes):
    return sum(G.nodes[n]['tokens'] for n in nodes)


def split_large_cluster(G_sub: nx.Graph, max_tokens: int) -> List[List[int]]:
    """
    Splits a large graph into atomic communities using Louvain.
    Ensures no single atom is larger than max_tokens (recursively).
    """
    current_total = get_subgraph_tokens(G_sub, G_sub.nodes())

    # Base case: Fits in one chunk
    if current_total <= max_tokens:
        return [list(G_sub.nodes())]

    # Edge case: Single huge file
    if G_sub.number_of_nodes() == 1:
        return [list(G_sub.nodes())]

    # 1. Detect Communities
    try:
        communities = nx.community.louvain_communities(G_sub, seed=42)
    except Exception:
        # Fallback: Connected components if graph became disconnected, or BFS split
        if not nx.is_connected(G_sub):
            communities = list(nx.connected_components(G_sub))
        else:
            # BFS Partition logic (simplified)
            nodes = list(G_sub.nodes())
            mid = len(nodes) // 2
            communities = [set(nodes[:mid]), set(nodes[mid:])]

    # 2. Recursive Check
    final_atoms = []
    for comm in communities:
        sub_G = G_sub.subgraph(comm)
        # If this community is still too big, split it further
        if get_subgraph_tokens(sub_G, comm) > max_tokens:
            final_atoms.extend(split_large_cluster(sub_G, max_tokens))
        else:
            final_atoms.append(list(comm))

    return final_atoms


def pack_atoms_topologically(
        G_original: nx.Graph,
        atoms: List[List[int]],
        files_info: List[Dict],
        max_tokens: int
) -> List[Dict]:
    """
    Recombines atoms into chunks based on their connectivity.
    Prioritizes merging adjacent atoms to maintain semantic flow.
    """
    # 1. Build Meta-Graph (Nodes = Atom Indices)
    # Map each file_node to its atom_index
    node_to_atom = {}
    for atom_idx, atom_nodes in enumerate(atoms):
        for n in atom_nodes:
            node_to_atom[n] = atom_idx

    MetaG = nx.Graph()
    for atom_idx in range(len(atoms)):
        # Calculate priority score for the atom (Sum of file scores)
        score = sum(files_info[n].get('score', 0) for n in atoms[atom_idx])
        size = sum(files_info[n]['tokens'] for n in atoms[atom_idx])
        MetaG.add_node(atom_idx, size=size, score=score, file_nodes=atoms[atom_idx])

    # Add edges between Atoms if their internal files are connected
    # Iterate original edges
    for u, v in G_original.edges():
        if u in node_to_atom and v in node_to_atom:
            a1, a2 = node_to_atom[u], node_to_atom[v]
            if a1 != a2:
                MetaG.add_edge(a1, a2)

    # 2. Topological Bin Packing (Greedy Traversal)
    chunks = []
    processed_atoms = set()

    # Get all atoms sorted by Priority Score (High risk first)
    sorted_atoms = sorted(MetaG.nodes(), key=lambda n: MetaG.nodes[n]['score'], reverse=True)

    for start_atom in sorted_atoms:
        if start_atom in processed_atoms:
            continue

        # Start a new Chunk
        current_chunk_atoms = [start_atom]
        current_tokens = MetaG.nodes[start_atom]['size']
        processed_atoms.add(start_atom)

        # Try to grow this chunk by absorbing neighbors
        # We use a frontier queue: neighbors of the current chunk
        while True:
            # Find potential candidates: neighbors of current_chunk_atoms that are not processed
            candidates = set()
            for ca in current_chunk_atoms:
                for nbr in MetaG.neighbors(ca):
                    if nbr not in processed_atoms:
                        candidates.add(nbr)

            if not candidates:
                break  # No connected neighbors left

            # Heuristic: Pick the neighbor that fits and has highest priority (or strongest connection)
            # For simplicity: Pick largest priority, then largest size that fits
            best_candidate = -1

            # Sort candidates
            sorted_candidates = sorted(list(candidates),
                                       key=lambda x: (MetaG.nodes[x]['score'], MetaG.nodes[x]['size']),
                                       reverse=True)

            found = False
            for cand in sorted_candidates:
                cand_size = MetaG.nodes[cand]['size']
                if current_tokens + cand_size <= max_tokens:
                    # Accept
                    processed_atoms.add(cand)
                    current_chunk_atoms.append(cand)
                    current_tokens += cand_size
                    found = True
                    break  # Restart loop to update frontier with new neighbor's neighbors

            if not found:
                break  # Cannot fit any neighbors

        # Build chunk object
        c_files = []
        c_content = ""
        c_tokens = 0

        for atom_idx in current_chunk_atoms:
            atom_file_indices = MetaG.nodes[atom_idx]['file_nodes']
            for f_idx in atom_file_indices:
                f = files_info[f_idx]
                c_files.append(f["rel_path"])
                c_content += f"\n// File: {f['rel_path']}\n{f['content']}\n"
                c_tokens += f["tokens"]

        chunks.append({
            "files": c_files,
            "tokens": c_tokens,
            "content": c_content,
            # "atoms": current_chunk_atoms # debug info
        })

    return chunks


def step_3_semantic_chunking(selected_data: Dict[str, Any]) -> Dict[str, Any]:
    files = selected_data.get("files", [])
    if not files:
        return {"chunks": []}

    # 1. Build Dependency Graph
    G = parse_dependencies(files)

    # 2. Identify Connected Components (Islands)
    raw_components = [G.subgraph(c).copy() for c in nx.connected_components(G)]

    final_chunks = []

    # 3. Process each Component separately (The Fix)
    # We do NOT mix components yet. We ensure internal cohesion first.

    for comp_G in raw_components:
        comp_tokens = get_subgraph_tokens(comp_G, comp_G.nodes())

        if comp_tokens <= ExperimentConfig.MAX_CHUNK_TOKENS:
            # Case A: Perfect fit. Keep the island intact.
            c_files = []
            c_content = ""
            for idx in comp_G.nodes():
                f = files[idx]
                c_files.append(f["rel_path"])
                c_content += f"\n// File: {f['rel_path']}\n{f['content']}\n"

            final_chunks.append({
                "files": c_files,
                "tokens": comp_tokens,
                "content": c_content
            })
        else:
            # Case B: Too big. Split into Atoms -> Re-aggregate Topologically.
            # This ensures we don't lose connectivity info during the split.
            atoms = split_large_cluster(comp_G, ExperimentConfig.MAX_CHUNK_TOKENS)

            # Pack these atoms back into chunks, prioritizing adjacency
            comp_chunks = pack_atoms_topologically(comp_G, atoms, files, ExperimentConfig.MAX_CHUNK_TOKENS)
            final_chunks.extend(comp_chunks)

    # 4. (Optional) Cleanup Phase: Merge tiny unrelated chunks
    # If we have many small chunks (e.g. < 10k tokens) that are unconnected,
    # we can merge them to save LLM calls, but we must mark them as unrelated.
    # For APK-Sleuth, we perform a simple 'Best Fit' merge on the results.

    optimized_chunks = []
    # Sort chunks by size descending to handle big ones first
    final_chunks.sort(key=lambda c: c['tokens'], reverse=True)

    # Simple bin packing for the resulting chunks
    for chunk in final_chunks:
        placed = False
        # Try to fit into an existing optimized chunk if it has space
        # AND if the current chunk is small (don't merge two 20k chunks)
        if chunk['tokens'] < (ExperimentConfig.MAX_CHUNK_TOKENS * 0.5):
            for target in optimized_chunks:
                if target['tokens'] + chunk['tokens'] <= ExperimentConfig.MAX_CHUNK_TOKENS:
                    # Merge
                    target['files'].extend(chunk['files'])
                    target['content'] += "\n\n// --- SEPARATOR: Unrelated Component ---\n\n" + chunk['content']
                    target['tokens'] += chunk['tokens']
                    placed = True
                    break

        if not placed:
            optimized_chunks.append(chunk)

    # Assign IDs
    for i, c in enumerate(optimized_chunks):
        c['id'] = i

    return {
        "chunks": optimized_chunks,
        "total_chunks": len(optimized_chunks)
    }


# =========================
# Step 4: Visualization
# =========================
def render_visualization_html(sample_hash: str, chunks_data: Dict[str, Any]) -> str:
    # 1. Graph Data Reconstruction (Same as before)
    nodes = []
    class_to_path = {}

    # Pass 1: Build Nodes
    for chunk in chunks_data.get("chunks", []):
        chunk_content = chunk.get("content", "")
        raw_parts = re.split(r'\n// File: (.+)\n', chunk_content)

        for i in range(1, len(raw_parts), 2):
            fname = raw_parts[i]
            fcontent = raw_parts[i + 1] if i + 1 < len(raw_parts) else ""
            cname = fname.replace('/', '.').replace('\\', '.')[:-5]
            class_to_path[cname] = fname

            nodes.append({
                "id": fname,
                "name": Path(fname).name,
                "group": chunk["id"],
                "tokens": len(fcontent) // 4,  # Approx
                "content_snippet": fcontent
            })

    # Pass 2: Build Edges
    import_re = re.compile(r'import\s+([\w\.]+);')
    node_indices = {n["id"]: i for i, n in enumerate(nodes)}
    links = []

    for i, node in enumerate(nodes):
        content = node.pop("content_snippet")
        seen_imports = set()
        for m in import_re.finditer(content):
            imp_cls = m.group(1)
            if imp_cls in class_to_path:
                target_path = class_to_path[imp_cls]
                if target_path != node["id"] and target_path in node_indices:
                    target_idx = node_indices[target_path]
                    if target_idx not in seen_imports:
                        links.append({"source": i, "target": target_idx})
                        seen_imports.add(target_idx)

    nodes_json = json.dumps(nodes)
    links_json = json.dumps(links)

    # 2. New HTML Template (White Theme & Improved Labeling)
    html_template = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<title>APK-Sleuth: {sample_hash}</title>
<style>
  :root {{
    --bg-color: #ffffff;
    --text-color: #333333;
    --line-color: #999999;
    --node-stroke: #666666;
    --panel-bg: rgba(245, 245, 245, 0.9);
    --panel-border: #dddddd;
  }}
  body {{ margin: 0; background: var(--bg-color); color: var(--text-color); font-family: "Segoe UI", Roboto, Helvetica, Arial, sans-serif; overflow: hidden; }}
  #graph {{ width: 100vw; height: 100vh; }}

  .node circle {{ stroke: var(--node-stroke); stroke-width: 1px; stroke-opacity: 0.5; transition: all 0.2s; }}
  .link {{ stroke: var(--line-color); stroke-opacity: 0.6; stroke-width: 1.2px; }}

  .label {{
    font-size: 12px; 
    fill: var(--text-color); 
    font-weight: 500;
    pointer-events: none;
    /* White Halo for readability over lines */
    paint-order: stroke;
    stroke: #ffffff;
    stroke-width: 3px;
    stroke-linejoin: round;
    transition: opacity 0.3s ease;
  }}

  /* Dimmed class for decluttering - drastically reduce opacity */
  .label.dimmed {{ opacity: 0.05; }}

  .tooltip {{
    position: absolute; background: rgba(255,255,255,0.95); color: #333; 
    padding: 10px; border-radius: 6px; pointer-events: none; font-size: 12px;
    border: 1px solid #ccc; box-shadow: 0 4px 15px rgba(0,0,0,0.15);
    max-width: 300px; word-wrap: break-word;
  }}

  #controls {{
    position: absolute; top: 20px; right: 20px; 
    background: var(--panel-bg); border: 1px solid var(--panel-border);
    padding: 15px; border-radius: 8px; backdrop-filter: blur(10px);
    max-width: 260px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
  }}
  .control-row {{ margin-bottom: 8px; display: flex; align-items: center; font-size: 13px; }}
  .control-row input {{ margin-right: 10px; }}
  h3 {{ margin: 0 0 10px 0; font-size: 15px; font-weight: 600; border-bottom: 1px solid #ccc; padding-bottom: 8px; color: #111; }}

  .legend-scroll {{ max-height: 300px; overflow-y: auto; margin-top: 10px; border-top: 1px solid #ccc; padding-top: 10px; }}
  .legend-item {{ display: flex; align-items: center; margin-top: 4px; font-size: 12px; }}
  .swatch {{ width: 12px; height: 12px; border-radius: 3px; margin-right: 8px; border: 1px solid rgba(0,0,0,0.1); }}

  /* Scrollbar styling */
  ::-webkit-scrollbar {{ width: 6px; }}
  ::-webkit-scrollbar-thumb {{ background: #ccc; border-radius: 3px; }}
</style>
</head>
<body>
<div id="graph"></div>
<div id="controls">
  <h3>{sample_hash[:8]}...</h3>
  <div class="control-row"><label><input type="checkbox" id="toggleLabels" checked> Show Labels</label></div>
  <div class="control-row"><label><input type="checkbox" id="toggleFreeze"> Freeze Layout</label></div>
  <div class="control-row" style="opacity:0.6; font-size:11px;">Drag nodes to fix position.<br>Hover for info.</div>
  <div id="legend" class="legend-scroll"></div>
</div>
<div id="tooltip" class="tooltip" style="opacity:0;"></div>

<script src="https://d3js.org/d3.v7.min.js"></script>
<script>
const nodes = {nodes_json};
const links = {links_json};

const width = window.innerWidth;
const height = window.innerHeight;

// Vibrant Academic Palette
const color = d3.scaleOrdinal(d3.schemeTableau10);

const svg = d3.select("#graph").append("svg")
    .attr("width", width).attr("height", height)
    .call(d3.zoom().scaleExtent([0.1, 8]).on("zoom", (event) => {{
        container.attr("transform", event.transform);
        currentTransform = event.transform;
        requestAnimationFrame(updateLabels); // Smooth update
    }}));

const container = svg.append("g");
let currentTransform = d3.zoomIdentity;
let isFrozen = false;

// Force Simulation - Tuned for rigidity
const simulation = d3.forceSimulation(nodes)
    .force("link", d3.forceLink(links).id(d => d.index).distance(60).strength(0.4))
    .force("charge", d3.forceManyBody().strength(-300))
    .force("center", d3.forceCenter(width / 2, height / 2))
    .force("collide", d3.forceCollide().radius(d => getNodeRadius(d) + 2).iterations(2));

// Draw Elements
const link = container.append("g").attr("class", "links")
    .selectAll("line").data(links).enter().append("line")
    .attr("class", "link");

const node = container.append("g").attr("class", "nodes")
    .selectAll("circle").data(nodes).enter().append("circle")
    .attr("r", d => getNodeRadius(d))
    .attr("fill", d => color(d.group))
    .call(d3.drag()
        .on("start", dragstarted)
        .on("drag", dragged)
        .on("end", dragended));

const label = container.append("g").attr("class", "labels")
    .selectAll("text").data(nodes).enter().append("text")
    .attr("class", "label")
    .text(d => d.name)
    .attr("dy", -10) // Shift above node
    .attr("text-anchor", "middle");

function getNodeRadius(d) {{
    // Logarithmic scale for better visualization of huge variances
    return Math.max(5, Math.log(d.tokens + 1) * 1.5);
}}

// Tooltip Logic
const tooltip = d3.select("#tooltip");
node.on("mouseover", (e, d) => {{
    tooltip.transition().duration(100).style("opacity", 1);
    tooltip.html(`<b>${{d.name}}</b><br>Path: ${{d.id}}<br>Chunk ID: <b>${{d.group}}</b><br>Size: ${{d.tokens}} tokens`)
           .style("left", (e.pageX + 15) + "px")
           .style("top", (e.pageY + 15) + "px");

    // Highlight connections
    link.style("stroke", l => (l.source === d || l.target === d) ? "#d62728" : null)
        .style("stroke-opacity", l => (l.source === d || l.target === d) ? 1 : 0.1)
        .style("stroke-width", l => (l.source === d || l.target === d) ? 2 : 1);

    label.filter(l => l === d).style("opacity", 1).classed("dimmed", false);

}}).on("mouseout", () => {{
    tooltip.transition().duration(300).style("opacity", 0);
    link.style("stroke", null).style("stroke-opacity", null).style("stroke-width", null);
    updateLabels(); // Restore declutter state
}});

// Simulation Tick
simulation.on("tick", () => {{
    link.attr("x1", d => d.source.x).attr("y1", d => d.source.y)
        .attr("x2", d => d.target.x).attr("y2", d => d.target.y);
    node.attr("cx", d => d.x).attr("cy", d => d.y);
    label.attr("x", d => d.x).attr("y", d => d.y);

    // Throttled label update
    if (simulation.alpha() < 0.1 || isFrozen) updateLabels(); 
}});

// --- Improved Decluttering Logic (Bounding Box Collision) ---
function updateLabels() {{
    if (!document.getElementById('toggleLabels').checked) {{
        label.style("opacity", 0);
        return;
    }}

    // Sort nodes by importance (tokens) so bigger nodes keep their labels
    // We process labels in importance order.
    const sortedIndices = d3.range(nodes.length).sort((a, b) => nodes[b].tokens - nodes[a].tokens);

    const occupied = []; // Store bounding boxes of visible labels
    const k = currentTransform.k;

    // Char width approx (Segoe UI 12px approx 7px width, 14px height)
    const charW = 7;
    const charH = 14;
    const padding = 4;

    label.classed("dimmed", (d, i) => {{
        // Map actual index to sorted rank, process high rank first
        // This loop structure is slightly complex in d3, so we iterate sortedIndices manually below
        return false; // Reset first
    }});

    // Manual iteration to apply class
    const isVisible = new Array(nodes.length).fill(false);

    sortedIndices.forEach(idx => {{
        const d = nodes[idx];
        const x = d.x * k + currentTransform.x;
        const y = d.y * k + currentTransform.y;

        // Estimate box in screen coordinates
        const w = (d.name.length * charW);
        const h = charH;

        // Box center is at (x, y-10*k) approx
        const box = {{
            l: x - w/2 - padding,
            r: x + w/2 + padding,
            t: y - 10*k - h - padding,
            b: y - 10*k + padding
        }};

        // Check collision
        let overlaps = false;
        for (const ob of occupied) {{
            if (box.l < ob.r && box.r > ob.l && box.t < ob.b && box.b > ob.t) {{
                overlaps = true;
                break;
            }}
        }}

        if (!overlaps) {{
            occupied.push(box);
            isVisible[idx] = true;
        }}
    }});

    label.classed("dimmed", (d, i) => !isVisible[i]);
    label.style("opacity", null); // Remove any manual overrides
}}

// Drag & Controls
function dragstarted(event, d) {{
    if (!event.active && !isFrozen) simulation.alphaTarget(0.3).restart();
    d.fx = d.x; d.fy = d.y;
}}
function dragged(event, d) {{ d.fx = event.x; d.fy = event.y; }}
function dragended(event, d) {{
    if (!event.active && !isFrozen) simulation.alphaTarget(0);
    if (!isFrozen) {{ d.fx = null; d.fy = null; }}
}}

document.getElementById('toggleLabels').addEventListener('change', updateLabels);
document.getElementById('toggleFreeze').addEventListener('change', (e) => {{
    isFrozen = e.target.checked;
    if (isFrozen) simulation.stop();
    else simulation.alpha(0.5).restart();
}});

// Legend
const legendDiv = d3.select("#legend");
const groups = [...new Set(nodes.map(d => d.group))].sort((a,b)=>a-b);
groups.forEach(g => {{
    const row = legendDiv.append("div").attr("class", "legend-item");
    row.append("div").attr("class", "swatch").style("background", color(g));
    row.append("span").text("Chunk " + g);
}});

// Initial update
setTimeout(updateLabels, 500);

</script>
</body>
</html>"""
    return html_template


def step_4_visualize(sample_hash: str, paths: Dict[str, Path]):
    """
    Generate visual representation of the semantic chunks.
    """
    chunk_path = paths["chunks"]
    if not chunk_path.exists():
        logging.warning(f"[{sample_hash}] No chunks found for visualization.")
        return

    try:
        chunks_data = read_json(chunk_path)
        html_content = render_visualization_html(sample_hash, chunks_data)

        out_file = out_dir(ExperimentConfig.DIR_VISUALIZATIONS) / f"{sample_hash}_viz.html"
        with open(fix_long_path(out_file), "w", encoding="utf-8") as f:
            f.write(html_content)

    except Exception as e:
        logging.error(f"[{sample_hash}] Visualization failed: {e}")


# =========================
# Step 5: Chunk Summarization
# =========================
def step_5_process_all(samples: List[Dict], model_mgr: ModelManager):
    logging.info(">>> Step 5: Chunk Summarization (Mode: BASE)")
    model_mgr.ensure_model('base')

    for sample in tqdm(samples, desc="Step 5"):
        h = sample['hash']
        paths = get_out_paths(h)

        if not ExperimentConfig.OVERWRITE_EXISTING and paths["chunk_summaries"].exists(): continue
        if not paths["chunks"].exists(): continue

        chunks_data = read_json(paths["chunks"])
        summaries = []
        prompts_log = []

        for chunk in chunks_data.get("chunks", []):
            content = chunk.get("content", "")
            if not content.strip(): continue

            prompt = ExperimentConfig.PROMPT_CHUNK_SUMMARY.format(code_chunk=content)
            prompts_log.append(f"### Chunk {chunk['id']}\n{prompt}\n")

            out_text = model_mgr.generate(prompt)
            summaries.append({
                "chunk_id": chunk["id"],
                "tokens": chunk.get("tokens", 0),
                "summary": out_text.strip()
            })

        write_json(paths["chunk_summaries"], {"hash": h, "chunk_summaries": summaries})
        if ExperimentConfig.SAVE_LLM_INPUTS:
            try:
                with open(fix_long_path(get_llm_input_path(paths["chunk_summaries"])), "w", encoding="utf-8") as f:
                    f.write("\n".join(prompts_log))
            except:
                pass


# =========================
# Step 7: Construct Final IR (CPU) - NOW CONSUMES STEP 5 DIRECTLY
# =========================
def _strip_intent_name(action: str) -> str:
    """
    Remove common Android Intent prefixes and use shorter names in IR.
    Use only in IR, without changing the original values ​​in metadata.json.
    """
    if not isinstance(action, str):
        return ""
    if action.startswith("android.intent.action."):
        return action[len("android.intent.action.") :]
    if action.startswith("android.provider.Telephony."):
        return action[len("android.provider.Telephony.") :]
    return action.split(".")[-1]


def _build_permission_group_counts(perms: List[str]) -> Dict[str, int]:
    """
    Based on a simple categorization of full permission names, a coarse-grained permission statistics are provided.
    """
    groups = {
        "sms": 0,
        "device_admin": 0,
        "storage": 0,
        "phone": 0,
        "location": 0,
    }
    for p in perms:
        up = p.upper()
        if "SMS" in up:
            groups["sms"] += 1
        if "BIND_DEVICE_ADMIN" in up:
            groups["device_admin"] += 1
        if "WRITE_EXTERNAL_STORAGE" in up or "READ_EXTERNAL_STORAGE" in up:
            groups["storage"] += 1
        if "READ_PHONE_STATE" in up or "CALL_PHONE" in up or "READ_CALL_LOG" in up:
            groups["phone"] += 1
        if "ACCESS_FINE_LOCATION" in up or "ACCESS_COARSE_LOCATION" in up or "ACCESS_BACKGROUND_LOCATION" in up:
            groups["location"] += 1
    return groups


def _build_metadata_for_ir(meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert the information (Manifest + strings) in metadata.json into the metadata structure used in IR.
    Note: Do not rely on any code digest information here; ensure that the two parts are independent during ablation experiments.
    """
    components = meta.get("components", {})
    acts = components.get("activities", []) or []
    srvs = components.get("services", []) or []
    rcvs = components.get("receivers", []) or []
    provs = components.get("providers", []) or []

    intent_actions = meta.get("intent_actions", {})
    recv_full = intent_actions.get("receiver_actions", []) or []
    srv_full = intent_actions.get("service_actions", []) or []
    act_full = intent_actions.get("activity_actions", []) or []

    recv_short = [_strip_intent_name(a) for a in recv_full]
    srv_short = [_strip_intent_name(a) for a in srv_full]
    act_short = [_strip_intent_name(a) for a in act_full]

    action_counts: Dict[str, int] = {}
    for name in recv_short + srv_short + act_short:
        if not name:
            continue
        action_counts[name] = action_counts.get(name, 0) + 1

    perms = meta.get("permissions", []) or []
    group_counts = _build_permission_group_counts(perms)

    string_stats = meta.get("string_stats", {}) or {}
    interesting_strings = meta.get("interesting_strings", []) or []
    ad_sdk_hits_map = string_stats.get("ad_sdk_hits", {}) or {}
    sdk_hits = [sdk for sdk, hit in ad_sdk_hits_map.items() if hit]

    manifest_signals = meta.get("manifest_signals", {}) or {}

    metadata_ir: Dict[str, Any] = {
        "app_info": {
            "hash": meta.get("hash"),
            "package_name": meta.get("package_name"),
            "valid_manifest": meta.get("valid_manifest", True),
        },
        "components": {
            "counts": {
                "activities": len(acts),
                "services": len(srvs),
                "receivers": len(rcvs),
                "providers": len(provs),
            },
            "activities": acts,
            "services": srvs,
            "receivers": rcvs,
            "providers": provs,
        },
        "intents": {
            "receiver_actions": recv_short,
            "service_actions": srv_short,
            "activity_actions": act_short,
            "action_counts": action_counts,
        },
        "permissions": {
            "list": perms,
            "group_counts": group_counts,
        },
        "strings": {
            "interesting": interesting_strings,
            "string_stats": string_stats,
            "sdk_hits": sdk_hits,
        },
        "manifest_signals": manifest_signals,
    }

    return metadata_ir


def _shrink_metadata_to_fit(ir_obj: Dict[str, Any], max_bytes: int = ExperimentConfig.MAX_IR_BYTES) -> None:
    """
    Compress IR only by modifying ir_obj['metadata'],
    until the length of json.dumps(ir_obj) is <= max_bytes, or if compression becomes impossible.
    - Strictly avoid modifying code_behaviors.
    - Gradually trim components/intents/strings in stages.
    """

    def _size(o: Any) -> int:
        return len(json.dumps(o, ensure_ascii=False))

    if _size(ir_obj) <= max_bytes:
        return

    md = ir_obj.get("metadata", {})
    intents = md.get("intents", {})
    comps = md.get("components", {})
    strings_md = md.get("strings", {})

    def _uniq_preserve(seq):
        seen = set()
        out = []
        for x in seq:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    for k in ["receiver_actions", "service_actions", "activity_actions"]:
        lst = intents.get(k)
        if isinstance(lst, list):
            intents[k] = _uniq_preserve(lst)

    if _size(ir_obj) <= max_bytes:
        return

    limits_seq = [
        {"activities": 64, "services": 32, "receivers": 32, "providers": 16},
        {"activities": 32, "services": 16, "receivers": 16, "providers": 8},
        {"activities": 16, "services": 8, "receivers": 8, "providers": 4},
    ]

    for limits in limits_seq:
        for k, limit in limits.items():
            lst = comps.get(k)
            if isinstance(lst, list) and len(lst) > limit:
                comps[k] = lst[:limit]
        if _size(ir_obj) <= max_bytes:
            return

    ilimits_seq = [
        {"receiver_actions": 64, "service_actions": 16, "activity_actions": 16},
        {"receiver_actions": 32, "service_actions": 8,  "activity_actions": 8},
        {"receiver_actions": 16, "service_actions": 4,  "activity_actions": 4},
    ]

    for ilimits in ilimits_seq:
        for k, limit in ilimits.items():
            lst = intents.get(k)
            if isinstance(lst, list) and len(lst) > limit:
                intents[k] = lst[:limit]
        if _size(ir_obj) <= max_bytes:
            return

    if "action_counts" in intents:
        intents.pop("action_counts")
    if _size(ir_obj) <= max_bytes:
        return

    top = strings_md.get("interesting", [])
    if not isinstance(top, list):
        top = []

    for newlen in [20, 10, 5]:
        if _size(ir_obj) <= max_bytes:
            return
        if len(top) > newlen:
            strings_md["interesting"] = top[:newlen]
            top = strings_md["interesting"]



def step_7_process_all(samples: List[Dict]):
    logging.info(">>> Step 7: Construct Final IR (Optimized & Cleaned)")

    for sample in tqdm(samples, desc="Step 7"):
        h = sample["hash"]
        paths = get_out_paths(h)

        if not ExperimentConfig.OVERWRITE_EXISTING and paths["ir"].exists():
            continue
        if not paths["chunk_summaries"].exists() or not paths["metadata"].exists():
            continue

        meta = read_json(paths["metadata"])
        summ_data = read_json(paths["chunk_summaries"])

        chunk_list = summ_data.get("chunk_summaries", [])
        formatted_chunks = [
            f"[Chunk {c['chunk_id']}] {c['summary']}"
            for c in chunk_list
        ]

        metadata_ir = _build_metadata_for_ir(meta)

        ir_obj = {
            "metadata": metadata_ir,
            "code_behaviors": formatted_chunks,
        }

        _shrink_metadata_to_fit(ir_obj, ExperimentConfig.MAX_IR_BYTES)

        write_json(paths["ir"], ir_obj)



# =========================
# Step 8 & 9: Classification & Rationale
# =========================
# [Modified] Step 8: Classification
def step_8_process_all(samples: List[Dict], model_mgr: ModelManager):
    logging.info(">>> Step 8: Classification")

    # Build a lookup map for Ground Truth from the dataset file
    # samples list already contains 'category' and 'family' from the input JSON
    gt_map = {s['hash']: s.get('category', 'Unknown') for s in samples}

    target_mode = 'lora' if ExperimentConfig.USE_LORA_FOR_CLASSIFICATION else 'base'
    model_mgr.ensure_model(target_mode)

    for sample in tqdm(samples, desc="Step 8"):
        h = sample['hash']
        paths = get_out_paths(h)

        if not ExperimentConfig.OVERWRITE_EXISTING and paths["classification"].exists(): continue
        if not paths["ir"].exists(): continue

        ir_obj = read_json(paths["ir"])

        # IR is already clean (no labels), use as is for prompt
        base_prompt_ir = ir_obj

        results = {
            "hash": h,
            "ground_truth": gt_map.get(h, "Unknown"),  # Fetched from dataset, not IR
            "variants": {}
        }

        # Variant 1: Full
        prompt_full = ExperimentConfig.PROMPT_CLASSIFICATION.format(ir_json_block=json.dumps(base_prompt_ir, indent=2))
        raw_full = model_mgr.generate(prompt_full)
        results["variants"]["full"] = _parse_classification_output(raw_full)

        # Ablations (Logic same as before, just using clean IR keys)
        if ExperimentConfig.RUN_ABLATION_E_META:
            ir_e_meta = {"code_behaviors": base_prompt_ir["code_behaviors"]}
            prompt_e_meta = ExperimentConfig.PROMPT_CLASSIFICATION.format(ir_json_block=json.dumps(ir_e_meta, indent=2))
            raw_e_meta = model_mgr.generate(prompt_e_meta)
            results["variants"]["E-Meta"] = _parse_classification_output(raw_e_meta)

        if ExperimentConfig.RUN_ABLATION_E_SUMMARY:
            ir_e_sum = {"metadata": base_prompt_ir["metadata"]}
            prompt_e_sum = ExperimentConfig.PROMPT_CLASSIFICATION.format(ir_json_block=json.dumps(ir_e_sum, indent=2))
            raw_e_sum = model_mgr.generate(prompt_e_sum)
            results["variants"]["E-Summary"] = _parse_classification_output(raw_e_sum)

        write_json(paths["classification"], results)


def _parse_classification_output(raw_text: str) -> Dict[str, str]:
    match = re.search(r'\[(BENIGN|RANSOMWARE|SMSMALWARE|ADWARE|SCAREWARE)\]', raw_text, re.IGNORECASE)
    prediction = match.group(1).upper() if match else "UNKNOWN"
    return {"prediction": prediction, "raw_output": raw_text}


def step_9_process_all(samples: List[Dict], model_mgr: ModelManager):
    logging.info(">>> Step 9: Rationale Generation (Mode: BASE)")
    # Rationale uses Base model (per instructions)
    model_mgr.ensure_model('base')

    for sample in tqdm(samples, desc="Step 9"):
        h = sample['hash']
        paths = get_out_paths(h)

        if not ExperimentConfig.OVERWRITE_EXISTING and paths["rationale"].exists(): continue
        if not paths["classification"].exists() or not paths["ir"].exists(): continue

        ir_obj = read_json(paths["ir"])
        clf_res = read_json(paths["classification"])
        full_pred = clf_res.get("variants", {}).get("full", {}).get("prediction", "UNKNOWN")

        prompt_ir = {k: v for k, v in ir_obj.items() if k != "metadata"}
        prompt_ir["metadata"] = {k: v for k, v in ir_obj["metadata"].items() if not k.startswith("gt_")}

        prompt = ExperimentConfig.PROMPT_RATIONALE.format(
            label=f"[{full_pred}]",
            ir_json_block=json.dumps(prompt_ir, indent=2)
        )

        rationale = model_mgr.generate(prompt)

        write_json(paths["rationale"], {
            "hash": h,
            "target_prediction": full_pred,
            "rationale": rationale.strip()
        })



# =========================
# Main Loop
# =========================
# [Modified] Main Orchestrator (BFS Style)
def main():
    setup_logging()

    if not ExperimentConfig.DATASET_JSON.exists():
        logging.error("Dataset not found.")
        return

    dataset = read_json(ExperimentConfig.DATASET_JSON)
    samples = dataset.get("samples", [])
    logging.info(f"Loaded {len(samples)} samples.")

    # Init Tokenizer (only needed for Step 2)
    tc = None
    if ExperimentConfig.RUN_STEP_2_CODE_SELECTION:
        tc = LocalTokenCounter(ExperimentConfig.LOCAL_TOKENIZER_PATH)

    # =================================================
    # BFS STEP 1: Metadata Extraction
    # =================================================
    if ExperimentConfig.RUN_STEP_1_METADATA:
        logging.info(">>> Starting BFS Step 1: Metadata Extraction")
        for sample in tqdm(samples, desc="Step 1: Metadata"):
            sample_hash = sample['hash']
            preproc_path = ExperimentConfig.PREPROC_DIR / sample_hash
            paths = get_out_paths(sample_hash)

            if not preproc_path.exists():
                continue
            if not ExperimentConfig.OVERWRITE_EXISTING and paths["metadata"].exists():
                continue

            meta = step_1_extract_metadata(sample_hash, preproc_path)
            # Don't Add label info
            # meta["label_category"] = sample.get("category")
            # meta["label_family"] = sample.get("family")
            write_json(paths["metadata"], meta)

    # =================================================
    # BFS STEP 2: Code Selection (Requires Tokenizer)
    # =================================================
    if ExperimentConfig.RUN_STEP_2_CODE_SELECTION:
        logging.info(">>> Starting BFS Step 2: Code Selection & Prioritization")
        for sample in tqdm(samples, desc="Step 2: Code Selection"):
            sample_hash = sample['hash']
            preproc_path = ExperimentConfig.PREPROC_DIR / sample_hash
            paths = get_out_paths(sample_hash)

            if not paths["metadata"].exists(): continue  # Prereq failed
            if not ExperimentConfig.OVERWRITE_EXISTING and paths["raw_code"].exists(): continue

            meta = read_json(paths["metadata"])
            raw_code_data = step_2_select_files(sample_hash, preproc_path, meta, tc)
            write_json(paths["raw_code"], raw_code_data)

    # =================================================
    # BFS STEP 3: Semantic Chunking
    # =================================================
    if ExperimentConfig.RUN_STEP_3_SEMANTIC_CHUNKING:
        logging.info(">>> Starting BFS Step 3: Semantic Chunking (Graph Clustering)")
        for sample in tqdm(samples, desc="Step 3: Chunking"):
            sample_hash = sample['hash']
            paths = get_out_paths(sample_hash)

            if not paths["raw_code"].exists(): continue
            if not ExperimentConfig.OVERWRITE_EXISTING and paths["chunks"].exists(): continue

            raw_code_data = read_json(paths["raw_code"])
            chunks_data = step_3_semantic_chunking(raw_code_data)
            chunks_data["hash"] = sample_hash
            write_json(paths["chunks"], chunks_data)

    # =================================================
    # BFS STEP 4: Visualization (HTML Generation)
    # =================================================
    if ExperimentConfig.RUN_STEP_4_VISUALIZATION:
        logging.info(">>> Starting BFS Step 4: Generating Visualizations")
        for sample in tqdm(samples, desc="Step 4: Visualization"):
            sample_hash = sample['hash']
            paths = get_out_paths(sample_hash)

            if not paths["chunks"].exists(): continue
            # Visualization is fast and often tweaked, so we default to overwrite or check specifically
            viz_file = out_dir(ExperimentConfig.DIR_VISUALIZATIONS) / f"{sample_hash}_viz.html"
            if not ExperimentConfig.OVERWRITE_EXISTING and viz_file.exists(): continue

            step_4_visualize(sample_hash, paths)

    model_mgr = ModelManager()

    # =================================================
    # BFS STEP 5-9
    # =================================================
    model_mgr = ModelManager()
    try:
        if ExperimentConfig.RUN_STEP_5_CHUNK_SUMMARIZATION:
            step_5_process_all(samples, model_mgr)

        # Step 6 Skipped intentionally (per instruction)

        if ExperimentConfig.RUN_STEP_7_FINAL_IR:
            step_7_process_all(samples)

        if ExperimentConfig.RUN_STEP_8_CLASSIFICATION:
            step_8_process_all(samples, model_mgr)

        if ExperimentConfig.RUN_STEP_9_RATIONALE:
            step_9_process_all(samples, model_mgr)
    finally:
        if model_mgr.client: model_mgr.client.unload()


    logging.info("APK-Sleuth Pipeline Finished.")


if __name__ == "__main__":
    main()