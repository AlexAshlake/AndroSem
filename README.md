## 1. Overview

AndroSem is a semantics-guided, LLM-based static Android malware detector designed for both accuracy and interpretability. Given an APK, AndroSem decompiles code and the manifest, performs **Semantic Abstraction** to extract concise behavior semantics from relevant code regions, and parses manifest fields and embedded strings into normalized static features. These heterogeneous static views are then fused into a human-readable intermediate representation, which is consumed by a base or LoRA-fine-tuned large language model (LLM) to produce **multi-class or binary predictions together with natural-language rationales**.

This repository contains the complete implementation used in our paper:

> **AndroSem: Semantics-Guided, LLM-based Interpretable Static Android Malware Detection**

For a detailed description of the methodology, semantic abstraction process, and evaluation on the full CIC-AndMal2017 APK corpus, please refer to the paper. This codebase is intended to be **research-friendly and reusable**, so that others can reproduce our experiments, inspect the intermediate representations, and extend AndroSem to new datasets or model backends.

---

## 2. Project structure

The repository is organized into two main parts: a **local client** that performs static preprocessing and orchestrates the analysis pipeline, and a **remote server** that hosts the LLM (base and/or LoRA-fine-tuned).

```text
AndroSem/
├── local_client_script/        # Local-side preprocessing, orchestration, analysis
│   ├── Batch_preprocess_apks.py
│   ├── Split_dataset.py
│   ├── MainLogic.py
│   ├── Analyse_result.py
│   ├── local_tokenizer.py
│   ├── remote_model_client.py
│   ├── PE-Sleuth.txt
│   ├── AndroSem.py
│   └── README.md               # Local-side usage and reproduction guide (to be filled)
│
└── remote_server_script/       # Remote-side LLM service (GPU server)
    ├── model_server.py
    ├── remote_llm_server.py
    ├── requirements.txt
    ├── run_server.sh           # Optional helper script for launching the service
    └── README.md               # Remote-side deployment guide (to be filled)

```
### A typical workflow is:

1. **Static preprocessing (local)**

   * `Batch_preprocess_apks.py`

     * Recursively scans the APK dataset, runs `apktool` and `jadx` on each APK, extracts basic string features, and stores all intermediate artifacts under a unified `data/preproc/<sha256>/` directory together with a `status.json` metadata file.

2. **Dataset definition & splitting (local)**

   * `Split_dataset.py`

     * Reads all valid preprocessed samples from `data/preproc/`, filters out incomplete or too-small apps, and creates stratified **train/test** splits (and small experimental subsets) on top of the CIC-AndMal2017 corpus.
     * Outputs dataset definition files such as `train_set.json`, `test_set.json`, `experiment_set_5pct.json`, etc. under `data/dataset_splits/`.

3. **Main analysis pipeline (local + remote)**

   * `MainLogic.py`

     * Implements the end-to-end AndroSem pipeline over a chosen dataset split:

       * loads preprocessed samples from `data/preproc/`,
       * extracts and ranks relevant code regions,
       * performs semantic abstraction and chunking,
       * builds the human-readable intermediate representation (IR),
       * queries the remote LLM (base or LoRA) for multi-class / binary predictions and natural-language rationales.
     * All intermediate products and final results are written under `data/main_output/<experiment_name>/` in step-wise subdirectories (`1_metadata/`, `2_selected_code/`, `3_semantic_chunks/`, `7_final_ir/`, `8_classification/`, `9_rationale/`, etc.).

   * `local_tokenizer.py`

     * Loads the local tokenizer (from the base Qwen3-14B checkpoint) to estimate token counts and enforce per-app token budgets before sending requests to the remote LLM server.

   * `remote_model_client.py`

     * Lightweight HTTP client for talking to the remote LLM server (`remote_llm_server.py`), providing simple methods such as “generate text” and “count tokens”, with optional LoRA usage.

4. **Result analysis and metrics (local)**

   * `Analyse_result.py`

     * Consumes dataset definition files (e.g., `test_set.json`) and the outputs under `data/main_output/<experiment_name>/`.
     * Aggregates predictions to compute:

       * multi-class metrics (per-family precision/recall/F1, macro-F1, accuracy),
       * binary benign vs. malicious metrics (confusion matrix and F1 on malicious).
     * Can optionally generate detailed JSON traces for correct and misclassified samples to support qualitative analysis and case studies.

5. **LLM hosting (remote server)**

   * `model_server.py`

     * Loads the **base Qwen3-14B model** and optionally the **AndroSem LoRA adapter** from absolute paths on the GPU server.
     * Exposes a unified interface that supports:

       * base-only inference,
       * LoRA-fine-tuned inference,
       * token counting and optional model unloading.

   * `remote_llm_server.py`

     * Wraps `model_server.py` into a simple **FastAPI** HTTP service (e.g., on `http://0.0.0.0:8000`).
     * Provides endpoints for:

       * text generation (`/generate`),
       * token counting (`/count_tokens`),
       * health checking (`/health`),
       * optional model unload (`/unload`).
     * This service is intended to run on a remote GPU machine and be accessed from the local client via SSH port forwarding or a similar secure channel.


## 3. Reproduction guide

This section describes how to reproduce the experiments in the paper using the released code and models. The pipeline is split into two logical parts:

- a **remote LLM server** running on a GPU node (e.g., AutoDL), and  
- a **local client** that performs preprocessing, semantic abstraction, IR construction, and evaluation.

You can run both on the same machine if you have sufficient resources, but in our experiments they were physically separated.

---

### 3.1 Reference hardware and model configuration

In our experiments, the LLM component of AndroSem runs on the AutoDL cloud platform, while the core processing pipeline runs on a local workstation.

- **Remote GPU node (AutoDL)**
  - 20 vCPUs (Intel Xeon Platinum 8470Q)
  - 1 × 48 GB virtual GPU
  - Classifier/explainer model: **Qwen3-14B** (14B parameters), loaded locally on the node (no external API calls)
  - Inference configuration:
    - 4-bit `nf4` quantization via BitsAndBytes
    - FlashAttention enabled
    - Maximum context length extended to **128k tokens** (via YaRN)
    - Maximum generation length: **512 tokens**
    - Sampling: temperature **0.5**, top-p **0.9**
    - Computation in bfloat16 or float16 depending on hardware support
  - Supervised fine-tuning:
    - Performed with LLaMA-Factory in SFT mode
    - LoRA rank **24**, alpha **48**, dropout **0.15**
    - 4-bit quantization during training

- **Local workstation**
  - Windows 11
  - AMD Ryzen 9 7940H CPU
  - 48 GB RAM (this is *more than strictly required*; AndroSem is CPU-bound on the local side and can run with less memory)
  - Communicates with the AutoDL node via HTTP and an SSH tunnel.

You do **not** need to exactly match this hardware. Any GPU with enough memory to host Qwen3-14B with 4-bit quantization should work, and the local machine only needs enough CPU/RAM to run APK toolchains and Python scripts.

---

### 3.2 Required downloads

Before setting up the environments, obtain the following:

- **Code repository** (this project):
  - GitHub: <https://github.com/AlexAshlake/AndroSem>
- **Dataset**:
  - CIC-AndMal2017 APK corpus: <https://www.unb.ca/cic/datasets/andmal2017.html>
- **Base LLM**:
  - Qwen3-14B: <https://huggingface.co/Qwen/Qwen3-14B>
- **LoRA weights for AndroSem**:
  - AndroSem-Qwen3-14B-LoRA: <https://huggingface.co/AlexAshlake/AndroSem-Qwen3-14B-LoRA>

The base model and LoRA weights will be used on the **remote** side, and the base model (or at least its tokenizer) will also be used on the **local** side.

---

### 3.3 Remote LLM server (GPU node)

**Goal:** run a FastAPI-based LLM server that exposes Qwen3-14B (base and LoRA) over HTTP.

#### 3.3.1 Clone repository and prepare environment

```bash
# On the remote GPU node
git clone https://github.com/AlexAshlake/AndroSem.git
cd AndroSem/remote_server_script

# Create and activate a virtual environment (example with venv)
python -m venv .venv
source .venv/bin/activate    # On Linux/macOS
# .venv\Scripts\activate     # On Windows, if applicable

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt
````

#### 3.3.2 Download and place model weights

On the remote node, download:

* the **base** Qwen3-14B model from Hugging Face, and
* the **AndroSem LoRA** adapter.

Place them in directories such as:

```text
/root/models/Qwen3-14B/
    # base model checkpoint

/root/models/AndroSem-Qwen3-14B-LoRA/
    # LoRA adapter
```

You are free to choose different paths; just make sure they are consistent with `model_server.py`.

#### 3.3.3 Configure `model_server.py`

Open `remote_server_script/model_server.py` and set the absolute paths for the base model and LoRA:

```python
class ModelConfig:
    """Configuration for the language model (shared by base & LoRA)."""

    # Absolute paths on the remote GPU server
    BASE_MODEL_PATH: str = "/root/models/Qwen3-14B"
    LORA_WEIGHTS_PATH: Optional[str] = "/root/models/AndroSem-Qwen3-14B-LoRA"
```

If you want to disable LoRA and use only the base model, set:

```python
LORA_WEIGHTS_PATH = None
```

Other options (quantization, FlashAttention, context length) are controlled inside `model_server.py` and can be kept as default to match the configuration used in our experiments.

#### 3.3.4 Start the LLM HTTP server

From the `remote_server_script` directory:

```bash
# Still inside the virtual environment
python remote_llm_server.py
```

By default this starts a FastAPI server on `http://0.0.0.0:8000`. You should see log messages indicating that the base model (and optionally LoRA) are loaded and the service is ready.

Make sure:

* Port **8000** is reachable from your local machine (via firewall rules or SSH tunneling).
* The local client’s `REMOTE_LLM_BASE_URL` matches the address you expose (e.g., `http://127.0.0.1:8000` if accessed through an SSH tunnel).

---

### 3.4 Local client (preprocessing, IR construction, evaluation)

**Goal:** run the AndroSem pipeline locally, using the remote GPU node only for LLM inference.

#### 3.4.1 Clone repository and prepare environment

```bash
# On the local machine
git clone https://github.com/AlexAshlake/AndroSem.git
cd AndroSem/local_client_script

# Create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\activate      # On Windows
# source .venv/bin/activate   # On Linux/macOS

# Install Python dependencies for the local pipeline
pip install --upgrade pip
pip install -r requirements.txt    # if you maintain a local-side requirements file
```

If you do not have a separate `requirements.txt` for the local side, you can either:

* copy the relevant parts from the remote requirements, or
* manually install standard packages such as `tqdm`, `numpy`, `pandas`, `networkx`, etc., depending on your environment.

#### 3.4.2 Install `apktool` and `jadx`

AndroSem uses `apktool` and `jadx` to decompile APKs:

1. Install `apktool` and `jadx` on your local machine and ensure they can be invoked from the command line.
2. Open `Batch_preprocess_apks.py` and configure the tool paths:

```python
APKTOOL_PATH = "apktool"   # or full path to apktool if not in PATH
JADX_PATH = "jadx"         # or full path to the jadx executable
```

You can keep these as simple command names if they are in your system `PATH`.

#### 3.4.3 Download base model (for tokenizer only)

On the local machine, download at least the **tokenizer files** of Qwen3-14B (you can copy them from the remote node or pull directly from Hugging Face). Place them, for example, under:

```text
AndroSem/local_client_script/models/Qwen3-14B/
```

Then open `MainLogic.py` and set:

```python
LOCAL_TOKENIZER_PATH: str = str(BASE_DIR / "models" / "Qwen3-14B")
```

This tokenizer is used locally to estimate token counts and enforce token budgets before sending data to the remote server. No local GPU is required.

#### 3.4.4 Place the CIC-AndMal2017 APKs

Create the data directory structure expected by the scripts (relative to `local_client_script`):

```text
local_client_script/
  data/
    APKs/
      Benign_2015/
      Benign_2016/
      Benign_2017/
      Ransomware/
      SMSMalware/
      Adware/
      Scareware/
      ...
```

Copy the **APK files** from the CIC-AndMal2017 dataset into the appropriate subdirectories under `data/APKs/`, preserving the category/family structure as much as possible. In particular, benign APKs should be placed under `Benign_2015`, `Benign_2016`, and `Benign_2017`, while malware families should go under their respective folders (e.g., `Ransomware`, `SMSMalware`, `Adware`, `Scareware`, etc.).

The scripts will automatically infer high-level labels from the directory names.

#### 3.4.5 Preprocess all APKs

From `local_client_script`:

```bash
python Batch_preprocess_apks.py
```

This will:

* recursively scan `data/APKs/` for `.apk` files,
* for each APK:

  * compute its SHA-256 hash,
  * copy it to `data/preproc/<hash>/raw.apk`,
  * run `apktool` into `data/preproc/<hash>/apktool_out/`,
  * run `jadx` into `data/preproc/<hash>/java_src/`,
  * extract basic string features into `data/preproc/<hash>/strings.txt`,
  * record status and paths in `data/preproc/<hash>/status.json`.

After this step, all preprocessed artifacts required by the pipeline will be under `data/preproc/`.

#### 3.4.6 Create dataset splits

```bash
python Split_dataset.py
```

This script:

* scans all valid preprocessed samples in `data/preproc/` (filters incomplete or too-small Java code),
* parses their original `apk_path` to infer category/family labels,
* produces stratified **70/30 train/test** splits and smaller experimental subsets.

The following JSON files are written to `data/dataset_splits/`:

* `train_set.json`
* `test_set.json`
* `experiment_set_5pct.json`
* `mini_set_0.5pct.json`

Each file contains a list of sample hashes and labels, which are consumed by `MainLogic.py` and `Analyse_result.py`.

#### 3.4.7 Connect to the remote LLM server

Before running the main pipeline, ensure that:

* `remote_llm_server.py` is already running on the remote GPU node (Section 3.3.4), and
* your local machine can reach it.

A typical setup is to use SSH tunneling from the local machine:

```bash
ssh -CNg -L 8000:127.0.0.1:8000 USER@REMOTE_HOST -p PORT
```

For example:

```bash
ssh -CNg -L 8000:127.0.0.1:8000 root@connect.bjb1.seetacloud.com -p 22003
```

Then in `MainLogic.py`, configure:

```python
REMOTE_LLM_BASE_URL: str = "http://127.0.0.1:8000"
USE_LORA_FOR_CLASSIFICATION: bool = True   # or False to use base model only
```

#### 3.4.8 Run the AndroSem pipeline

Choose which dataset split you want to run on by setting, in `MainLogic.py`:

```python
DATASET_JSON: Path = BASE_DIR / "data" / "dataset_splits" / "test_set.json"
OUTPUT_DIR: Path = BASE_DIR / "data" / "main_output" / "cicandmal2017_full"
```

Then run:

```bash
python MainLogic.py
```

The script will:

1. Load all samples listed in `DATASET_JSON`.
2. For each sample:

   * gather metadata and code paths,
   * select relevant code regions,
   * perform semantic chunking and abstraction,
   * build a human-readable intermediate representation (IR),
   * send IR to the remote LLM server for classification and rationales.
3. Save outputs under `OUTPUT_DIR`, with subdirectories such as:

   * `1_metadata/`
   * `2_selected_code/`
   * `3_semantic_chunks/`
   * `5_chunk_summaries/`
   * `7_final_ir/`
   * `8_classification/`
   * `9_rationale/`

You can run multiple experiments by changing `DATASET_JSON` and `OUTPUT_DIR` and re-running `MainLogic.py`.

#### 3.4.9 Evaluate detection performance and collect error cases

Finally, run:

```bash
python Analyse_result.py test_set.json cicandmal2017_full
```

* The first argument selects the dataset definition (e.g., `test_set.json`, `train_set.json`, `experiment_set_5pct.json`, or `mini_set_0.5pct.json`).
* The second argument selects the experiment directory under `data/main_output/`.

`Analyse_result.py` will:

* aggregate predictions from `7_final_ir/` and `8_classification/`,
* compute:

  * **multi-class** metrics (per-class precision/recall/F1, macro-F1, overall accuracy),
  * **binary** metrics (benign vs malicious confusion matrix and F1 on malicious),
* write summary statistics to:

```text
data/main_output/cicandmal2017_full/llm_classification_stats.json
```

Optional trace files for misclassified samples can also be generated (controlled by flags inside `Analyse_result.py`), allowing fine-grained inspection of AndroSem’s decisions and explanations.

---

### 3.5 End-to-end checklist

To fully reproduce the experiments reported in the paper:

1. ✅ Clone the repository on both the remote GPU node and local machine.
2. ✅ Download Qwen3-14B (base) and AndroSem LoRA weights to the remote node, configure `model_server.py`.
3. ✅ Download at least the base tokenizer to the local machine and configure `LOCAL_TOKENIZER_PATH` in `MainLogic.py`.
4. ✅ Install `apktool` and `jadx` locally and configure their paths in `Batch_preprocess_apks.py`.
5. ✅ Place CIC-AndMal2017 APKs under `local_client_script/data/APKs/` following the category/family structure.
6. ✅ Run `Batch_preprocess_apks.py` → `Split_dataset.py`.
7. ✅ Start `remote_llm_server.py` on the remote node and set up an SSH tunnel or other network route.
8. ✅ Run `MainLogic.py` for the desired dataset split.
9. ✅ Run `Analyse_result.py` to compute metrics and inspect error cases.

Following these steps should allow you to reproduce the main results of **AndroSem: Semantics-Guided, LLM-based Interpretable Static Android Malware Detection** and to explore the intermediate representations and explanations produced by the system.

```
```

