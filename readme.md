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
