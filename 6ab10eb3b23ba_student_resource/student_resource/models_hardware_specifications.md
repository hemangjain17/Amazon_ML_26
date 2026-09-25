# Models, Memory, and Hardware Resource Specifications

This document outlines the detailed resource footprint (VRAM, RAM, Disk Space), core operational purpose, and high-performance alternative suggestions for every model used across the **Amazon ML Challenge 2026 Entity Resolution Pipelines**.

---

### **1. Approach 1: Contrastive VAE Model**

* **Backbone Architecture:** `paraphrase-multilingual-MiniLM-L12-v2` (Transformer) + Variational Heads + Decoder MLP.
* **Core Operational Work:** 
  * Concatenates business name + address + country into a single normalized string sequence.
  * Encodes inputs into a **256-dimensional normalized latent space $z$**.
  * Optimized during training using a compound loss function:
    $$\mathcal{L} = \mathcal{L}_{\text{recon}} + \beta \mathcal{L}_{\text{KL}} + \lambda \mathcal{L}_{\text{InfoNCE}}$$
  * Triplet loss pulls S1-S2-S3 positive matching records close together while pushing non-matching elements far apart.

#### **Resource Specifications & Footprint:**
* **Model Size on Disk:** **~471 MB** (Transformer backbone) + VAE projection heads (~20 MB).
* **RAM Requirement (CPU):** **~1.5 GB** (loaded in float32).
* **VRAM Requirement (GPU):**
  * **Batch Size = 32:** **~2.8 GB VRAM** (with Mixed Precision FP16 Enabled).
  * **Batch Size = 128:** **~9.5 GB VRAM** (FP16).
* **Hardware Recommendation:** **GPU (T4, P100, A10G, or A100)** to ensure rapid backpropagation and batch encoding of millions of test rows.

#### **Best Replacement / Upgrade Options:**
If compute is not a constraint and you want to scale up for maximum leaderboard $F_{0.5}$ accuracy:
* **Option A: `BAAI/bge-m3`** — Multilingual model that natively generates dense, sparse, and multi-vector representations.
  * *VRAM at BS=32:* ~8.5 GB.
  * *Score Gain:* Expected $+2.5\%$ to $+4.0\%$ $F_{0.5}$.
* **Option B: `intfloat/multilingual-e5-base`** — State-of-the-art multilingual text encoder built specifically for retrieval tasks.
  * *VRAM at BS=32:* ~4.2 GB.

---

### **2. Approach 2: Simple Unsupervised VAE Model**

* **Backbone Architecture:** `paraphrase-multilingual-MiniLM-L12-v2` + Variational Bottleneck Projection.
* **Core Operational Work:**
  * Learns an unsupervised joint representation space $z \in \mathbb{R}^{256}$ of business identity text over S1, S2, and S3 using ELBO loss without requiring ground-truth matches.
  * Good as a baseline representation and for ensembling.

#### **Resource Specifications & Footprint:**
* **Model Size on Disk:** **~471 MB** (backbone).
* **RAM Requirement (CPU):** **~1.2 GB**.
* **VRAM Requirement (GPU):**
  * **Batch Size = 32:** **~1.4 GB VRAM** (Mixed Precision).
  * **Batch Size = 128:** **~4.8 GB VRAM** (Mixed Precision).
* **Hardware Recommendation:** CPU or GPU (runs very fast).

#### **Best Replacement / Upgrade Options:**
* **Upgrade Option: `microsoft/mdeberta-v3-base`** — Incredible zero-shot representation capability using ELECTRA pre-training, which significantly improves typing noise tolerance.

---

### **3. Approach 3: Two-Tower Bi-Encoder Model**

* **Backbone Architecture:** `SentenceTransformer` (Two-Tower architecture based on `paraphrase-multilingual-MiniLM-L12-v2`).
* **Core Operational Work:**
  * Separately encodes S1 (Tower A) and S2/S3 (Tower B).
  * Fine-tuned using **MultipleNegativesRankingLoss (MNR)** on positive match pairs.
  * Outputs **384-dimensional dense vectors** optimized for direct cosine similarity.

#### **Resource Specifications & Footprint:**
* **Model Size on Disk:** **~471 MB**.
* **RAM Requirement (CPU):** **~1.5 GB**.
* **VRAM Requirement (GPU):**
  * **Batch Size = 32:** **~2.2 GB VRAM** (FP16).
  * **Batch Size = 128:** **~7.8 GB VRAM** (FP16).
* **Hardware Recommendation:** GPU is preferred for MNR in-batch negatives computation.

#### **Best Replacement / Upgrade Options:**
* **Upgrade Option: `sentence-transformers/all-mpnet-base-v2`** — Top-tier general-purpose sentence transformer (English-heavy, so use if US records dominate your validation focus).
* **Upgrade Option: `sentence-transformers/LaBSE`** — Language-Agnostic BERT Sentence Embeddings, optimized for 109+ languages (ideal for Tamil/Hindi script transliteration).

---

### **4. FAISS Candidate Blocking Index**

* **Architecture:** `faiss.IndexFlatIP` (Flat Inner Product index) or `faiss.IndexHNSW` (Hierarchical Navigable Small World graphs).
* **Core Operational Work:**
  * Compares S1 latent vectors against candidate S2 + S3 vectors partitioned by country (`US`, `India`, `France`).
  * Yields the Top-30 candidate matches per S1 entity in logarithmic time ($O(\log N)$).

#### **Resource Specifications & Footprint:**
* **Disk Space (Indices):** **~1.8 GB** (for 11M candidate float16 vectors).
* **RAM Requirement (CPU):** **~5.5 GB** (with country-partitioned garbage collection enabled).
* **VRAM Requirement (GPU):** Uses negligible VRAM if run on CPU, or ~2.2 GB if run on GPU via `faiss.index_cpu_to_gpu`.
* **Hardware Recommendation:** **CPU-only execution** is recommended because country partitioning and garbage collection keep RAM well below your 16 GB system memory threshold.

---

### **5. GBDT Match Reranker (CatBoost)**

* **Architecture:** Symmetric Gradient Boosted Decision Trees (`CatBoostClassifier`).
* **Core Operational Work:**
  * Evaluates 15 Million+ candidate pairs generated by FAISS blocking.
  * Computes 8 high-performance string, token, character n-gram, PIN, and latent similarity features.
  * Predicts matching probability $P(\text{Match}_{i,j})$.

#### **Resource Specifications & Footprint:**
* **Model Size on Disk:** **~5 MB** (Symmetric tree configuration is extremely lightweight).
* **RAM Requirement (CPU):** **~1.2 GB** (Sparsified matrix handling).
* **Training Time:** **~1.5 minutes** for 15M rows.
* **Hardware Recommendation:** CPU (utilizes multi-threaded parallel execution natively).

#### **Best Replacement / Upgrade Options:**
* **Upgrade Option: XGBoost / LightGBM Ensemble** — Stacking CatBoost with LightGBM features captures diverse tabular interaction paths and raises precision by $+1.0\%$ to $+2.0\%$ $F_{0.5}$.
