# Amazon ML Challenge 2026 — Business Entity Resolution

End-to-End Business Entity Resolution pipeline supporting Contrastive VAE, Unsupervised Simple VAE, and Two-Tower Bi-Encoder approaches for large-scale multi-source entity matching across US, India, and France.

## Repository Structure

```
Amazon_ML_26/
└── 6ab10eb3b23ba_student_resource/
    └── student_resource/
        ├── global_notebooks/
        │   ├── 01_global_data_download_and_s3_sync.ipynb        # Data ingestion (gdown) & S3 sync
        │   └── 03_global_feature_extraction_and_reranking.ipynb # Pairwise feature extractor & CatBoost
        ├── approach_1_contrastiveVAE/                           # Approach 1: Contrastive VAE Model
        │   ├── 02_train_and_encode_contrastive_vae.ipynb
        │   ├── config.py
        │   └── src/
        ├── approach_2_simpleVAE/                                # Approach 2: Simple Unsupervised VAE
        │   ├── 02_train_and_encode_simple_vae.ipynb
        │   ├── config.py
        │   └── src/
        ├── approach_3_biEncoder/                                # Approach 3: Two-Tower Bi-Encoder
        │   ├── 02_train_and_encode_bi_encoder.ipynb
        │   ├── config.py
        │   └── src/
        └── utils/
            └── validate_submission.py                           # Official competition validator
```

## Quick Start on AWS SageMaker

1. Clone this repository in SageMaker Terminal:
   ```bash
   git clone https://github.com/hemangjain17/Amazon_ML_26.git
   cd Amazon_ML_26/6ab10eb3b23ba_student_resource/student_resource
   ```

2. Open and run **Step 1 (Global Data Download & S3 Sync)**:
   - `global_notebooks/01_global_data_download_and_s3_sync.ipynb`

3. Open and run **Step 2 (Model Training & FAISS Candidate Blocking)**:
   - `approach_1_contrastiveVAE/02_train_and_encode_contrastive_vae.ipynb`

4. Open and run **Step 3 (Feature Extraction, CatBoost Reranking & Validation)**:
   - `global_notebooks/03_global_feature_extraction_and_reranking.ipynb`
