# Approach 4: Gemini Embedding + Cascade Pruner + DeBERTa-v3-Large Cross-Encoder

This approach implements a **3-stage neural entity resolution pipeline** designed for high precision ($F_{0.5}$) and VRAM efficiency on a 6 GB GPU.

---

## 🚀 Execution Instructions for New Setup

### 1. Prerequisites & Environment Setup

Ensure Python 3.10+ and Git are installed. Open terminal / PowerShell and run:

```bash
# Clone the repository
git clone https://github.com/hemangjain17/Amazon_ML_26.git
cd Amazon_ML_26

# Create and activate virtual environment
python -m venv .venv
# On Windows PowerShell:
.venv\Scripts\Activate.ps1
# On Linux / macOS:
# source .venv/bin/activate

# Install required dependencies
pip install -r 6ab10eb3b23ba_student_resource/student_resource/approach_4_gemini_deberta/requirements.txt
```

---

### 2. Configure API Keys (`.env`)

Create a file named `.env` in `6ab10eb3b23ba_student_resource/student_resource/approach_4_gemini_deberta/`:

```env
# Gemini API Key (Required for Stage 1 embeddings)
GEMINI_API_KEY=your_actual_gemini_api_key_here

# Telegram Bot Notifications (Optional but recommended for progress updates)
TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here
TELEGRAM_CHAT_ID=your_telegram_chat_id_here
```

---

### 3. Verify System with Unit Tests

Run the full unit test suite to confirm all modules (retriever, cascade pruner, cross-encoder, evaluator, Telegram logger) function properly:

```bash
python 6ab10eb3b23ba_student_resource/student_resource/approach_4_gemini_deberta/tests/run_all_tests.py
```

Expected Output: `✅ ALL UNIT TESTS PASSED SUCCESSFULLY!`

---

### 4. Run Pipeline (Dry-Run Test)

Run a fast dry-run on a small subset (10 rows) to ensure GPU memory, model downloads, and file writers work without errors:

```bash
python 6ab10eb3b23ba_student_resource/student_resource/approach_4_gemini_deberta/run_pipeline.py --dry-run --subset-size 10
```

---

### 5. Launch Full Pipeline Execution

Run the complete pipeline on the full dataset:

```bash
python 6ab10eb3b23ba_student_resource/student_resource/approach_4_gemini_deberta/run_pipeline.py
```

---

## ⚡ Embedding Caching & Persistence

- Computed Gemini embeddings are **automatically saved** as `.npy` arrays under:
  `6ab10eb3b23ba_student_resource/student_resource/approach_4_gemini_deberta/artifacts/embeddings/`
  - `train_s1_gemini.npy`
  - `train_cand_gemini.npy`
  - `test_s1_gemini.npy`
  - `test_cand_gemini.npy`
- On future runs, the pipeline detects these files, loads them instantly from disk, and **skips recomputing Gemini API calls**.

---

## 📊 Pipeline Outputs

Output submission files are saved to `6ab10eb3b23ba_student_resource/student_resource/output/`:
1. `matching_results.tsv` (Leaderboard submission file with 1,732,544 rows).
2. `candidate_pairs.tsv` (Blocking candidate pairs file).

Both files are automatically verified using `utils/validate_submission.py` upon completion.
