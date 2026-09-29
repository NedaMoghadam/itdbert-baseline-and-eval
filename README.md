# ITDBERT — Debugging, Baseline & Evaluation

Contribution to [Jules Gouy's ITDBERT_experiences](https://github.com/JulesGouy/ITDBERT_experiences),
a reproduction of [cgly/ITDBERT](https://github.com/cgly/ITDBERT) (insider threat
detection via RoBERTa + Bi-LSTM on the CERT r4.2 dataset). Shared here with Jules's
permission, crediting both him and the original authors.

**What I did:**
- Fixed several bugs in the original pipeline: a tokenizer that silently never added
  `<s>`/`</s>` special tokens (missing post-processor), broken file paths, a Windows
  path-handling bug, and a checkpoint-saving bug that crashed training.
- Added real early stopping (validation-based, keeps the best epoch) and a fixed
  random seed for reproducible runs.
- Built and tested an XGBoost baseline on simple engineered features — it beats the
  RoBERTa deep model on both datasets tested.
- Replicated the original paper's benchmark on the authors' own data and identified
  why it scores higher than a fairer evaluation: their negative (normal) sequences
  come from different users than the attackers, making the task easier than
  detecting an insider's bad days against their own normal behavior.
- Extended the preprocessing to retain userID/date/scenario metadata
  (`dataclean_meta.py`), verified to reproduce the original output exactly.

**Results (AUC, threshold not chosen on test):**

| Dataset | XGBoost (simple features) | RoBERTa (pretrained, fine-tuned) |
|---|---|---|
| Authors' original benchmark | 0.954 | 0.916 |
| Fairer "reduced" split (same insiders, clean vs. attack days) | 0.854 | 0.776 ± 0.016 (3 seeds) |

**Takeaway:** with ~250 labeled malicious examples, a simple baseline outperforms
the deep model — more labeled data, not more model complexity, is the likely path
to further improvement. The gap between the two datasets above also shows how much
a benchmark's negative-sampling choices can inflate reported results.

**Scripts:**
- `train_xgboost.py` / `train_xgboost_authors.py` — the XGBoost baseline, on the
  reduced and authors' datasets respectively.
- `robertaBERT.py` / `robertaLSTM.py` — fixed versions of the original classifiers.
- `robertaBERT_authors.py` / `robertaBERT_clean.py` — RoBERTa with clean,
  test-label-free evaluation, multi-seed.
- `trans_corpus.py` — MLM pretraining, now with real early stopping.
- `dataclean_meta.py` / `compare_with_jules.py` — preprocessing patch adding
  userID/date/scenario metadata, plus a verification script.

**Not included:** the CERT r4.2 dataset itself — obtain it from the
[official CERT Insider Threat Dataset](https://kilthub.cmu.edu/articles/dataset/Insider_Threat_Test_Dataset/12841247).
