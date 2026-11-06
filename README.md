# SMS Scam and Fraud Shield

DSC306 Data Mining semester project (Project 7).

A web application that lets an ordinary phone user paste an SMS and receive:
- a verdict (scam or safe),
- a scam probability,
- the specific words and patterns that made the message suspicious, with a short
  plain-language explanation of the trick behind each one.

## Quick start

Requires Python 3.10 or newer.

```bash
python -m venv venv
# Windows:    venv\Scripts\activate
# Mac/Linux:  source venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

The app opens at http://localhost:8501. Trained models and cleaned data are
included in the repository, so nothing needs to be downloaded or trained first.

## The six screens

| # | Screen | Purpose |
|---|--------|---------|
| 1 | Data & quality | Audit report, before/after repair table, download of cleaned CSV |
| 2 | Model performance | Comparison of 3 models with baseline, ROC curves, confusion matrix |
| 3 | Single prediction | Paste one SMS, get verdict, confidence and reasons |
| 4 | Batch scoring | Upload a CSV with a `text` column, get a ranked, filterable table |
| 5 | Settings | Choose model, move threshold, edit cost matrix, see consequences live |
| 6 | Learn the tricks | Suspicious words highlighted inside the message with explanations |

## Data

- UCI SMS Spam Collection (5,572 messages after loading).
- After cleaning and de-duplication: 5,157 messages (4,516 ham, 641 spam).
- Optional: `data/raw/own_scams.csv` (columns `text,label,source`) for locally collected scams.

See `docs/data_dictionary.md` and `docs/leakage_statement.md`.

## Models

| Model | Family | Text representation |
|-------|--------|---------------------|
| Multinomial Naive Bayes | Naive Bayes | Word counts |
| Decision Tree | Tree | Binary word presence |
| k-Nearest Neighbours (cosine) | Nearest neighbour | TF-IDF |

Hyper-parameters were chosen with GridSearchCV (5-fold stratified CV, F1 on the
scam class) on the training portion only. Naive Bayes smoothing (alpha) is
demonstrated in `src/models.py` (zero-probability demo and alpha vs F1 plot).

## Evaluation

- Stratified random 80/20 split (the data has no date column). Duplicates were
  removed before splitting to avoid train/test leakage.
- Majority-class baseline is reported beside every accuracy figure.
- Metrics beyond accuracy: precision and recall for the scam class
  (missing a scam is costly, false alarms erode trust), plus F1, AUC and PR-AUC.
- Cost matrix: missed scam = 10, false alarm = 1. The decision threshold is chosen
  from out-of-fold predictions on the training data, not left at 0.5.
- 5-fold cross-validation on the training portion, reported as mean and standard deviation.

## Re-running the full pipeline (optional)

Run from the project root, in this order:

```bash
python src/load_data.py    # raw UCI file -> data/raw/uci_sms.csv
python src/audit.py        # audit, repair, validate() -> data/clean/sms_clean.csv
python src/features.py     # hand-built attributes -> data/clean/sms_features.csv
python src/models.py       # split, train, tune, zero-probability demo
python src/evaluate.py     # ROC, cost matrix, thresholds
python src/explain.py      # quick test of the explanation module
```
