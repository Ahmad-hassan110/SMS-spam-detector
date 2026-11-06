import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
from features import normalize_text

df = pd.read_csv("data/clean/sms_clean.csv")
df["norm"] = df["text"].map(normalize_text)
df["y"] = (df["label"] == "spam").astype(int)

# ---------- SPLIT (dedup pehle ho chuki hai) ----------
train, test = train_test_split(df, test_size=0.2, stratify=df["y"], random_state=42)
train.to_csv("data/clean/train.csv", index=False)
test.to_csv("data/clean/test.csv", index=False)
Xtr, ytr, Xte, yte = train["norm"], train["y"], test["norm"], test["y"]
baseline = 1 - yte.mean()
print(f"Train {len(train)} | Test {len(test)} | Baseline accuracy (sab ham): {baseline:.3f}")

cv = StratifiedKFold(5, shuffle=True, random_state=42)

# ---------- 3 MODELS + TUNING ----------
specs = {
    "nb": (Pipeline([("vec", CountVectorizer()), ("clf", MultinomialNB())]),
           {"vec__min_df": [1, 2], "clf__alpha": [0.01, 0.1, 0.5, 1, 2]}),
    "tree": (Pipeline([("vec", CountVectorizer(binary=True, min_df=2)),
                       ("clf", DecisionTreeClassifier(random_state=42))]),
             {"clf__max_depth": [5, 10, 20, None], "clf__min_samples_leaf": [1, 2, 5]}),
    "knn": (Pipeline([("vec", TfidfVectorizer(min_df=2)),
                      ("clf", KNeighborsClassifier(metric="cosine", algorithm="brute"))]),
            {"clf__n_neighbors": [3, 5, 9], "clf__weights": ["uniform", "distance"]}),
}

rows = []
for name, (pipe, grid) in specs.items():
    gs = GridSearchCV(pipe, grid, cv=cv, scoring="f1", n_jobs=-1).fit(Xtr, ytr)
    best = gs.best_estimator_
    cvs = cross_val_score(best, Xtr, ytr, cv=cv, scoring="f1")
    p = best.predict(Xte)
    rows.append({"model": name, "best_params": gs.best_params_,
                 "cv_f1_mean": cvs.mean(), "cv_f1_std": cvs.std(),
                 "test_acc": accuracy_score(yte, p), "baseline": baseline,
                 "test_precision": precision_score(yte, p),
                 "test_recall": recall_score(yte, p), "test_f1": f1_score(yte, p)})
    joblib.dump(best, f"models/{name}.joblib")
res = pd.DataFrame(rows)
pd.set_option("display.width", 250)
print("\n=== MODEL COMPARISON ===")
print(res.drop(columns="best_params").round(4).to_string(index=False))
print(res[["model", "best_params"]].to_string(index=False))
res.to_csv("data/clean/model_comparison.csv", index=False)

# ---------- ZERO-PROBABILITY DEMO ----------
vec = CountVectorizer()
X = vec.fit_transform(Xtr)
vocab = vec.get_feature_names_out()
c_s = np.asarray(X[ytr.values == 1].sum(0)).ravel()
c_h = np.asarray(X[ytr.values == 0].sum(0)).ravel()
w_s = int(np.argmax(np.where(c_h == 0, c_s, -1)))   # sirf spam mein aane wala lafz
w_h = int(np.argmax(np.where(c_s == 0, c_h, -1)))   # sirf ham mein aane wala lafz
print(f"\n=== ZERO-PROBABILITY DEMO ===")
print(f"Message = '{vocab[w_s]} {vocab[w_h]}'  (pehla lafz sirf spam mein, doosra sirf ham mein dekha gaya)")
V = len(vocab)
for label, c in [("spam", c_s), ("ham", c_h)]:
    p0 = c / c.sum()                       # bina smoothing
    p1 = (c + 1) / (c.sum() + V)           # Laplace alpha=1
    print(f"{label}: unsmoothed P = {p0[w_s]:.2e} x {p0[w_h]:.2e} = {p0[w_s]*p0[w_h]:.2e} | "
          f"smoothed P = {p1[w_s]*p1[w_h]:.2e}")
print("Unsmoothed: dono classes ka score 0 -> faisla nahi ho sakta. Smoothing se dono non-zero.")

# ---------- ALPHA vs F1 ----------
alphas = [1e-4, 1e-3, 0.01, 0.1, 0.5, 1, 2, 5, 10]
f1s = []
for a in alphas:
    pipe = Pipeline([("vec", CountVectorizer()), ("clf", MultinomialNB(alpha=a))])
    f1s.append(cross_val_score(pipe, Xtr, ytr, cv=cv, scoring="f1").mean())
plt.figure(figsize=(6, 4))
plt.semilogx(alphas, f1s, "o-")
plt.xlabel("alpha (smoothing)"); plt.ylabel("CV F1 (scam)"); plt.title("Naive Bayes: alpha vs F1")
plt.grid(alpha=0.3); plt.tight_layout(); plt.savefig("docs/alpha_vs_f1.png", dpi=150)
print("\nalpha vs F1:", dict(zip(alphas, np.round(f1s, 4))))

# ---------- SELECTION vs REDUCTION (same model: kNN) ----------
def knn(): return KNeighborsClassifier(n_neighbors=5, metric="cosine", algorithm="brute")
cmp = {
    "All features (baseline)": Pipeline([("v", TfidfVectorizer(min_df=2)), ("c", knn())]),
    "SelectKBest chi2 k=300": Pipeline([("v", TfidfVectorizer(min_df=2)),
                                        ("s", SelectKBest(chi2, k=300)), ("c", knn())]),
    "TruncatedSVD 300 comps": Pipeline([("v", TfidfVectorizer(min_df=2)),
                                        ("s", TruncatedSVD(300, random_state=42)), ("c", knn())]),
}
print("\n=== SELECTION vs REDUCTION (kNN, 5-fold CV on train) ===")
for k, p in cmp.items():
    s = cross_val_score(p, Xtr, ytr, cv=cv, scoring="f1")
    print(f"{k:28s} F1 = {s.mean():.4f} +/- {s.std():.4f}")