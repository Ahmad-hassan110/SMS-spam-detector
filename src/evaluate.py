import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_curve, roc_auc_score, confusion_matrix,
                             average_precision_score)
from sklearn.model_selection import cross_val_predict, StratifiedKFold

train = pd.read_csv("data/clean/train.csv")
test = pd.read_csv("data/clean/test.csv")
X, y = test["norm"], test["y"]

COST_FN, COST_FP = 10, 1       
names = ["nb", "tree", "knn"]
ths = np.linspace(0.01, 0.99, 99)
cv = StratifiedKFold(5, shuffle=True, random_state=42)

plt.figure(figsize=(6, 5))
rows = []
for n in names:
    m = joblib.load(f"models/{n}.joblib")
    p = m.predict_proba(X)[:, 1]

    # ---- ROC (test set) ----
    fpr, tpr, _ = roc_curve(y, p)
    auc = roc_auc_score(y, p)
    plt.plot(fpr, tpr, label=f"{n} (AUC={auc:.3f})")

    # ---- threshold: train ke out-of-fold predictions se (test ko nahi dekha) ----
    oof = cross_val_predict(m, train["norm"], train["y"], cv=cv,
                            method="predict_proba")[:, 1]
    costs = []
    for t in ths:
        tn, fp, fn, tp = confusion_matrix(train["y"], (oof >= t).astype(int)).ravel()
        costs.append(COST_FN * fn + COST_FP * fp)
    best_t = float(ths[int(np.argmin(costs))])

    # ---- test set par final score ----
    tn, fp, fn, tp = confusion_matrix(y, (p >= best_t).astype(int)).ravel()
    tn5, fp5, fn5, tp5 = confusion_matrix(y, (p >= 0.5).astype(int)).ravel()

    rows.append({"model": n,
                 "AUC": auc,
                 "PR_AUC": average_precision_score(y, p),
                 "best_threshold": best_t,
                 "cost_at_best": COST_FN * fn + COST_FP * fp,
                 "cost_at_0.5": COST_FN * fn5 + COST_FP * fp5,
                 "FN@best": fn,
                 "FP@best": fp})
    joblib.dump(best_t, f"models/{n}_threshold.joblib")

plt.plot([0, 1], [0, 1], "k--", label="random")
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.title("ROC curves")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("docs/roc.png", dpi=150)

res = pd.DataFrame(rows)
pd.set_option("display.width", 250)
print(res.round(4).to_string(index=False))
res.to_csv("data/clean/threshold_results.csv", index=False)