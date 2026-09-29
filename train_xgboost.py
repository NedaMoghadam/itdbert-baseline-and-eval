"""
train_xgboost.py  (v3)
----------------------
XGBoost sur features tabulaires extraites des sequences.

Corrections par rapport a la v1 :
  1. Decodage des tokens : token = 24*type + heure, en partant de 0
     (0=Logon, 1=Logoff, 2=http, 3=email, 4=file, 5=Connect, 6=Disconnect).
     La v1 supposait que les tokens commencaient a 1 (decalage d'une heure).
  2. Evaluation honnete : AUC / PR-AUC (sans seuil) + un seuil choisi
     UNIQUEMENT avec le train (validation croisee), pas avec le test.
     Le seuil "optimise sur le test" est affiche a part, comme borne optimiste.

Usage :  python ITDBERT\\train_xgboost.py
"""
import os, csv
import numpy as np
from collections import Counter

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
TRAIN_CSV = os.path.join(BASE_DIR, "data", "preprocessing_try", "reduced_THNS24_2010.csv")
TEST_CSV  = os.path.join(BASE_DIR, "data", "preprocessing_try", "reduced_THNS24_2011.csv")
TYPE_NAMES = ["Logon", "Logoff", "http", "email", "file", "Connect", "Disconnect"]


def extract_features(tokens):
    types = [t // 24 for t in tokens]
    hours = [t % 24 for t in tokens]
    n = len(tokens)
    tc = Counter(types)
    f = {"seq_len": n}
    for ty, name in enumerate(TYPE_NAMES):
        f[f"{name}_ratio"] = tc.get(ty, 0) / n
    f["off_hours_ratio"] = sum(1 for h in hours if h < 8 or h >= 18) / n
    f["unique_ratio"] = len(set(tokens)) / n
    p = np.array(list(tc.values())) / n
    f["type_entropy"] = -np.sum(p * np.log2(p + 1e-12))
    f["most_common_ratio"] = tc.most_common(1)[0][1] / n
    f["token_mean"] = float(np.mean(tokens))
    f["token_std"] = float(np.std(tokens))
    run = best = 1
    for i in range(1, n):
        run = run + 1 if types[i] == types[i - 1] else 1
        best = max(best, run)
    f["max_consecutive_run"] = best
    return f


def load(path):
    X, y = [], []
    with open(path) as fh:
        for row in csv.reader(fh):
            if not row:
                continue
            tokens = [int(t) for t in row[1:] if t.strip()]
            if not tokens:
                continue
            X.append(extract_features(tokens))
            y.append(int(row[0]))
    return X, np.array(y)


Xd_tr, y_tr = load(TRAIN_CSV)
Xd_te, y_te = load(TEST_CSV)
names = sorted(Xd_tr[0])
X_tr = np.array([[d[k] for k in names] for d in Xd_tr])
X_te = np.array([[d[k] for k in names] for d in Xd_te])
print(f"Train : {len(y_tr)} sequences ({y_tr.mean()*100:.1f}% malicious)")
print(f"Test  : {len(y_te)} sequences ({y_te.mean()*100:.1f}% malicious)")

from xgboost import XGBClassifier
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold, cross_val_predict

spw = (y_tr == 0).sum() / (y_tr == 1).sum()
model = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                      scale_pos_weight=spw, eval_metric="logloss", random_state=42)


def best_f1_threshold(y, p):
    pr, rc, th = metrics.precision_recall_curve(y, p)
    f1 = 2 * pr * rc / (pr + rc + 1e-12)
    return th[np.argmax(f1[:-1])]


def report(title, y, p, thr):
    pred = (p >= thr).astype(int)
    print(f"\n--- {title} (seuil = {thr:.3f}) ---")
    print(f"Accuracy  : {metrics.accuracy_score(y, pred):.4f}")
    print(f"Precision : {metrics.precision_score(y, pred, zero_division=0):.4f}")
    print(f"Recall    : {metrics.recall_score(y, pred, zero_division=0):.4f}")
    print(f"F1-score  : {metrics.f1_score(y, pred, zero_division=0):.4f}")
    print(f"Matrice de confusion :\n{metrics.confusion_matrix(y, pred)}")


# Seuil choisi avec le TRAIN uniquement (predictions out-of-fold)
oof = cross_val_predict(model, X_tr, y_tr, method="predict_proba",
                        cv=StratifiedKFold(5, shuffle=True, random_state=42))[:, 1]
thr_train = best_f1_threshold(y_tr, oof)

model.fit(X_tr, y_tr)
p_te = model.predict_proba(X_te)[:, 1]

print("\n" + "=" * 60)
print("EVALUATION XGBoost (v3)")
print("=" * 60)
print(f"\nAUC train (validation croisee) : {metrics.roc_auc_score(y_tr, oof):.4f}")
print(f"AUC test                       : {metrics.roc_auc_score(y_te, p_te):.4f}")
print(f"PR-AUC test                    : {metrics.average_precision_score(y_te, p_te):.4f}"
      f"   (hasard = {y_te.mean():.3f})")

report("Seuil par defaut", y_te, p_te, 0.5)
report("Seuil choisi sur le TRAIN (honnete)", y_te, p_te, thr_train)
report("Seuil optimise sur le TEST (borne optimiste)", y_te, p_te, best_f1_threshold(y_te, p_te))

print("\n--- Importance des features ---")
for n_, imp in sorted(zip(names, model.feature_importances_), key=lambda x: -x[1]):
    print(f"  {n_:22s} : {imp:.4f}")