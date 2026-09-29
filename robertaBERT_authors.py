"""
robertaBERT_authors.py
----------------------
RoBERTa pre-entraine (models\\42_H1_24_42) + tete de classification sur [CLS],
entraine sur les donnees ORIGINALES des auteurs (Data\\original data).
Labels 1, 2, 3 -> 1 (malveillant), 0 -> normal.

Evaluation propre : 15 % du train sert de VALIDATION (choix de l'epoch, arret
anticipe, seuil). Le test n'est utilise qu'UNE seule fois, a la fin.

Ctrl+C pendant l'entrainement : arrete proprement et evalue le meilleur epoch
obtenu jusque-la (aucun resultat perdu).

Usage :  python ITDBERT\\robertaBERT_authors.py            (complet)
         python ITDBERT\\robertaBERT_authors.py --quick    (test rapide, peu de donnees)
"""
import os, sys, time, random, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preTrain.MaskedLM.MaskedUtils import load_dataset, bulid_iterator

from tokenizers import ByteLevelBPETokenizer as _BPE
from tokenizers.processors import RobertaProcessing
from transformers import PreTrainedTokenizerFast, RobertaModel


def load_tokenizer(model_dir):
    vocab_file  = os.path.join(model_dir, "vocab.json").replace("\\", "/")
    merges_file = os.path.join(model_dir, "merges.txt").replace("\\", "/")
    _bpe = _BPE(vocab=vocab_file, merges=merges_file)
    _bpe._tokenizer.post_processor = RobertaProcessing(
        ("</s>", _bpe.token_to_id("</s>")),
        ("<s>", _bpe.token_to_id("<s>")),
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=_bpe, model_max_length=512,
        bos_token="<s>", eos_token="</s>", unk_token="<unk>",
        sep_token="</s>", cls_token="<s>", pad_token="<pad>", mask_token="<mask>",
    )


class Config(object):
    def __init__(self, quick):
        base = os.path.dirname(os.path.abspath(__file__))
        data = os.path.join(base, "..", "Data", "original data")
        self.train_path = os.path.join(data, "THNS24_2010.csv")
        self.test_path  = os.path.join(data, "THNS24_2011.csv")
        self.device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = 64
        self.pad_size   = 154
        self.learning_rate = 1e-4
        self.max_epochs = 2 if quick else 20
        self.patience   = 4
        models = os.path.join(base, "preTrain", "MaskedLM", "models")
        self.reberta_path   = os.path.join(models, "42_H1_24_42")
        self.tokenizer_path = os.path.join(models, "tokenizer")
        self.save_path = os.path.join(base, "preTrain", "MaskedLM", "save_model",
                                      "robertaBERT_authors.pth")
        print("Chargement du tokenizer et du modele pre-entraine...")
        self.rebertaTokenizer = load_tokenizer(self.tokenizer_path)
        self.rebertaModel = RobertaModel.from_pretrained(self.reberta_path)
        print(f"Device : {self.device}")


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.roberta = config.rebertaModel
        self.fc2 = nn.Linear(768, 128)
        self.fc3 = nn.Linear(128, 2)

    def forward(self, x):
        pooled = self.roberta(x[0], attention_mask=x[2]).pooler_output
        return self.fc3(F.relu(self.fc2(pooled)))


def predict(model, data_iter):
    """Retourne (labels, proba_malveillant, loss_moyenne)."""
    model.eval()
    ys, ps, loss_sum, n = [], [], 0.0, 0
    with torch.no_grad():
        for x, y in data_iter:
            if y.shape[0] == 0:
                continue
            out = model(x)
            loss_sum += F.cross_entropy(out, y, reduction="sum").item()
            n += y.shape[0]
            ys.append(y.cpu().numpy())
            ps.append(F.softmax(out, dim=1)[:, 1].cpu().numpy())
    return np.concatenate(ys), np.concatenate(ps), loss_sum / max(n, 1)


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


def main(quick):
    cfg = Config(quick)
    train_all = load_dataset(cfg.train_path, cfg)
    test      = load_dataset(cfg.test_path, cfg)
    if quick:
        train_all, test = train_all[:600], test[:200]

    train, val = train_test_split(train_all, test_size=0.15, random_state=SEED,
                                  stratify=[it[1] for it in train_all])
    print(f"Train : {len(train)} | Validation : {len(val)} | Test : {len(test)}")

    train_iter = bulid_iterator(train, cfg)
    val_iter   = bulid_iterator(val, cfg)
    test_iter  = bulid_iterator(test, cfg)

    model = Model(cfg).to(cfg.device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=1e-4)
    w = compute_class_weight("balanced", classes=np.array([0, 1]),
                             y=np.array([it[1] for it in train]))
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float).to(cfg.device))

    os.makedirs(os.path.dirname(cfg.save_path), exist_ok=True)
    best_auc, no_imp = -1.0, 0
    try:
        for epoch in range(1, cfg.max_epochs + 1):
            t0 = time.time()
            model.train()
            random.shuffle(train)
            for x, y in train_iter:
                if y.shape[0] == 0:
                    continue
                optimizer.zero_grad()
                loss = criterion(model(x), y)
                loss.backward()
                optimizer.step()
            vy, vp, vloss = predict(model, val_iter)
            vauc = metrics.roc_auc_score(vy, vp)
            improved = vauc > best_auc
            if improved:
                best_auc, no_imp = vauc, 0
                torch.save(model.state_dict(), cfg.save_path)
            else:
                no_imp += 1
            print(f"Epoch {epoch:2d} | val AUC {vauc:.4f} | val loss {vloss:.4f} | "
                  f"meilleur {best_auc:.4f} | {time.time() - t0:.0f}s"
                  f"{'  <- sauvegarde' if improved else ''}")
            if no_imp >= cfg.patience:
                print(f"Arret anticipe (pas d'amelioration depuis {cfg.patience} epochs)")
                break
    except KeyboardInterrupt:
        print("\nInterruption (Ctrl+C) : evaluation du meilleur epoch obtenu jusque-la...")
        if best_auc < 0:                      # aucun epoch termine : on garde l'etat actuel
            torch.save(model.state_dict(), cfg.save_path)

    model.load_state_dict(torch.load(cfg.save_path, map_location=cfg.device))
    vy, vp, _ = predict(model, val_iter)
    ty, tp, _ = predict(model, test_iter)
    thr_val = best_f1_threshold(vy, vp)

    print("\n" + "=" * 60)
    print("EVALUATION RoBERTa (donnees des auteurs, test utilise une seule fois)")
    print("=" * 60)
    print(f"\nAUC validation : {metrics.roc_auc_score(vy, vp):.4f}")
    print(f"AUC test       : {metrics.roc_auc_score(ty, tp):.4f}")
    print(f"PR-AUC test    : {metrics.average_precision_score(ty, tp):.4f}"
          f"   (hasard = {ty.mean():.3f})")
    report("Seuil par defaut", ty, tp, 0.5)
    report("Seuil choisi sur la VALIDATION (honnete)", ty, tp, thr_val)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="test rapide sur peu de donnees")
    main(parser.parse_args().quick)