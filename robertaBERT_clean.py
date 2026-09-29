"""
robertaBERT_clean.py
--------------------
RoBERTa pre-entraine + tete de classification sur [CLS], protocole PROPRE,
plusieurs graines (seeds) pour mesurer la variabilite.

  --data reduced   fichiers reduced de Jules (ITDBERT\\data\\preprocessing_try)  [defaut]
  --data authors   donnees originales des auteurs (Data\\original data)
  --seeds 42 43 44 une execution complete par graine ; moyenne +/- ecart-type a la fin
  --quick          test rapide (peu de donnees, 2 epochs)

Protocole : 15 % du train = validation (choix de l'epoch, arret anticipe, seuil).
Le test n'est utilise qu'UNE fois par graine, a la fin.
Padding dynamique par lots de longueurs proches (~3x plus rapide, meme modele).
Ctrl+C : arrete proprement, evalue le meilleur epoch de la graine en cours,
puis affiche le resume des graines terminees.
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preTrain.MaskedLM.MaskedUtils import load_dataset

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
    def __init__(self, data_kind, quick):
        base = os.path.dirname(os.path.abspath(__file__))
        if data_kind == "authors":
            d = os.path.join(base, "..", "Data", "original data")
            self.train_path = os.path.join(d, "THNS24_2010.csv")
            self.test_path  = os.path.join(d, "THNS24_2011.csv")
        else:
            d = os.path.join(base, "data", "preprocessing_try")
            self.train_path = os.path.join(d, "reduced_THNS24_2010.csv")
            self.test_path  = os.path.join(d, "reduced_THNS24_2011.csv")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = 64
        self.pad_size = 154            # troncature max (utilisee par load_dataset)
        self.learning_rate = 1e-4
        self.max_epochs = 2 if quick else 20
        self.patience = 4
        models = os.path.join(base, "preTrain", "MaskedLM", "models")
        self.reberta_path = os.path.join(models, "42_H1_24_42")
        self.rebertaTokenizer = load_tokenizer(os.path.join(models, "tokenizer"))
        self.pad_id = self.rebertaTokenizer.pad_token_id


def to_examples(items):
    """(ids_pad, label, seq_len, mask) -> (ids_sans_padding, label)."""
    return [(it[0][:sum(it[3])], it[1]) for it in items]


def batches(examples, cfg, shuffle, rng):
    """Lots au padding dynamique ; en entrainement, lots de longueurs proches."""
    idx = list(range(len(examples)))
    bs = cfg.batch_size
    if shuffle:
        rng.shuffle(idx)
        chunk = bs * 50
        idx = [j for i in range(0, len(idx), chunk)
               for j in sorted(idx[i:i + chunk], key=lambda k: len(examples[k][0]))]
        groups = [idx[i:i + bs] for i in range(0, len(idx), bs)]
        rng.shuffle(groups)
    else:
        idx.sort(key=lambda k: len(examples[k][0]))
        groups = [idx[i:i + bs] for i in range(0, len(idx), bs)]
    for g in groups:
        m = max(len(examples[k][0]) for k in g)
        ids = torch.full((len(g), m), cfg.pad_id, dtype=torch.long)
        att = torch.zeros((len(g), m), dtype=torch.long)
        for r, k in enumerate(g):
            seq = examples[k][0]
            ids[r, :len(seq)] = torch.tensor(seq)
            att[r, :len(seq)] = 1
        y = torch.tensor([examples[k][1] for k in g], dtype=torch.long)
        yield ids.to(cfg.device), att.to(cfg.device), y.to(cfg.device)


class Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained(cfg.reberta_path)
        self.fc2 = nn.Linear(768, 128)
        self.fc3 = nn.Linear(128, 2)

    def forward(self, ids, att):
        pooled = self.roberta(ids, attention_mask=att).pooler_output
        return self.fc3(F.relu(self.fc2(pooled)))


def predict(model, examples, cfg):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for ids, att, y in batches(examples, cfg, False, None):
            out = model(ids, att)
            ys.append(y.cpu().numpy())
            ps.append(F.softmax(out, dim=1)[:, 1].cpu().numpy())
    return np.concatenate(ys), np.concatenate(ps)


def best_f1_threshold(y, p):
    pr, rc, th = metrics.precision_recall_curve(y, p)
    f1 = 2 * pr * rc / (pr + rc + 1e-12)
    return th[np.argmax(f1[:-1])]


def prf(y, p, thr):
    pred = (p >= thr).astype(int)
    return (metrics.precision_score(y, pred, zero_division=0),
            metrics.recall_score(y, pred, zero_division=0),
            metrics.f1_score(y, pred, zero_division=0))


def run_seed(seed, train_all, test, cfg):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    rng = random.Random(seed)
    train, val = train_test_split(train_all, test_size=0.15, random_state=seed,
                                  stratify=[e[1] for e in train_all])
    print(f"\n=== Graine {seed} | train {len(train)} | validation {len(val)} | test {len(test)} ===")
    model = Model(cfg).to(cfg.device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=1e-4)
    w = compute_class_weight("balanced", classes=np.array([0, 1]),
                             y=np.array([e[1] for e in train]))
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float).to(cfg.device))

    best_auc, best_state, no_imp, interrupted = -1.0, None, 0, False
    try:
        for epoch in range(1, cfg.max_epochs + 1):
            t0 = time.time()
            model.train()
            for ids, att, y in batches(train, cfg, True, rng):
                optimizer.zero_grad()
                criterion(model(ids, att), y).backward()
                optimizer.step()
            vy, vp = predict(model, val, cfg)
            vauc = metrics.roc_auc_score(vy, vp)
            if vauc > best_auc:
                best_auc, no_imp = vauc, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                no_imp += 1
            print(f"  Epoch {epoch:2d} | val AUC {vauc:.4f} | meilleur {best_auc:.4f} | "
                  f"{time.time() - t0:.0f}s{'  <- meilleur' if no_imp == 0 else ''}")
            if no_imp >= cfg.patience:
                print(f"  Arret anticipe (pas d'amelioration depuis {cfg.patience} epochs)")
                break
    except KeyboardInterrupt:
        interrupted = True
        print("\n  Interruption (Ctrl+C) : evaluation du meilleur epoch de cette graine...")
    if best_state is not None:
        model.load_state_dict(best_state)

    vy, vp = predict(model, val, cfg)
    ty, tp = predict(model, test, cfg)
    thr = best_f1_threshold(vy, vp)
    res = {"seed": seed, "auc": metrics.roc_auc_score(ty, tp),
           "prauc": metrics.average_precision_score(ty, tp), "thr": thr}
    res["p05"], res["r05"], res["f05"] = prf(ty, tp, 0.5)
    res["pv"], res["rv"], res["fv"] = prf(ty, tp, thr)
    print(f"  TEST | AUC {res['auc']:.4f} | PR-AUC {res['prauc']:.4f} (hasard {ty.mean():.3f}) | "
          f"seuil 0.5 : P {res['p05']:.3f} R {res['r05']:.3f} F1 {res['f05']:.3f} | "
          f"seuil validation {thr:.3f} : P {res['pv']:.3f} R {res['rv']:.3f} F1 {res['fv']:.3f}")
    return res, interrupted


def main(data_kind, seeds, quick):
    cfg = Config(data_kind, quick)
    train_all = to_examples(load_dataset(cfg.train_path, cfg))
    test = to_examples(load_dataset(cfg.test_path, cfg))
    if quick:
        train_all, test = train_all[:600], test[:200]
    print(f"Donnees : {data_kind} | train {len(train_all)} ({np.mean([e[1] for e in train_all])*100:.1f}% malveillant)"
          f" | test {len(test)} ({np.mean([e[1] for e in test])*100:.1f}% malveillant)")

    results = []
    for seed in seeds:
        res, interrupted = run_seed(seed, train_all, test, cfg)
        results.append(res)
        if interrupted:
            break

    print("\n" + "=" * 64)
    print(f"RESUME RoBERTa ({data_kind}) - {len(results)} graine(s), moyenne +/- ecart-type")
    print("=" * 64)
    for name, key in (("AUC test", "auc"), ("PR-AUC test", "prauc"),
                      ("Precision (seuil validation)", "pv"),
                      ("Recall    (seuil validation)", "rv"),
                      ("F1        (seuil validation)", "fv"),
                      ("F1        (seuil 0.5)", "f05")):
        v = np.array([r[key] for r in results])
        print(f"{name:30s}: {v.mean():.4f} +/- {v.std():.4f}   {np.round(v, 4).tolist()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", choices=["reduced", "authors"], default="reduced")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    main(a.data, [a.seeds[0]] if a.quick and a.seeds == [42, 43, 44] else a.seeds, a.quick)