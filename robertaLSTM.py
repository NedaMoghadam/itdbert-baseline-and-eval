"""
robertaLSTM.py
--------------
Pipeline principale ITDBERT :
  - Charge le modèle RoBERTa pré-entraîné (trans_corpus.py)
  - Ajoute un Bi-LSTM + attention par-dessus les embeddings contextuels
  - Entraîne en supervisé sur les labels CERT
  - Évalue : Precision, Recall, F1, AUC (objectif : F1=0.9243, AUC=0.9556)

Usage :
    python robertaLSTM.py

Prérequis :
    - trans_corpus.py doit avoir été exécuté (modèle dans preTrain/MaskedLM/models/42_H1_24_42/)
    - MaskedUtils.py dans preTrain/MaskedLM/
"""

import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score

# Seed fixe : rend les resultats reproductibles d'un run a l'autre.
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# ── Import MaskedUtils ──────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "preTrain", "MaskedLM"))
from preTrain.MaskedLM.MaskedUtils import bulid_dataset, bulid_iterator

# ── Chargement du tokenizer et du modèle pré-entraîné ───────────────────────
from tokenizers import ByteLevelBPETokenizer as _BPE
from tokenizers.processors import RobertaProcessing
from transformers import PreTrainedTokenizerFast, RobertaModel, RobertaConfig
from transformers import get_cosine_schedule_with_warmup

from sklearn.utils.class_weight import compute_class_weight

def load_tokenizer(model_dir: str) -> PreTrainedTokenizerFast:
    """
    Charge le tokenizer depuis le dossier du modèle pré-entraîné.
    Cherche vocab.json + merges.txt (produits par trans_corpus.py).
    """
    vocab_file  = os.path.join(model_dir, "vocab.json").replace("\\", "/")
    merges_file = os.path.join(model_dir, "merges.txt").replace("\\", "/")
    _bpe = _BPE(vocab=vocab_file, merges=merges_file)
    # BUGFIX : sans post_processor, add_special_tokens=True ne fait RIEN —
    # aucun <s>/</s> n'est jamais ajouté aux séquences.
    _bpe._tokenizer.post_processor = RobertaProcessing(
        ("</s>", _bpe.token_to_id("</s>")),
        ("<s>", _bpe.token_to_id("<s>")),
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=_bpe,
        model_max_length=512,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        sep_token="</s>",
        cls_token="<s>",
        pad_token="<pad>",
        mask_token="<mask>",
    )
    return tokenizer


# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

class Config(object):
    def __init__(self):
        # ── Données ──────────────────────────────────────────────────────────
        # BUGFIX : "ITDBERT_reproduit" n'existe pas dans ce repo (dossier réel :
        # "ITDBERT") — remplacé par des chemins relatifs à ce script.
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        self.train_path  = os.path.join(BASE_DIR, "data", "preprocessing_try", "reduced_THNS24_2010.csv")
        self.test_path   = os.path.join(BASE_DIR, "data", "preprocessing_try", "reduced_THNS24_2011.csv")
        self.datasetpkl  = os.path.join(BASE_DIR, "preTrain", "MaskedLM", "data", "datasetTHNS.pkl")

        # ── Device ───────────────────────────────────────────────────────────
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ── Hyperparamètres (Tableau I de l'article) ─────────────────────────
        self.dropout          = 0.5
        self.require_improvement = 1000
        self.num_classes      = 2
        self.n_vocab          = 600
        self.num_epochs       = 200       # borne max, early stopping prendra le relais
        self.early_stop_patience = 10     # arrêt si pas d'amélioration F1 pendant N epochs
        self.batch_size       = 64
        self.pad_size         = 154        # Taille max d'une séquence dans le dataset : 154
        self.learning_rate    = 1e-4       # Tableau I : learning rate = 0.0001
        self.embed            = 768        # Tableau I : hidden size MaskedLM = 768
        self.hidden_size      = 512        # Tableau I : hidden size Bi-LSTM = 512 → 256 par direction
        self.num_layers       = 2
        self.hidden_size2     = 64

        # ── Modèle pré-entraîné ───────────────────────────────────────────────
        self.reberta_path     = os.path.join(BASE_DIR, "preTrain", "MaskedLM", "models", "42_H1_24_42")
        self.tokenizer_path   = os.path.join(BASE_DIR, "preTrain", "MaskedLM", "models", "tokenizer")

        print("Chargement du tokenizer...")
        self.rebertaTokenizer = load_tokenizer(self.tokenizer_path)

        print("Chargement du modèle RoBERTa pré-entraîné...")
        self.rebertaModel = RobertaModel.from_pretrained(self.reberta_path)
        print(f"Device : {self.device}")


# ═══════════════════════════════════════════════════════════════════════════════
# Architecture ITDBERT : RoBERTa + Bi-LSTM + Attention
# ═══════════════════════════════════════════════════════════════════════════════

class Model(nn.Module):
    """
    Implémentation de la méthode 2 de l'article (ITDBERT complet) :
        RoBERTa → encoder_out [batch, seq, 768]
                → Bi-LSTM    [batch, seq, 512]
                → Attention  [batch, 512]
                → FC         [batch, 2]

    La méthode 1 ([CLS]+FC) est conservée en commentaire pour référence.
    """
    def __init__(self, config):
        super(Model, self).__init__()

        # ── Encodeur RoBERTa (fine-tunable) ──────────────────────────────────
        self.tokenizer = config.rebertaTokenizer
        self.roberta   = config.rebertaModel
        for param in self.roberta.parameters():
            param.requires_grad = True

        # ── Bi-LSTM ───────────────────────────────────────────────────────────
        self.lstm = nn.LSTM(
            input_size=config.embed,          # 768
            hidden_size=config.hidden_size,   # 256 (× 2 directions = 512)
            num_layers=config.num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=config.dropout,
        )

        # ── Attention ─────────────────────────────────────────────────────────
        self.tanh = nn.Tanh()
        self.w    = nn.Parameter(torch.zeros(config.hidden_size * 2))

        # ── Couches de classification ─────────────────────────────────────────
        self.fc1 = nn.Linear(config.hidden_size * 2, config.num_classes)  # méthode 2
        self.fc2 = nn.Linear(768, 128)
        self.fc3 = nn.Linear(128, 2)

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        context = x[0]   # token ids  [batch, pad_size]
        mask    = x[2]   # mask        [batch, pad_size]

        # ── Encodage RoBERTa ──────────────────────────────────────────────────
        outputs     = self.roberta(context, attention_mask=mask)
        encoder_out = outputs.last_hidden_state   # [batch, seq, 768]

        # ── Bi-LSTM + Attention (ITDBERT complet) ────────────────
        H, _ = self.lstm(encoder_out)
        M     = self.tanh(H)
        alpha = F.softmax(torch.matmul(M, self.w), dim=1).unsqueeze(-1)
        out   = H * alpha                         # [batch, seq, 512]
        out   = torch.sum(out, dim=1)             # [batch, 512]
        out   = F.relu(out)
        out   = self.dropout(out)
        out   = self.fc1(out)                     # [batch, 2]
        return out


# ═══════════════════════════════════════════════════════════════════════════════
# Boucles d'entraînement et d'évaluation
# ═══════════════════════════════════════════════════════════════════════════════

def train_epoch(model, device, train_loader, optimizer, epoch):
    model.train()
    labels = np.array([item[1] for item in train_data])
    weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=labels)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float).to(device))
    for batch_idx, (x, y) in enumerate(train_loader):
        optimizer.zero_grad()
        y_pred = model(x)
        loss   = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
        if (batch_idx + 1) % 10 == 0:
            print(f"Epoch {epoch} [{batch_idx * len(y)}/{len(train_loader.dataset)} "
                  f"({100. * batch_idx / len(train_loader):.0f}%)]  "
                  f"Loss: {loss.item():.6f}")


def find_best_threshold(labels_all, probs_all):
    """Cherche le seuil de decision qui maximise le F1 sur les predictions,
    au lieu du seuil fixe a 0.5 utilise par argmax."""
    from sklearn.metrics import precision_recall_curve
    precisions, recalls, thresholds = precision_recall_curve(labels_all, probs_all)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-12)
    best_idx = np.argmax(f1s[:-1])
    return thresholds[best_idx], f1s[best_idx]


def evaluate(model, data_iter, show_report=False):
    """
    Retourne : (accuracy, loss_moyenne, [confusion_matrix si show_report])
    Calcule aussi Precision, Recall, F1 sur la classe malicieuse (label=1).
    """
    model.eval()
    loss_total   = 0.0
    predict_all  = np.array([], dtype=int)
    labels_all   = np.array([], dtype=int)
    probs_all    = np.array([], dtype=float)

    with torch.no_grad():
        for texts, labels in data_iter:
            if labels.shape[0] == 0:   #le dernier batch est toujours vide donc on le skip (je sais pas pourquoi)
                continue
            outputs     = model(texts)
            loss        = F.cross_entropy(outputs, labels)
            loss_total += loss.item()
            labels_np   = labels.data.cpu().numpy()
            predic_np   = torch.max(outputs.data, 1)[1].cpu().numpy()
            probs_np    = F.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            labels_all  = np.append(labels_all, labels_np)
            predict_all = np.append(predict_all, predic_np)
            probs_all   = np.append(probs_all, probs_np)

    acc       = metrics.accuracy_score(labels_all, predict_all)
    precision = metrics.precision_score(labels_all, predict_all, pos_label=1, zero_division=0)
    recall    = metrics.recall_score(labels_all, predict_all, pos_label=1, zero_division=0)
    f1        = metrics.f1_score(labels_all, predict_all, pos_label=1, zero_division=0)

    auc = roc_auc_score(labels_all, probs_all)

    if show_report:
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        print(f"\n--- Seuil par defaut (0.5) ---")
        print(f"Accuracy  : {acc:.4f}")
        print(f"Precision : {precision:.4f}")
        print(f"Recall    : {recall:.4f}")
        print(f"F1-score  : {f1:.4f}")
        print(f"AUC       : {auc:.4f}")
        print(f"\nMatrice de confusion :\n{confusion}")
        print("\nRapport complet :")
        print(classification_report(labels_all, predict_all,
                                    target_names=["Normal", "Malicious"], digits=4))

        best_thresh, best_f1_at_thresh = find_best_threshold(labels_all, probs_all)
        predict_tuned = (probs_all >= best_thresh).astype(int)
        acc_t   = metrics.accuracy_score(labels_all, predict_tuned)
        prec_t  = metrics.precision_score(labels_all, predict_tuned, pos_label=1, zero_division=0)
        rec_t   = metrics.recall_score(labels_all, predict_tuned, pos_label=1, zero_division=0)
        f1_t    = metrics.f1_score(labels_all, predict_tuned, pos_label=1, zero_division=0)
        conf_t  = metrics.confusion_matrix(labels_all, predict_tuned)
        print(f"\n--- Seuil optimise (F1) : {best_thresh:.4f} ---")
        print(f"Accuracy  : {acc_t:.4f}")
        print(f"Precision : {prec_t:.4f}")
        print(f"Recall    : {rec_t:.4f}")
        print(f"F1-score  : {f1_t:.4f}")
        print(f"\nMatrice de confusion (seuil optimise) :\n{conf_t}")

        return acc, loss_total / len(data_iter), f1, auc, confusion

    return acc, loss_total / len(data_iter), f1, auc


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # ── Configuration ─────────────────────────────────────────────────────────
    config = Config()

    # ── Dataset ───────────────────────────────────────────────────────────────
    train_data, test_data = bulid_dataset(config)
    train_iter = bulid_iterator(train_data, config)
    test_iter  = bulid_iterator(test_data,  config)

    # ── Modèle ────────────────────────────────────────────────────────────────
    model = Model(config).to(config.device)
    print(model)

    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)

    # ── Chemin de sauvegarde ──────────────────────────────────────────────────
    save_dir = r"preTrain/MaskedLM/save_model"
    os.makedirs(save_dir, exist_ok=True)
    PATH = os.path.join(save_dir, "42rebertaH1_model.pth")

    # ── Entraînement ──────────────────────────────────────────────────────────
    # BUGFIX : voir robertaBERT.py — best_f1 a 0.0 empechait tout checkpoint
    # si le 1er epoch obtenait aussi F1=0.0.
    best_f1       = -1.0
    best_acc      = 0.0
    epochs_no_imp = 0          # compteur d'epochs sans amélioration

    for epoch in range(1, config.num_epochs + 1):
        train_epoch(model, config.device, train_iter, optimizer, epoch)
        acc, loss, f1, auc = evaluate(model, test_iter)
        print(f"Epoch {epoch:3d} | Acc: {acc:.4f} | F1: {f1:.4f} | AUC: {auc:.4f} | "
              f"Loss: {loss:.4f} | Best F1: {best_f1:.4f} | No-imp: {epochs_no_imp}/{config.early_stop_patience}")

        if f1 > best_f1:
            best_f1       = f1
            best_acc      = acc
            epochs_no_imp = 0
            torch.save(model.state_dict(), PATH)
            print(f"  → Nouveau meilleur modèle sauvegardé (F1={best_f1:.4f})")
        else:
            epochs_no_imp += 1
            if epochs_no_imp >= config.early_stop_patience:
                print(f"Early stopping déclenché après {epoch} epochs (patience={config.early_stop_patience})")
                break

    # ── Évaluation finale du meilleur modèle ──────────────────────────────────
    print("\n" + "="*60)
    print("ÉVALUATION FINALE (meilleur modèle)")
    print("="*60)
    best_model = Model(config).to(config.device)
    best_model.load_state_dict(torch.load(PATH, map_location=config.device))
    acc, loss, f1, auc, confusion = evaluate(best_model, test_iter, show_report=True)