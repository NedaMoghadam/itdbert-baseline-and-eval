
"""
trans_corpus.py
---------------
Pré-entraîne un modèle RoBERTa (MaskedLM) from scratch sur le corpus
de séquences comportementales CERT, avec arrêt anticipé (early stopping)
basé sur un petit ensemble de validation.
"""

import os
import torch

print(f"CUDA disponible : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU : {torch.cuda.get_device_name(0)}")

# ─── Chemins ────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
CORPUS_TXT    = os.path.join(BASE_DIR, "data", "330CertAdd.txt")
TOKENIZER_DIR = os.path.join(BASE_DIR, "models", "tokenizer")
MODEL_DIR     = os.path.join(BASE_DIR, "models", "42_H1_24_42")

os.makedirs(TOKENIZER_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ─── 1. Entraînement du tokenizer BPE ───────────────────────────────────────
print("\n[1/4] Entraînement du tokenizer BPE...")

from tokenizers import ByteLevelBPETokenizer

tokenizer_bpe = ByteLevelBPETokenizer()
tokenizer_bpe.train(
    files=CORPUS_TXT.replace("\\", "/"),
    vocab_size=600,
    min_frequency=1,
    special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"],
)
tokenizer_bpe.save_model(TOKENIZER_DIR)
print(f"   Tokenizer sauvegardé dans : {TOKENIZER_DIR}")

# ─── 2. Configuration du modèle RoBERTa ─────────────────────────────────────
print("\n[2/4] Configuration RoBERTa...")

from transformers import RobertaConfig

config = RobertaConfig(
    vocab_size=600,
    max_position_embeddings=512,
    num_attention_heads=8,
    num_hidden_layers=2,
    type_vocab_size=1,
)

from transformers import RobertaForMaskedLM
model = RobertaForMaskedLM(config=config)
print(f"   Nombre de paramètres : {model.num_parameters():,}")

# ─── 3. Chargement du tokenizer rapide ──────────────────────────────────────
print("\n[3/4] Chargement du tokenizer rapide...")

from tokenizers import ByteLevelBPETokenizer as _BPE
from tokenizers.processors import RobertaProcessing
from transformers import PreTrainedTokenizerFast

vocab_file  = os.path.join(TOKENIZER_DIR, "vocab.json")
merges_file = os.path.join(TOKENIZER_DIR, "merges.txt")

_bpe = _BPE(vocab=vocab_file, merges=merges_file)
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
print(f"   Taille du vocabulaire : {tokenizer.vocab_size}")

# ─── 4. Dataset ─────────────────────────────────────────────────────────────
print("\n[4/4] Chargement du dataset...")

from torch.utils.data import Dataset

class BehaviorLineDataset(Dataset):
    def __init__(self, tokenizer, file_path: str, block_size: int = 128, lines=None):
        self.examples = []
        if lines is None:
            assert os.path.isfile(file_path), f"Fichier introuvable : {file_path}"
            print(f"   Lecture de {file_path}...")
            with open(file_path, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]
            print(f"   {len(lines)} séquences chargées.")

        for line in lines:
            ids = tokenizer.encode(line, add_special_tokens=True)
            if len(ids) > block_size:
                ids = ids[:block_size]
            self.examples.append(ids)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i], dtype=torch.long)


# EARLY STOPPING : on reserve 10% du corpus comme ensemble de validation,
# pour pouvoir mesurer si le modele s'ameliore encore ou non.
with open(CORPUS_TXT, "r", encoding="utf-8") as f:
    all_lines = [line.strip() for line in f if line.strip()]

split_idx    = int(len(all_lines) * 0.9)
train_lines  = all_lines[:split_idx]
eval_lines   = all_lines[split_idx:]
print(f"   Train : {len(train_lines)} séquences | Validation : {len(eval_lines)} séquences")

train_dataset = BehaviorLineDataset(tokenizer=tokenizer, file_path=CORPUS_TXT, block_size=128, lines=train_lines)
eval_dataset  = BehaviorLineDataset(tokenizer=tokenizer, file_path=CORPUS_TXT, block_size=128, lines=eval_lines)

# ─── 5. Data Collator (masquage aléatoire 15 %) ─────────────────────────────
from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15,
)

# ─── 6. Arguments d'entraînement ────────────────────────────────────────────
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback

training_args = TrainingArguments(
    output_dir=MODEL_DIR,
    num_train_epochs=25,               # plafond haut ; l'early stopping coupera avant si possible
    per_device_train_batch_size=128,
    eval_strategy="epoch",             # EARLY STOPPING : evalue a chaque epoch
    save_strategy="epoch",             # doit correspondre a eval_strategy pour load_best_model_at_end
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    logging_steps=50,
    report_to="none",
)

# ─── 7. Entraînement ────────────────────────────────────────────────────────
print("\n[5/5] Démarrage du pré-entraînement MaskedLM...")
print(f"   Epochs (max) : {training_args.num_train_epochs}")
print(f"   Batch size   : {training_args.per_device_train_batch_size}")
print(f"   Device       : {'GPU' if torch.cuda.is_available() else 'CPU'}\n")

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    # EARLY STOPPING : arrete automatiquement si la loss de validation
    # ne s'ameliore plus pendant 3 evaluations (= 3 epochs) d'affilee.
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

trainer.train()

trainer.save_model(MODEL_DIR)
tokenizer.save_pretrained(MODEL_DIR)

print(f"\nModèle pré-entraîné sauvegardé dans : {MODEL_DIR}")
print("Contenu du dossier :")
for f in os.listdir(MODEL_DIR):
    print(f"  {f}")