"""
MaskedUtils.py
--------------
Utilitaires pour charger et itérer sur le dataset CERT
dans la pipeline MaskedLM (robertaLSTM.py).

Fonctions exportées :
    bulid_dataset(config)  -> (train, test)
    bulid_iterator(dataset, config) -> DatasetIterator
"""

from tqdm import tqdm
import torch
import time
from datetime import timedelta
import pickle as pkl
import os
import csv
import random

# Tokens spéciaux utilisés lors de la tokenisation
PAD, CLS = '<pad>', '<s>'


def load_dataset(file_path, config):
    """
    Lit un CSV CERT prétraité et retourne une liste de tuples :
        (token_ids, label, seq_len, mask)

    Format CSV attendu : label,token1,token2,...,tokenN
    Le label 0 = normal, tout autre = malicieux.
    """
    contents = []

    with open(file_path, 'r', encoding='UTF-8') as f:
        reader = csv.reader(f)
        for line in tqdm(reader, desc=f"Chargement {os.path.basename(file_path)}"):
            if not line:
                continue

            data  = ' '.join(line[1:])
            label = 0 if int(line[0]) == 0 else 1

            token_ids = config.rebertaTokenizer.encode(
                data, add_special_tokens=True
            )

            seq_len = len(token_ids)
            pad_size = config.pad_size

            # ── Padding / troncature ─────────────────────────────────────
            if pad_size:
                if len(token_ids) < pad_size:
                    mask     = [1] * len(token_ids) + [0] * (pad_size - len(token_ids))
                    token_ids = token_ids + [0] * (pad_size - len(token_ids))
                else:
                    mask      = [1] * pad_size
                    token_ids = token_ids[:pad_size]
                    seq_len   = pad_size

            contents.append((token_ids, label, seq_len, mask))

    random.shuffle(contents)
    return contents


def bulid_dataset(config):
    """
    Construit (ou charge depuis cache .pkl) les datasets train et test.

    Si config.datasetpkl existe déjà, on le recharge directement pour
    éviter de re-tokeniser à chaque run.
    """
    if os.path.exists(config.datasetpkl):
        print(f"Chargement du cache : {config.datasetpkl}")
        dataset = pkl.load(open(config.datasetpkl, 'rb'))
        train   = dataset['train']
        test    = dataset['test']
    else:
        print("Construction du dataset (première fois, peut prendre quelques minutes)...")
        train = load_dataset(config.train_path, config)
        test  = load_dataset(config.test_path,  config)
        dataset = {'train': train, 'test': test}
        os.makedirs(os.path.dirname(config.datasetpkl), exist_ok=True)
        pkl.dump(dataset, open(config.datasetpkl, 'wb'))
        print(f"Dataset mis en cache : {config.datasetpkl}")

    print(f"Train : {len(train)} séquences | Test : {len(test)} séquences")
    return train, test


class DatasetIterator(object):
    """
    Itérateur custom qui découpe le dataset en mini-batchs et les
    convertit en tenseurs PyTorch sur le bon device.
    """

    def __init__(self, dataset, batch_size, device):
        self.batch_size = batch_size
        self.dataset    = dataset
        self.n_batches  = len(dataset) // batch_size
        self.residue    = (len(dataset) % batch_size != 0)  # batch final incomplet
        self.index      = 0
        self.device     = device

    def _to_tensor(self, datas):
        x       = torch.LongTensor([item[0] for item in datas]).to(self.device)
        y       = torch.LongTensor([item[1] for item in datas]).to(self.device)
        seq_len = torch.LongTensor([item[2] for item in datas]).to(self.device)
        mask    = torch.LongTensor([item[3] for item in datas]).to(self.device)
        return (x, seq_len, mask), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            # Dernier batch (taille < batch_size)
            batches = self.dataset[self.index * self.batch_size:]
            self.index += 1
            return self._to_tensor(batches)
        elif self.index > self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.dataset[
                self.index * self.batch_size: (self.index + 1) * self.batch_size
            ]
            self.index += 1
            return self._to_tensor(batches)

    def __iter__(self):
        return self

    def __len__(self):
        return self.n_batches + (1 if self.residue else 0)


def bulid_iterator(dataset, config):
    return DatasetIterator(dataset, config.batch_size, config.device)


def get_time_dif(start_time):
    end_time = time.time()
    return timedelta(seconds=int(round(end_time - start_time)))