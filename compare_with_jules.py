"""
compare_with_jules.py
Compare tes fichiers regeneres avec ceux de Jules (l'ordre des lignes est ignore).
Usage : python compare_with_jules.py <fichier_regenere.csv> <fichier_de_jules.csv>
"""
import sys, csv
from collections import Counter

def load(path):
    return Counter(tuple(r) for r in csv.reader(open(path)) if r)

a, b = load(sys.argv[1]), load(sys.argv[2])
na, nb = sum(a.values()), sum(b.values())
common = sum((a & b).values())
lab = lambda c: dict(Counter(k[0] for k, v in c.items() for _ in range(v)))
print(f"Regenere : {na} lignes, labels {lab(a)}")
print(f"Jules    : {nb} lignes, labels {lab(b)}")
print(f"Lignes identiques (ordre ignore) : {common}")
print(f"Seulement chez toi : {na - common} | seulement chez Jules : {nb - common}")
print("=> IDENTIQUES" if na == nb == common else "=> DIFFERENTS (voir les nombres ci-dessus)")
