#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Extraction tarifs EDF (Vert/Bleu) — full Camelot, sans pdfplumber.

pip install "camelot-py[cv]" requests
# + ghostscript / tk installés côté système si nécessaire.

Sortie: insère 6 kVA (abo €/mois, kWh €/kWh) dans ta DB.
"""

import hashlib
import re
import tempfile
import unicodedata
from decimal import Decimal
from pathlib import Path

import camelot
import requests
from requests.adapters import HTTPAdapter, Retry

VERT_URL = "https://particulier.edf.fr/content/dam/2-Actifs/Documents/Offres/grille-prix-vert-electrique.pdf"
BLEU_URL = "https://particulier.edf.fr/content/dam/2-Actifs/Documents/Offres/Grille_prix_Tarif_Bleu.pdf"


# ====================== Réseau / Téléchargement ======================

def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": "tarif-scraper/1.0"})
    retries = Retry(
        total=3, backoff_factor=0.6,
        status_forcelist=[429, 500, 502, 503, 504],
        method_whitelist=["GET", "HEAD"]
    )
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.mount("http://", HTTPAdapter(max_retries=retries))
    return s

def download_to_tmp(url: str) -> Path:
    fn = Path(tempfile.gettempdir()) / Path(url).name
    with _session().get(url, stream=True, timeout=20) as r:
        r.raise_for_status()
        with open(fn, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 15):
                if chunk:
                    f.write(chunk)
    return fn

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ====================== Normalisation / Parsing ======================

def _norm(s: str) -> str:
    """Minuscule, accents retirés, espaces normalisés."""
    if s is None:
        return ""
    s = str(s).replace("\xa0", " ")
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _to_decimal_fr(s: str) -> Decimal:
    """'0,2964 €' -> Decimal('0.2964'); '27,56' -> Decimal('27.56')."""
    s = re.sub(r"[^\d,.\-]", "", s or "")
    if s.count(",") == 1 and s.count(".") == 0:
        s = s.replace(",", ".")
    return Decimal(s)

def parse_pdf_tables(path: Path):
    """Essaye stream puis lattice pour tolérer les maquettes EDF."""
    last_exc = None
    for flavor in ("stream", "lattice"):
        try:
            tables = camelot.read_pdf(str(path), pages="1-end", flavor=flavor)
            if getattr(tables, "n", 0) > 0:
                return tables
        except Exception as e:
            last_exc = e
            continue
    raise RuntimeError(f"Impossible d'extraire des tables depuis {path.name} ({last_exc})")

def _iter_rows(tables):
    for tbl in tables:
        for row in tbl.data:
            yield row

def _guess_header_indices(rows):
    """
    Détecte les index de colonnes:
      - puissance (kVA)
      - abonnement (€/mois)
      - kwh (€/kWh ou cts €/kWh)
    Et un flag kwh_in_cents si l'en-tête contient 'cts'.
    """
    power_idx = None
    abo_idx = None
    kwh_idx = None
    kwh_in_cents = False

    scan = rows[:12] if len(rows) > 12 else rows
    for r in scan:
        for i, cell in enumerate(r):
            n = _norm(cell)
            # Colonne puissance
            if power_idx is None and (re.search(r"\bpuissance\b", n) or re.search(r"\(kva\)", n)):
                power_idx = i
            # Colonne abonnement
            if abo_idx is None and (re.search(r"\babonnement\b|\babo\b", n) or "€/mois" in n or "euro/mois" in n):
                abo_idx = i
            # Colonne kWh
            if kwh_idx is None and (re.search(r"\bkwh\b", n) or re.search(r"prix.*kwh", n) or re.fullmatch(r"base", n)):
                kwh_idx = i
            # Détection "cts"
            if "cts" in n and "kwh" in n:
                kwh_in_cents = True

    return power_idx, abo_idx, kwh_idx, kwh_in_cents


def _find_row_power(rows, power_idx, target="6"):
    """
    Si power_idx est connu, retourne l'index de la ligne dont la cellule power == target (ex: '6').
    Sinon, fallback: cherche une ligne contenant exactement '6' et un en-tête 'kva' dans les 10 premières lignes.
    """
    tnorm = _norm(target)
    if power_idx is not None:
        for idx, r in enumerate(rows):
            cell = r[power_idx] if power_idx < len(r) else ""
            if _norm(cell) == tnorm:
                return idx

    # Fallback tolérant
    header_has_kva = any(re.search(r"\(kva\)", _norm(c)) for rr in rows[:10] for c in rr)
    if header_has_kva:
        for idx, r in enumerate(rows):
            for c in r:
                if _norm(c) == tnorm:
                    return idx
    return None

def _find_row_6kva(rows):
    """Renvoie l'index de la première ligne contenant '6 kVA' (toutes variantes)."""
    for idx, r in enumerate(rows):
        nrow = " | ".join(_norm(c) for c in r)
        if re.search(r"\b6\s*kva\b", nrow):
            return idx
    return None

def _first_decimal_right_of(row, start_idx):
    for j in range(start_idx + 1, len(row)):
        try:
            return _to_decimal_fr(row[j])
        except Exception:
            continue
    return None

def extract_tarifs_6kva_from_tables(tables):
    rows = list(_iter_rows(tables))
    if not rows:
        raise RuntimeError("Aucune table détectée dans le PDF.")

    power_idx, abo_idx, kwh_idx, kwh_in_cents = _guess_header_indices(rows)

    ridx = _find_row_power(rows, power_idx, target="6")
    if ridx is None:
        # Debug utile pour ajuster si la maquette change encore
        preview = "\n".join(" | ".join(r[:4]) for r in rows[:8])
        raise ValueError("Ligne pour '6 (kVA)' introuvable. Aperçu (début):\n" + preview)

    row = rows[ridx]

    # Fallback indices “classiques” si en-têtes absents
    if abo_idx is None and len(row) >= 2:
        abo_idx = 1
    if kwh_idx is None and len(row) >= 3:
        kwh_idx = 2

    abo = None
    kwh = None

    if abo_idx is not None and abo_idx < len(row):
        try:
            abo = _to_decimal_fr(row[abo_idx])
        except Exception:
            abo = _first_decimal_right_of(row, 0)

    if kwh_idx is not None and kwh_idx < len(row):
        try:
            kwh = _to_decimal_fr(row[kwh_idx])
        except Exception:
            base_idx = abo_idx if abo_idx is not None else 0
            kwh = _first_decimal_right_of(row, base_idx)

    # Dernier filet si l'une des valeurs manque
    if abo is None or kwh is None:
        vals = []
        for cell in row:
            try:
                vals.append(_to_decimal_fr(cell))
            except Exception:
                pass
        if len(vals) >= 2:
            # Si kWh est en cts, les deux valeurs seront >1 : on ne peut plus
            # utiliser la règle "kWh<1". On prend l'hypothèse classique :
            # - l’abo est la plus grande valeur monétaire de la ligne
            # - le kWh la plus petite valeur monétaire
            vals_sorted = sorted(vals)
            if kwh is None:
                kwh = vals_sorted[0]
            if abo is None:
                abo = vals_sorted[-1]

    if abo is None or kwh is None:
        raise ValueError(f"Impossible d'extraire abo/kWh sur la ligne 6 kVA: {row}")

    # Conversion si la colonne kWh est en centimes
    if kwh_in_cents:
        kwh = kwh / Decimal("100")

    return (abo, kwh)



# ====================== Facades spécifiques Vert / Bleu ======================

def extract_vert(path: Path):
    tables = parse_pdf_tables(path)
    return extract_tarifs_6kva_from_tables(tables)

def extract_bleu(path: Path):
    tables = parse_pdf_tables(path)
    return extract_tarifs_6kva_from_tables(tables)


# ====================== Main ======================

def main():
    print("Téléchargement des grilles…")
    f_vert = download_to_tmp(VERT_URL)
    f_bleu = download_to_tmp(BLEU_URL)

    print(f"VERT: {f_vert.name}  sha256={sha256_file(f_vert)}")
    print(f"BLEU: {f_bleu.name}  sha256={sha256_file(f_bleu)}")

    print("Parsing PDF (Vert)…")
    abo_vert, kwh_vert = extract_vert(f_vert)

    print("Parsing PDF (Bleu)…")
    abo_bleu, kwh_bleu = extract_bleu(f_bleu)

    print(f"[VERT]  abo(6kVA) = {abo_vert} €/mois   kWh = {kwh_vert} €/kWh")
    print(f"[BLEU]  abo(6kVA) = {abo_bleu} €/mois   kWh = {kwh_bleu} €/kWh")

    print("Insertion en base OK.")

if __name__ == "__main__":
    main()
