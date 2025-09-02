# edftarifs.py
# -*- coding: utf-8 -*-
"""
Mini-lib + script pour extraire les tarifs EDF (Vert/Bleu) avec Camelot
et afficher des logs exactement au format demandé.

Dépendances:
  pip install "camelot-py[cv]" requests

Côté système: Ghostscript + Tk/Qt selon l'install Camelot.
"""

from __future__ import annotations
import re, unicodedata, hashlib, tempfile
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import camelot
import requests
from requests.adapters import HTTPAdapter, Retry


VERT_URL = "https://particulier.edf.fr/content/dam/2-Actifs/Documents/Offres/grille-prix-vert-electrique.pdf"
BLEU_URL = "https://particulier.edf.fr/content/dam/2-Actifs/Documents/Offres/Grille_prix_Tarif_Bleu.pdf"


# ========================= Réseau / Téléchargement =========================

def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": "edftarifs/0.1"})
    try:
        retries = Retry(total=3, backoff_factor=0.6,
                        status_forcelist=[429, 500, 502, 503, 504],
                        allowed_methods=["GET", "HEAD"])
    except TypeError:
        # fallback vieux urllib3
        retries = Retry(total=3, backoff_factor=0.6,
                        status_forcelist=[429, 500, 502, 503, 504],
                        method_whitelist=["GET", "HEAD"])
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.mount("http://", HTTPAdapter(max_retries=retries))
    return s

def _download_to_tmp(url: str) -> Path:
    fn = Path(tempfile.gettempdir()) / Path(url).name
    with _session().get(url, stream=True, timeout=20) as r:
        r.raise_for_status()
        with open(fn, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 15):
                if chunk: f.write(chunk)
    return fn

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""): h.update(chunk)
    return h.hexdigest()


# ========================= Normalisation / Parsing =========================

def _norm(s: str) -> str:
    if s is None: return ""
    s = str(s).replace("\xa0", " ")
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _to_decimal_fr(s: str) -> Decimal:
    s = re.sub(r"[^\d,.\-]", "", s or "")
    if s.count(",") == 1 and s.count(".") == 0:
        s = s.replace(",", ".")
    return Decimal(s)

def _read_tables(path: Path):
    # Essaye stream puis lattice pour couvrir les deux maquettes
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


@dataclass
class TarifsLigne:
    puissance_kva: str
    abo_eur_mois: Decimal
    base_eur_kwh: Optional[Decimal]
    hp_eur_kwh:   Optional[Decimal]
    hc_eur_kwh:   Optional[Decimal]

    def pick(self, option: str) -> Tuple[Decimal, Optional[Decimal]]:
        o = option.lower()
        if o == "base":
            return (self.abo_eur_mois, self.base_eur_kwh)
        elif o in ("hp/hc", "hp_hc", "heures pleines/creuses", "heures pleines - heures creuses"):
            return (self.abo_eur_mois, None)
        else:
            raise ValueError("option inconnue (attendu: 'base' ou 'hp/hc')")


class EDFTarifs:
    def __init__(self):
        self.paths: Dict[str, Path] = {}
        self.tables: Dict[str, List[List[str]]] = {}
        self.meta: Dict[str, dict] = {}

    def load_default(self) -> "EDFTarifs":
        self.load_url("vert", VERT_URL)
        self.load_url("bleu", BLEU_URL)
        return self

    def load_url(self, name: str, url: str) -> "EDFTarifs":
        p = _download_to_tmp(url)
        self.paths[name] = p
        self.meta[name] = {"sha256": sha256_file(p), "url": url}
        self.tables[name] = list(_iter_rows(_read_tables(p)))
        return self

    def _guess_columns(self, rows: List[List[str]]) -> dict:
        power = abo = base = hp = hc = None
        base_cents = hp_cents = hc_cents = False

        scan = rows[:14] if len(rows) > 14 else rows
        for r in scan:
            for i, cell in enumerate(r):
                n = _norm(cell)
                if power is None and (re.search(r"\bpuissance\b", n) or "(kva)" in n):
                    power = i
                if abo is None and (re.search(r"\babonnement\b|\babo\b", n) or "€/mois" in n or "euro/mois" in n):
                    abo = i
                if base is None and (n == "base" or re.search(r"\bprix.*kwh\b", n)):
                    base = i
                    base_cents = ("cts" in n and "kwh" in n)
                if hp is None and re.search(r"\bhp\b|heures pleines", n):
                    hp = i
                    hp_cents = ("cts" in n and "kwh" in n)
                if hc is None and re.search(r"\bhc\b|heures creuses", n):
                    hc = i
                    hc_cents = ("cts" in n and "kwh" in n)

        return dict(power=power, abo=abo, base=base, hp=hp, hc=hc,
                    base_cents=base_cents, hp_cents=hp_cents, hc_cents=hc_cents)

    def _find_row_by_power(self, rows: List[List[str]], power_idx: Optional[int], target: str) -> int:
        t = _norm(target)
        if power_idx is not None:
            for idx, r in enumerate(rows):
                cell = r[power_idx] if power_idx < len(r) else ""
                if _norm(cell) == t:
                    return idx
        # fallback si l'entête contient (kVA)
        header_has_kva = any("(kva)" in _norm(c) for rr in rows[:10] for c in rr)
        if header_has_kva:
            for idx, r in enumerate(rows):
                for c in r:
                    if _norm(c) == t:
                        return idx
        raise ValueError(f"Ligne puissance '{target}' introuvable")

    @staticmethod
    def _get_decimal(row: List[str], idx: Optional[int]) -> Optional[Decimal]:
        if idx is None or idx >= len(row): return None
        try:
            return _to_decimal_fr(row[idx])
        except Exception:
            return None

    def _convert_if_cents(self, val: Optional[Decimal], flag_cents: bool) -> Optional[Decimal]:
        if val is None: return None
        return (val / Decimal("100")) if flag_cents else val

    def get_ligne(self, name: str, puissance_kva: str) -> TarifsLigne:
        if name not in self.tables:
            raise KeyError(f"'{name}' non chargé. Utilise load_default() ou load_url().")

        rows = self.tables[name]
        cols = self._guess_columns(rows)
        ridx = self._find_row_by_power(rows, cols["power"], puissance_kva)
        row = rows[ridx]

        abo = self._get_decimal(row, cols["abo"])
        base = self._get_decimal(row, cols["base"])
        hp   = self._get_decimal(row, cols["hp"])
        hc   = self._get_decimal(row, cols["hc"])

        base = self._convert_if_cents(base, cols["base_cents"])
        hp   = self._convert_if_cents(hp,   cols["hp_cents"])
        hc   = self._convert_if_cents(hc,   cols["hc_cents"])

        # Filets de sécurité
        if abo is None or (base is None and hp is None and hc is None):
            vals = []
            for c in row:
                try: vals.append(_to_decimal_fr(c))
                except Exception: pass
            if vals:
                vals_sorted = sorted(vals)
                if abo is None: abo = vals_sorted[-1]
                if base is None and hp is None and hc is None:
                    base = vals_sorted[0]

        if abo is None:
            raise ValueError(f"Abonnement introuvable pour {name} {puissance_kva} kVA (ligne={row})")

        return TarifsLigne(
            puissance_kva=puissance_kva,
            abo_eur_mois=abo,
            base_eur_kwh=base,
            hp_eur_kwh=hp,
            hc_eur_kwh=hc,
        )


# ========================= Script (logs exacts) =========================

def main():
    print("Téléchargement des grilles…")
    t = EDFTarifs().load_default()

    vert_name = t.paths["vert"].name
    bleu_name = t.paths["bleu"].name
    print(f"VERT: {vert_name}  sha256={t.meta['vert']['sha256']}")
    print(f"BLEU: {bleu_name}  sha256={t.meta['bleu']['sha256']}")

    print("Parsing PDF (Vert)…")
    ligne_vert_6 = t.get_ligne("vert", "6")   # 6 kVA
    print("Parsing PDF (Bleu)…")
    ligne_bleu_6 = t.get_ligne("bleu", "6")   # 6 kVA

    # On privilégie 'base' ; si base None (HP/HC only), on affiche HP/HC moyenné simple
    abo_v, base_v = ligne_vert_6.pick("base")
    abo_b, base_b = ligne_bleu_6.pick("base")

    # Format: 2 décimales pour abo, 4 pour kWh
    def fmt_abo(x: Decimal) -> str:
        return f"{float(x):.2f}"
    def fmt_kwh(x: Optional[Decimal]) -> str:
        return f"{float(x):.4f}" if x is not None else "N/A"

    print(f"[VERT]  abo(6kVA) = {fmt_abo(abo_v)} €/mois   kWh = {fmt_kwh(base_v)} €/kWh")
    print(f"[BLEU]  abo(6kVA) = {fmt_abo(abo_b)} €/mois   kWh = {fmt_kwh(base_b)} €/kWh")



# ========================= Helpers internes =========================

def _to_decimal_fr(s: str) -> Decimal:
    s = re.sub(r"[^\d,.\-]", "", s or "")
    if s.count(",") == 1 and s.count(".") == 0:
        s = s.replace(",", ".")
    return Decimal(s)


if __name__ == "__main__":
    main()
