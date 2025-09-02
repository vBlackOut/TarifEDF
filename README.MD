

<img src="https://github.com/vBlackOut/TarifEDF/blob/main/tarif.png"/>



le lancement du script ce fait simplement.
la fonction main est içi.

```
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
    ligne_bleu_6 = t.get_ligne("bleu", "9")   # 6 kVA

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

    # Insertion DB (adapte les champs selon ton ORM)
    Prix_EDF_vert.create(prix=float(base_v) if base_v is not None else None,
                         abo=float(abo_v))
    Prix_EDF_bleu.create(prix=float(base_b) if base_b is not None else None,
                         abo=float(abo_b))

    print("Insertion en base OK.")
```
