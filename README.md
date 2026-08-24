# TCN_Autoencoders_Grupp
Grupp projekt i TCN/ Autoencoders

**Valt spår:** Autoencoders — denoising på Fashion-MNIST (brusiga klädbilder → rena bilder).
Inget dataset behöver laddas ner manuellt, Keras hämtar och cachar det automatiskt.

## Kom igång

> **Viktigt:** TensorFlow stödjer ännu inte Python 3.14. Använd Python 3.12 (eller 3.10).
> Kolla vilka versioner du har med `py -0` om du är osäker.

```bash
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Kör pipelinen

Kör i den här ordningen från `src/`-mappen:

```bash
cd src
python data_prep.py   # laddar Fashion-MNIST, lägger på brus, skriver ut shapes
python train.py       # tränar autoencodern, sparar modell + loss-kurva till reports/
python evaluate.py    # räknar RMSE/PSNR, sparar exempelbilder till reports/
```

Redan körd en gång — i `reports/` finns:
- `training_loss.png` — train/val-loss över 20 epoker (konvergerar runt MSE ≈ 0.008)
- `denoising_examples.png` — brusig/rekonstruerad/ren bild sida vid sida
- `autoencoder.keras` — den tränade modellen (ignoreras av git, se `.gitignore`)

Resultat hittills: **RMSE ≈ 0.09, PSNR ≈ 20.9 dB** på testsetet — modellen tar bort bruset tydligt,
även om detaljer (t.ex. mönster på skjortor) blir något suddiga. Bra utgångspunkt att bygga vidare på.

## Filstruktur

```
src/
├── data_prep.py   # laddning, normalisering, brus
├── model.py       # convolutional autoencoder-arkitektur
├── train.py       # träningsloop
└── evaluate.py     # RMSE/PSNR + visualisering
```

Se docstrings/TODO-kommentarer i varje fil för vad som går att justera (brusnivå, antal filter, epoker).

