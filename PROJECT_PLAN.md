# Projektbas: TCN / Autoencoders (grupp om 3)

Baserat på kraven:
> Välj ett valfritt dataset ni kan jobba med TCN eller Autoencoders. Bygg motsvarande modell. Utvärdera modellen.
> Presentera: beskriv datasetet, presentera resultat/utvärdering, gör en databerättelse (~15 min).

Från lektionerna vet vi vilka exempel som redan är körda i kursen (så vi undviker att bara upprepa dem):
- **TCN**: UrbanSound8K — ljudklassificering med MFCC + dilated convolutions.
- **Autoencoders**: Credit Card Fraud (anomalidetektion), Titanic (imputering av saknade värden), MNIST (denoising), U-Net på ansiktsbilder (bildrestaurering).

Vi väljer alltså ett *eget* dataset i samma anda men inte identiskt, så gruppen visar egen förståelse.

---

## Steg 1 — Välj spår: TCN eller Autoencoder

| | TCN | Autoencoder |
|---|---|---|
| Passar för | Sekvens-/tidsseriedata där ordningen spelar roll (klassificering, prognos) | Rekonstruktion, komprimering, anomalidetektion, denoising, imputering |
| Komplexitet | Kräver att förstå dilated convolutions + residual connections | Enklare grundarkitektur (encoder → latent → decoder) |
| Bra om gruppen vill | Bygga en klassificerare/prognosmodell på tidsserier | Hitta avvikelser, komprimera data eller rensa brus |

**Rekommendation:** bestäm spår gemensamt vecka 1 utifrån vilket dataset ni tycker är mest intressant — datasetet ska styra valet av modell, inte tvärtom.

---

## Steg 2 — Datasetförslag (inte redan körda i kursen)

### TCN-spår (tidsserier/sekvenser)
- **UCI HAR (Human Activity Recognition)** — accelerometer/gyroskop, klassificera aktivitet (gång, sitta, springa). Bra, ren tidsseriedata, tydliga klasser.
- **ECG5000 / PhysioNet arytmi-data** — klassificera EKG-signaler som normala/onormala.
- **GTZAN Music Genre** — ljud, likt UrbanSound8K men annan uppgift (genre istället för ljudtyp).
- **Air Quality / energiförbrukning (t.ex. PJM Hourly Energy, eller svenska SMHI-data)** — prognos (regression) istället för klassificering, ger mer variation mot lektionsexemplet.
- **Stocknet / aktiekurser** — prognos, men var tydlig med att detta är en pedagogisk övning, inte investeringsrådgivning.

### Autoencoder-spår
- **Fashion-MNIST** — denoising eller anomalidetektion (t.ex. träna bara på skor, se om modellen "reagerar" på väskor).
- **KDD Cup 99 / NSL-KDD (nätverksintrång)** — samma idé som Credit Card Fraud men för cybersäkerhet.
- **NASA Turbofan Engine Degradation / bearing vibration dataset** — anomalidetektion i sensordata, industriellt underhåll.
- **Wine Quality / annat tabulärt dataset med saknade värden** — samma upplägg som Titanic-imputeringen men nytt dataset.
- **Egen bildsamling (t.ex. Kaggle "Anomaly Detection" eller MVTec AD)** — Convolutional Autoencoder för att hitta defekter i bilder.

> Tips: leta på Kaggle, UCI Machine Learning Repository eller Hugging Face Datasets. Välj något med rimlig storlek (inte för stort för att träna lokalt, inte för litet för att ge meningsfulla resultat).

---

## Steg 3 — Rollfördelning för 3 personer

Alla bör förstå hela pipelinen (för presentationen), men ni driver var sin del parallellt:

**Person A — Data & förberedelse**
- Hitta och ladda ner datasetet, beskriv det (storlek, källa, klasser/variabler).
- EDA (exploratory data analysis): fördelningar, saknade värden, obalans, visualiseringar.
- Preprocessing: normalisering, train/val/test-split, ev. augmentation (som tidsförskjutning/pitch shift i TCN-exemplet, eller maskering för autoencoder-imputering).

**Person B — Modellbygge & träning**
- Bygg modellarkitekturen (TCN med dilated convolutions + residual connections, eller Autoencoder med encoder/latent/decoder).
- Sätt hyperparametrar (kernel_size, dilations, lagerantal, latent-dimension) och motivera valen.
- Träna modellen, logga träningshistorik (loss-kurvor).

**Person C — Utvärdering & databerättelse**
- Utvärderingsmetrik: classification report/confusion matrix (TCN-klassificering) eller rekonstruktionsfel/RMSE/tröskelvärde (Autoencoder).
- Jämför mot en enkel baseline (t.ex. Isolation Forest som i fraud-exemplet, eller en enklare modell).
- Bygger presentationens narrativ: sätter ihop allas resultat till en 15-minuters databerättelse.

Byt gärna roller halvvägs eller para ihop er för kodgranskning — men ha en tydlig ägare per del så inget faller mellan stolarna.

---

## Steg 4 — Föreslagen repo-struktur

```
TCN_Autoencoders_Grupp/
├── data/                # rådata (eller skript som laddar ner den) — lägg stora filer i .gitignore
├── notebooks/           # EDA och experiment, en notebook per person/fas
├── src/
│   ├── data_prep.py     # laddning, rensning, split
│   ├── model.py          # TCN- eller Autoencoder-arkitektur
│   ├── train.py          # träningsloop
│   └── evaluate.py       # metrik, plots
├── reports/              # figurer, tränings-loss, confusion matrix etc.
├── presentation/         # slides för databerättelsen
├── requirements.txt
└── README.md
```

---

## Steg 5 — Tidsplan (justera efter er deadline)

| Vecka | Fokus |
|---|---|
| 1 | Välj spår + dataset, sätt upp repo, gör EDA |
| 2 | Bygg och träna första modellversionen |
| 3 | Iterera på modellen, utvärdera, jämför med baseline |
| 4 | Sätt ihop resultat, bygg presentation/databerättelse, repetera |

---

## Steg 6 — Presentationens struktur (~15 min, databerättelse)

1. **Hook** (1 min) — varför är detta problem intressant/viktigt?
2. **Datasetet** (2–3 min) — vad är det, hur stort, vilka utmaningar (obalans, brus, saknade värden)?
3. **Metoden** (3–4 min) — varför TCN eller Autoencoder passar just detta problem; kort om arkitekturen (undvik att bara rada upp lager — förklara *varför*).
4. **Resultat** (4–5 min) — nyckeltal, grafer (loss-kurva, confusion matrix eller rekonstruktionsfel), konkreta exempel (rätt/fel klassificerade, upptäckta anomalier).
5. **Tolkning/insikt** (2 min) — vad betyder resultaten i praktiken? Var brister modellen?
6. **Avslutning** (1 min) — sammanfattning + ev. nästa steg.

Gör det till en berättelse, inte en resultatlista: koppla varje del till problemet ni ville lösa.

---

## Checklista innan inlämning

- [ ] Dataset valt och beskrivet (källa, storlek, variabler/klasser)
- [ ] Modell byggd (TCN eller Autoencoder) med motiverade hyperparametrar
- [ ] Modell tränad, träningskurvor sparade
- [ ] Utvärdering gjord med relevant metrik + ev. baseline-jämförelse
- [ ] Presentation (~15 min) förberedd som databerättelse
- [ ] Kod och resultat i repo, roller tydliga
