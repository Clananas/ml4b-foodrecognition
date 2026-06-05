# Training — wo, womit und wie

Dieses Dokument beantwortet *eine* Frage so konkret wie möglich:
**„Wo und wie wird das Machine-Learning-Modell trainiert?"**

Wenn dich der Code interessiert, findest du am Ende eine Liste der relevanten
Dateien. Wenn du nur verstehen willst, was passiert, lies die ersten Abschnitte.

---

## 1. Was wir überhaupt trainieren

Drei der vier ML-Bausteine in dieser App sind **vortrainierte** Modelle aus dem
Internet — wir trainieren sie *nicht* selbst, weil das ohne GPU und ohne riesige
Bilddatensätze unrealistisch wäre:

| Baustein | Modell | Trainiert von |
|---|---|---|
| Segmentierung (wo ist das Essen?) | FastSAM | Ultralytics (vortrainiert) |
| Erkennung (welches Essen?) | CLIP | OpenAI (vortrainiert) |
| Tiefenschätzung (optional) | Depth Anything V2 | DepthAnything-Team (vortrainiert) |

**Wir trainieren genau einen Baustein selbst**: das **Portions-/Volumen-Modell**.
Es bekommt zwei Zahlen pro Speise (Fläche und Höhe, beide in Zentimetern) und
soll daraus das Volumen in Millilitern vorhersagen. Aus dem Volumen wird über
eine Nährwerttabelle die Masse, daraus die Kalorien und Makronährstoffe.

Konzeptuell:

```
   Eingabe:  Fläche [cm²]   Höhe [cm]      ←  aus dem Foto gemessen
                  ↓               ↓
              ┌───────────────────────┐
              │   trainiertes Modell  │   ←  HIER findet das Training statt
              └───────────────────────┘
                          ↓
   Ausgabe:           Volumen [mL]        →  × Dichte → Masse → Kalorien & Makros
```

---

## 2. Woher die Trainingsdaten kommen

Wir benutzen den öffentlichen Datensatz **ECUSTFD**
([Liang & Li, 2017](https://arxiv.org/abs/1705.07632)), der genau für dieses
Problem gebaut wurde: 145 Speise-Portionen, jede gewogen mit einer Küchenwaage
und volumenmäßig per Wasserverdrängung gemessen.

```
145 Portionen × { Foto von oben + Foto von der Seite + 1-Yuan-Münze als Maßstab + Gewicht + Volumen }
```

Beispielzeile aus den extrahierten Features
(siehe [artifacts/ecustfd_features.csv](artifacts/ecustfd_features.csv)):

```
portion_id  food_type  area_cm2  height_cm  volume_ml  weight_g
apple001    apple      72.2      7.7        310.0      244.5
banana003   banana     94.7      3.2        150.0      149.8
```

Aus jedem **Top-Foto** rechnen wir per Münzen-Maßstab die Fläche in Quadrat­
zentimetern aus; aus jedem **Side-Foto** die Höhe in Zentimetern. Das gewogene
Volumen ist das **Trainingsziel**.

Wichtig zu wissen: ECUSTFD enthält nur **rundes Obst und Snacks** (apple,
banana, bread, bun, doughnut, egg, grape, lemon, litchi, mango, mooncake,
orange, peach, pear, plum, kiwi, sachima, tomato). **Keine Pizza, keinen Burger,
keine Suppe.** Für solche Gerichte greift die App auf von Hand geschätzte
Klassen-Priors zurück (siehe Abschnitt 6), bis eigene Daten gesammelt werden.

---

## 3. Wo das Training konkret im Code passiert

Es gibt zwei Stellen, an denen wir ein Modell trainieren — eine einfache und
eine ausführliche:

### 3a) Einfaches Training (in [`foodvol/volume.py`](foodvol/volume.py))

Die Methode `VolumeEstimator.fit(areas, heights, volumes_ml, model_kind=...)`
lernt das Modell. Für die **ausgelieferte** Variante `proportional` ist das
buchstäblich eine Zeile Mathematik:

```python
# foodvol/volume.py, in VolumeEstimator.fit(..., model_kind="proportional")
ah = areas * heights
self.shape_factor = float(np.sum(ah * volumes_ml) / np.sum(ah * ah))
```

Das ist die geschlossene Lösung der linearen Regression durch den Ursprung —
Schul­mathematik. Das gelernte Ergebnis ist eine einzige Zahl `k ≈ 0,51`,
gespeichert in [artifacts/volume_regressor.joblib](artifacts/volume_regressor.joblib).
Die App nutzt sie als robusten Default für unbekannte Speiseklassen.

### 3b) Ausführliches Training ([`notebooks/01_training.ipynb`](notebooks/01_training.ipynb))

Das richtige ML-Trainings-Notebook mit allem Drum und Dran:

| Zelle | Was passiert |
|---|---|
| 1 | Features laden ([artifacts/ecustfd_features_extended.csv](artifacts/ecustfd_features_extended.csv)) |
| 2 | **Train/Validation/Test-Split** (70 / 15 / 15), stratifiziert nach Speisetyp |
| 3 | Zwei Feature-Sets vorbereiten: nur Geometrie vs. Geometrie + Form-Deskriptoren |
| 4 | **6 Modellfamilien** + Hyperparameter-Raster definieren: Ridge, Lasso, Huber, Random Forest, Gradient Boosting, MLP |
| 5 | **GridSearchCV** über alle Modelle und Feature-Sets — sucht die besten Hyperparameter |
| 6 | Vergleichs-Diagramm aller Modelle |
| 7 | Gewinner auswählen |
| 8 | **Lernkurve** — bringen mehr Daten was? |
| 9 | **Permutation-Importance** — welches Feature trägt das meiste Signal? |
| 10 | **Einmalige Test-Set-Evaluation** mit dem Gewinner — die ehrliche Zahl |
| 11 | Modell speichern unter [artifacts/volume_model_trained.joblib](artifacts/volume_model_trained.joblib) |

Wenn man das Notebook ausführt (Kernel „Python (ml4b-foodvol)", *Run All*),
trainiert es das Modell von Grund auf neu. Die letzte Lauf-Bestzahl:

```
Gewinner:   GradientBoosting (base features)
            Hyperparameter: learning_rate=0.03, max_depth=2, n_estimators=200
Validation: MAPE 13,8 %   |  R² 0,93
Test-Set:   MAPE 19,4 %   |  R² 0,84
```

Vergleich: das einfache Ein-Parameter-Modell aus 3a) kommt auf MAPE 24 %.
Das ausführliche Training bringt also einen messbaren Genauigkeits­gewinn.

---

## 4. Wie wir das Training bewerten

Drei Methoden, in steigender Strenge:

### 4a) 5-fache Kreuzvalidierung (Notebook 00, Zelle 9)

Die Daten werden in 5 gleich große Teile geschnitten. Fünfmal: vier davon zum
Trainieren, der fünfte zum Testen. Ergebnis: jeder Datenpunkt war einmal Test­
punkt → ehrlicher Out-of-Sample-Fehler.

### 4b) Train / Validation / Test (Notebook 01)

* **Train (70 %)** → Modell lernen
* **Validation (15 %)** → Hyperparameter wählen
* **Test (15 %)** → einmalige finale Evaluierung. **Nicht** zum Tunen verwendet.

Diese Trennung ist wichtig: nur so ist die finale Zahl eine ehrliche Schätzung,
wie gut das Modell auf *neuen, ungesehenen* Daten arbeitet.

### 4c) Metriken

* **MAPE** — Mean Absolute Percentage Error, der mittlere prozentuale Fehler
* **MAE** — Mean Absolute Error in mL bzw. g
* **R²** — Bestimmtheitsmaß (1,0 = perfekt, 0 = nur der Mittelwert wäre genauso gut)

Code: [`foodvol/volume.py`](foodvol/volume.py) → `evaluate()`.

---

## 5. Wie ich das Training selbst ausführe

Voraussetzung: Setup wie im README. Dann:

```bash
# Umgebung aktivieren (Linux/macOS)
source .venv/bin/activate

# Datensatz herunterladen (~125 MB, einmalig)
python data/download_ecustfd.py

# Einfaches Training (10 Sekunden — eine Zelle im Notebook)
jupyter lab notebooks/00_feasibility.ipynb     # Zelle 16 ausführen

# Vollständiges Training mit Hyperparameter-Tuning (~3 Minuten)
jupyter lab notebooks/01_training.ipynb        # "Run All"
```

Nach dem Training liegen die Modelle hier:

```
artifacts/volume_regressor.joblib        # 1-Parameter-Modell (Default der App)
artifacts/volume_model_trained.joblib    # Gradient-Boosting-Modell (genauer)
```

---

## 6. Was wir *nicht* aus Daten lernen (ehrlich)

Für Speisen, die **nicht** in ECUSTFD vorkommen (Pizza, Burger, Suppe, Salat,
Pasta, …), kennt das trainierte Modell die typische Form nicht. Wir greifen dort
auf **handgeschätzte Klassen-Priors** aus
[`foodvol/data/nutrition_db.csv`](foodvol/data/nutrition_db.csv) zurück:

```csv
class,density_g_per_ml,kcal_per_100g,...,typical_height_cm,shape_factor
pizza,0.70,270,...,1.5,0.9
hamburger,0.90,250,...,5.5,0.75
soup,1.00,50,...,3.5,1.0
```

`typical_height_cm` und `shape_factor` sind *plausible Schätzwerte*, keine aus
Daten gelernten Größen. Für eine produktreife Lösung müssten diese Klassen mit
echten gewogenen Fotos trainiert werden — dafür ist die Pipeline vorbereitet
([`foodvol/benchmark.py`](foodvol/benchmark.py) → `fit_final(df, ...)`):
einfach eigene Datenpunkte ins DataFrame anhängen und das Training erneut
ausführen.

---

## 7. Datei-Übersicht (für Lesende, die in den Code wollen)

| Datei | Rolle |
|---|---|
| [`data/download_ecustfd.py`](data/download_ecustfd.py) | lädt ECUSTFD herunter |
| [`foodvol/datasets.py`](foodvol/datasets.py) | liest ECUSTFD ein (Bilder, Boxen, GT) |
| [`foodvol/benchmark.py`](foodvol/benchmark.py) | **extrahiert Features** aus den Fotos (`extract_features`, `extract_features_extended`) |
| [`foodvol/volume.py`](foodvol/volume.py) | **trainiert** + speichert das Modell (`VolumeEstimator.fit/save/load`) |
| [`notebooks/00_feasibility.ipynb`](notebooks/00_feasibility.ipynb) | einfaches Training + Baseline |
| [`notebooks/01_training.ipynb`](notebooks/01_training.ipynb) | **vollständiger Trainings-Workflow mit Split, Tuning, Test-Eval** |
| [`artifacts/ecustfd_features.csv`](artifacts/ecustfd_features.csv) | gecachte Trainings-Features |
| [`artifacts/volume_regressor.joblib`](artifacts/volume_regressor.joblib) | trainiertes 1-Parameter-Modell (ausgeliefert) |
| [`artifacts/volume_model_trained.joblib`](artifacts/volume_model_trained.joblib) | trainiertes Gradient-Boosting-Modell |
