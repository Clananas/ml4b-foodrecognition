# Food Volume & Calorie Estimation

Erkennt Speisen in einem Foto und schätzt **Masse, Kalorien und Makros** —
**ohne Teller, ohne Münze, ohne Marker**. Die App **kalibriert sich selbst** über
die erkannte Speiseklasse: wenn CLIP „pizza" sagt, weiß sie, dass eine Pizza-
Slice typisch ca. 16 cm lang ist, und leitet daraus den Maßstab cm/Pixel ab.

Die App läuft komplett auf der CPU bzw. Apple-MPS eines normalen Laptops —
**ohne dedizierte GPU**.

> **Wo wird das Modell trainiert?** Siehe [`TRAINING.md`](TRAINING.md) — dort steht
> ausführlich erklärt, welche Modelle vortrainiert aus dem Internet kommen, welchen
> Teil wir selbst trainieren, mit welchem Datensatz und an welcher Stelle im Code.
>
> **Wie ist das System aufgebaut, welche ML-Methoden stecken drin?** Siehe
> [`ARCHITECTURE.md`](ARCHITECTURE.md) — erklärt jede Pipeline-Stufe, was sie tut
> und warum genau diese Methode (z. B. „warum kein k-Means hier?").

## Schnellüberblick — die drei Notebooks und die App

| Was | Datei | Dauer | Wozu |
|---|---|---|---|
| Demo-App | [`app.py`](app.py) | sofort | Foto/Video hochladen → Masse + Kalorien sehen |
| Phase 0 — Machbarkeit | [`notebooks/00_feasibility.ipynb`](notebooks/00_feasibility.ipynb) | ~1 Min | Beweist, dass die Methode funktioniert |
| **Phase 1 — Training** | [`notebooks/01_training.ipynb`](notebooks/01_training.ipynb) | ~3 Min | Volles ML-Training: Train/Val/Test, GridSearch, Lernkurve, Test-Eval |

## In 5 Minuten loslegen

```bash
# 1. Setup (legt venv an, installiert alles, registriert Jupyter-Kernel)
python setup_env.py            # auf Windows: gleicher Befehl, läuft genauso

# 2. Aktivieren
source .venv/bin/activate      # Windows PowerShell: .venv\Scripts\Activate.ps1

# 3. App starten
streamlit run app.py
```

Im Browser: Foto eines Tellers hochladen, Tellerdurchmesser eintragen, **Estimate**
klicken. Fertig.

Für das Training: `python data/download_ecustfd.py` (lädt 125 MB Trainingsdaten),
dann `jupyter lab notebooks/01_training.ipynb` und *Run All*.

---

## 1. Idee in einem Absatz

Kalorien zu schätzen heißt, **Masse** zu schätzen, und Masse = **Volumen × Dichte**.
Dichte und Energiegehalt liefert eine Nachschlagetabelle, sobald die **Speise klassifiziert**
ist. Das eigentlich schwere Problem ist das **Volumen**, denn Volumen braucht 3D-Information.
Aus einem einzelnen 2D-Foto ist Höhe physikalisch nicht eindeutig. Deshalb nutzen wir
**zwei Ansichten** — eine **Draufsicht** (liefert die Grundfläche) und eine **Seitenansicht**
(liefert die Höhe) — und einen runden Referenzkörper bekannter Größe (Teller bzw. in den
Trainingsdaten eine Münze), der beide Ansichten in metrische Einheiten umrechnet.

```
Kalorien = Masse × (kcal/g)         kcal/g       ← Nährwerttabelle (Klasse)
Masse    = Volumen × Dichte         Dichte       ← Nährwerttabelle (Klasse)
Volumen  = Grundfläche × Höhe × Formfaktor       ← Formfaktor + Höhe pro Klasse
Grundfläche = (cm/px)² × Pixelfläche             ← cm/px aus typischer Klassengröße
```

**Der Trick mit dem Maßstab**: cm/px = typical_long_cm / measured_long_px.
Die App misst die Bounding-Box-Länge des erkannten Items im Bild und teilt durch
den typischen Wert für diese Klasse aus der Nährwerttabelle. Verifiziert: derselbe
Apfel bei 180/280/420 Pixel Bildgröße liefert 249/205/225 g vs. GT 244,5 g.

## 2. Pipeline

| Stufe | Modul | Methode | Modell |
|------|-------|---------|--------|
| A Selbstkalibrierung | `foodvol/pipeline.py` (`_per_item_scale`) | typische Klassengröße → cm/Pixel pro Item | Lookup |
| B Segmentierung | `foodvol/segmentation.py` | Region-Kandidaten | FastSAM (vortrainiert) |
| C Erkennung + Essen-Tor | `foodvol/recognition.py` | Speiseklasse + „ist das Essen?" | CLIP zero-shot (ViT/Food-101 Fallback) |
| D Tiefe (optional) | `foodvol/depth.py` | relative Tiefenkarte als Höhen-Hinweis | Depth Anything V2 (vortrainiert) |
| E Volumen | `foodvol/volume.py` | Grundfläche × Höhe × Formfaktor | **trainierter Regressor** |
| F Nährwerte | `foodvol/nutrition.py` | Dichte + Energie/Makros pro Klasse | Tabelle (CSV) |
| — Orchestrierung | `foodvol/pipeline.py` | verbindet A–F end to end | — |

Schwere Modelle werden **lazy** geladen und bei fehlendem Download **sauber degradiert**,
damit das Paket auch offline importierbar bleibt.

> Welche ML-Methode an welcher Stelle steckt (und warum kein k-Means), steht ausführlich
> in **[ARCHITECTURE.md](ARCHITECTURE.md)**.

## 3. Genauigkeit — ehrliche Einordnung

Volumen aus Bildern ist physikalisch unterbestimmt. Realistische Erwartung für offene
Speisen ohne Tiefensensor: **~15–30 % Massenfehler**. Das reicht für sinnvolles Tracking,
nicht „auf das Gramm". Hebel für mehr Genauigkeit (dokumentiert, teils als Upgrade-Pfad):

- **Zwei Ansichten statt einer** (bereits umgesetzt) — der größte Hebel ohne Sensor.
- **Eigene, gewogene Trainingsdaten** auf den Zielspeisen (Küchenwaage als Ground Truth).
- **Echte Tiefe** (LiDAR) oder **Video-3D-Rekonstruktion** — späterer Ausbau, GPU empfohlen.

Der Volumen-Regressor wird auf **ECUSTFD** trainiert/evaluiert (Top- + Seitenansicht,
Referenzobjekt, gemessenes Ground-Truth-Volumen und -Gewicht) — der öffentliche Datensatz,
der unser Aufnahme-Szenario am genauesten trifft. Siehe `notebooks/00_feasibility.ipynb`
für den gemessenen Baseline-Fehler.

## 4. Setup

Läuft auf **Windows, Linux und macOS**. Voraussetzung: **Python 3.10–3.12**
(3.13/3.14 sind für einige ML-Wheels noch zu neu). GPU optional — CUDA (Linux/Windows)
und Apple-MPS werden automatisch genutzt, sonst CPU.

**Ein-Befehl-Setup** (legt das venv an, installiert alles, registriert den Kernel):

```bash
python setup_env.py          # bzw. python3 setup_env.py
```

Danach Umgebung aktivieren:

| OS | Befehl |
|----|--------|
| Windows (PowerShell) | `.venv\Scripts\Activate.ps1` |
| Windows (cmd) | `.venv\Scripts\activate.bat` |
| Linux / macOS | `source .venv/bin/activate` |

<details><summary>Manuell statt per Skript</summary>

```bash
# 1) venv anlegen
python -m venv .venv
# 2) aktivieren (siehe Tabelle oben), dann:
python -m pip install --upgrade pip wheel setuptools
python -m pip install -r requirements.txt
python -m pip install -e .          # macht `import foodvol` überall verfügbar
python -m ipykernel install --user --name ml4b-foodvol --display-name "Python (ml4b-foodvol)"
```
</details>

## 5. Nutzung

Alle Befehle unten setzen ein **aktiviertes venv** voraus (siehe Setup-Tabelle).

**App starten:**
```bash
streamlit run app.py
```
Im Browser: Draufsicht (+ optional Seitenansicht **oder ein Video**) hochladen,
Tellerdurchmesser in cm eingeben, Ergebnis ablesen (Masse, Kalorien und Makros pro
Speise und gesamt).

**Datensatz laden und Notebooks öffnen:**
```bash
python data/download_ecustfd.py                  # lädt ECUSTFD nach data/ECUSTFD/
jupyter lab notebooks/                            # Kernel: "Python (ml4b-foodvol)"
```

Zwei Notebooks:

* [`00_feasibility.ipynb`](notebooks/00_feasibility.ipynb) — Phase 0: kann die Methode
  überhaupt funktionieren? Misst die Baseline-Genauigkeit mit dem ausgelieferten
  1-Parameter-Modell.
* [`01_training.ipynb`](notebooks/01_training.ipynb) — **Voller Trainings-Workflow**:
  erweiterte Feature-Extraktion, Train/Validation/Test-Split, `GridSearchCV` über fünf
  Modellfamilien (Ridge, Lasso, Huber, Random Forest, Gradient Boosting, MLP), Lernkurve,
  Permutation-Importance, einmalige Test-Set-Evaluation. Ergebnis bei aktuellem Datensatz:
  Gradient Boosting senkt den Volumen-MAPE von 24 % → 19 % gegenüber der Baseline.

**Programmatic:**
```python
from foodvol.pipeline import FoodVolumePipeline
pipe = FoodVolumePipeline()
result = pipe.estimate("top.jpg", side_image="side.jpg", plate_diameter_cm=26.0)
print(result.summary())
```

## 6. Projektstruktur

```
foodvol/                 # importierbares Paket (eine Datei pro Pipeline-Stufe)
  data/nutrition_db.csv  # gebündelte Nährwerttabelle (Dichte + Energie/Makros)
  recognition.py         # CLIP-Erkennung (offenes Vokabular) + Nicht-Essen-Tor
  video.py               # Frame-Extraktion + Top/Side-Auswahl aus einem Video
data/download_ecustfd.py # Datensatz-Downloader
notebooks/00_feasibility.ipynb  # Phase-0-Machbarkeit: Baseline-Fehler messen
app.py                   # Streamlit-Demo (Fotos oder Video)
setup_env.py             # plattformübergreifendes Setup (Windows/Linux/macOS)
ARCHITECTURE.md          # ML-Methoden & Aufbau im Detail
tests/                   # Smoke-/Unit-Tests
requirements.txt         # kuratierte Abhängigkeiten (+ requirements-lock.txt = exakt)
```

## 7. Grenzen & nächste Schritte (ehrlich)

- **Trainingsdaten** decken nur 19 Klassen ab (Obst und Snacks aus ECUSTFD).
  Für andere Speisen (Pizza, Burger, Suppe, …) greift die App auf von Hand
  geschätzte Klassen-Priors zurück — funktioniert, ist aber kein Training auf
  diesen Klassen. Behoben wird das mit eigenen gewogenen Fotos; siehe
  [`TRAINING.md`](TRAINING.md) Abschnitt 6.
- **Nährwerttabelle** enthält Näherungswerte für 152 Klassen; für den Produktiv-
  einsatz durch USDA FoodData Central oder den deutschen Bundeslebensmittel­
  schlüssel (BLS) ersetzen.
- **Volumen-Regressor**: aktuell leichtgewichtig (CPU-tauglich). Mit GPU lässt
  sich ein tiefes Multi-View-Netz trainieren (Upgrade-Pfad in `foodvol/volume.py`
  dokumentiert).

## 8. Weiterführende Doku

- [`TRAINING.md`](TRAINING.md) — wo das Modell trainiert wird, mit welchen Daten,
  wie man es selbst ausführt und welche Bewertungsmaße verwendet werden.
- [`ARCHITECTURE.md`](ARCHITECTURE.md) — welche Machine-Learning-Methode an
  welcher Stelle der Pipeline steckt und warum.

## Lizenz

[MIT](LICENSE) — frei verwendbar mit Namensnennung.
