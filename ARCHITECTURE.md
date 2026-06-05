# Architektur & ML-Methoden

Dieses Dokument erklärt, **wie** das System aufgebaut ist und **welche Machine-Learning-
Methode an welcher Stelle** zum Einsatz kommt — und warum. Gedacht als Grundlage zum
Nachvollziehen und Erklären (z. B. für die Abgabe).

## 1. Grundidee

Das Ziel „Menge erkennen" zerfällt in eine Kette, in der jeder Schritt ein eigenes,
gut definiertes Teilproblem ist:

```
                      ┌──────────────── Top-Ansicht (Foto/Video-Frame)
                      │
 Teller (Ø bekannt) ──┤  A. Kalibrierung  →  Maßstab  cm/Pixel
                      │
                      ├─ B. Segmentierung →  Maske pro Speise
                      │          │
                      │          ├─ C. Klassifikation → Speiseklasse
                      │          └─ Fläche (cm²)  ◄── Maßstab
                      │
 Seiten-Ansicht ──────┘  → Höhe (cm)  ◄── Maßstab

 Fläche × Höhe ──→ E. Volumenmodell (trainiert) ──→ Volumen (mL)
 Volumen × Dichte ──→ Masse (g) ──→ × kcal/g, Makros ──→ F. Nährwerte
```

Der Kerngedanke: **Volumen ist reine Geometrie**, **Nährwerte hängen an der Klasse**.
Deshalb sind beide getrennt — das Volumenmodell ist *klassen-agnostisch* und muss nicht
für jede Speise neu gelernt werden.

## 2. Methode pro Stufe

| Stufe | Aufgabe | ML-Typ | Konkretes Modell / Algorithmus | Von uns trainiert? |
|------|---------|--------|-------------------------------|:--:|
| **A** Kalibrierung | Maßstab cm/px aus dem Teller | *kein ML* (klassische CV) | Ellipsen-Fit (algebraischer Kegelschnitt-Fit), Hough-Kreise | – |
| **B** Segmentierung | Region-Kandidaten | Deep Learning (vortrainiert) | **FastSAM** (CNN, YOLOv8-seg-Backbone) | nein |
| **C** Erkennung + Essen-Tor | Welche Speise? Überhaupt Essen? | Zero-Shot Deep Learning (vortrainiert) | **CLIP** (offenes Vokabular + Nicht-Essen-Tor); ViT/Food-101 als Fallback | nein |
| **D** Tiefe (optional) | Höhen-Hinweis aus 1 Bild | Supervised Deep Learning (vortrainiert) | **Depth Anything V2** (DINOv2-Encoder + DPT-Decoder) | nein |
| **E** Volumen | Fläche+Höhe → Volumen | **Supervised Regression** | lineares Modell / Gradient Boosting | **ja** |
| **F** Nährwerte | Klasse → Dichte, kcal, Makros | *kein ML* | Tabellen-Lookup | – |

### A. Kalibrierung — klassische Computer Vision (kein ML)
Ein runder Teller projiziert sich als **Ellipse**. Wir fitten diese Ellipse
(`cv2.fitEllipse`, ein algebraischer Least-Squares-Fit eines Kegelschnitts) und nehmen
die **große Halbachse** als den nicht verzerrten Durchmesser. Aus „bekannter Durchmesser
in cm ÷ Durchmesser in Pixel" folgt der Maßstab. Hier ist **kein Lernen nötig**, weil die
Geometrie exakt und robust lösbar ist — ein gelerntes Modell wäre unnötig und fehleranfälliger.

### B. Segmentierung — FastSAM (CNN)
**FastSAM** ist ein faltendes neuronales Netz (Architektur von YOLOv8-Segmentation), das
darauf trainiert wurde, das große „Segment Anything Model" (SAM) nachzuahmen — bei
Bruchteilen der Rechenkosten, also CPU-tauglich. Es liefert Masken für Objekte im Bild;
wir beschränken sie auf den Tellerinnenraum und filtern den Teller selbst heraus.
*Warum gelernt:* Essensformen sind beliebig; eine gelernte Segmentierung generalisiert
weit besser als feste Farb-/Kantenregeln.

### C. Erkennung + Essen-Tor — CLIP (Zero-Shot)
**CLIP** bewertet ein Bild gegen beliebige Text-Labels. Wir scoren jede Region gegen
(a) ein **breites Essens-Vokabular** (alle Klassen der Nährwerttabelle, inkl. Obst) und
(b) **Nicht-Essen-Anker** („a checkerboard pattern", „a plate", „a table"…). Gewinnt ein
Nicht-Essen-Anker, wird die Region **verworfen**. Das löst zwei Probleme der ersten
Version auf einmal: es **erkennt echtes Essen und Obst** *und* **filtert Nicht-Essen**
(Schachbrett, Matte, Teller) heraus — der fehlende „Essen-ja/nein"-Schritt, an dem die App
zuvor mit 23 Phantom-Items gescheitert ist.

*Warum CLIP/Zero-Shot:* offenes Vokabular (nicht auf 101 Klassen beschränkt, Obst inklusive)
und ein Tor, das ein reiner Klassifikator nicht hat. Der fein-getunte **ViT/Food-101** bleibt
als Fallback, falls CLIP nicht geladen werden kann (dann ohne Tor). Text-Features werden
einmal vorab berechnet und gecacht, danach ist jede Region nur ein Bild-Encode plus
Matrixmultiplikation.

Verschachtelte Mehrfacherkennungen derselben Speise (FastSAM liefert sie auf mehreren
Skalen) werden per **Containment-NMS** zusammengeführt; bei plausiblem Teller wird zusätzlich
auf den **Tellerinnenraum** beschränkt, sodass tellergroße Masken herausfallen.

### D. Tiefe — Depth Anything V2 (optional)
Schätzt aus **einem** Bild eine *relative* Tiefenkarte (Transformer: DINOv2-Encoder +
DPT-Decoder, trainiert auf sehr vielen Bildern). Wir nutzen sie nur als zusätzlichen
**Höhen-Hinweis**, wenn keine Seitenansicht vorliegt. Standardmäßig aus.

### E. Volumen — das ist der trainierte Teil (Regression) + Klassen-Priors
Aus der Geometrie haben wir zwei metrische Messgrößen: **Fläche** (Top) und **Höhe** (Seite).
Ein umschließendes Prisma hätte das Volumen `Fläche × Höhe`; das echte Volumen ist ein
Bruchteil davon (ein „Formfaktor"). Statt den Faktor blind anzunehmen, kombinieren wir
zwei Mechanismen:

1. **Klassen-bewusste Priors** in der Nährwerttabelle (`typical_height_cm`,
   `shape_factor` pro Klasse). Wenn die Erkennung „pizza" sagt, weiß die Pipeline, dass
   eine Pizza typisch ~1,5 cm dick ist (`h`) und nahezu ein Prisma ist (`k=0,9`). Eine
   Suppe hat dagegen `h≈3,5 cm`, `k=1,0` (Schüsseltiefe), ein Apfel `h≈6,5 cm`, `k=0,55`
   (halbkugelähnlich). 130 von 152 Klassen haben solche Priors, organisiert nach Kategorie
   (flache Gerichte, Schüsselgerichte, rundes Obst, gestapelte Burger, …).
2. **Trainiertes Regressionsmodell** als Fallback für unbekannte Klassen — eine
   überwachte Regression auf den ECUSTFD-Ground-Truth (siehe unten).

Höhen-Auflösung pro Item:
* Seitenansicht vorhanden → **gemessene Höhe** verwenden.
* Sonst, falls Klasse einen Prior hat → **Klassen-Höhe**.
* Sonst → **geometrischer Fallback-Prior** (`0,8 · √(A/π)`).

Volumen pro Item:
* Falls Klasse einen `shape_factor` hat → `V = k_klasse · A · h`.
* Sonst → trainierter Regressor `V = 0,51 · A · h`.

**Warum so?** Eine reine klassen-agnostische Regression (was wir vorher hatten) lernt einen
Mittelwert-`k` aus Trainingsobst und liefert bei einem flachen Pizzastück absurde Volumina
(testweise 336 mL / 235 g / 636 kcal für ein Slice). Die Klassen-Priors sind die einfache,
sehr effektive Brücke zwischen „was die Erkennung weiß" und „wie das Volumen daraus folgt".

**Trainiertes Regressionsmodell** (Fallback / Vergleichsbaseline):
Eine **überwachte Regression** (kontinuierliche Vorhersage in mL). Drei Varianten:

* **`proportional`** — `Volumen = k · Fläche · Höhe`, ein einziger Parameter `k` per
  Least-Squares durch den Ursprung. **Das ausgelieferte Fallback-Modell** (k ≈ 0,51).
* **`huber` / `linear`** — lineare Regression über `[Fläche, Höhe, Fläche·Höhe]`.
* **`gbr`** — Gradient-Boosted Trees (`HistGradientBoostingRegressor`); in-distribution
  am genauesten (R² ≈ 0,83), extrapoliert aber nicht.

## 3. „Nutzen wir k-Means?" — ML-Methoden eingeordnet

Kurz: **Nein, k-Means passt hier nicht** — und zwar aus einem grundsätzlichen Grund.
ML-Verfahren teilen sich grob in:

* **Supervised (überwacht):** lernt aus *gelabelten* Beispielen eine Zielgröße.
  * **Klassifikation** → diskrete Klasse  → unsere **Speiseerkennung (ViT)**.
  * **Regression** → kontinuierliche Zahl → unser **Volumenmodell** (mL).
* **Unsupervised (unüberwacht):** findet Struktur in *ungelabelten* Daten.
  * **Clustering**, z. B. **k-Means** → gruppiert Punkte ohne Zielwert.
  * Dimensionsreduktion, z. B. PCA.

Wir wollen pro Teller eine **Zahl** (Volumen/Masse) bzw. eine **Klasse** vorhersagen und
haben dafür **gelabelte** Daten (ECUSTFD: bekanntes Volumen/Gewicht; Food-101: Klassen).
Das ist genau **Supervised Learning**. **k-Means** würde nur ähnliche Bilder/Pixel
*gruppieren*, ohne je „Volumen = 310 mL" auszugeben — es kennt die Zielgröße nicht.

*Wo k-Means trotzdem sinnvoll sein könnte:* als simple Farb-Quantisierung zur
Vorsegmentierung (Pixel nach Farbe clustern) oder zum Gruppieren ähnlicher Gerichte für
eine Analyse. Wir verwenden aber zielgerichtete, stärkere Verfahren (FastSAM, ViT,
Regression), die das Problem direkt lösen.

## 4. Training & Evaluation des Volumenmodells

* **Datensatz:** ECUSTFD, 145 Einzelspeise-Portionen mit Top+Side, Münze als Maßstab,
  gemessenem Volumen (mL) und Gewicht (g). (Die Münze ist nur das Referenzobjekt *des
  Datensatzes* — die App nutzt den Teller.)
* **Features:** `Fläche`, `Höhe`, `Fläche·Höhe` (der dominante, physikalisch motivierte Term).
* **Validierung:** **5-fache Kreuzvalidierung** → Fehler auf *ungesehenen* Portionen.
* **Metriken:** MAE, RMSE, **MAPE** (mittlerer prozentualer Fehler), R².
* **Ergebnis:** Volumen-MAPE ≈ **22 %** (gbr) bis **24 %** (proportional), R² bis 0,83;
  end-to-end Masse-MAPE ≈ **25–27 %**. Vergleichsbaseline ohne Lernen (`0,5·Fläche·Höhe`):
  24,1 %. Reproduzierbar in [notebooks/00_feasibility.ipynb](notebooks/00_feasibility.ipynb),
  Code in [foodvol/benchmark.py](foodvol/benchmark.py).

## 5. Warum dieser Aufbau

* **Modular:** jede Stufe ist einzeln testbar und austauschbar (eine Datei pro Stufe).
* **Klassen-agnostisches Volumen:** Geometrie statt pro-Speise-Modell → generalisiert.
* **So viel vortrainiert wie möglich:** ohne GPU trainieren wir nur das kleine
  Regressionsmodell (Sekunden auf der CPU) — der Rest kommt fertig.
* **Ehrliche Grenzen:** trainiert auf rundem Obst, angewandt auf beliebige Gerichte
  (Domänenlücke). Der Weg nach oben ist klar definiert (Abschnitt 6).

## 6. Ausbau / nächste Schritte

1. **Eigene, gewogene Daten** der Zielgerichte sammeln und `foodvol.benchmark.fit_final`
   erneut aufrufen — der größte Genauigkeitshebel.
2. **Nährwerttabelle** durch USDA FoodData Central / Bundeslebensmittelschlüssel ersetzen
   (Schema: [foodvol/data/nutrition_db.csv](foodvol/data/nutrition_db.csv)).
3. **Breitere Gericht-Erkennung:** größeres Klassifikationsmodell (z. B. auf Food-2K)
   hinter [foodvol/classification.py](foodvol/classification.py).
4. **Mit GPU:** ein tiefes Multi-View-Netz, das Volumen direkt aus den Bildern regrediert
   — gleiche `VolumeEstimator`-Schnittstelle, kein Umbau der übrigen Pipeline.
```
