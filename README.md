# Sapiens Pose Estimation Pipeline (COCO17)

Pipeline per la stima della posa umana da video di esercizi di fisiologia.

## Modelli utilizzati
- **YOLO v8**: Person detection
- **Meta Sapiens 0.3B**: Pose estimation (COCO 17 keypoints nativi)
- **Meta Sapiens 0.6B**: Pose estimation (COCO 17 keypoints nativi)
- **Meta Sapiens 1B**: Pose estimation (COCO 17 keypoints nativi)

## Features
- Elaborazione singolo video o dataset completo
- Output JSONL con keypoints + video overlay opzionale
- Estrazione automatica del soggetto dal path
- Configurazione via YAML
- Struttura modulare
- **Stabilizzazione keypoints** con One Euro Filter e Hold (riduzione jitter)
- **Analisi del movimento** con rilevamento automatico intervalli

## Installazione

```bash
# Crea ambiente virtuale e installa le dipendenze con
pip install -r requirements.txt
```

## Utilizzo

### Singolo video
```bash
python scripts/run_pose_estimation.py --video data/input/videos/soggetto001/nome_video.mp4
```

Il soggetto viene estratto automaticamente dal path. Output:
```
data/output/soggetto001/nome_video.jsonl
data/output/soggetto001/nome_video.mp4
```

### Dataset completo
```bash
python scripts/run_pose_estimation.py --dataset data/input/videos
```

Elabora tutti i video in `data/input/videos/soggettoNNN/*.mp4`

### Opzioni utili
```bash
--no-video          # Solo JSONL, salta video overlay
--stride N          # Elabora 1 frame ogni N
--skip-existing     # Salta video già elaborati
--device cuda       # Forza GPU (default: auto)
```
---

## Analisi del Movimento

Lo script `scripts/run_motion_analysis.py` analizza i file JSONL prodotti dalla pose estimation per rilevare intervalli di movimento e calcolare tempi di esecuzione degli esercizi.

### Funzionalità

- **Calcolo segnale di movimento**: Aggregazione pesata degli spostamenti di tutti i keypoints
- **Normalizzazione**: Per dimensione del corpo (diagonale bounding box)
- **Rilevamento intervalli**: Con isteresi (soglia ON/OFF) e logica di merge
- **Soglie automatiche**: Metodo K-Means per separare "fermo" da "in movimento"
- **Generazione grafici**: Visualizzazione segnale + intervalli rilevati

### Utilizzo

```bash
# Singolo file
python scripts/run_motion_analysis.py --input file.jsonl

# Cartella batch
python scripts/run_motion_analysis.py --input cartella/ --batch

# Con soglie automatiche (consigliato)
python scripts/run_motion_analysis.py --input file.jsonl --auto_threshold
```

### Opzioni

| Opzione | Default | Descrizione |
|---------|---------|-------------|
| `--input`, `-i` | - | File JSONL o cartella (required) |
| `--output`, `-o` | `data/output/test_tempi` | Cartella output |
| `--batch` | False | Processa tutti i .jsonl nella cartella |
| `--conf_th` | 0.25 | Soglia confidenza keypoints |
| `--smooth` | 5 | Finestra smoothing (moving average) |
| `--ton` | auto | Soglia inizio movimento |
| `--toff` | auto | Soglia fine movimento |
| `--auto_threshold` | False | Calcola soglie automaticamente (K-Means) |
| `--min_frames` | 3 | Frame consecutivi per cambio stato |
| `--no_weights` | False | Non usare pesi per keypoint |
| `--no_grafici` | False | Non generare grafici |


### Output

Per ogni file JSONL elaborato, viene creata una cartella con:

```
data/output/test_tempi/
└── nome_video/
    ├── risultato.json    # Statistiche e intervalli
    └── grafico.png       # Visualizzazione movimento
```

#### Formato risultato.json

```json
{
  "file": "video.jsonl",
  "durata_totale_sec": 45.2,
  "tempo_movimento_sec": 28.5,
  "percentuale_movimento": 63.1,
  "num_intervalli": 3,
  "soglia_on": 0.0234,
  "soglia_off": 0.0117,
  "intervalli": [
    {"inizio": 2.5, "fine": 12.3, "durata": 9.8},
    {"inizio": 15.0, "fine": 25.2, "durata": 10.2},
    {"inizio": 30.1, "fine": 38.5, "durata": 8.4}
  ]
}
```

### Notebook di Analisi

Il file `motion_analysis.ipynb` permette di eseguire l'analisi batch su tutti i file JSONL di una cartella, utile per processare l'intero dataset di esercizi (es. 4SST, 5xSTS, TUG).

## Struttura del progetto

```
sapiens_pose2/
├── config/
│   ├── default.yaml          # Configurazione generale
│   └── keypoints.yaml        # Definizione COCO 17 keypoints
├── scripts/
│   ├── run_pose_estimation.py  # Script CLI pose estimation
│   └── run_motion_analysis.py  # Script CLI analisi movimento
├── src/
│   ├── detection/
│   │   └── person_detector.py  # Utilities bounding box
│   ├── filters/
│   │   └── one_euro_filter.py  # Stabilizzazione keypoints
│   ├── io/
│   │   ├── jsonl_writer.py     # Output JSONL
│   │   ├── video_reader.py     # Lettura video
│   │   └── video_writer.py     # Scrittura video
│   ├── loaders/
│   │   ├── sapiens_loader.py   # Wrapper Sapiens
│   │   └── yolo_loader.py      # Wrapper YOLO
│   ├── pipeline/
│   │   └── pipeline_video.py   # Orchestrazione pipeline
│   ├── pose/
│   │   ├── decode.py           # Decodifica heatmaps -> keypoints
│   │   └── preprocess.py       # Preprocessing immagini
│   └── viz/
│       └── draw.py             # Disegno skeleton
├── data/
│   ├── input/videos/
│   │   └── soggettoNNN/        # Video input per soggetto
│   └── output/
│       ├── soggettoNNN/        # Output pose per soggetto
│       │   ├── video.jsonl
│       │   └── video_overlay.mp4
│       └── test_tempi/         # Output analisi movimento
│           └── nome_video/
│               ├── risultato.json
│               └── grafico.png
├── weights/
│   └── yolo/
│       └── yolov8n.pt          # Pesi YOLO
├── sapiens_host/
│   ├── detector/checkpoints/   # Checkpoint detector
│   └── pose/checkpoints/
│       ├── sapiens_0.3b/
│       │   └── sapiens_0.3b_coco_best_coco_AP_796_torchscript.pt2
│       ├── sapiens_0.6b/
│       │   └── sapiens_0.6b_coco_best_coco_AP_812_torchscript.pt2
│       └── sapiens_1b/
│           └── sapiens_1b_coco_best_coco_AP_820_torchscript.pt2
├── motion_analysis.ipynb       # Notebook analisi batch
├── requirements.txt
└── README.md
```

## Formato output JSONL delle pose

Ogni riga contiene un frame:

```json
{
  "frame": 0,
  "time_sec": 0.0,
  "person_box_xyxy": [100, 50, 300, 400],
  "person_box_conf": 0.95,
  "status": "ok",
  "keypoints17": [
    [150.0, 80.0, 0.92],
    [155.0, 75.0, 0.88],
    ...
  ]
}
```

### COCO 17 Keypoints
| Indice | Nome |
|--------|------|
| 0 | nose |
| 1 | left_eye |
| 2 | right_eye |
| 3 | left_ear |
| 4 | right_ear |
| 5 | left_shoulder |
| 6 | right_shoulder |
| 7 | left_elbow |
| 8 | right_elbow |
| 9 | left_wrist |
| 10 | right_wrist |
| 11 | left_hip |
| 12 | right_hip |
| 13 | left_knee |
| 14 | right_knee |
| 15 | left_ankle |
| 16 | right_ankle |

## Note

- I pesi dei modelli NON sono inclusi nel repository
- Il checkpoint Sapiens produce direttamente 17 keypoints COCO (nessun mapping necessario)
