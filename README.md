# Incident Classification System (Assignment 2B)

This repository contains a minimal end-to-end Incident Classification System (ICS) that integrates:

- Machine learning (CNN) to classify the severity of a traffic incident from an image
- A small road-network graph around Kuching heritage area
- Path search (k-shortest paths) using ML-adjusted travel times
- An HTML map (Folium/Leaflet) visualisation similar to the provided example

## What you can do

- Train a tiny CNN on your dataset in `data/` and save it under `models/incident_cnn.h5`
- Provide an incident image to classify severity (minor/moderate/severe)
- Generate top-k routes from origin to destination(s), where travel times on camera-monitored roads are multiplied according to the predicted severity
- Produce an interactive HTML map under `maps/ics_map.html`

## Quick start (Windows PowerShell)

This project requires ML classification and won’t run without it.

1) Install dependencies (inside a virtual environment if you like):

```
pip install -r requirements.txt
```

2) Run (incident image REQUIRED). This will train a model if missing, classify severity, compute routes using the severity-adjusted times, and open the map:

```
python -m src.ics_cli --incident-image path\to\incident.jpg --model-arch simple --epochs 5 --open
```
### Choose routing algorithm and travel-time model (Part A vs Part B)

You can switch between NetworkX k-shortest paths and your own Uniform Cost Search (UCS) implementation, and optionally compute edge times from distances assuming a speed (default 60 km/h):

```
# NetworkX k-shortest paths (default) with distance-based times at 60 km/h
python -m src.ics_cli --incident-image path\to\incident.jpg --algorithm ksp --k 5 --use-distance-time --speed-kmh 60

# Your UCS (single best path per destination) using distance-based times
python -m src.ics_cli --incident-image path\to\incident.jpg --algorithm ucs --use-distance-time
```


The output HTML is saved to `maps/ics_map.html`.

---

## Install

Create a virtual environment (recommended) and install the requirements:

```
pip install -r requirements.txt
```

Note: TensorFlow is required because ML is mandatory for Assignment 2B.

## Train or load the ML model

Images are expected in the following structure:

```
data/
  training/
    01-minor/
    02-moderate/
    03-severe/
  validation/
    01-minor/
    02-moderate/
    03-severe/
```

Running the CLI with `--incident-image` will train a small CNN if a saved model is not found, then use it to predict the severity.

### Compare three custom CNN models (required by assignment)

We provide 3 lightweight, from-scratch architectures (no large pre-trained networks):

1. `simple` – small baseline CNN.
2. `deep` – deeper CNN with more filters & dropout.
3. `bn` – CNN with BatchNorm + Dropout blocks for better regularization.

Evaluate all three and save metrics (accuracy, confusion matrix, classification report) under `models/metrics/`:

```
python -m src.ics_eval --data-dir data --epochs 8
```

Pick the model at runtime (example using batchnorm model):

```
python -m src.ics_cli --incident-image path\to\incident.jpg --model-arch bn
```

## Run the ICS (generate the HTML map)

From the project root (custom origin/dests and output path; incident image REQUIRED):

```
python -m src.ics_cli --origin 1 --dest 10 13 --k 5 --incident-image path\to\incident.jpg --output maps\ics_map.html
```

```
python -m src.ics_cli --origin 1 --dest 10 13 --k 5 --incident-image path\to\incident.jpg --data-dir data --model models\incident_cnn.h5 --output maps\ics_map.html --open
```

Open the generated HTML file in your browser. The map includes an incident panel (top-right) showing the image, predicted severity, confidence, and model.

## Test cases (for marking scheme)

Run a suite of 10+ O-D scenarios across algorithms, severities, and timing modes:

```
python -m src.ics_testcases
```

## Notes

- Road times can be derived from edge distances at 60 km/h (`--use-distance-time`) or use built-in illustrative times.
- Severity multipliers (for camera-monitored roads): minor×1.2, moderate×1.8, severe×3.0.
- The CLI highlights up to K routes (first is green-like, others use distinct colours).
- You can extend the graph (`src/ics_graph.py`) with more nodes, edges, and better travel-time rules.

## Notes

- Road times are illustrative and loosely matched to the example HTML.
- Severity multipliers (for camera-monitored roads): minor×1.2, moderate×1.8, severe×3.0.
- The CLI highlights up to K routes (first is green-like, others use distinct colours).
- You can extend the graph (`src/ics_graph.py`) with more nodes, edges, and better travel-time rules.
