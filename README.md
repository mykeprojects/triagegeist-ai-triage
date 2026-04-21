# TriageGeist AI Triage

TriageGeist AI Triage is a clinical decision-support demo for estimating a patient's urgency level using the ESI (Emergency Severity Index) framework. The project combines a web interface for capturing vital signs and clinical context with a FastAPI service that runs inference through an XGBoost model enriched with biomedical embeddings.

> Important: this project is intended for demonstration and research. It does not replace clinical judgment and should not be used as the sole basis for care decisions.

## Features

- Modern web interface for recording vital signs and the chief complaint.
- ESI prediction from level 1 to level 5.
- Confidence and probability visualization by triage level.
- Visual clinical flags such as hypotension, hypoxemia, tachypnea, fever, and severe pain.
- Health check endpoint at `/health` and prediction endpoint at `/predict`.
- Combined model using structured variables and free-text complaint input.

## Architecture

The project is divided into two parts:

- Static frontend in `Demo/index.html`.
- Backend in `Demo/Backend/main.py` with FastAPI.

The frontend sends patient data to the API, the API processes the clinical variables, generates a free-text embedding with Bio_ClinicalBERT, and runs inference with the trained model.

## Technology Stack

- Frontend: plain HTML, CSS, and JavaScript.
- Backend: FastAPI, Uvicorn, NumPy, Joblib, and Sentence Transformers.
- Model: XGBoost.

## Project Structure

```text
.
├── README.md
├── notebook.md
└── Demo/
		├── index.html
		└── Backend/
				├── main.py
				├── requirements.txt
				└── models/
						├── xgb_model_demo.pkl
						└── features_demo.pkl
```

## Requirements

- Python 3.10 or later recommended.
- Internet access on first run to download `emilyalsentzer/Bio_ClinicalBERT` from Hugging Face.
- A modern web browser.

## Installation

1. Clone or open the repository on your machine.
2. Open a terminal in `Demo/Backend`.
3. Install the dependencies:

```bash
pip install -r requirements.txt
```

## Running the Backend

From `Demo/Backend`:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

If you prefer, you can also run it directly:

```bash
python main.py
```

The API will be available at `http://localhost:8000`.

## Running the Frontend

Open `Demo/index.html` in your browser. The interface will automatically try to connect to the API in the following order:

1. The `?api=` value in the URL.
2. The URL saved in browser storage.
3. `http://127.0.0.1:8000`.
4. `http://localhost:8000`.

If the frontend and backend are served from different origins, make sure the API is running and reachable from the browser.

## Endpoints

### `GET /health`

Checks whether the service is running.

Expected response:

```json
{"status":"ok"}
```

### `POST /predict`

Receives clinical data and returns the estimated ESI level, model confidence, and class probabilities.

Example payload:

```json
{
	"heart_rate": 92,
	"respiratory_rate": 18,
	"spo2": 97,
	"systolic_bp": 118,
	"diastolic_bp": 76,
	"temperature_c": 37.1,
	"pain_score": 4,
	"age": 45,
	"mental_status_triage": "alert",
	"arrival_mode": "walk-in",
	"sex": "F",
	"chief_complaint": "Chest pain and shortness of breath"
}
```

Example response:

```json
{
	"esi_level": 3,
	"confidence": 84.2,
	"probabilities": {
		"ESI_1": 1.1,
		"ESI_2": 8.4,
		"ESI_3": 84.2,
		"ESI_4": 5.3,
		"ESI_5": 1.0
	}
}
```

## Model and Inputs

Inference combines two groups of signals:

- Structured variables: vital signs, age, sex, arrival mode, mental status, and pain score.
- Free text: the main complaint, converted into a clinical embedding.

The backend loads the artifacts from `Demo/Backend/models/`:

- `xgb_model_demo.pkl`
- `features_demo.pkl`

## Important Deployment Note

The server uses `allow_origins=["*"]` to keep the demo simple. If you move this to a real environment, restrict allowed origins, add authentication, and review how clinical data is handled.

## Development

If you want to adjust the workflow or adapt the form, the main entry points are:

- `Demo/index.html`: visual logic, field validation, and API consumption.
- `Demo/Backend/main.py`: preprocessing, model loading, and the prediction endpoint.

## Project Status

This repository contains a functional demo that can be run locally with the included model and artifacts.