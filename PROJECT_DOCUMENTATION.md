# Parkinson's AI Assistant - Comprehensive Project Architecture & Technical Documentation

## 1. Executive Summary

This document serves as an exhaustive, government-grade technical overview of the "Parkinson's AI Assistant" software system. The system is a full-stack, web-based diagnostic support application designed to predict the likelihood of Parkinson's Disease in patients based on variations in 22 distinct vocal/speech features (such as Jitter, Shimmer, HNR, NHR).

The software utilizes a robust `StackingClassifier` Machine Learning ensemble for prediction and incorporates a natural language chatbot (powered by the Mistral-7B LLM via OpenRouter) to educate and guide users. The architecture is decoupled, consisting of a rigorous Python/FastAPI backend and a modern HTML/CSS/JS frontend.

---

## 2. High-Level System Architecture

- **Backend Framework:** Python 3.8+ with FastAPI.
- **Machine Learning Core:** Scikit-Learn (Random Forest, Gradient Boosting, Logistic Regression, SVC, KNN, MLP, united via a StackingClassifier).
- **Audio Processing:** `Parselmouth` (Praat integration) for algorithmic pitch and perturbation analysis; `pydub` for Audio format standardization (WebM to WAV).
- **Frontend Stack:** HTML5, modern CSS3 (custom design system), and Vanilla JavaScript utilizing `localStorage` for state management across pages.
- **External Integrations:** HuggingFace Inference API (`mistralai/Mistral-7B-Instruct-v0.2`) for generative AI Chat.

---

## 3. Directory Structure

```text
Parkinson/
│
├── README.md                          # General overview and setup guide
├── requirements.txt                   # Python dependencies matrix
├── .env                               # Environment variables (API Keys)
├── .gitignore                         # Git exclusion rules
│
├── backend/                           # Server-side Application & ML
│   ├── main.py                        # Central FastAPI Server Entry Point
│   ├── ml_models.py                   # ML Pipeline (Data prep, training, saving)
│   ├── test_accuracy.py               # Post-training evaluation script
│   ├── test_audio.py                  # Pydub & FFMpeg dependency validation script
│   │
│   ├── models/                        # Serialized ML Artifacts
│   │   ├── parkinsons_predictor.pkl   # The trained StackingClassifier object
│   │   ├── scaler.pkl                 # StandardScaler for feature normalization
│   │   └── feature_names.pkl          # Authorized feature sequence list
│   │
│   └── data/                          # Datasets and Statistical Scripts
│       ├── parkinsons_hospital.csv    # Primary training dataset 1
│       ├── Parkinsson_data.csv        # Primary training dataset 2
│       ├── user_predictions_log.csv   # Audit trail of system usage
│       ├── stat.py                    # 10-fold cross-validation & statistical analysis
│       ├── t_test_visualization.py    # Hypothesis testing graphing utility
│       ├── a.py                       # Exploratory Data Analysis & visualizer
│       └── image/                     # Output directory for EDA plots
│
└── frontend/                          # Client-side Web Interface
    ├── prediction.html                # Voice recording and manual input UI
    ├── chatbot.html                   # Interactive LLM chat UI
    ├── report_dashboard.html          # Prediction results and PDF export UI
    ├── generate_matrix.py             # Utility for dummy confusion matrix generation
    │
    ├── css/                           # Cascading Style Sheets
    │   ├── style.css                  # Global design tokens and layout
    │   ├── prediction_styling.css     # Domain-specific styles for prediction UX
    │   └── report_style.css           # Styling for the PDF/Dashboard views
    │
    └── js/                            # Client-Side Logic
        ├── prediction.js              # MediaRecorder API and API client
        ├── chatbot.js                 # Chatbot state and dynamic suggestion engine
        └── report_dashboard.js        # SVG Gauge rendering and jsPDF orchestration
```

---

## 4. File-by-File Detailed Technical Analysis

### 4.1 Root Level Configuration Files

#### `README.md`

Contains the developer documentation, feature lists, technology stack breakdowns, and explicit instructions on system configuration, preparation, and standard execution (`uvicorn backend.main:app --reload`).

#### `requirements.txt`

The strict dependency specification file ensuring reproducible environments. Critical libraries include `fastapi==0.111.0`, `scikit-learn==1.5.0`, `praat-parselmouth`, `pydub`, `pandas`, `numpy`, and `httpx`.

#### `.env`

The security vault for environment variables. Currently stores `HUGGINGFACE_API_KEY` for secure injection into the backend to facilitate the Mistral-7B chatbot queries.

#### `.gitignore`

Standard repository masking for Python `__pycache__` and IDE configurations to prevent uploading transient data to version control.

---

### 4.2 Backend Core Executables (`backend/`)

#### `main.py`

This is the mission-critical system kernel.

- **API Setup:** Initializes FastAPI and defines a strict CORS middleware policy allowing requests only from authorized local ports (e.g., `http://127.0.0.1:5500`, `http://localhost:8000`).
- **Dependencies & Environment:** Patches system `PATH` dynamically to ensure `FFMPEG` is accessible to `pydub`. Loads trained model artifacts securely via `joblib`.
- **Audio Processing (`/extract-features`):** Acts as an endpoint for receiving WebM vocal files. Translates WebM to WAV via `pydub`. Utilizes `parselmouth.Sound()` to evaluate intrinsic human vocal frequencies. Programmatically maps mathematical phenomena to 22 critical vocal markers (MDVP:Flo, Jitter(%), etc.). _Note: Several non-linear dynamical markers (e.g., RPDE, DFA, PPE) are currently hardcoded to mean placeholder values to bypass complex dependency restrictions._
- **Prediction Engine (`/predict`):** Receives 22 quantitative markers via JSON, passes them through the pre-loaded `StandardScaler`, and executes `.predict_proba()` against the `StackingClassifier`.
- **System Auditing:** Utilizes `log_prediction()` to continuously append `prediction`, `probability`, and all 22 inputs to `user_predictions_log.csv` for downstream auditing.
- **LLM Proxy (`/chat`):** Forwards conversational dialogue via `httpx` to HuggingFace Inference API with a strict system prompt prohibiting the AI from diagnosing conditions, ensuring ethical compliance.

#### `ml_models.py`

The data preparation and predictive model generation pipeline.

- Loads raw clinical data from both `parkinsons_hospital.csv` and `Parkinsson_data.csv`.
- Normalizes column orders, eliminates obfuscated non-features (like patient `name`), and unifies the resultant `pandas.DataFrame`.
- Segregates data into 80% Training / 20% Testing subsets.
- Iteratively trains an arsenal of classifiers (LogisticRegression, RandomForest, SVC, KNN, GradientBoosting, MLP).
- Constructs a `StackingClassifier` where the disparate estimators vote on outcomes, and a final `LogisticRegression` meta-classifier synthesizes the ultimate predictive model.
- Serializes the artifacts (`parkinsons_predictor.pkl`, `scaler.pkl`, `feature_names.pkl`) and evaluates the final accuracy.

#### `test_accuracy.py`

A modular evaluation script decoupled from the training pipeline. Independently loads the serialized artifacts and tests them against the raw data to confirm the uncorrupted persistence of the machine learning model. Outputs the Confusion Matrix and Classification Report (Precision, Recall, F1).

#### `test_audio.py`

A unit-test for the host machine's environment. Generates a 1000-millisecond silent `.webm` file programmatically via `pydub`, attempts a conversion to `.wav`, and executes cleanup, guaranteeing the integrity of the FFmpeg binary pathway.

---

### 4.3 Backend Statistical & Data Utilities (`backend/data/`)

#### `parkinsons_hospital.csv` & `Parkinsson_data.csv`

The tabular ground-truth data powering the intelligence of the system. Consists of thousands of diagnostic instances spanning 22 acoustic properties.

#### `user_predictions_log.csv`

The living audit trail. Captures every interaction mapped against the `/predict` endpoint to evaluate real-time usage metrics and potentially uncover concept drift.

#### `stat.py`

An advanced analytics script built for rigorous academic or clinical review. Implements a 10-fold Stratified K-Fold cross-validation over the ML models. Conducts definitive Paired T-Tests proving with statistical significance (`p < 0.05`) whether the Stacking Classifier outperforms its base constituents. Includes `seaborn` boxplot generation for visual confirmation.

#### `t_test_visualization.py`

Generates a `matplotlib` T-Distribution graph specifically demonstrating the hypothesis test between the `Stacking Classifier` and the `Logistic Regression` models. Shades the Alpha rejection region and explicitly marks the T-statistic to validate the null-hypothesis rejection.

#### `a.py`

Exploratory Data Analysis toolkit. Generates an exhaustive feature-correlation heatmap and extracts the 'Top 10' most statistically significant features relative to the disease `status`. Drops these generated artifacts into `image/` for developer distribution.

---

### 4.4 Frontend Interactivity & Display (`frontend/`)

#### HTML Layouts (`prediction.html`, `chatbot.html`, `report_dashboard.html`)

The structural representation of the client UI. Engineered with semantic HTML5 for accessibility (`role` definitions, `aria` tags). Integrates the Phosphor-Icons SVG vector library and enforces responsive, grid-based layouts suitable for various viewport dimensions.

#### Application Styles (`css/*.css`)

- `style.css`: Instantiates structural CSS variables (Root configurations) and layout controls like the Nav-bar and Theme toggle.
- `prediction_styling.css`: Custom animations (e.g., CSS keyframe loader simulating waveform processing) and tab-list interactions for the manual input forms.
- `report_style.css`: Strict design elements tailored to mimic formal clinical documentation, optimizing element margins for A4 PDF exports.

#### `js/prediction.js`

The intelligence behind the data-capture phase.

- Instantiates the `MediaRecorder` API to capture streams from the user's microphone.
- Leverages the Web Audio API (`AudioContext`, `AnalyserNode`) to draw live time-domain voice visualizations across an HTML `<canvas>`.
- Emits multi-part `FormData` HTTP POST requests back to the FastAPI `/extract-features` server.
- Upon retrieval, automatically injects the payload into the manual entry form, validates values, issues the final `/predict` POST, intercepts the response, and commits the data tuple (Feature vectors + Probability score + Timestamp) to `localStorage`.

#### `js/chatbot.js`

The conversational orchestrator.

- Maintains `chatHistory` state vectors for conversation continuity.
- Features a sophisticated dynamic suggestion engine: it scans incoming `mistral-7b` string outputs via regex/keyword filters (e.g., detecting "symptoms", "diagnose", "results") and actively shifts the Quick-Reply user buttons. For example, if "symptoms" are actively discussed, the script generates a highly visible `<button>` linking directly to the Voice Prediction tool.
- Handles markdown rendering for beautiful text formatting via `marked.js`.

#### `js/report_dashboard.js`

The final diagnostic readout logic.

- Polls `localStorage` for `parkinsonsPrediction` data keys.
- Procedurally maps the 22 user feature values against hardcoded `Healthy Ranges` (E.g. Jitter MDVP `< 0.005`).
- Generates a mathematically proportionate SVG radial gauge chart simulating risk likelihood based directly on the model's `predict_proba`.
- Initiates the PDF conversion cycle: It masks UI elements (buttons), leverages `html2canvas` to screenshot the DOM with augmented `scale: 2` (enhancing resolution), integrates it with `jspdf`, constructs an A4 dimensions document, concatenates pages if the canvas transcends vertical parameters, and triggers a localized download for the user.

#### `generate_matrix.py` (Residing in Frontend)

A standalone procedural script designed specifically to scaffold simulated dummy classification data (`make_classification`). It prints a standard `[TN, FP, FN, TP]` array to standard output, providing developers with immediate dummy integers to prototype confusing matrix interfaces.

---

**End of Document.** Prepared for high-fidelity technical review and operational verification.
