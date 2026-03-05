Parkinson's AI Assistant
This project is a web-based application designed to assist in the preliminary prediction of Parkinson's disease based on vocal features. It integrates machine learning models with an interactive web interface, including a prediction tool, a chatbot for information, and a report dashboard.

Features
Vocal Feature-Based Prediction: Predicts the likelihood of Parkinson's disease using 19 voice-related features.

Ensemble Model (Stacking Classifier): Utilizes a robust machine learning ensemble for accurate predictions.

- **Interactive Chatbot**: Ask questions about Parkinson's, and get detailed, symptom-specific answers powered by the advanced Mistral 7B LLM (via Hugging Face API).

Prediction Report Dashboard: Visualizes prediction results and feature analysis.

User-Friendly Interface: Built with modern web technologies for an intuitive experience.

Technologies Used
Backend:

Python

FastAPI: Web framework for building the API.

Scikit-learn: For machine learning models (Stacking Classifier, Logistic Regression, Random Forest, SVM, KNN, Gradient Boosting, MLP).

Pandas: For data manipulation and analysis.

Joblib: For model persistence (saving/loading models).

httpx: For making HTTP requests (e.g., to external LLM APIs if configured).

python-dotenv: For managing environment variables.

Frontend:

HTML5

CSS3 (Custom styling with a design system)

JavaScript (Vanilla JS)

Phosphor Icons: For vector icons.

marked.js: For rendering Markdown in the chatbot.

jspdf & html2canvas: (Potentially for PDF report generation, though commented out in report_dashboard.html).

Getting Started
Follow these steps to set up and launch the project locally.

Prerequisites
Python 3.8+

pip (Python package installer)

2. Install Dependencies
   Navigate to the backend directory in your terminal and install the required Python packages:

pip install -r requirements.txt
Note: If you don't have a requirements.txt file, you can create one by listing the dependencies:

fastapi
uvicorn
scikit-learn
pandas
numpy
python-dotenv
httpx
joblib

Then run pip install -r requirements.txt.

3. Prepare Data
   Ensure that the parkinsons_hospital.csv and Parkinsson_data.csv files are placed inside the backend/data/ directory. Create this directory if it doesn't exist.

4. Configure Hugging Face API Key (for Chatbot)
   1. Get an API key from [Hugging Face](https://huggingface.co/). A `Read` access token is sufficient for the Inference API.
   2. In the `Parkinson` directory, create a `.env` file or modify the existing one.
   3. Add your key: `HUGGINGFACE_API_KEY=your_key_here`

5. Train the Machine Learning Model
   Before running the web application, you need to train the machine learning model. This script will preprocess the data, train the Stacking Classifier, and save the model components (.pkl files) into the backend/models/ directory.

From the backend directory, run:

python ml_models.py

You should see output indicating the training progress and model evaluation results for various classifiers, including the Stacking Classifier. This step is crucial as it generates the parkinsons_predictor.pkl, scaler.pkl, and feature_names.pkl files that the FastAPI application will load.

6. Launch the FastAPI Backend
   Once the model is trained and saved, you can start the FastAPI server.

From the root directory of your project (the directory containing both backend/ and frontend/), run:

uvicorn backend.main:app --reload

backend.main: Refers to the main.py file located inside the backend directory.

app: Refers to the FastAPI() instance named app inside main.py.

--reload: (Optional) This flag enables auto-reloading of the server when code changes are detected, which is useful during development.

You should see output similar to:

INFO: Will watch for changes in these directories: ['D:\\Programming\\Projects\\Parkinson\\parkinson v7\\parkinson v6\\backend']
INFO: Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO: Started reloader process [PID]
INFO: Started server process [PID]
INFO: Waiting for application startup.
INFO: Application startup complete.

Usage
Once the FastAPI server is running:

Access the Prediction Page: Open your web browser and go to http://127.0.0.1:8000/. This will load the prediction.html page where you can input vocal features and get a prediction.

Access the Chatbot: Navigate to http://127.0.0.1:8000/chatbot.html (or click the "Chatbot" link in the navigation).

Access the Report Dashboard: Navigate to http://127.0.0.1:8000/report_dashboard.html (or click the "Report" link in the navigation).

Disclaimer
This application is for informational and educational purposes only and is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of a qualified healthcare professional for any medical concerns. The predictions provided by this model are based on statistical analysis and should not be used as the sole basis for medical decisions.
