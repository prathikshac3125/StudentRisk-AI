# StudentRisk AI
### Predicting At-Risk Students with Early Intervention Using Machine Learning

A student mini-project that trains a Random Forest classifier to identify students who may need academic support. It includes dataset loading, preprocessing, training, evaluation, prediction, feature importance, intervention suggestions, and an optional Streamlit interface.

## Important scope note
The project uses the public UCI Student Performance dataset from Portuguese secondary schools. The label is defined for this project as **at risk = final grade G3 below 10/20**. The model excludes G2 and G3 from its inputs so it does not use the final outcome to predict itself. This is an educational prototype, not a validated tool for high-stakes decisions or a guarantee of future performance.

## Folder structure
```text
StudentRisk-AI/
├── data/
│   ├── raw/                 # downloaded source CSV
│   └── processed/           # optional processed datasets
├── models/                  # trained model is saved here
├── notebooks/               # exploratory analysis notebook
├── reports/                 # metrics JSON
├── src/
│   ├── load_data.py
│   ├── preprocessing.py
│   ├── train_model.py
│   ├── predict_risk.py
│   ├── explainability.py
│   ├── risk_progression.py
│   ├── intervention.py
│   └── evaluation.py
├── utils/visualization.py
├── app.py
└── requirements.txt
```

## 1. Install Python
Use Python 3.10 or newer. If you are new to Python, Google Colab can be used for notebook experiments; the Streamlit app is easiest to run locally.

## 2. Open a terminal in the project folder
Create and activate a virtual environment if possible.

Windows:
```bash
python -m venv .venv
.venv\Scripts\activate
```

macOS/Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 3. Install dependencies
```bash
pip install -r requirements.txt
```

## 4. Train the model
From the root folder (`StudentRisk-AI`), run:
```bash
python -m src.train_model
```
The first run downloads the dataset using `ucimlrepo`. Internet access is required. It saves:
- `data/raw/student_data.csv`
- `models/best_model.pkl`
- `reports/metrics.json`

## 5. Launch the app
```bash
streamlit run app.py
```
Your browser should open the local app. If it does not, follow the local URL shown in the terminal.

## 6. Understand the evaluation
- **Accuracy:** overall share of correct predictions.
- **Precision for at-risk:** among predicted at-risk students, how many were actually at risk in the test data.
- **Recall for at-risk:** among truly at-risk students in the test data, how many the model identified.
- **F1:** balance between precision and recall.
- **Confusion matrix:** shows correct and incorrect predictions for both classes.

For this use case, review at-risk recall as well as precision. A single train/test split on a small dataset is only an initial evaluation; cross-validation and external validation would be stronger.

## Project modules
- `load_data.py`: download/read the public dataset.
- `preprocessing.py`: choose features and define the target label.
- `train_model.py`: build preprocessing + Random Forest pipeline, evaluate and save it.
- `predict_risk.py`: load the trained model and predict for one student.
- `explainability.py`: extract Random Forest feature importances.
- `risk_progression.py`: retrospective grade summaries (not used as early prediction features).
- `intervention.py`: simple, transparent support suggestions.
- `evaluation.py`: load saved metrics.
- `app.py`: interactive demonstration UI.

## Ethics and limitations
Use synthetic or authorized, de-identified student records for demonstrations. Avoid storing student names, student IDs, contact information, or sensitive details in this prototype. Predictions can be wrong and should trigger supportive review, not punishment or automatic academic decisions. Do not assume this dataset generalizes to a different country, institution, or student population.

## Dataset citation
Cortez, P. and Silva, A. (2008). *Using data mining to predict secondary school student performance.* UCI Machine Learning Repository, Student Performance dataset. https://archive.ics.uci.edu/dataset/320/student+performance


## Application Screenshots

### StudentRisk AI Dashboard

![StudentRisk AI Dashboard](Screenshot%202026-10-22%20000225.png)

### Application Screenshot 2

![Application Screenshot 2](Screenshot%202026-10-22%20000209.png)

### Application Screenshot 3

![Application Screenshot 3](Screenshot%202026-10-22%20000241.png)

### Application Screenshot 4

![Application Screenshot 4](Screenshot%202026-10-22%20000151.png)

