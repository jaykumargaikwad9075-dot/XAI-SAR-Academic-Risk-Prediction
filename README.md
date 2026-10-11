# XAI-SAR — Industry Dashboard (201 Students)

**Project:** Explainable AI for Personalized Student Success and Academic Risk Prediction

## Run on Windows PowerShell (Python 3.13)
Open PowerShell in this `xai_project` folder and run:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m streamlit run app.py
```

Open the local URL Streamlit prints, normally `http://localhost:8501`.

If PowerShell blocks activation, run:
```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```
Or double-click `run_windows.bat`.

## Dataset validation
- Main dataset: `data/student_dataset_201.csv`
- Compatibility copy: `data/student_dataset_200.csv` (legacy filename retained; it contains the same 201 records)
- Excel workbook: `data/Student_Dataset_201.xlsx`
- Total records: **201**, with unique student IDs and names.
- Dataset is **synthetic demonstration data**, not real student records.

## Dashboard sections
- **Overview:** KPI cards, risk distribution, feature impact, marks vs attendance, behaviour analysis, highest-risk students.
- **Student Explorer:** select a student or enter sample values; view predicted risk, probabilities, SHAP drivers, recommendations, and a what-if plan.
- **Email Center:** preview drafted emails. Review all drafts; real sending requires SMTP configuration.
- **Model Performance:** model comparison, confusion matrix, feature importance, and class metrics.
- **About:** project purpose and responsible-use notes.

## Optional retraining
From this folder run:
```powershell
python src/train_and_explain.py
```
This retrains and refreshes evaluation outputs using the 201-row cohort. It may take a few minutes. The included prebuilt model is sufficient to launch the dashboard without retraining.

**Evaluation note:** the included model artifact, model-comparison metrics, and student predictions have been regenerated using all 201 records. The dataset is synthetic; predictions must not be used for real academic decisions.


## Dashboard refresh
This package uses a refreshed enterprise UI (navy/teal palette, hero header, compact navigation, section headers, refined KPI cards, and chart containers). The cohort CSV and Excel workbook should each contain 201 records.
