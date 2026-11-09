# Student Performance Prediction System

A classroom prototype that flags students who may be at risk of failing the final Mathematics grade.

## Problem
Academic risk is often noticed only after poor results. This prototype explores early identification using available student indicators (study time, absences, past failures, G1, G2, and others).

## Dataset
UCI Machine Learning Repository, *Student Performance* (ID 320):
https://archive.ics.uci.edu/dataset/320/student+performance
Mathematics file: `student-mat.csv` (395 records, Portuguese secondary school).

## Prediction target
`G3 < 10` = **AT RISK** (1); `G3 >= 10` = **NOT AT RISK** (0).
G3 is the target only and is never used as an input feature.

## Technologies
Python 3.10+, pandas, scikit-learn (Random Forest), joblib, Streamlit, Git/GitHub.

## Installation and run
```bash
pip install -r requirements.txt
python train_model.py      # downloads data, trains model, saves student_model.pkl
streamlit run app.py       # opens http://localhost:8501
```
Use the "Download sample input CSV" button in the app to get a valid test file.

## Limitations
- Trained on one Portuguese secondary-school dataset; may not generalize to other institutions.
- Small dataset; results vary with the train/test split.
- G1 and G2 are strong predictors, so the model is not an "early" predictor unless they are removed.
- Predictions are not diagnoses and must not be the sole basis for real student decisions.
