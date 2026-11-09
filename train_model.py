# train_model.py
import os
import pandas as pd

# ---------------------------------------------------------------
# Part 1: Retrieve the dataset (Activity A)
# ---------------------------------------------------------------
try:
    from ucimlrepo import fetch_ucirepo
    # Fetch UCI Student Performance dataset (ID 320)
    student = fetch_ucirepo(id=320)
    # Combine student features and grade target
    df = pd.concat([student.data.features, student.data.targets], axis=1)
    print("Downloaded dataset from UCI.")
except Exception as e:
    # Fallback: use student-mat.csv downloaded manually from the UCI page
    # (the zip contains student-mat.csv, which is the Mathematics file, sep=';')
    print("UCI download failed:", e)
    if os.path.exists("student-mat.csv"):
        df = pd.read_csv("student-mat.csv", sep=";")
        print("Using local student-mat.csv instead.")
    else:
        raise SystemExit(
            "Could not download the data. Download the zip from "
            "https://archive.ics.uci.edu/dataset/320/student+performance, "
            "extract student-mat.csv into this folder, and run again."
        )

df.to_csv("student_data.csv", index=False)

print("Rows and columns:", df.shape)
print(df.head())
print(df.dtypes)
print("Missing values per column:")
print(df.isnull().sum())

# The Mathematics file has 395 records; the Portuguese file has 649.
if df.shape[0] != 395:
    print(f"\nWARNING: expected 395 Mathematics rows but got {df.shape[0]}. "
          "If you need the Math file exactly, use student-mat.csv from the UCI zip.\n")

# ---------------------------------------------------------------
# Part 2: Preprocess, train, evaluate (Activity B)
# ---------------------------------------------------------------
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Target: 1 = at risk (G3 < 10); 0 = not at risk
df = df.drop_duplicates()
y = (df["G3"] < 10).astype(int)

# Never use G3 as a predictor
X = df.drop(columns=["G3"])

numeric_cols = X.select_dtypes(include="number").columns.tolist()
categorical_cols = X.select_dtypes(exclude="number").columns.tolist()

numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])
categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])
preprocessor = ColumnTransformer([
    ("numeric", numeric_pipe, numeric_cols),
    ("categorical", categorical_pipe, categorical_cols),
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced",
    )),
])

model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("Accuracy:", round(accuracy_score(y_test, predictions), 3))
print(classification_report(
    y_test, predictions,
    target_names=["Not at risk", "At risk"],
    zero_division=0,
))

joblib.dump(model, "student_model.pkl")
print("Saved student_model.pkl")
