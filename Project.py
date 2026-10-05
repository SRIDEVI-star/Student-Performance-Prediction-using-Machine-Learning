"""
Student Performance Prediction using Machine Learning
-----------------------------------------------------
Predicts a student's performance class (L = Low, M = Medium, H = High)
from LMS engagement data, and compares five classifiers.

Run:  pip install -r requirements.txt
      python Project.py
"""

import time as t

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report)
from sklearn.model_selection import (StratifiedKFold, cross_val_score,
                                     train_test_split)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# ---------------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------------
data = pd.read_csv("AI-Data.csv")

# ---------------------------------------------------------------
# 2. CORRELATION HEATMAP
# The heatmap needs numbers, so we make a numeric copy where the
# text columns "Class" and "StudentAbsenceDays" are converted too.
# (Previously these two were left out of the heatmap.)
# ---------------------------------------------------------------
heat = data.select_dtypes(include="number").copy()
heat["StudentAbsenceDays"] = data["StudentAbsenceDays"].map({"Under-7": 0, "Above-7": 1})
heat["Class"] = data["Class"].map({"L": 0, "M": 1, "H": 2})

plt.figure(figsize=(10, 7))
sb.heatmap(heat.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# ---------------------------------------------------------------
# 3. GRAPH MENU
# Each option is (title, column, optional x-axis order).
# One dictionary replaces nine copy-pasted if/elif blocks.
# ---------------------------------------------------------------
GRAPHS = {
    1: ("Marks Class Count Graph", "Class", ["L", "M", "H"]),
    2: ("Marks Class Semester-wise Graph", "Semester", None),
    3: ("Marks Class Gender-wise Graph", "gender", ["M", "F"]),
    4: ("Marks Class Nationality-wise Graph", "NationalITy", None),
    5: ("Marks Class Grade-wise Graph", "GradeID",
        ["G-02", "G-04", "G-05", "G-06", "G-07", "G-08", "G-09", "G-10", "G-11", "G-12"]),
    6: ("Marks Class Section-wise Graph", "SectionID", None),
    7: ("Marks Class Topic-wise Graph", "Topic", None),
    8: ("Marks Class Stage-wise Graph", "StageID", None),
    9: ("Marks Class Absent Days-wise Graph", "StudentAbsenceDays", None),
}

MENU = (
    "\n1.Marks Class Count Graph\t2.Marks Class Semester-wise Graph\n"
    "3.Marks Class Gender-wise Graph\t4.Marks Class Nationality-wise Graph\n"
    "5.Marks Class Grade-wise Graph\t6.Marks Class Section-wise Graph\n"
    "7.Marks Class Topic-wise Graph\t8.Marks Class Stage-wise Graph\n"
    "9.Marks Class Absent Days-wise\t10.No Graph (continue)\n"
)

while True:
    print(MENU)
    try:
        ch = int(input("Enter Choice: "))
    except ValueError:                      # user typed letters, not a number
        print("Please enter a number from 1 to 10.")
        continue

    if ch == 10:
        break
    if ch not in GRAPHS:
        print("Please enter a number from 1 to 10.")
        continue

    title, column, order = GRAPHS[ch]
    print("Loading Graph....\n")
    t.sleep(1)
    fig, ax = plt.subplots(figsize=(10, 6))
    if column == "Class":                   # plain count of L / M / H
        sb.countplot(x="Class", data=data, order=order, ax=ax)
    else:                                   # count split by Class
        sb.countplot(x=column, hue="Class", data=data, order=order,
                     hue_order=["L", "M", "H"], ax=ax)
    ax.set_title(title)
    plt.show()

# ---------------------------------------------------------------
# 4. FEATURES AND LABEL
# Only these 4 columns are used as inputs. The label stays as the
# text "L"/"M"/"H", so predictions come back as letters directly
# (no manual number-to-letter mapping, which was the old M/L bug).
# StudentAbsenceDays is text ("Under-7"/"Above-7"), so we turn it
# into 0/1 first.
# ---------------------------------------------------------------
FEATURES = ["raisedhands", "VisITedResources", "Discussion", "StudentAbsenceDays"]

X = data[FEATURES].copy()
X["StudentAbsenceDays"] = X["StudentAbsenceDays"].map({"Under-7": 0, "Above-7": 1})
y = data["Class"]

# ---------------------------------------------------------------
# 5. TRAIN / TEST SPLIT
# train_test_split shuffles the rows (the old code never really
# shuffled) and stratify=y keeps the L/M/H proportions equal in
# both parts. random_state makes the result repeatable.
# ---------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42
)

# ---------------------------------------------------------------
# 6. MODELS
# Perceptron, Logistic Regression and MLP are sensitive to the
# scale of the features, so they get a StandardScaler first.
# Tree-based models don't need it.
# ---------------------------------------------------------------
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Perceptron": make_pipeline(StandardScaler(), Perceptron(random_state=42)),
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "MLP Classifier": make_pipeline(
        StandardScaler(), MLPClassifier(activation="logistic", max_iter=1000, random_state=42)
    ),
}

# 5-fold cross-validation: a more reliable score than one split,
# because every row gets used for testing exactly once.
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    cv_scores = cross_val_score(model, X, y, cv=cv)

    print(f"\n===== {name} =====")
    print(classification_report(y_test, pred))
    print(f"Test accuracy: {acc:.3f}")
    print(f"5-fold CV accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
    results.append((name, acc, cv_scores.mean(), cv_scores.std()))
    t.sleep(0.5)

# ---------------------------------------------------------------
# 7. SUMMARY TABLE
# ---------------------------------------------------------------
summary = pd.DataFrame(results, columns=["Model", "Test Acc", "CV Mean", "CV Std"])
summary = summary.sort_values("CV Mean", ascending=False).reset_index(drop=True)
print("\n===== SUMMARY (sorted by cross-validation accuracy) =====")
print(summary.round(3).to_string(index=False))

best_name = summary.loc[0, "Model"]
print(f"\nBest model: {best_name}")

# ---------------------------------------------------------------
# 8. CONFUSION MATRIX (best model) + FEATURE IMPORTANCE (Random Forest)
# ---------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ConfusionMatrixDisplay.from_estimator(
    models[best_name], X_test, y_test, labels=["L", "M", "H"], ax=axes[0], cmap="Blues"
)
axes[0].set_title(f"Confusion Matrix - {best_name}")

importances = pd.Series(models["Random Forest"].feature_importances_, index=FEATURES)
importances.sort_values().plot(kind="barh", ax=axes[1])
axes[1].set_title("Random Forest Feature Importance")
axes[1].set_xlabel("Importance")

plt.tight_layout()
plt.show()

# ---------------------------------------------------------------
# 9. PREDICT FOR A NEW STUDENT
# Only the 4 features the models were trained on are asked for.
# ---------------------------------------------------------------
def ask_int(prompt, low=0, high=100):
    """Keep asking until the user types a whole number in range."""
    while True:
        try:
            value = int(input(prompt))
            if low <= value <= high:
                return value
        except ValueError:
            pass
        print(f"  Please enter a whole number between {low} and {high}.")


choice = input("\nDo you want to test a specific student (y or n): ").strip().lower()
if choice == "y":
    raised = ask_int("Enter raised hands (0-100): ")
    resources = ask_int("Enter visited resources (0-100): ")
    discussion = ask_int("Enter number of discussions (0-100): ")

    while True:
        absence = input("Enter absences (Under-7 or Above-7): ").strip().lower()
        if absence in ("under-7", "above-7"):
            break
        print("  Please type exactly Under-7 or Above-7.")
    absence = 0 if absence == "under-7" else 1

    # Same column names and order as training
    student = pd.DataFrame([[raised, resources, discussion, absence]], columns=FEATURES)

    print()
    for name, model in models.items():
        print(f"Using {name}: {model.predict(student)[0]}")
        t.sleep(0.5)

print("\nExiting..")
