
# ============================================================
# UNDERGRADUATE STUDENT ACADEMIC PERFORMANCE PREDICTION
# Complete Machine Learning Pipeline
# Author: Dibyajyoti Swain
# IBCP | Class XII | Sai International School
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from pathlib import Path

from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent

DATA_PATH = ROOT / "data" / "student_performance_synthetic.csv"
OUTPUT = ROOT / "results"

OUTPUT.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid")

RANDOM_STATE = 42

print("\n" + "=" * 65)
print(" UNDERGRADUATE STUDENT PERFORMANCE PREDICTION")
print("=" * 65)

# ============================================================
# 2. LOAD DATASET
# ============================================================

df = pd.read_csv(DATA_PATH)

df.drop_duplicates(inplace=True)
df.reset_index(drop=True, inplace=True)

print("\nDATASET INFORMATION")
print("-" * 45)

print("Total records:", len(df))
print("Unique students:", df["student_id"].nunique())
print("Features available:", len(df.columns))

# ============================================================
# 3. DEFINE FEATURES AND TARGET
# ============================================================

TARGET = "final_marks"

# Exclude identifiers and target-related leakage columns.
excluded = [
    "student_id",
    "academic_year",
    "previous_semester_sgpa",
    "final_marks",
    "final_sgpa"
]

X = df.drop(columns=excluded)
y = df[TARGET]

groups = df["student_id"]

categorical_features = ["program"]

numerical_features = [
    "semester",
    "attendance_percentage",
    "study_hours_per_day",
    "assignment_average",
    "internal_assessment_score",
    "quiz_average",
    "class_participation_score",
    "previous_semester_marks"
]

# ============================================================
# 4. GROUP-BASED TRAIN / VALIDATION / TEST SPLIT
# ============================================================

# First: reserve 20% of students for final testing.

split1 = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=RANDOM_STATE
)

trainval_idx, test_idx = next(
    split1.split(X, y, groups)
)

X_trainval = X.iloc[trainval_idx]
y_trainval = y.iloc[trainval_idx]
groups_trainval = groups.iloc[trainval_idx]

X_test = X.iloc[test_idx]
y_test = y.iloc[test_idx]
groups_test = groups.iloc[test_idx]

# Second: split remaining students into 60% training
# and 20% validation of the complete dataset.

split2 = GroupShuffleSplit(
    n_splits=1,
    test_size=0.25,
    random_state=RANDOM_STATE
)

train_idx, val_idx = next(
    split2.split(
        X_trainval,
        y_trainval,
        groups_trainval
    )
)

X_train = X_trainval.iloc[train_idx]
y_train = y_trainval.iloc[train_idx]

X_val = X_trainval.iloc[val_idx]
y_val = y_trainval.iloc[val_idx]

# Verify no student appears in multiple sets.

train_ids = set(groups.iloc[trainval_idx].iloc[train_idx])
val_ids = set(groups.iloc[trainval_idx].iloc[val_idx])
test_ids = set(groups_test)

assert not train_ids.intersection(val_ids)
assert not train_ids.intersection(test_ids)
assert not val_ids.intersection(test_ids)

print("\nDATA SPLITTING")
print("-" * 45)
print("Training records:", len(X_train))
print("Validation records:", len(X_val))
print("Testing records:", len(X_test))
print("Training students:", len(train_ids))
print("Validation students:", len(val_ids))
print("Testing students:", len(test_ids))
print("Student-level data leakage: NONE")

# ============================================================
# 5. PREPROCESSING PIPELINE
# ============================================================

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numerical_features),
    ("categorical", categorical_pipeline, categorical_features)
])

# ============================================================
# 6. DEFINE MACHINE LEARNING MODELS
# ============================================================

models = {
    "Linear Regression": LinearRegression(),

    "Decision Tree": DecisionTreeRegressor(
        max_depth=12,
        min_samples_leaf=5,
        random_state=RANDOM_STATE
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=250,
        max_depth=18,
        min_samples_leaf=3,
        n_jobs=-1,
        random_state=RANDOM_STATE
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=3,
        random_state=RANDOM_STATE
    )
}

# ============================================================
# 7. TRAIN AND VALIDATE ALL MODELS
# ============================================================

validation_results = []
trained_models = {}

print("\nMODEL TRAINING")
print("=" * 65)

for name, algorithm in models.items():

    print(f"\nTraining {name}...")

    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("model", algorithm)
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_val)

    mae = mean_absolute_error(y_val, predictions)
    rmse = np.sqrt(mean_squared_error(y_val, predictions))
    r2 = r2_score(y_val, predictions)

    validation_results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })

    trained_models[name] = pipeline

    print(f"Validation MAE: {mae:.3f}")
    print(f"Validation RMSE: {rmse:.3f}")
    print(f"Validation R2: {r2:.4f}")

# ============================================================
# 8. SELECT BEST MODEL USING VALIDATION MAE
# ============================================================

validation_df = pd.DataFrame(validation_results)

validation_df.sort_values(
    "MAE",
    ascending=True,
    inplace=True
)

best_name = validation_df.iloc[0]["Model"]

print("\nMODEL COMPARISON")
print("=" * 65)
print(validation_df.to_string(index=False))

print("\nBEST MODEL:", best_name)

# ============================================================
# 9. REFIT BEST MODEL ON TRAINING + VALIDATION DATA
# ============================================================

X_development = pd.concat([X_train, X_val])
y_development = pd.concat([y_train, y_val])

best_model = Pipeline([
    ("preprocessing", preprocessor),
    ("model", models[best_name])
])

best_model.fit(X_development, y_development)

# ============================================================
# 10. FINAL TEST EVALUATION
# ============================================================

test_predictions = best_model.predict(X_test)

test_predictions = np.clip(test_predictions, 0, 100)

mae = mean_absolute_error(y_test, test_predictions)
mse = mean_squared_error(y_test, test_predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, test_predictions)

print("\nFINAL TEST RESULTS")
print("=" * 65)

print("Selected Model:", best_name)
print(f"MAE:  {mae:.3f}")
print(f"MSE:  {mse:.3f}")
print(f"RMSE: {rmse:.3f}")
print(f"R2:   {r2:.4f}")

# ============================================================
# 11. CONVERT PREDICTED MARKS TO ILLUSTRATIVE SGPA
# ============================================================

def marks_to_sgpa(marks):

    if marks >= 90:
        return 10.0
    elif marks >= 80:
        return 9.0
    elif marks >= 70:
        return 8.0
    elif marks >= 60:
        return 7.0
    elif marks >= 50:
        return 6.0
    elif marks >= 45:
        return 5.0
    elif marks >= 40:
        return 4.0
    else:
        return 0.0

predicted_sgpa = [
    marks_to_sgpa(mark)
    for mark in test_predictions
]

# ============================================================
# 12. EXPORT PREDICTIONS
# ============================================================

prediction_df = pd.DataFrame({
    "Actual_Marks": y_test.values,
    "Predicted_Marks": test_predictions,
    "Prediction_Error": y_test.values - test_predictions,
    "Predicted_SGPA": predicted_sgpa
})

prediction_df.to_csv(
    OUTPUT / "student_predictions.csv",
    index=False
)

# Export model comparison and final evaluation.

validation_df.to_csv(
    OUTPUT / "model_comparison.csv",
    index=False
)

pd.DataFrame([{
    "Best_Model": best_name,
    "MAE": mae,
    "MSE": mse,
    "RMSE": rmse,
    "R2": r2
}]).to_csv(
    OUTPUT / "final_evaluation.csv",
    index=False
)

# ============================================================
# 13. FEATURE IMPORTANCE
# ============================================================

try:

    model = best_model.named_steps["model"]
    fitted_preprocessor = best_model.named_steps["preprocessing"]

    feature_names = fitted_preprocessor.get_feature_names_out()

    if hasattr(model, "feature_importances_"):

        importance = model.feature_importances_

    elif hasattr(model, "coef_"):

        importance = np.abs(np.ravel(model.coef_))

    else:
        importance = None

    if importance is not None:

        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importance
        }).sort_values(
            "Importance",
            ascending=False
        )

        importance_df.to_csv(
            OUTPUT / "feature_importance.csv",
            index=False
        )

        plt.figure(figsize=(10, 6))

        sns.barplot(
            data=importance_df.head(12),
            x="Importance",
            y="Feature"
        )

        plt.title("Top Academic Performance Predictors")
        plt.tight_layout()

        plt.savefig(
            OUTPUT / "feature_importance.png",
            dpi=300
        )

        plt.close()

except Exception as error:
    print("Feature importance unavailable:", error)

# ============================================================
# 14. GENERATE EVALUATION CHARTS
# ============================================================

# Actual vs predicted marks

plt.figure(figsize=(8, 6))

sns.scatterplot(
    x=y_test,
    y=test_predictions,
    alpha=0.5
)

plt.plot(
    [0, 100],
    [0, 100],
    linestyle="--",
    color="red"
)

plt.xlabel("Actual Marks")
plt.ylabel("Predicted Marks")
plt.title("Actual vs Predicted Student Marks")

plt.tight_layout()

plt.savefig(
    OUTPUT / "actual_vs_predicted.png",
    dpi=300
)

plt.close()

# Model comparison chart

plt.figure(figsize=(9, 5))

sns.barplot(
    data=validation_df,
    x="MAE",
    y="Model"
)

plt.title("Machine Learning Model Comparison")
plt.xlabel("Validation Mean Absolute Error")

plt.tight_layout()

plt.savefig(
    OUTPUT / "model_comparison.png",
    dpi=300
)

plt.close()

# Residual distribution

residuals = y_test.values - test_predictions

plt.figure(figsize=(8, 5))

sns.histplot(
    residuals,
    bins=25,
    kde=True
)

plt.title("Prediction Error Distribution")
plt.xlabel("Actual Marks - Predicted Marks")

plt.tight_layout()

plt.savefig(
    OUTPUT / "prediction_residuals.png",
    dpi=300
)

plt.close()

# ============================================================
# 15. SAVE TRAINED MODEL
# ============================================================

joblib.dump(
    best_model,
    OUTPUT / "best_student_performance_model.joblib"
)

# ============================================================
# 16. FINAL PROJECT SUMMARY
# ============================================================

print("\n" + "=" * 65)
print(" MACHINE LEARNING PIPELINE COMPLETED SUCCESSFULLY")
print("=" * 65)

print("Best Model:", best_name)
print(f"Test MAE: {mae:.3f}")
print(f"Test RMSE: {rmse:.3f}")
print(f"Test R2: {r2:.4f}")

print("\nGenerated Files:")

for file in sorted(OUTPUT.iterdir()):
    if file.is_file():
        print(" -", file.name)

print("\nAll outputs saved in:", OUTPUT)
print("=" * 65)
