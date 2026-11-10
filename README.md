# Undergraduate Student Academic Performance Prediction

A machine-learning and educational data analytics project that explores synthetic undergraduate student-semester records across four academic years and predicts semester marks.

## Project specifications

- Dataset: synthetic, generated with Python's standard library
- Students represented: 1,250
- Academic structure: 4 years / 8 semesters
- Records: 10,000 student-semester rows
- Primary prediction target: `final_marks` (0–100)
- Secondary academic measure: `final_sgpa` (0–10, illustrative conversion)

## Dataset

See [`data/DATA_DICTIONARY.md`](data/DATA_DICTIONARY.md) for column definitions, target definitions, limitations and modelling cautions.

- Full dataset: [`data/student_performance_synthetic.csv`](data/student_performance_synthetic.csv)
- 100-row sample: [`data/student_performance_sample.csv`](data/student_performance_sample.csv)

**Important:** The dataset is entirely synthetic. It is suitable for demonstrating a machine-learning workflow, not for drawing conclusions about real students or universities.

## Initial setup

Use a supported Python version for which the required scientific packages provide compatible wheels. Python 3.12 or 3.13 is a sensible starting point for this project if your current Python installation has package compatibility problems.

Create and activate a virtual environment, then install:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Regenerate the synthetic dataset (standard library only):

```bash
python src/generate_dataset.py
```

## Planned development stages

1. Dataset validation and data-quality checks
2. Exploratory data analysis
3. Preprocessing and feature engineering
4. Regression model training
5. Model evaluation and comparison
6. Interactive dashboard
7. Documentation, screenshots and final GitHub presentation

## Modelling principles

- Exclude `student_id` from model inputs.
- Treat `final_marks` and `final_sgpa` as target columns, not predictors.
- Avoid using information that would not be available at prediction time.
- Use group-aware validation by student to reduce leakage between semesters belonging to the same student.
- Report MAE, RMSE and R² on held-out data.

## Academic use and limitations

This project is for educational demonstration. Predictions are estimates, not guaranteed academic outcomes. The synthetic data and illustrative SGPA conversion do not represent any particular institution's grading policy.
