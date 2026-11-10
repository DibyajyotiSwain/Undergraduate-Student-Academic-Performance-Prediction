# Undergraduate Student Academic Performance Prediction

**Machine Learning | Academic Analytics | Interactive Dashboard**

A machine learning project that analyses synthetic undergraduate student-semester records and predicts final marks using academic and study-related indicators. The project compares four regression algorithms and provides an interactive Dash dashboard for exploring academic patterns and generating example predictions.

> **Dataset notice:** This project uses synthetic data created for educational and demonstration purposes. It does not contain real student records. Predictions are experimental estimates and must not be used for actual grading, admissions, or other high-stakes decisions.

## Project details

| Item | Description |
|---|---|
| Project title | Undergraduate Student Academic Performance Prediction |
| Author | Dibyajyoti Swain |
| Academic context | IBCP, Class XII, Sai International School |
| Dataset | Synthetic student-semester data |
| Intended scope | Undergraduate Years 1–4 |
| Prediction target | Final marks (0–100) |
| Problem type | Supervised machine learning, regression |
| Interface | Interactive web dashboard built with Dash |

## Objectives

- Explore academic performance patterns across undergraduate semesters and programmes.
- Examine relationships between attendance, study habits, assessments, participation and final marks.
- Prepare data for supervised machine learning.
- Train and compare multiple regression algorithms.
- Evaluate the selected model on a held-out test set.
- Build an interactive dashboard for academic exploration and marks prediction.

## Dataset

The project was designed around **10,000 synthetic student-semester records**. Each record represents a student's academic performance in a semester.

The dataset includes fields such as:

| Feature | Description |
|---|---|
| `student_id` | Synthetic student identifier |
| `program` | Undergraduate programme |
| `academic_year` | Undergraduate year (Year 1–Year 4) |
| `semester` | Semester number |
| `attendance_percentage` | Attendance percentage |
| `study_hours_per_day` | Average daily study hours |
| `assignment_average` | Assignment average |
| `internal_assessment_score` | Internal assessment score |
| `quiz_average` | Quiz average |
| `class_participation_score` | Class participation score |
| `previous_semester_marks` | Marks from the previous semester, where available |
| `final_marks` | Target variable predicted by the regression models |
| `final_sgpa` | Synthetic SGPA field included in the generated dataset |

The project also contains a data dictionary and a smaller sample dataset, if included in the repository.

## Methodology

1. **Dataset generation and validation** – generate synthetic student-semester observations and validate the dataset structure.
2. **Exploratory data analysis** – inspect distributions and relationships between numerical academic variables.
3. **Preprocessing** – impute missing numerical and categorical values, scale numerical features and one-hot encode the programme category.
4. **Data splitting** – create training, validation and test sets using student-grouped splits so that the same student does not appear across multiple sets.
5. **Model training** – train four regression algorithms.
6. **Model selection** – compare validation metrics and select the model with the lowest validation MAE.
7. **Final evaluation** – evaluate the selected model on the held-out test set.
8. **Dashboard development** – present dataset summaries, visual analysis, model evaluation and an interactive prediction form.

## Machine learning models

The following regression algorithms were compared:

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor

### Validation comparison

The following values are from `results/model_comparison.csv`. These are validation-set metrics used to compare the candidate models. Lower MAE and RMSE are better; a higher R² indicates a better fit.

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| **Linear Regression** | **4.6362** | **5.8358** | **0.6909** |
| Gradient Boosting | 4.7619 | 5.9879 | 0.6746 |
| Random Forest | 4.9014 | 6.1348 | 0.6584 |
| Decision Tree | 5.9582 | 7.3764 | 0.5061 |

**Selected model: Linear Regression**, because it achieved the lowest validation MAE among the four tested models.

### Final held-out test evaluation

The following values are from `results/final_evaluation.csv`. They represent the selected model's performance on the held-out test set.

| Metric | Result |
|---|---:|
| Selected model | Linear Regression |
| MAE | **4.6907** |
| MSE | **34.7719** |
| RMSE | **5.8968** |
| R² | **0.6623** |

On this synthetic test set, the model's predictions had a mean absolute error of approximately **4.69 marks**. The R² value was approximately **0.662**. These results describe this generated dataset and should not be interpreted as expected performance on real university data.

## Exploratory data analysis

The correlation analysis showed the following relationships with `final_marks` in the generated dataset:

| Feature | Correlation with final marks |
|---|---:|
| Previous semester marks | 0.59 |
| Assignment average | 0.55 |
| Internal assessment score | 0.52 |
| Quiz average | 0.46 |
| Study hours per day | 0.37 |
| Attendance percentage | 0.22 |
| Class participation score | 0.11 |

These are descriptive correlations, not evidence of causation or a substitute for model-based feature analysis.

The heatmap also showed a very high correlation between `academic_year` and `semester` (approximately 0.98). The training workflow excludes `academic_year` from the model inputs and uses `semester` instead.

## Interactive dashboard

The Dash application provides:

- Dataset summary cards.
- Final-mark distribution.
- Year-wise and semester-wise average marks.
- Programme-wise performance comparison.
- Attendance versus final marks.
- Study hours versus final marks.
- Candidate model comparison and final test metrics.
- Actual-versus-predicted marks visualisation.
- An interactive form that predicts marks from entered academic information.

The dashboard loads the saved model pipeline; it does not retrain the models when it starts.

### Dashboard screenshots

Add your screenshots to the `screenshots/` folder and use these filenames, or update the paths below to match your filenames.

| Dashboard view | Screenshot |
|---|---|
| Dashboard overview | `screenshots/dashboard-overview.png` |
| Academic performance insights | `screenshots/academic-insights.png` |
| Model evaluation | `screenshots/model-evaluation.png` |
| Student marks prediction | `screenshots/marks-prediction.png` |

![Student marks prediction dashboard](screenshots/marks-prediction.png)

## Project structure

```text
undergraduate_student_performance_project/
│
├── data/
│   ├── student_performance_synthetic.csv
│   ├── student_performance_sample.csv
│   └── DATA_DICTIONARY...
│
├── models/
│
├── notebooks/
│
├── screenshots/
│
├── results/
│   ├── best_student_performance_model.joblib
│   ├── student_predictions.csv
│   ├── model_comparison.csv
│   ├── final_evaluation.csv
│   ├── feature_importance.csv
│   └── generated charts
│
├── src/
│   ├── generate_dataset.py
│   └── eda.py
│
├── validate_dataset.py
├── train_model.py
├── dashboard.py
├── requirements.txt
├── .gitignore
└── README.md
```

The exact files in `data/` and `results/` may vary depending on which generated outputs are retained in the repository.

## Technologies and libraries

- **Python** – project implementation
- **Pandas** and **NumPy** – data handling and numerical operations
- **Matplotlib** and **Seaborn** – exploratory visualisation
- **Plotly** – interactive charts
- **Scikit-learn** – preprocessing, regression models and evaluation
- **Joblib** – saving and loading the trained model
- **Dash** – interactive web dashboard

## Installation and execution

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd undergraduate_student_performance_project
```

### 2. Create and activate a virtual environment

**Windows PowerShell:**

```powershell
py -3.15 -m venv .venv
.venv\Scripts\Activate.ps1
```

If your environment uses a different Python version, replace `-3.15` with the installed version you intend to use.

### 3. Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Generate and validate the dataset (optional)

If you want to recreate the synthetic dataset:

```powershell
python src/generate_dataset.py
python validate_dataset.py
```

Run these only if you intend to regenerate the data. They may overwrite existing generated files, depending on the script configuration.

### 5. Run exploratory data analysis

```powershell
python src/eda.py
```

### 6. Train and evaluate the models

```powershell
python train_model.py
```

This creates the saved model and evaluation outputs in `results/`.

### 7. Launch the dashboard

```powershell
python dashboard.py
```

Open the local address shown in the terminal, normally:

```text
http://127.0.0.1:8050
```

## Prediction and grading note

The dashboard converts predicted marks into an **illustrative SGPA estimate** using the project's example mark bands. This conversion is not an official grading scale and may not match any university's academic regulations. The model predicts marks; the SGPA display is a separate demonstration calculation.

## Limitations

- The dataset is synthetic and may not represent the complexity, distributions or policies of real universities.
- Model performance on synthetic data does not establish performance on real student populations.
- The illustrative SGPA conversion is not institution-specific.
- Correlation does not imply causation.
- Predictions should not be used to make decisions about real students.

## Future improvements

- Evaluate the model on an appropriately consented, anonymised real-world dataset.
- Add cross-validation and further model tuning.
- Investigate fairness and performance across programmes and academic groups.
- Add prediction uncertainty or prediction intervals.
- Improve the dashboard with downloadable reports and richer filters.

## Author

**Dibyajyoti Swain**  
IBCP | Class XII  
Sai International School

---

*Developed as an educational machine learning and data analytics project.*
