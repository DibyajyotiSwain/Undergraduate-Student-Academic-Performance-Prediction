# Undergraduate Student Academic Performance Prediction Using Machine Learning

<p align="center">

**An Academic Analytics and Machine Learning Project**

Developed by **Dibyajyoti Swain**  
IBCP | Class XII | Sai International School

</p>

---

## 1. Introduction

The Undergraduate Student Academic Performance Prediction project is a machine learning-based academic analytics system designed to analyse student performance and predict final academic marks using various educational and study-related indicators.

The project focuses on undergraduate students pursuing a four-year degree programme, covering Years 1 to 4 and their respective semesters.

The primary idea behind this project is to explore how factors such as attendance, daily study hours, assignment performance, internal assessment scores, quiz performance, class participation and previous semester marks relate to academic outcomes.

Using these factors, the system applies supervised machine learning regression algorithms to estimate students' final marks. An interactive web dashboard has also been developed to present academic insights, compare machine learning models and allow users to experiment with individual student performance predictions.

The project demonstrates the complete machine learning workflow, from dataset generation and exploratory data analysis to preprocessing, model training, evaluation and interactive application development.

## 2. Project Objectives

The major objectives of this project are:

- To generate and analyse an undergraduate academic performance dataset.
- To examine academic performance patterns across four undergraduate years.
- To identify relationships between attendance, study habits, assessment scores and final marks.
- To perform data cleaning, preprocessing and feature engineering.
- To implement and compare multiple machine learning regression algorithms.
- To evaluate prediction performance using standard regression metrics.
- To identify the best-performing model based on validation results.
- To develop an interactive dashboard for academic analysis and marks prediction.
- To demonstrate how machine learning can be integrated into an educational analytics application.

## 3. Dataset Description

A synthetic dataset containing **10,000 student-semester records** was generated for this project.

The dataset represents 1,250 fictional undergraduate students, with each student contributing eight semester records across four academic years.

Each row represents an individual student's academic information for a particular semester.

The dataset includes the following variables:

| Feature | Description |
|---|---|
| `student_id` | Unique synthetic student identifier |
| `program` | Undergraduate academic programme |
| `academic_year` | Current undergraduate year (1–4) |
| `semester` | Semester number (1–8) |
| `attendance_percentage` | Student attendance percentage |
| `study_hours_per_day` | Average daily study hours |
| `assignment_average` | Average assignment score |
| `internal_assessment_score` | Internal assessment marks |
| `quiz_average` | Average quiz performance |
| `class_participation_score` | Class participation score |
| `previous_semester_marks` | Previous semester marks, where available |
| `previous_semester_sgpa` | Previous semester SGPA, where available |
| `final_marks` | Final semester marks and primary prediction target |
| `final_sgpa` | Final SGPA field included in the synthetic dataset |

**Dataset size:** 10,000 records  
**Unique students:** 1,250  
**Academic duration:** 4 years  
**Total semesters:** 8

The dataset is entirely synthetic and was created for educational and experimental purposes. It does not contain actual university student records.

## 4. Exploratory Data Analysis (EDA)

Exploratory Data Analysis was conducted to understand the dataset, examine distributions and identify relationships between academic variables.

The analysis included:

- Overall distribution of final marks.
- Average marks across academic Years 1–4.
- Attendance percentage versus final marks.
- Daily study hours versus final marks.
- Programme-wise academic performance.
- Correlation analysis between numerical variables.
- Identification of important academic indicators.

### Correlation Analysis

The correlation analysis revealed the following relationships with final marks:

| Academic Variable | Correlation |
|---|---:|
| Previous semester marks | 0.59 |
| Assignment average | 0.55 |
| Internal assessment score | 0.52 |
| Quiz average | 0.46 |
| Study hours per day | 0.37 |
| Attendance percentage | 0.22 |
| Class participation score | 0.11 |

Previous semester marks, assignment averages and internal assessment scores demonstrated relatively stronger positive linear relationships with final marks in the generated dataset.

Attendance and study hours also showed positive relationships.

These findings are descriptive observations from synthetic data. Correlation does not establish causation or independently determine which features will be most useful to a trained model.

The analysis also identified a high correlation between academic year and semester. To reduce redundant progression information, academic year was excluded from the initial model input features.

## 5. Data Preprocessing and Feature Engineering

Before training the machine learning models, the dataset was prepared through a structured preprocessing workflow.

The following operations were performed:

1. Removal of duplicate records.
2. Separation of input features and the prediction target.
3. Exclusion of student identifiers from model training.
4. Exclusion of final SGPA and other target-related information to prevent data leakage.
5. Handling of missing numerical values using median imputation.
6. Handling of missing categorical values using most-frequent imputation.
7. Numerical feature scaling using StandardScaler.
8. Categorical feature encoding using OneHotEncoder.
9. Group-based splitting of student records into training, validation and testing datasets.

### Data Splitting Strategy

The dataset was divided into:

- Training set: approximately 60%.
- Validation set: approximately 20%.
- Testing set: approximately 20%.

The splitting process was performed using student-based grouping rather than randomly separating individual semester records.

This ensured that records belonging to the same student were not distributed across the training, validation and testing sets.

This approach helps reduce student-level data leakage and provides a more meaningful evaluation of the model's ability to generalise to previously unseen students.

## 6. Machine Learning Algorithms

Four supervised machine learning regression algorithms were implemented and compared.

### 6.1 Linear Regression

Linear Regression was used as a baseline model to understand the relationship between academic input variables and final marks.

It estimates the target variable using a linear combination of the input features.

### 6.2 Decision Tree Regressor

Decision Tree Regression was implemented to capture non-linear relationships and decision-based patterns within academic performance data.

### 6.3 Random Forest Regressor

Random Forest Regression uses an ensemble of decision trees to generate predictions. It was included to evaluate whether combining multiple trees could improve prediction performance.

### 6.4 Gradient Boosting Regressor

Gradient Boosting Regression was implemented as an ensemble learning approach that builds successive models to improve prediction performance by learning from previous errors.

## 7. Model Evaluation and Comparison

The four models were evaluated using standard regression metrics:

- **Mean Absolute Error (MAE):** Measures the average absolute difference between actual and predicted marks.
- **Mean Squared Error (MSE):** Measures the average squared prediction error.
- **Root Mean Squared Error (RMSE):** Represents prediction error in the original marks scale.
- **R² Score:** Measures the proportion of variance in the target variable explained by the model.

### Validation Results

The following table presents the actual validation results obtained during model comparison.

| Machine Learning Model | MAE | RMSE | R² Score |
|---|---:|---:|---:|
| Linear Regression | **4.6362** | **5.8358** | **0.6909** |
| Gradient Boosting | 4.7619 | 5.9879 | 0.6746 |
| Random Forest | 4.9014 | 6.1348 | 0.6584 |
| Decision Tree | 5.9582 | 7.3764 | 0.5061 |

Based on the validation results, Linear Regression achieved the lowest MAE and was selected as the best-performing model among the four algorithms evaluated.

### Final Test Performance

The selected Linear Regression model was subsequently evaluated using the held-out testing dataset.

| Evaluation Metric | Final Result |
|---|---:|
| Selected Model | Linear Regression |
| Mean Absolute Error (MAE) | **4.6907** |
| Mean Squared Error (MSE) | **34.7719** |
| Root Mean Squared Error (RMSE) | **5.8968** |
| R² Score | **0.6623** |

The final test results indicate an average absolute prediction error of approximately 4.69 marks on the synthetic testing dataset.

The R² score of 0.6623 indicates that the selected model explains approximately 66.23% of the variation in final marks within this particular test dataset.

These results are specific to the generated synthetic data and should not be interpreted as a guarantee of performance on real-world student populations.

## 8. Interactive Academic Analytics Dashboard

An interactive web dashboard was developed using Python and Dash to make the project more accessible and visually understandable.

The dashboard integrates dataset analysis, visualisation, model evaluation and interactive marks prediction into a single application.

### Dashboard Features

**Academic Overview**

- Total student-semester records.
- Number of unique students.
- Average final marks.
- Average attendance percentage.

**Academic Performance Analysis**

- Distribution of final marks.
- Year-wise academic performance.
- Semester-wise performance trends.
- Programme-wise average marks.

**Academic Performance Insights**

- Attendance versus final marks.
- Study hours versus final marks.
- Exploration of relationships between academic indicators.

**Machine Learning Evaluation**

- Comparison of the four regression algorithms.
- Final test MAE, RMSE and R² scores.
- Actual versus predicted marks visualisation.

**Interactive Marks Prediction**

Users can enter academic information, including:

- Undergraduate programme.
- Current semester.
- Attendance percentage.
- Daily study hours.
- Assignment average.
- Internal assessment score.
- Quiz average.
- Class participation.
- Previous semester marks, where available.

The saved machine learning pipeline processes the entered information and generates an estimated final marks prediction.

The dashboard also displays an illustrative SGPA estimate using the project's example grading conversion scale.

The SGPA conversion is for demonstration purposes only and is not based on an official university grading policy.

### Dashboard Screenshots

The project's dashboard screenshots can be found in the `screenshots/` directory.

Here are some samples of the Screenshots: 
<img width="2031" height="700" alt="Screenshot 2026-11-10 150350" src="https://github.com/user-attachments/assets/c0e7419f-7bac-44bd-a4aa-ebd9dcab7082" />
<img width="2063" height="813" alt="Screenshot 2026-11-10 150021" src="https://github.com/user-attachments/assets/b8c90490-eae5-4108-87ac-3dc101f3711d" />

Suggested screenshots include:

1. Dashboard overview.
2. Academic performance insights.
3. Machine learning model comparison.
4. Actual versus predicted marks.
5. Interactive student marks prediction.

## 9. Technologies and Libraries Used

The project was developed using Python and the following technologies:

| Technology / Library | Application |
|---|---|
| Python | Main programming language |
| Pandas | Dataset manipulation and analysis |
| NumPy | Numerical operations |
| Matplotlib | Data visualisation |
| Seaborn | Statistical charts and heatmaps |
| Plotly | Interactive visualisations |
| Scikit-learn | Machine learning and evaluation |
| SciPy | Scientific and statistical operations |
| Joblib | Saving and loading trained models |
| Dash | Interactive web dashboard |
| JupyterLab | Notebook-based analysis |
| Openpyxl | Excel file support |
| PyArrow | Columnar data file support |
| Statsmodels | Statistical analysis |
| SHAP | Model explainability tools |

## 10. Project Structure

The repository is organised into the following structure:

```text
undergraduate_student_performance_project/
│
├── data/
│   ├── student_performance_synthetic.csv
│   ├── student_performance_sample.csv
│   └── DATA_DICTIONARY.md
│
├── notebooks/
│
├── src/
│   ├── generate_dataset.py
│   └── eda.py
│
├── models/
│
├── results/
│   ├── best_student_performance_model.joblib
│   ├── student_predictions.csv
│   ├── model_comparison.csv
│   ├── final_evaluation.csv
│   ├── feature_importance.csv
│   ├── actual_vs_predicted.png
│   ├── model_comparison.png
│   └── prediction_residuals.png
│
├── screenshots/
│
├── validate_dataset.py
├── train_model.py
├── dashboard.py
├── requirements.txt
├── .gitignore
└── README.md
```


## 11. Installation and Execution

The project can be run locally using Python and Visual Studio Code.

### Step 1: Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
```

Navigate into the project directory:

```bash
cd undergraduate_student_performance_project
```

### Step 2: Create a Virtual Environment

For Windows PowerShell:

```powershell
py -3.15 -m venv .venv
```

Activate the environment:

```powershell
.venv\Scripts\Activate.ps1
```

### Step 3: Install Required Libraries

```powershell
python -m pip install --upgrade pip
```

Install project dependencies:

```powershell
python -m pip install -r requirements.txt
```

### Step 4: Validate the Dataset

```powershell
python validate_dataset.py
```

### Step 5: Perform Exploratory Data Analysis

```powershell
python src/eda.py
```

### Step 6: Train and Evaluate the Machine Learning Models

```powershell
python train_model.py
```

This process trains the four regression models, compares their validation performance, selects the best model and generates the final test evaluation results.

The trained model and generated outputs are saved in the `results/` directory.

### Step 7: Launch the Interactive Dashboard

```powershell
python dashboard.py
```

Once the application starts, open the following address in your browser:

**http://127.0.0.1:8050**

The dashboard loads the previously saved machine learning pipeline and does not retrain the models each time it is launched.

## 12. Limitations

Although the project demonstrates an end-to-end machine learning workflow, it has certain limitations.

- The dataset is synthetically generated and does not represent actual university student records.
- The relationships between academic variables are based on the assumptions used during synthetic data generation.
- Model performance on synthetic data may differ significantly from performance on real-world academic datasets.
- The SGPA conversion is illustrative and may not match the grading regulations of individual universities.
- Correlation analysis does not establish causal relationships.
- Predictions should not be used for actual student grading, admissions or other high-stakes academic decisions.

## 13. Future Scope

The project can be further developed through the following improvements:

- Evaluation using appropriately anonymised real-world academic datasets.
- Implementation of additional machine learning and ensemble algorithms.
- Hyperparameter optimisation and cross-validation.
- More detailed feature importance and explainability analysis.
- Prediction uncertainty estimation.
- Additional academic performance indicators.
- Interactive filtering and downloadable analytical reports.
- Further improvements to dashboard design and usability.

## 14. Conclusion

The Undergraduate Student Academic Performance Prediction project demonstrates how data analysis and machine learning can be applied to an educational performance prediction problem.

Starting with the generation of a synthetic dataset, the project progressed through data validation, exploratory data analysis, preprocessing, feature selection, model development, model comparison and final evaluation.

The implementation of four regression algorithms provided an opportunity to compare different approaches to predicting academic marks. Linear Regression achieved the best validation performance among the models tested and was selected for final evaluation.

The development of an interactive Dash dashboard further demonstrates how a trained machine learning model can be integrated into a user-facing analytical application.

Overall, this project provided practical experience in Python programming, data science, machine learning, statistical evaluation and interactive application development.

---

## Author

**Dibyajyoti Swain**  


**Project:** Undergraduate Student Academic Performance Prediction Using Machine Learning

*An educational project developed using synthetic academic data.*
