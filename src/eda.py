
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Project paths
DATA_PATH = Path("data/student_performance_synthetic.csv")
OUTPUT_PATH = Path("screenshots/eda")

OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

# Load dataset
df = pd.read_csv(DATA_PATH)

sns.set_theme(style="whitegrid")

print("\nUNDERGRADUATE STUDENT PERFORMANCE ANALYSIS")
print("=" * 55)

# 1. Overall academic performance
print("\n1. Overall Marks Statistics")
print(df["final_marks"].describe())

# 2. Average marks by academic year
year_analysis = df.groupby("academic_year")["final_marks"].mean()

print("\n2. Average Marks by Academic Year")
print(year_analysis)

plt.figure(figsize=(9, 5))
sns.barplot(x=year_analysis.index, y=year_analysis.values)
plt.title("Average Student Marks by Academic Year")
plt.xlabel("Academic Year")
plt.ylabel("Average Final Marks")
plt.ylim(0, 100)
plt.tight_layout()
plt.savefig(OUTPUT_PATH / "yearly_performance.png", dpi=300)
plt.show()

# 3. Distribution of final marks
plt.figure(figsize=(9, 5))
sns.histplot(df["final_marks"], bins=25, kde=True)
plt.title("Distribution of Student Final Marks")
plt.xlabel("Final Marks")
plt.ylabel("Number of Records")
plt.tight_layout()
plt.savefig(OUTPUT_PATH / "marks_distribution.png", dpi=300)
plt.show()

# 4. Attendance vs final marks
plt.figure(figsize=(9, 5))
sns.scatterplot(
    data=df,
    x="attendance_percentage",
    y="final_marks",
    alpha=0.35
)
plt.title("Attendance vs Academic Performance")
plt.xlabel("Attendance Percentage")
plt.ylabel("Final Marks")
plt.tight_layout()
plt.savefig(OUTPUT_PATH / "attendance_vs_marks.png", dpi=300)
plt.show()

# 5. Study hours vs final marks
plt.figure(figsize=(9, 5))
sns.regplot(
    data=df,
    x="study_hours_per_day",
    y="final_marks",
    scatter_kws={"alpha": 0.25}
)
plt.title("Study Hours vs Academic Performance")
plt.xlabel("Daily Study Hours")
plt.ylabel("Final Marks")
plt.tight_layout()
plt.savefig(OUTPUT_PATH / "study_hours_vs_marks.png", dpi=300)
plt.show()

# 6. Correlation heatmap
numeric_columns = [
    "academic_year",
    "semester",
    "attendance_percentage",
    "study_hours_per_day",
    "assignment_average",
    "internal_assessment_score",
    "quiz_average",
    "class_participation_score",
    "previous_semester_marks",
    "final_marks"
]

correlation = df[numeric_columns].corr()

plt.figure(figsize=(12, 9))
sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5
)
plt.title("Academic Performance Correlation Heatmap")
plt.tight_layout()
plt.savefig(OUTPUT_PATH / "correlation_heatmap.png", dpi=300)
plt.show()

# 7. Programme-wise performance
program_analysis = (
    df.groupby("program")["final_marks"]
    .mean()
    .sort_values(ascending=False)
)

print("\n3. Programme-wise Average Marks")
print(program_analysis)

print("\nEDA COMPLETED SUCCESSFULLY")
print("Charts saved in:", OUTPUT_PATH)
