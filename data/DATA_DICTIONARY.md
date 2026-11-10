# Dataset Dictionary

File: `student_performance_synthetic.csv`

This is a fully synthetic dataset generated for an educational machine-learning project. It contains 10,000 student-semester records representing 1,250 fictional undergraduate students across eight semesters (four academic years).

| Column | Type | Meaning | Role |
|---|---|---|---|
| student_id | string | Anonymous synthetic student identifier | Grouping only; exclude from model |
| program | category | Fictional programme category | Input feature |
| academic_year | integer (1–4) | Undergraduate year | Input feature |
| semester | integer (1–8) | Semester number | Input feature |
| attendance_percentage | float (35–100) | Attendance percentage | Input feature |
| study_hours_per_day | float | Average daily study hours | Input feature |
| assignment_average | float (20–100) | Average assignment score | Input feature |
| internal_assessment_score | float (15–100) | Internal assessment score | Input feature |
| quiz_average | float (10–100) | Average quiz score | Input feature |
| class_participation_score | float (0–10) | Synthetic participation score | Input feature |
| previous_semester_marks | float or blank | Previous semester final marks; blank in semester 1 | Input feature |
| previous_semester_sgpa | float or blank | Previous semester SGPA; blank in semester 1 | Input feature |
| final_marks | float (0–100) | Current semester final marks | Primary target |
| final_sgpa | float (0–10) | SGPA mapped from final marks | Secondary target |

## Important modelling notes

- One row represents one student's semester, not one unique student.
- There are 1,250 synthetic students × 8 semesters = 10,000 rows.
- The first semester has no previous-semester values; these are intentionally blank.
- `student_id` is an identifier, not a predictive feature.
- Train/test splitting must be designed to prevent student-level leakage. A group-based split by `student_id` is recommended when evaluating generalisation to unseen students.
- `final_marks` and `final_sgpa` are target columns. Do not use either as an input feature when predicting the other without clearly defining a separate modelling experiment.
- `final_sgpa` uses an illustrative mark-to-SGPA mapping (90+=10, 80–89.9=9, 70–79.9=8, 60–69.9=7, 50–59.9=6, 45–49.9=5, 40–44.9=4, below 40=0). Actual university grading rules differ. This mapping is not an official grading standard.
- All records are simulated. They are not actual university records and should not be presented as empirical evidence about real students.
