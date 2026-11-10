"""
Generate a reproducible synthetic undergraduate student-semester dataset.
Uses only Python's standard library.
Run: python src/generate_dataset.py
"""
import csv
import random
from pathlib import Path

SEED = 20261109
STUDENTS = 1250
PROGRAMS = [
    ("Computer Science", 1.5),
    ("Business Administration", 0.0),
    ("Electrical Engineering", 1.0),
    ("Psychology", 0.3),
    ("Biotechnology", 0.6),
]

def clamp(value, low, high):
    return max(low, min(high, value))

def marks_to_sgpa(marks):
    # Illustrative mapping for this synthetic project; not a universal university rule.
    if marks >= 90: return 10.0
    if marks >= 80: return 9.0
    if marks >= 70: return 8.0
    if marks >= 60: return 7.0
    if marks >= 50: return 6.0
    if marks >= 45: return 5.0
    if marks >= 40: return 4.0
    return 0.0

def main():
    rng = random.Random(SEED)
    output = Path(__file__).resolve().parents[1] / "data" / "student_performance_synthetic.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "student_id", "program", "academic_year", "semester",
        "attendance_percentage", "study_hours_per_day",
        "assignment_average", "internal_assessment_score",
        "quiz_average", "class_participation_score",
        "previous_semester_marks", "previous_semester_sgpa",
        "final_marks", "final_sgpa"
    ]

    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()

        for student_number in range(1, STUDENTS + 1):
            student_id = f"STU{student_number:05d}"
            program, program_effect = rng.choice(PROGRAMS)
            ability = rng.gauss(0, 7.0)
            study_habit = clamp(rng.gauss(3.8, 1.25), 0.5, 8.0)
            attendance_base = clamp(rng.gauss(82, 9), 45, 100)
            previous_marks = None

            for semester in range(1, 9):
                year = (semester - 1) // 2 + 1
                study_hours = round(clamp(study_habit + rng.gauss(0, 0.7), 0.5, 10), 1)
                attendance = round(clamp(
                    attendance_base + rng.gauss(0, 6) + (study_hours - 4) * 1.1, 35, 100
                ), 1)
                assignment = round(clamp(
                    68 + ability * 0.55 + (study_hours - 3.5) * 2.2 + rng.gauss(0, 9), 20, 100
                ), 1)
                internal = round(clamp(
                    65 + ability * 0.60 + (attendance - 75) * 0.20 + rng.gauss(0, 10), 15, 100
                ), 1)
                quiz = round(clamp(
                    64 + ability * 0.58 + (study_hours - 3.5) * 2.0 + rng.gauss(0, 11), 10, 100
                ), 1)
                participation = round(clamp(
                    5.5 + (attendance - 75) * 0.045 + rng.gauss(0, 1.4), 0, 10
                ), 1)

                previous_effect = 0 if previous_marks is None else (previous_marks - 65) * 0.12
                raw = (
                    0.28 * assignment + 0.27 * internal + 0.18 * quiz
                    + 0.10 * attendance + 1.25 * study_hours
                    + 0.35 * participation + program_effect
                    + ability * 0.42 + previous_effect + rng.gauss(0, 5.5)
                )
                final_marks = round(clamp(raw - 9.0, 0, 100), 1)
                writer.writerow({
                    "student_id": student_id,
                    "program": program,
                    "academic_year": year,
                    "semester": semester,
                    "attendance_percentage": attendance,
                    "study_hours_per_day": study_hours,
                    "assignment_average": assignment,
                    "internal_assessment_score": internal,
                    "quiz_average": quiz,
                    "class_participation_score": participation,
                    "previous_semester_marks": "" if previous_marks is None else round(previous_marks, 1),
                    "previous_semester_sgpa": "" if previous_marks is None else marks_to_sgpa(previous_marks),
                    "final_marks": final_marks,
                    "final_sgpa": marks_to_sgpa(final_marks),
                })
                previous_marks = final_marks

    print(f"Generated {STUDENTS * 8:,} records: {output}")

if __name__ == "__main__":
    main()
