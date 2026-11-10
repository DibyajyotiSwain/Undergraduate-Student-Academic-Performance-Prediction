"""
Student Performance Analytics Dashboard
Author: Dibyajyoti Swain | IBCP, Class XII | Sai International School

Run from the project root:
    python dashboard.py

Loads the saved pipeline from results/ and does not retrain the model.
"""

from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "student_performance_synthetic.csv"
RESULTS = ROOT / "results"
MODEL_PATH = RESULTS / "best_student_performance_model.joblib"
EVALUATION_PATH = RESULTS / "final_evaluation.csv"
COMPARISON_PATH = RESULTS / "model_comparison.csv"
PREDICTIONS_PATH = RESULTS / "student_predictions.csv"

# ---------- Load project files ----------
for path in [DATA_PATH, MODEL_PATH]:
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}\n"
            "Keep dashboard.py in the project root and run train_model.py first."
        )

df = pd.read_csv(DATA_PATH)
model = joblib.load(MODEL_PATH)
evaluation = pd.read_csv(EVALUATION_PATH) if EVALUATION_PATH.exists() else pd.DataFrame()
comparison = pd.read_csv(COMPARISON_PATH) if COMPARISON_PATH.exists() else pd.DataFrame()
predictions = pd.read_csv(PREDICTIONS_PATH) if PREDICTIONS_PATH.exists() else pd.DataFrame()

required = {
    "student_id", "program", "academic_year", "semester",
    "attendance_percentage", "study_hours_per_day",
    "assignment_average", "internal_assessment_score",
    "quiz_average", "class_participation_score",
    "previous_semester_marks", "final_marks"
}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Dataset is missing columns: {sorted(missing)}")

# ---------- Theme and helpers ----------
NAVY = "#172554"
BLUE = "#3157D5"
PURPLE = "#7656D6"
GREEN = "#15966B"
MUTED = "#667085"
BORDER = "#E5E9F2"
BG = "#F4F6FB"

def sgpa_from_marks(marks):
    """Illustrative conversion for this synthetic project, not a universal scale."""
    marks = float(np.clip(marks, 0, 100))
    if marks >= 90: return 10.0
    if marks >= 80: return 9.0
    if marks >= 70: return 8.0
    if marks >= 60: return 7.0
    if marks >= 50: return 6.0
    if marks >= 45: return 5.0
    if marks >= 40: return 4.0
    return 0.0

def metric(name):
    if evaluation.empty or name not in evaluation.columns:
        return "—"
    value = evaluation.iloc[0][name]
    return "—" if pd.isna(value) else f"{float(value):.3f}"

def make_card(title, value, note, accent=BLUE):
    return html.Div([
        html.Div(title, style={"fontSize": "12px", "color": MUTED, "fontWeight": "600"}),
        html.Div(value, style={"fontSize": "27px", "fontWeight": "750", "color": accent, "marginTop": "8px"}),
        html.Div(note, style={"fontSize": "11px", "color": MUTED, "marginTop": "6px"})
    ], style={
        "background": "white", "border": f"1px solid {BORDER}",
        "borderRadius": "13px", "padding": "18px",
        "boxShadow": "0 3px 12px rgba(20,35,80,.04)"
    })

def panel(child):
    return html.Div(child, style={
        "background": "white", "border": f"1px solid {BORDER}",
        "borderRadius": "13px", "padding": "12px",
        "boxShadow": "0 3px 12px rgba(20,35,80,.04)", "minWidth": "0"
    })

def style_figure(fig, height=330):
    fig.update_layout(
        height=height, margin=dict(l=25, r=20, t=35, b=35),
        paper_bgcolor="white", plot_bgcolor="white",
        font=dict(family="Arial", color="#172033", size=11),
        legend=dict(orientation="h", y=1.08, x=1, xanchor="right")
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="#EDF0F5")
    return fig

# ---------- Summary data and charts ----------
students = df["student_id"].nunique()
records = len(df)
avg_marks = df["final_marks"].mean()
avg_attendance = df["attendance_percentage"].mean()

year_data = df.groupby("academic_year", as_index=False)["final_marks"].mean()
year_data["Year"] = year_data["academic_year"].map(lambda x: f"Year {int(x)}")
semester_data = df.groupby("semester", as_index=False)["final_marks"].mean()
program_data = df.groupby("program", as_index=False)["final_marks"].mean().sort_values("final_marks")

fig_distribution = px.histogram(
    df, x="final_marks", nbins=25,
    labels={"final_marks": "Final marks", "count": "Records"},
    color_discrete_sequence=[BLUE]
)
fig_year = px.bar(
    year_data, x="Year", y="final_marks",
    labels={"final_marks": "Average marks"},
    color_discrete_sequence=[PURPLE]
)
fig_semester = px.line(
    semester_data, x="semester", y="final_marks", markers=True,
    labels={"semester": "Semester", "final_marks": "Average marks"},
    color_discrete_sequence=[BLUE]
)
fig_program = px.bar(
    program_data, x="final_marks", y="program", orientation="h",
    labels={"final_marks": "Average marks", "program": "Programme"},
    color_discrete_sequence=[GREEN]
)

sample = df.sample(min(1500, len(df)), random_state=42)
fig_attendance = px.scatter(
    sample, x="attendance_percentage", y="final_marks",
    color="academic_year", opacity=.55,
    labels={"attendance_percentage": "Attendance (%)", "final_marks": "Final marks",
            "academic_year": "Academic year"}
)
fig_study = px.scatter(
    sample, x="study_hours_per_day", y="final_marks",
    color="academic_year", opacity=.55,
    labels={"study_hours_per_day": "Study hours per day", "final_marks": "Final marks",
            "academic_year": "Academic year"}
)

if not comparison.empty and {"Model", "MAE"}.issubset(comparison.columns):
    fig_comparison = px.bar(
        comparison.sort_values("MAE"), x="MAE", y="Model", orientation="h",
        labels={"MAE": "Validation MAE (lower is better)"},
        color_discrete_sequence=[BLUE]
    )
else:
    fig_comparison = go.Figure()
    fig_comparison.add_annotation(text="Model comparison results unavailable.",
                                   x=.5, y=.5, xref="paper", yref="paper", showarrow=False)

if not predictions.empty and {"Actual_Marks", "Predicted_Marks"}.issubset(predictions.columns):
    fig_actual = px.scatter(
        predictions, x="Actual_Marks", y="Predicted_Marks", opacity=.5,
        labels={"Actual_Marks": "Actual marks", "Predicted_Marks": "Predicted marks"},
        color_discrete_sequence=[PURPLE]
    )
    fig_actual.add_trace(go.Scatter(
        x=[0, 100], y=[0, 100], mode="lines", name="Ideal prediction",
        line={"color": "#D94B4B", "dash": "dash"}
    ))
else:
    fig_actual = go.Figure()
    fig_actual.add_annotation(text="Prediction results unavailable.",
                              x=.5, y=.5, xref="paper", yref="paper", showarrow=False)

figures = [
    fig_distribution, fig_year, fig_semester, fig_program,
    fig_attendance, fig_study, fig_comparison, fig_actual
]
for fig in figures:
    style_figure(fig)

# ---------- Dash application ----------
app = Dash(__name__)
app.title = "Student Performance Analytics"

label_style = {"display": "block", "fontSize": "12px", "fontWeight": "600",
               "color": MUTED, "marginBottom": "6px"}
input_style = {"width": "100%", "padding": "10px", "border": f"1px solid {BORDER}",
               "borderRadius": "8px", "boxSizing": "border-box", "background": "white"}

def input_field(label, component):
    return html.Div([html.Label(label, style=label_style), component],
                    style={"marginBottom": "13px"})

best_name = (
    str(evaluation.iloc[0]["Best_Model"])
    if not evaluation.empty and "Best_Model" in evaluation.columns
    else "Saved machine learning model"
)

app.layout = html.Div([
    # Header
    html.Div([
        html.Div([
            html.Div("ACADEMIC DATA SCIENCE PROJECT",
                     style={"fontSize": "10px", "letterSpacing": "2px",
                            "fontWeight": "700", "color": "#C7D2FE"}),
            html.H1("Student Performance Analytics",
                    style={"fontSize": "clamp(23px, 4vw, 32px)", "margin": "9px 0"}),
            html.P("Undergraduate academic analysis and machine learning-based marks prediction",
                   style={"fontSize": "13px", "color": "#D9E2FF", "margin": 0})
        ], style={"flex": "1"}),
        html.Div([
            html.Div("SELECTED MODEL", style={"fontSize": "10px", "color": "#C7D2FE"}),
            html.Div(best_name, style={"fontWeight": "700", "marginTop": "6px"})
        ], style={"background": "rgba(255,255,255,.12)", "padding": "13px 16px",
                  "borderRadius": "10px", "minWidth": "155px"})
    ], style={
        "background": f"linear-gradient(120deg, {NAVY}, #293F83)",
        "color": "white", "padding": "27px 4%", "display": "flex",
        "alignItems": "center", "justifyContent": "space-between",
        "gap": "20px", "flexWrap": "wrap"
    }),

    html.Div([
        html.Div([
            html.B("SYNTHETIC DATA  ", style={"color": BLUE}),
            "This dashboard uses simulated student records for educational demonstration. "
            "Predictions are estimates, not actual academic results."
        ], style={"background": "#EAF0FF", "border": "1px solid #D7E2FF",
                  "padding": "11px 14px", "borderRadius": "9px",
                  "fontSize": "11px", "color": NAVY, "marginBottom": "18px"}),

        # KPI cards
        html.Div([
            make_card("STUDENT-SEMESTER RECORDS", f"{records:,}", "Dataset observations"),
            make_card("UNIQUE STUDENTS", f"{students:,}", "Across four academic years", PURPLE),
            make_card("AVERAGE FINAL MARKS", f"{avg_marks:.1f}/100", "Across synthetic records", GREEN),
            make_card("AVERAGE ATTENDANCE", f"{avg_attendance:.1f}%", "Across synthetic records", "#B7791F")
        ], style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit,minmax(190px,1fr))",
                  "gap": "13px", "marginBottom": "24px"}),

        html.H3("Academic overview", style={"color": NAVY, "fontSize": "18px"}),
        html.P("Explore marks distribution and performance across the undergraduate programme.",
               style={"fontSize": "12px", "color": MUTED}),
        html.Div([
            panel(dcc.Graph(figure=fig_distribution, config={"displayModeBar": False})),
            panel(dcc.Graph(figure=fig_year, config={"displayModeBar": False})),
            panel(dcc.Graph(figure=fig_semester, config={"displayModeBar": False})),
            panel(dcc.Graph(figure=fig_program, config={"displayModeBar": False}))
        ], style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit,minmax(310px,1fr))",
                  "gap": "14px", "marginBottom": "24px"}),

        html.H3("Academic performance insights", style={"color": NAVY, "fontSize": "18px"}),
        html.P("Explore relationships between attendance, study habits and final marks.",
               style={"fontSize": "12px", "color": MUTED}),
        html.Div([
            panel(dcc.Graph(figure=fig_attendance, config={"displayModeBar": False})),
            panel(dcc.Graph(figure=fig_study, config={"displayModeBar": False}))
        ], style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit,minmax(310px,1fr))",
                  "gap": "14px", "marginBottom": "24px"}),

        html.H3("Machine learning model evaluation", style={"color": NAVY, "fontSize": "18px"}),
        html.P("Validation model comparison and final held-out test performance.",
               style={"fontSize": "12px", "color": MUTED}),
        html.Div([
            panel(html.Div([
                dcc.Graph(figure=fig_comparison, config={"displayModeBar": False}),
                html.Div([
                    make_card("FINAL TEST MAE", metric("MAE"), "Lower is better"),
                    make_card("FINAL TEST RMSE", metric("RMSE"), "Lower is better", PURPLE),
                    make_card("FINAL TEST R²", metric("R2"), "Closer to 1 is generally better", GREEN)
                ], style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit,minmax(130px,1fr))",
                          "gap": "9px"})
            ])),
            panel(dcc.Graph(figure=fig_actual, config={"displayModeBar": False}))
        ], style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit,minmax(310px,1fr))",
                  "gap": "14px", "marginBottom": "24px"}),

        # Prediction section
        html.H3("Student marks prediction", style={"color": NAVY, "fontSize": "18px"}),
        html.P("Enter academic details to estimate final semester marks using the saved model.",
               style={"fontSize": "12px", "color": MUTED}),
        html.Div([
            html.Div([
                html.Div([
                    input_field("Undergraduate programme", dcc.Dropdown(
                        id="program", options=[{"label": p, "value": p}
                                               for p in sorted(df["program"].dropna().unique())],
                        value=sorted(df["program"].dropna().unique())[0], clearable=False)),
                    input_field("Semester", dcc.Dropdown(
                        id="semester", options=[{"label": f"Semester {s}", "value": s}
                                                for s in range(1, 9)],
                        value=1, clearable=False)),
                    input_field("Attendance (%)", dcc.Input(id="attendance", type="number",
                        min=0, max=100, value=85, step=.1, style=input_style)),
                    input_field("Study hours per day", dcc.Input(id="study", type="number",
                        min=0, max=24, value=4, step=.1, style=input_style))
                ], style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit,minmax(180px,1fr))",
                          "gap": "0 14px"}),
                html.Div([
                    input_field("Assignment average (0–100)", dcc.Input(id="assignment", type="number",
                        min=0, max=100, value=80, step=.1, style=input_style)),
                    input_field("Internal assessment (0–100)", dcc.Input(id="internal", type="number",
                        min=0, max=100, value=75, step=.1, style=input_style)),
                    input_field("Quiz average (0–100)", dcc.Input(id="quiz", type="number",
                        min=0, max=100, value=75, step=.1, style=input_style)),
                    input_field("Class participation (0–10)", dcc.Input(id="participation", type="number",
                        min=0, max=10, value=7, step=.1, style=input_style))
                ], style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit,minmax(180px,1fr))",
                          "gap": "0 14px"}),
                input_field("Previous semester marks (optional for Semester 1)",
                    dcc.Input(id="previous_marks", type="number", min=0, max=100,
                              value=None, placeholder="Leave blank if unavailable",
                              step=.1, style=input_style)),
                html.Button("Predict student marks", id="predict", n_clicks=0,
                    style={"background": BLUE, "color": "white", "border": "none",
                           "borderRadius": "9px", "padding": "12px 20px",
                           "fontWeight": "700", "cursor": "pointer"}),
                html.Div(id="prediction-message", style={"marginTop": "13px"})
            ], style={"background": "white", "border": f"1px solid {BORDER}",
                      "borderRadius": "13px", "padding": "18px", "flex": "1.5", "minWidth": "280px"}),

            html.Div([
                html.Div("PREDICTION RESULT", style={"fontSize": "11px", "fontWeight": "800",
                    "letterSpacing": "1.5px", "color": MUTED}),
                html.Div("—", id="predicted-marks", style={"fontSize": "46px",
                    "fontWeight": "800", "color": BLUE, "marginTop": "18px"}),
                html.Div("Predicted marks out of 100", style={"fontSize": "12px", "color": MUTED}),
                html.Div("Estimated SGPA: —", id="predicted-sgpa",
                    style={"fontSize": "19px", "fontWeight": "700",
                           "color": GREEN, "marginTop": "24px"}),
                html.P("SGPA uses this project's illustrative conversion scale, not an official university grading policy.",
                    style={"fontSize": "11px", "color": MUTED, "lineHeight": "1.6", "marginTop": "18px"})
            ], style={"background": "#F9FAFF", "border": f"1px solid {BORDER}",
                      "borderRadius": "13px", "padding": "22px", "flex": ".8", "minWidth": "230px"})
        ], style={"display": "flex", "flexWrap": "wrap", "gap": "14px", "alignItems": "stretch"}),

        html.Div([
            html.B("Project by Dibyajyoti Swain"),
            html.Div("IBCP | Class XII | Sai International School", style={"marginTop": "4px"}),
            html.Div("Educational demonstration using synthetic data. Predictions should not be used for grading, admissions or other high-stakes decisions.",
                     style={"marginTop": "8px", "maxWidth": "850px", "marginLeft": "auto", "marginRight": "auto"})
        ], style={"textAlign": "center", "fontSize": "11px", "color": MUTED, "padding": "25px 10px"})
    ], style={"maxWidth": "1400px", "margin": "0 auto", "padding": "22px 3% 35px"})
], style={"background": BG, "minHeight": "100vh", "fontFamily": "Arial, sans-serif",
          "color": "#172033"})


# ---------- Interactive prediction ----------
@app.callback(
    Output("predicted-marks", "children"),
    Output("predicted-sgpa", "children"),
    Output("prediction-message", "children"),
    Input("predict", "n_clicks"),
    State("program", "value"),
    State("semester", "value"),
    State("attendance", "value"),
    State("study", "value"),
    State("assignment", "value"),
    State("internal", "value"),
    State("quiz", "value"),
    State("participation", "value"),
    State("previous_marks", "value"),
    prevent_initial_call=True
)
def predict_student_marks(n_clicks, program, semester, attendance, study,
                          assignment, internal, quiz, participation, previous_marks):
    required_values = [attendance, study, assignment, internal, quiz, participation]
    if any(value is None for value in required_values):
        return "—", "Estimated SGPA: —", html.Div(
            "Please complete all required fields.", style={"color": "#B42318"})

    if not 0 <= attendance <= 100:
        return "—", "Estimated SGPA: —", html.Div("Attendance must be between 0 and 100.", style={"color": "#B42318"})
    if not 0 <= study <= 24:
        return "—", "Estimated SGPA: —", html.Div("Study hours must be between 0 and 24.", style={"color": "#B42318"})
    if any(not 0 <= value <= 100 for value in [assignment, internal, quiz]):
        return "—", "Estimated SGPA: —", html.Div("Assessment scores must be between 0 and 100.", style={"color": "#B42318"})
    if not 0 <= participation <= 10:
        return "—", "Estimated SGPA: —", html.Div("Participation must be between 0 and 10.", style={"color": "#B42318"})
    if previous_marks is not None and not 0 <= previous_marks <= 100:
        return "—", "Estimated SGPA: —", html.Div("Previous marks must be between 0 and 100.", style={"color": "#B42318"})

    # Feature names and order match the training script's X dataframe.
    row = pd.DataFrame([{
        "program": program,
        "semester": int(semester),
        "attendance_percentage": float(attendance),
        "study_hours_per_day": float(study),
        "assignment_average": float(assignment),
        "internal_assessment_score": float(internal),
        "quiz_average": float(quiz),
        "class_participation_score": float(participation),
        "previous_semester_marks": np.nan if previous_marks is None else float(previous_marks)
    }])

    try:
        mark = float(np.clip(model.predict(row)[0], 0, 100))
        sgpa = sgpa_from_marks(mark)
        return (
            f"{mark:.1f}/100",
            f"Estimated SGPA: {sgpa:.1f}/10",
            html.Div("Prediction generated using the saved machine learning pipeline.",
                     style={"color": GREEN, "fontSize": "12px"})
        )
    except Exception as exc:
        return "—", "Estimated SGPA: —", html.Div(
            f"Prediction error: {exc}", style={"color": "#B42318", "fontSize": "12px"})


if __name__ == "__main__":
    print("\nStarting Student Performance Analytics Dashboard...")
    print("Open http://127.0.0.1:8050 in your browser.")
    app.run(debug=False, host="127.0.0.1", port=8050)
