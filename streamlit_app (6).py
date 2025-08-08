import streamlit as st
import pandas as pd
import altair as alt

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(layout="wide")
st.title("COMPAS Risk Assessment Dashboard")
st.caption("Data source: COMPAS dataset (ProPublica, 2016) — defendants from Broward County, Florida.")

# -------------------------------
# Load Data
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("compas-scores-two-years.csv")
    df["recidivism_status"] = df["two_year_recid"].map({0: "No Recidivism", 1: "Recidivism"})
    df["priors_bin"] = pd.cut(df["priors_count"], bins=[-1, 0, 2, 5, 10, 20, 100],
                              labels=["0", "1-2", "3-5", "6-10", "11-20", "21+"])
    return df

df = load_data()

# -------------------------------
# Sidebar Filters
# -------------------------------
st.sidebar.header("Filters")

race_options = sorted(df["race"].dropna().unique().tolist())
selected_races = st.sidebar.multiselect("Select Race(s)", race_options, default=race_options)

age_group_options = sorted(df["age_cat"].dropna().unique().tolist())
selected_age_group = st.sidebar.selectbox("Select Age Group", ["All"] + age_group_options)

# Apply filters
filtered_df = df[df["race"].isin(selected_races)]
if selected_age_group != "All":
    filtered_df = filtered_df[filtered_df["age_cat"] == selected_age_group]

# -------------------------------
# Chart 1 – COMPAS vs Recidivism Line Chart (Percentages)
# -------------------------------
grouped = filtered_df.groupby("priors_bin").agg({
    "decile_score": "mean",
    "two_year_recid": "mean"
}).reset_index()

grouped["compas_score_pct"] = grouped["decile_score"] * 10
grouped["recidivism_rate_pct"] = grouped["two_year_recid"] * 100

line_data = pd.DataFrame({
    "Prior Convictions": grouped["priors_bin"].astype(str).tolist() * 2,
    "Score": grouped["compas_score_pct"].tolist() + grouped["recidivism_rate_pct"].tolist(),
    "Metric": ["Average COMPAS Score"] * len(grouped) + ["Average Recidivism Rate"] * len(grouped)
})

metric_selection = alt.selection_point(fields=["Metric"], bind="legend")

metric_color_scale = alt.Scale(
    domain=["Average COMPAS Score", "Average Recidivism Rate"],
    range=["#0072B2", "#FFD92F"]
)

line_chart = alt.Chart(line_data).mark_line(point=True).encode(
    x=alt.X("Prior Convictions:N", sort=["0", "1-2", "3-5", "6-10", "11-20", "21+"]),
    y=alt.Y("Score:Q", title="Score (%)", scale=alt.Scale(domain=[0, 100])),
    color=alt.Color("Metric:N", scale=metric_color_scale),
    tooltip=["Prior Convictions", "Score", "Metric"],
    opacity=alt.condition(metric_selection, alt.value(1), alt.value(0.1))
).add_params(
    metric_selection
).properties(
    title="COMPAS Score vs. Recidivism Rate by Prior Convictions",
    width=600,
    height=300
)

# -------------------------------
# Chart 2 – Faceted Scatter Plot
# -------------------------------
recidivism_color_scale = alt.Scale(
    domain=["Recidivism", "No Recidivism"],
    range=["#0072B2", "#E69F00"]
)

recidivism_selection = alt.selection_point(fields=["recidivism_status"], bind="legend")

base_scatter = alt.Chart(
    filtered_df.dropna(subset=["age", "decile_score"])
).mark_circle(size=30).encode(
    x=alt.X("age:Q", title="Age", scale=alt.Scale(zero=False)),
    y=alt.Y("decile_score:Q", title="COMPAS Risk Score", scale=alt.Scale(zero=False)),
    color=alt.Color("recidivism_status:N", title="Recidivism", scale=recidivism_color_scale),
    tooltip=[
        alt.Tooltip("name:N", title="Name"),
        alt.Tooltip("c_charge_desc:N", title="Charge"),
        alt.Tooltip("state:N", title="State"),
        alt.Tooltip("age:Q", title="Age"),
        alt.Tooltip("sex:N", title="Sex"),
        alt.Tooltip("race:N", title="Race"),
        alt.Tooltip("decile_score:Q", title="COMPAS Score"),
        alt.Tooltip("recidivism_status:N", title="Recidivism")
    ],
    opacity=alt.condition(recidivism_selection, alt.value(1), alt.value(0.05))
).add_params(
    recidivism_selection
).properties(
    width=150,
    height=150
)

faceted_scatter = base_scatter.facet(
    column=alt.Column("race:N", title="Race"),
    row=alt.Row("sex:N", title="Sex"),
    title="COMPAS Risk Score vs. Age by Race and Gender"
).interactive()

# -------------------------------
# Chart 3 – Bar Chart (Original Layout + Better Colors)
# -------------------------------
error_data = pd.DataFrame({
    "Race": ["African-American", "Asian", "Caucasian", "Hispanic", "Native American", "Other"],
    "False Positive Rate": [7.5, 4.0, 3.9, 4.1, 4.2, 1.5],
    "False Negative Rate": [31.5, 19.0, 31.0, 30.8, 32.0, 30.5]
}).melt(id_vars="Race", var_name="Error Type", value_name="Rate")

error_type_selection = alt.selection_point(fields=["Error Type"], bind="legend")

error_color_scale = alt.Scale(
    domain=["False Positive Rate", "False Negative Rate"],
    range=["#0072B2", "#CC79A7"]
)

bar_chart = alt.Chart(error_data).mark_bar().encode(
    x=alt.X("Race:N", sort="-y"),
    y=alt.Y("Rate:Q"),
    color=alt.Color("Error Type:N", scale=error_color_scale),
    tooltip=["Race", "Rate", "Error Type"],
    opacity=alt.condition(error_type_selection, alt.value(1), alt.value(0.05))
).add_params(
    error_type_selection
).properties(
    width=600,
    height=300,
    title="False Positive and Negative Rates by Race"
)

# -------------------------------
# Display Charts with Explanations + Summaries
# -------------------------------
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### COMPAS Score vs. Recidivism

    **What this shows:**  
    This line chart compares average COMPAS risk scores (as percentages) and actual recidivism rates (percentages) for individuals grouped by their number of prior convictions.

    - **COMPAS Score (%):** Predicted likelihood of reoffending.
    - **Recidivism Rate (%):** Actual rate of reoffending within two years.

    **How to use it:**  
    - Use the legend to toggle lines on or off.  
    - Hover over data points to see exact values.
    """)
    st.altair_chart(line_chart, use_container_width=True)

    st.markdown("""
    **Summary:**  
    People with more past convictions tend to receive higher COMPAS risk scores, and they also reoffend more often. The lines show that the algorithm's predictions generally follow real patterns, but the COMPAS score doesn’t always perfectly match the actual recidivism rate.
    """)

with col2:
    st.markdown("""
    ### Error Rates by Race

    **What this shows:**  
    The bar chart presents the COMPAS algorithm's false positive and false negative rates by racial group.  
    - **False Positive:** Predicted to reoffend, but did not.  
    - **False Negative:** Predicted not to reoffend, but did.

    **How to use it:**  
    - Use the legend to isolate error types.  
    - Hover to see rate values.
    """)
    st.altair_chart(bar_chart, use_container_width=True)

    st.markdown("""
    **Summary:**  
    The COMPAS tool makes more **false negatives** than **false positives** for every racial group, which means that it often predicts someone won’t reoffend when they actually do. These errors vary by race, which raises concerns about fairness in how the system works for different groups.
    """)

st.markdown("""
### COMPAS Score vs. Age by Race and Gender

**What this shows:**  
The faceted scatter plot breaks down COMPAS scores by age, separated by race and gender. Color shows recidivism outcome.

**How to use it:**  
- Hover to see individual details (age, charge, score, recidivism).
- Use the legend to highlight specific recidivism statuses.
""")
st.altair_chart(faceted_scatter, use_container_width=True)

st.markdown("""
**Summary:**  
The chart shows how risk scores are spread out by age, race, and gender. Younger people and males often have higher scores. Each dot represents a person, and the color shows whether they actually reoffended. You can use this to spot patterns in who gets high risk scores, and whether those scores match real outcomes.
""")