import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# -------------------------------
# Synthetic dataset
# -------------------------------
def generate_data(n=500):
    np.random.seed(42)
    age = np.random.randint(18, 65, n)
    gender = np.random.choice(['Male', 'Female', 'Other'], n, p=[0.45, 0.45, 0.1])
    noise_level = np.random.normal(65, 10, n).clip(40, 100)
    stress_score = np.random.normal(50 + (noise_level-65)*0.6, 10, n).clip(0, 100)
    depression_score = np.random.normal(40 + (noise_level-65)*0.4, 12, n).clip(0, 100)
    return pd.DataFrame({
        'Age': age,
        'Gender': gender,
        'Noise_dB': noise_level,
        'Stress_Score': stress_score,
        'Depression_Score': depression_score
    })

# -------------------------------
# Initialize session state
# -------------------------------
if "data" not in st.session_state:
    st.session_state.data = generate_data()

data = st.session_state.data
user_entry = None  # store latest input

# -------------------------------
# Streamlit frontend
# -------------------------------
st.title("Urban Noise Pollution & Mental Health EDA")

st.subheader("Manual Survey Entry")
with st.form("survey_form"):
    age_input = st.number_input("Age", min_value=10, max_value=100, step=1)
    gender_input = st.selectbox("Gender", ["Male", "Female", "Other"])
    noise_input = st.slider("Noise Level (dB)", 40, 120, 65)
    stress_input = st.slider("Stress Score", 0, 100, 50)
    depression_input = st.slider("Depression Score", 0, 100, 40)
    submitted = st.form_submit_button("Add Record")

if submitted:
    user_entry = pd.DataFrame([{
        'Age': int(age_input),
        'Gender': gender_input,
        'Noise_dB': noise_input,
        'Stress_Score': stress_input,
        'Depression_Score': depression_input
    }])
    st.session_state.data = pd.concat([st.session_state.data, user_entry], ignore_index=True)
    st.success("✅ Manual entry added!")

    st.subheader("Your Submitted Record")
    st.write(user_entry)

# -------------------------------
# Data Overview
# -------------------------------
st.subheader("Dataset Overview")
st.write(data.head())
st.write("Summary Stats:")
st.write(data.describe())

# -------------------------------
# Visualizations
# -------------------------------
st.subheader("Visualizations")

# 1. Noise distribution
fig, ax = plt.subplots(figsize=(8,5))
sns.histplot(x='Noise_dB', data=data, bins=20, kde=True, color='skyblue', ax=ax)
if user_entry is not None:
    ax.axvline(noise_input, color='red', linestyle='--', label='Your Noise Level')
    ax.legend()
ax.set_title('Distribution of Noise Levels (dB)')
st.pyplot(fig)

# 2. Stress by gender
fig, ax = plt.subplots(figsize=(8,5))
sns.boxplot(x='Gender', y='Stress_Score', data=data, palette='Set2', ax=ax)
if user_entry is not None:
    ax.scatter(gender_input, stress_input, color='red', s=100, label="You")
    ax.legend()
ax.set_title('Stress Score Distribution by Gender')
st.pyplot(fig)

# 3. Noise vs Stress
fig, ax = plt.subplots(figsize=(8,5))
sns.scatterplot(x='Noise_dB', y='Stress_Score', hue='Gender', data=data, alpha=0.6, ax=ax)
if user_entry is not None:
    ax.scatter(noise_input, stress_input, color='red', s=120, edgecolor='black', label='You')
    ax.legend()
ax.set_title('Noise Levels vs Stress Scores')
st.pyplot(fig)

# 4. Noise vs Depression
fig, ax = plt.subplots(figsize=(8,5))
sns.scatterplot(x='Noise_dB', y='Depression_Score', data=data, alpha=0.6, ax=ax)
if user_entry is not None:
    ax.scatter(noise_input, depression_input, color='red', s=120, edgecolor='black', label='You')
    ax.legend()
ax.set_title('Noise Levels vs Depression Scores')
st.pyplot(fig)

# 5. Correlation Heatmap
fig, ax = plt.subplots(figsize=(6,4))
sns.heatmap(data[['Noise_dB','Stress_Score','Depression_Score']].corr(), annot=True, cmap='coolwarm', ax=ax)
ax.set_title('Correlation Heatmap')
st.pyplot(fig)

# -------------------------------
# Group Analysis
# -------------------------------
mean_scores = data.groupby('Gender')[['Stress_Score','Depression_Score']].mean()
st.subheader("Average Stress & Depression Scores by Gender")
st.write(mean_scores)

fig, ax = plt.subplots(figsize=(7,5))
mean_scores.plot(kind='bar', ax=ax, color=['skyblue','lightgreen'])
if user_entry is not None:
    ax.scatter(
        [list(mean_scores.index).index(gender_input), list(mean_scores.index).index(gender_input)],
        [stress_input, depression_input],
        color='red', s=120, zorder=5, label='You'
    )
    ax.legend()
ax.set_title('Average Mental Health Scores by Gender (with Your Input)')
ax.set_ylabel('Score')
ax.set_xticklabels(mean_scores.index, rotation=0)
st.pyplot(fig)
