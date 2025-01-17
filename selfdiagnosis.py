import streamlit as st
import pandas as pd
import numpy as np
import shap
from joblib import load
from script import dropcols2
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

# Define the questions
questions = [
    "Do you constantly remember stressful events that have occurred in the past?",
    "Do you constantly feel nervous?",
    "Are you always tired/fatigued?",
    "Have you been having trouble working at your job?",
    "Do you experience nightmares frequently while sleeping?",
    "Do you consistently experience feelings of hopelessness?",
    "Would you say you have close friends?",
    "Do you feel yourself panicking frequently?",
    "Do you have trouble processing emotions relating to anger/frustration?",
    "Are you addicted to social media?"
]

# Initialize session state to store answers if not already initialized
if "answers" not in st.session_state:
    st.session_state.answers = [None] * len(questions)
st.image("Header.png", use_column_width= True)

st.subheader("Diagnose yourself!")

st.write("Please answer the following questions:")

# Display the questions with yes/no options
col1, col2 = st.columns(2)
for i, question in enumerate(questions):
    if i < len(questions) // 2:
        with col1:
            st.session_state.answers[i] = st.radio(
                question, ["Yes", "No"], index=0 if st.session_state.answers[i] == "Yes" else 1, key=f"q{i}"
            )
    else:
        with col2:
            st.session_state.answers[i] = st.radio(
                question, ["Yes", "No"], index=0 if st.session_state.answers[i] == "Yes" else 1, key=f"q{i}"
            )

# Save the answers to a Pandas DataFrame
if st.button("Submit"):
    fnames= ["POPPING_UP_STRESSFUL_MEMORY", "FEELING_NERVOUS", "FEELING_TIRED", "HAVING_TROUBLE_WITH_WORK", "HAVING_NIGHTMARES", 
             "HOPELESSNESS", "CLOSE_FRIEND", "PANIC", "ANGER", "SOCIAL_MEDIA_ADDICTION"]
    answers_df = pd.DataFrame([st.session_state.answers], columns=fnames)
    answers_df= answers_df.apply(lambda x: x.str.lower())
    file= "multiclassxgb.pkl"
    model= load(file)
    prediction= model.predict(answers_df)
    predprob= model.predict_proba(answers_df)
    st.write("### Your Diagnosis and Prediction Probabilities:")
    classnames= ["ANXIETY", "DEPRESSION", "LONELINESS", "NORMAL", "STRESS"]
    classdict= {0: "Anxiety", 1: "Depression", 2: "Loneliness", 3: "Normal", 4: "Stress"}
    probs= pd.DataFrame(predprob, columns= classnames, index= answers_df.index)
    classoutput = [classdict.get(value, "Unknown") for value in prediction]
    odf= answers_df.copy()
    st.write("You have been diagnosed with:", classoutput[0])
    clrlist= ["#30bdcc", "#e9d6df", "#2c7eb4", "#364059", "#f5cfb3"]
    row= probs.iloc[0]
    st.markdown("""
    <style>
    .big-font {
        font-size:20px !important;
    }
    </style>
    """, unsafe_allow_html=True)
    st.markdown('<p class="big-font">Here is a graph of the probabilities of each possible diagnosis:</p>', unsafe_allow_html=True)
    #fig= px.bar(row, x= row.index, y= row.values, labels= {"x": "Predicted Class", "y": "Probability"}, color= row.index, color_discrete_sequence= clrlist)
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=row.index,
        y=row.values,
        marker=dict(color= clrlist),
        text=[f"{p*100:.1f}%" for p in row.values],  # Add percentages above bars
        textposition="outside"  # Display text above the bars
    ))
    
    # Update layout
    fig.update_layout(
        {'plot_bgcolor': 'rgba(0, 0, 0, 0)', 'paper_bgcolor': 'rgba(0, 0, 0, 0)',},
        xaxis=dict(title="Disorder", title_font=dict(size=18)),
        yaxis=dict(title="Probability", title_font=dict(size=18)),
        yaxis_range=[0,1.1],
        template="plotly_white"
    )
    
    st.plotly_chart(fig)
    shapm= model.named_steps["classifier"]
    fpdf= model.named_steps["preprocessor"].transform(answers_df)
    exp=shap.TreeExplainer(shapm)
    shapvals= exp(fpdf)
    predshapval= shapvals.values[0, :, prediction[0]]
    predexpval= exp.expected_value[prediction[0]]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax= shap.plots._waterfall.waterfall_legacy(predexpval, predshapval, feature_names= fpdf.columns.tolist(), max_display = 5)
    st.markdown('<p class="big-font">Here were the factors that contributed most to your diagnosis:</p>', unsafe_allow_html=True)
    st.pyplot(fig)