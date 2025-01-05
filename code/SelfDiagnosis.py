import streamlit as st
import pandas as pd
import numpy as np
import shap
import streamlit.components.v1 as components

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
    maxprobs= probs.max(axis=1)
    classoutput = [classdict.get(value, "Unknown") for value in prediction]
    odf= answers_df.copy()
    odf["PREDICTION"]= classoutput[0]
    odf["PROBABILITY"]= maxprobs
    dispcols= ["PREDICTION", "PROBABILITY"]
    st.dataframe(odf[dispcols])
    st.write("You have been diagnosed with:", classoutput[0])
    shapm= model.named_steps["classifier"]
    fpdf= model.named_steps["preprocessor"].transform(answers_df)
    exp=shap.TreeExplainer(shapm)
    shapvals= exp(fpdf)
    predshapval= shapvals.values[0, :, prediction[0]]
    predexpval= exp.expected_value[prediction[0]]
    fp= shap.plots.force(predexpval, predshapval, fpdf.iloc[0], link= "logit", show= False)
    shap.save_html("force_plot.html", fp)
    with open("force_plot.html", "r") as f:
        html_content = f.read()
    html_with_scroll = f"""
    <div style="width: 100%; overflow-x: auto;">
        {html_content}
    </div>
    """
    components.html(html_with_scroll, height=400)