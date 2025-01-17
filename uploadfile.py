import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
from script import dropcols2
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, FunctionTransformer
import plotly.express as px
st.image("Header.png", use_column_width= True)
st.subheader("Upload Your File and Obtain Patient Predictions")
uploaded_file = st.file_uploader("Choose a csv file", type=["csv"])

if uploaded_file is not None:
    st.success("File uploaded successfully!")

    # Display content for the CSV file
    try:
        df = pd.read_csv(uploaded_file)
        df.columns= [col.upper().replace(".","_").strip() for col in df.columns]
        file= "multiclassxgb.pkl"
        model= load(file)
        predictions = model.predict(df)  # Update with appropriate features
        predprobs= model.predict_proba(df)
        classnames= ["ANXIETY", "DEPRESSION", "LONELINESS", "NORMAL", "STRESS"]
        classdict= {0: "Anxiety", 1: "Depression", 2: "Loneliness", 3: "Normal", 4: "Stress"}
        classoutput = [classdict.get(value, "Unknown") for value in predictions]
        df["PREDICTIONS"] = classoutput
        probs= pd.DataFrame(predprobs, columns= classnames, index= df.index)
        maxprobs= probs.max(axis=1)
        df["PROBABILITY"]= maxprobs
        st.write("Predictions:")
        st.dataframe(df)
        clrlist= ["#30bdcc", "#e9d6df", "#2c7eb4", "#f5cfb3", "#364059"]
        clrlist2= ["#a9545e", "#364059"]
        counts= df["PREDICTIONS"].value_counts()
        fig= px.bar(counts, x= counts.index, y= counts.values, labels= {"x": "Predicted Class", "y": "Count"}, color= counts.index, color_discrete_sequence= clrlist)
        st.subheader("Batch Prediction Distribution")
        fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)', 'paper_bgcolor': 'rgba(0, 0, 0, 0)',})
        st.plotly_chart(fig)
        df["binary"]= df["PREDICTIONS"].apply(lambda x: "Normal" if (x == "Normal") else "Mentally Ill")
        counts= df["binary"].value_counts()
        fig2= px.pie(names= counts.index, values= counts, hole= 0.3, color= counts.index, color_discrete_sequence= clrlist2)
        fig2.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)', 'paper_bgcolor': 'rgba(0, 0, 0, 0)',})
        st.subheader("Diagnosis Percentage")
        st.plotly_chart(fig2)
    except Exception as e:
        st.error(f"An error occurred: {e}")
    # Select a row (e.g., patient)
    df2= df.copy()
    df2= df2.drop(["PREDICTIONS", "binary"], axis=1)
    row_index = st.selectbox("Select a row to edit:", df2.index)
    selected_row = df2.loc[row_index]
    st.write("Selected Row:", selected_row)

    # Edit specific fields
    new_values = {}
    editable_columns = [col for col in df2.columns if col not in ["PREDICTIONS", "BREATHING_RAPIDLY", "AVOIDS_PEOPLE_OR_ACTIVITIES", "WEIGHT_GAIN", "OVER_REACT", "FEELING_NEGATIVE","HAVING_TROUBLE_IN_SLEEPING", "SWEATING", "TROUBLE_IN_CONCENTRATION", "BLAMMING_YOURSELF", "CHANGE_IN_EATING", "TROUBLE_CONCENTRATING","MATERIAL_POSSESSIONS", "INTROVERT", "SUICIDAL_THOUGHT","ID", "PROBABILITY"]]
    cols= st.columns(3)
    for i, col in enumerate(editable_columns):
        with cols[i % 3]:
            new_value = st.text_input(f"Edit {col}", str(selected_row[col]))
            new_values[col] = new_value

    # Generate predictions based on the edited row before updating the DataFrame
    if st.button("Preview Prediction with Edited Row"):
        temp_row = selected_row.copy()
        for col, new_value in new_values.items():
            temp_row[col] = new_value.lower()
        try:
            temp_prediction = model.predict(pd.DataFrame([temp_row]))
            class_prediction= [classdict.get(value, "Unknown") for value in temp_prediction]
            st.write(f"Prediction for edited row: {class_prediction[0]}")
        except Exception as e:
            st.error(f"Error generating prediction: {e}")

    # Update the DataFrame with the edited row and regenerate predictions
    if st.button("Update and Predict"):
        for col, new_value in new_values.items():
            df2.at[row_index, col] = new_value

        st.success("Row updated successfully!")

        # Regenerate predictions for the entire dataset
        try:
            updatedpreds = model.predict(df2)
            updatedpredprobs= model.predict_proba(df2)
            updatedclassoutput = [classdict.get(value, "Unknown") for value in updatedpreds]
            df2["NEW PREDICTIONS"] = updatedclassoutput
            updatedprobs= pd.DataFrame(updatedpredprobs, columns= classnames, index= df2.index)
            updatedprobsmax= updatedprobs.max(axis=1)
            df2["PROBABILITY"]= updatedprobsmax
            st.write("Updated Predictions:")
            st.dataframe(df2)
            counts= df2["NEW PREDICTIONS"].value_counts()
            fig3= px.bar(counts, x= counts.index, y= counts.values, labels= {"x": "Predicted Class", "y": "Count"}, color= counts.index, color_discrete_sequence= clrlist)
            fig3.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)', 'paper_bgcolor': 'rgba(0, 0, 0, 0)',})
            st.subheader("Updated Batch Prediction Distribution")
            st.plotly_chart(fig3)
            df2["binary"]= df2["NEW PREDICTIONS"].apply(lambda x: "Normal" if (x == "Normal") else "Mentally Ill")
            counts= df2["binary"].value_counts()
            fig4= px.pie(names= counts.index, values= counts, hole= 0.3, color= counts.index, color_discrete_sequence= clrlist2)
            fig4.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)', 'paper_bgcolor': 'rgba(0, 0, 0, 0)',})
            st.subheader("Updated Diagnosis Percentage")
            st.plotly_chart(fig4)
        except Exception as e:
            st.error(f"Prediction error: {e}")