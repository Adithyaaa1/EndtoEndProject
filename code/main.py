import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
from script import dropcols2
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, FunctionTransformer

def main():

    st.title("Mental Health AI App")
    
    # Create a sidebar menu
    menu = ["Mental Health Crisis Overview", "Upload File", "Self Diagnosis"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Mental Health Crisis Overview":
        st.image("Header.png", use_column_width= True)
        st.subheader("Mental Health Crisis Overview")
        col1, col2= st.columns(2)
        with col1: 
            st.image("stateperc.png", use_column_width= True)
            st.write("As of 2024, at least 16% of every state's population is suffering from a mental illness, up to a maximum of almost 27%. The states marked in red (Washington, Ohio, Oregon, West Virginia) suffer the most from these ailments.")
            st.image("usperc.png", use_column_width= True)
            st.write("As of 2024, the most common mental illnesses were anxiety disorders and major depression which accounted for 27.4% of people suffering from a mental health issue.")
        with col2:
            st.image("agperc.png", use_column_width= True)
            st.write("The youth of the United States suffers disproportionately from mental illnesses in comparison to the other age groups, with the percentage almost doubling from 2008-2021. It can be inferred that this percentage has only gone up since then, with around 2 in 5 adolescents/young adults having mental health issues.")
            st.image("mhspend.png", use_column_width= True)
            st.write("The annual mental health spending by the National Institue of Health has more than doubled since 2013, with 2025's projected spendings expected to be around almost $4.5 billion USD.")

    elif choice == "Upload File":
        st.image("Header.png", use_column_width= True)
        st.subheader("Upload Your File and Obtain Patient Predictions")
        uploaded_file = st.file_uploader("Choose a csv file", type=["csv"])

        if uploaded_file is not None:
            st.success("File uploaded successfully!")

            # Display content for the CSV file
            try:
                df = pd.read_csv(uploaded_file)
                st.dataframe(df)
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
            except Exception as e:
                st.error(f"An error occurred: {e}")
            # Select a row (e.g., patient)
            row_index = st.selectbox("Select a row to edit:", df.index)
            selected_row = df.loc[row_index]
            st.write("Selected Row:", selected_row)

            # Edit specific fields
            new_values = {}
            editable_columns = [col for col in df.columns if col not in ["PREDICTIONS", "BREATHING_RAPIDLY", "AVOIDS_PEOPLE_OR_ACTIVITIES", "WEIGHT_GAIN", "OVER_REACT", "FEELING_NEGATIVE","HAVING_TROUBLE_IN_SLEEPING", "SWEATING", "TROUBLE_IN_CONCENTRATION", "BLAMMING_YOURSELF", "CHANGE_IN_EATING", "TROUBLE_CONCENTRATING","MATERIAL_POSSESSIONS", "INTROVERT", "SUICIDAL_THOUGHT","ID", "PROBABILITY"]]
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
                    df.at[row_index, col] = new_value

                st.success("Row updated successfully!")

                # Regenerate predictions for the entire dataset
                try:
                    updatedpreds = model.predict(df)
                    updatedpredprobs= model.predict_proba(df)
                    updatedclassoutput = [classdict.get(value, "Unknown") for value in updatedpreds]
                    df["PREDICTIONS"] = updatedclassoutput
                    updatedprobs= pd.DataFrame(updatedpredprobs, columns= classnames, index= df.index)
                    updatedprobsmax= updatedprobs.max(axis=1)
                    df["PROBABILITY"]= updatedprobsmax
                    st.write("Updated Predictions:")
                    st.dataframe(df)
                except Exception as e:
                    st.error(f"Prediction error: {e}")
    elif choice == "Self Diagnosis":
        exec(open("SelfDiagnosis.py").read())

if __name__ == "__main__":
    main()
