import streamlit as st
import pandas as pd
import numpy as np
import shap
import streamlit.components.v1 as components
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
                classdict2= {0: 'Anxiety', 1: 'Depression', 2: 'Loneliness', 3: 'Normal', 4: 'Stress'}
                classoutput = [classdict.get(value, "Unknown") for value in predictions]
                df["PREDICTIONS"] = classoutput
                probs= pd.DataFrame(predprobs, columns= classnames, index= df.index)
                maxprobs= probs.max(axis=1)
                df["PROBABILITY"]= maxprobs
                st.write("Predictions:")
                st.dataframe(df)
                custom_palette = sns.color_palette(["#30bdcc", "#f5cfb3", "#2c7eb4", "#a9545e", "#364059"])
                st.write("Patient Diagnosis Chart:")
                fig,ax= plt.subplots()
                ax= sns.countplot(data=df, x= "PREDICTIONS", palette= custom_palette)
                for p in ax.patches:
                       ax.annotate('{:.0f}'.format(p.get_height()), (p.get_x()+0.25, p.get_height()+.3))
                st.pyplot(fig)
            except Exception as e:
                st.error(f"An error occurred: {e}")
            # Select a row (e.g., patient)
            df2= df.copy()
            df2= df2.drop(["PREDICTIONS"], axis=1)
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
                    updatedpreds = model.predict(df)
                    updatedpredprobs= model.predict_proba(df)
                    updatedclassoutput = [classdict.get(value, "Unknown") for value in updatedpreds]
                    df2["NEW PREDICTIONS"] = updatedclassoutput
                    updatedprobs= pd.DataFrame(updatedpredprobs, columns= classnames, index= df.index)
                    updatedprobsmax= updatedprobs.max(axis=1)
                    df2["PROBABILITY"]= updatedprobsmax
                    st.write("Updated Predictions:")
                    st.dataframe(df2)
                    st.write("Updated Patient Diagnosis Chart:")
                    fig2,ax2= plt.subplots()
                    ax2= sns.countplot(data=df2, x= "NEW PREDICTIONS", palette= custom_palette)
                    for p in ax2.patches:
                       ax2.annotate('{:.0f}'.format(p.get_height()), (p.get_x()+0.25, p.get_height()+.01))
                    st.pyplot(fig2)
                except Exception as e:
                    st.error(f"Prediction error: {e}")
    elif choice == "Self Diagnosis":
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
            fnames= ["POPPING_UP_STRESSFUL_MEMORY", "FEELING_NERVOUS", "FEELING_TIRED", "HAVING_TROUBLE_WITH_WORK", 
                     "HAVING_NIGHTMARES", "HOPELESSNESS", "CLOSE_FRIEND", "PANIC", "ANGER", "SOCIAL_MEDIA_ADDICTION"]
            answers_df = pd.DataFrame([st.session_state.answers], columns=fnames)
            answers_df= answers_df.apply(lambda x: x.str.lower())
            file= "multiclassxgb.pkl"
            model= load(file)
            prediction= model.predict(answers_df)
            predprob= model.predict_proba(answers_df)
            st.write("### Your Diagnosis and Prediction Probabilities:")
            classnames= ["ANXIETY", "DEPRESSION", "LONELINESS", "NORMAL", "STRESS"]
            classdict2= {0: 'Anxiety', 1: 'Depression', 2: 'Loneliness', 3: 'Normal', 4: 'Stress'}
            probs= pd.DataFrame(predprob, columns= classnames, index= answers_df.index)
            maxprobs= probs.max(axis=1)
            classoutput = [classdict2.get(value, "Unknown") for value in prediction]
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

if __name__ == "__main__":
    main()
