import streamlit as st
from streamlit_navigation_bar import st_navbar



pages = ["Home", "Upload File", "Self Diagnosis", "Chat Bot", "Facial Recognition"]
styles = {
    "nav": {
        "background-color": "rgba(0, 0, 255, 2)",
    },
    "div": {
        "max-width": "40rem",
    },
    "span": {
        "border-radius": "0.5rem",
        "color": "rgb(255, 255, 255)",
        "margin": "0 0.125rem",
        "padding": "0.4375rem 0.625rem",
    },
    "active": {
        "background-color": "rgba(255, 255, 255, 0.25)",
    },
    "hover": {
        "background-color": "rgba(255, 255, 255, 0.35)",
    },
}


page = st_navbar(pages, styles=styles)

st.header('Mental Health AI App')

if page == "Home":
    exec(open("overview.py").read())
elif page == "Upload File":
    exec(open("uploadfile.py").read())
elif page == "Self Diagnosis":
     exec(open("selfdiagnosis.py").read())
elif page == "Chat Bot":
    st.image("Header.png", use_column_width= True)
    st.write(" Coming Soon! Under construction.")
    #exec(open("/Users/adithya/Downloads/app/chatbot.py").read())
elif page == "Facial Recognition":
    st.image("Header.png", use_column_width= True)
    st.write(" Coming Soon! Under construction.")
with st.sidebar:
    st.header("Sidebar")