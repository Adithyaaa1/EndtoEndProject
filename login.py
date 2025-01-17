import streamlit as st

st.set_page_config(initial_sidebar_state="collapsed", page_title= "Mental Health AI App")

custom_css = """
<style>
[data-testid="stAppViewContainer"] {
background-image: url("https://media0.giphy.com/media/v1.Y2lkPTc5MGI3NjExdWI5cWl4YnZodDB2NG1hYjQ1bDh2amc4cmV5ZDd6NjZqNmZ1bW13MyZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/e963xncvsxxfhtcGW3/giphy.gif");
background-size: cover;
}
[data-testid="stHeader"] {
background-color: rgba(0, 0, 0, 0);
}
</style>
"""

# Inject the CSS into the Streamlit app
st.markdown(custom_css, unsafe_allow_html=True)
usernames= ["Admin"]
passwords= ["Admin2025*"]
if "validated" not in st.session_state:
    st.session_state.validated = False
if not st.session_state.validated:
    st.title("Welcome to the Mental Health AI Application.")
    st.subheader("Log In")
    string1 = st.text_input("Username:")
    string2 = st.text_input("Password:", type= 'password')
    if st.button("Log In"):
        if string1 in usernames and string2 in passwords:
            st.success("Log in successful.")
            st.session_state.validated = True
            st.rerun()
        else:
            st.error("Username and/or Password are incorrect and case sensitive. Please try again.")
else:
    exec(open("main.py").read())
    