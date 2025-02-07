import streamlit as st
from streamlit.logger import get_logger

st.set_page_config(
    page_title="Welcome Screen",
    page_icon="👋",
)


st.markdown("<h1 style='text-align: center; color: white;'>Liver Disease Prediction</h1>", unsafe_allow_html=True)
st.image("img/healthy-liver-happy-you.webp", caption="Healthy Liver Happy You")
st.write("Liver disease includes a range of health conditions resulting from various factors that impact the normal functioning of the liver over an extended period. Effective treatment of Hepatitis C, a type of liver disease, can be significantly enhanced by accurately and timely predicting risk factors and severity, covering different stages like fibrosis and cirrhosis. This app does exactly that.")


