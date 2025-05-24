import streamlit as st
import os

st.set_page_config(page_title="Streamlit ADK Frontend", page_icon="💡")

st.title("Welcome to the Streamlit ADK Frontend!")

st.write(
    """
    This is a basic Streamlit application to serve as a frontend for your ADK backend.
    You can expand this application to interact with your ADK agent.
    """
)

adk_backend_url = os.environ.get("ADK_BACKEND_URL", "http://localhost:8001")
st.info(f"ADK Backend URL: {adk_backend_url}")

if st.button("Say Hello"):
    st.write("Hello from Streamlit!")
    # Example of how you might use the URL:
    # response = requests.get(f"{adk_backend_url}/some_adk_endpoint")
    # st.write(response.json())