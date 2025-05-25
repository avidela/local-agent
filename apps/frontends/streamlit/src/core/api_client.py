import requests
import streamlit as st
from src.core.config import ADK_BACKEND_URL

class APIClient:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.session = requests.Session()

    def _request(self, method: str, endpoint: str, **kwargs):
        url = f"{self.base_url}{endpoint}"
        try:
            response = self.session.request(method, url, **kwargs)
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
            return response
        except requests.exceptions.HTTPError as e:
            st.error(f"HTTP Error: {e.response.status_code} - {e.response.text}")
            raise
        except requests.exceptions.ConnectionError as e:
            st.error(f"Connection Error: Could not connect to {self.base_url}. Is the backend running?")
            raise
        except requests.exceptions.Timeout as e:
            st.error(f"Timeout Error: Request to {self.base_url} timed out.")
            raise
        except requests.exceptions.RequestException as e:
            st.error(f"An unexpected error occurred: {e}")
            raise

    def get(self, endpoint: str, **kwargs):
        return self._request("GET", endpoint, **kwargs)

    def post(self, endpoint: str, **kwargs):
        return self._request("POST", endpoint, **kwargs)

    def delete(self, endpoint: str, **kwargs):
        return self._request("DELETE", endpoint, **kwargs)

# Initialize a global API client instance
api_client = APIClient(ADK_BACKEND_URL)