# Use a lightweight Python base image
FROM python:3.13-slim-buster

# Set the working directory in the container
WORKDIR /app

# Copy the Streamlit application
COPY app.py .

# Install Streamlit
RUN pip install streamlit

# Expose the port Streamlit runs on
EXPOSE 8501

# Command to run the Streamlit application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]