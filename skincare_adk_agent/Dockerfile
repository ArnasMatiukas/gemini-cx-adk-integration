FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY agent.py server.py ./skincare_adk_agent/
ENV PORT=8080 PYTHONPATH=/app GOOGLE_CLOUD_LOCATION=global
CMD ["python3", "-m", "uvicorn", "skincare_adk_agent.server:app", "--host", "0.0.0.0", "--port", "8080"]
