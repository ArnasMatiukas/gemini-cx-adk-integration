FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py ./skincare_api/app.py
ENV PORT=8080 GOOGLE_CLOUD_LOCATION=global VISION_MODEL=gemini-3.8-flash
CMD ["python3", "-m", "uvicorn", "skincare_api.app:app", "--host", "0.0.0.0", "--port", "8080"]
