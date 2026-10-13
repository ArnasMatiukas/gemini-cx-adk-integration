FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app/skincare_api

ENV PORT=8080
ENV GOOGLE_CLOUD_LOCATION=global
ENV NANO_BANANA_MODEL=gemini-3.1-flash-image

EXPOSE 8080

CMD ["python3", "-m", "uvicorn", "skincare_api.app:app", "--host", "0.0.0.0", "--port", "8080"]
