FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY skincare_adk_agent /app/skincare_adk_agent

ENV PORT=8080
ENV PYTHONPATH=/app
ENV GOOGLE_CLOUD_LOCATION=global

EXPOSE 8080

CMD ["python3", "-m", "uvicorn", "skincare_adk_agent.a2a_server:app", "--host", "0.0.0.0", "--port", "8080"]
