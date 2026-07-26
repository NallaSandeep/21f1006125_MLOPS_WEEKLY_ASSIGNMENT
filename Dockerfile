FROM python:3.10-slim

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

# Download the model from MLflow
RUN python download_best_model.py

EXPOSE 8200

CMD ["uvicorn", "iris_fastapi:app", "--host", "0.0.0.0", "--port", "8200"]