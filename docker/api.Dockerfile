FROM python:3.12-slim
WORKDIR /app
COPY . /app
RUN pip install -U pip && pip install -r requirements.txt
CMD ["uvicorn", "app.main:app", "--app-dir", "apps/api", "--host", "0.0.0.0", "--port", "8000"]
