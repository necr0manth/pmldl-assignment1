FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/app/code
WORKDIR /app
COPY requirements/model.txt requirements/api.txt /app/requirements/
RUN pip install --no-cache-dir -r requirements/api.txt \
    && useradd --create-home --uid 10001 appuser
COPY --chown=appuser:appuser code/pmldl /app/code/pmldl
COPY --chown=appuser:appuser models/model.joblib models/metadata.json /app/models/
USER appuser
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "pmldl.api:app", "--host", "0.0.0.0", "--port", "8000"]
