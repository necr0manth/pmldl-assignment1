FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 HOME=/tmp \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false STREAMLIT_SERVER_HEADLESS=true
WORKDIR /app
COPY requirements/app.txt /app/requirements/app.txt
RUN pip install --no-cache-dir -r requirements/app.txt \
    && useradd --create-home --uid 10001 appuser
COPY --chown=appuser:appuser code/pmldl/webapp.py /app/webapp.py
USER appuser
EXPOSE 8501
CMD ["python", "-m", "streamlit", "run", "/app/webapp.py", "--server.address=0.0.0.0", "--server.port=8501"]
