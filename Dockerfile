FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY enclim/ ./enclim/
COPY config/ ./config/

ENTRYPOINT ["python", "-m", "enclim.run_ensemble"]
CMD ["--config", "config/ensemble_config.yaml"]
