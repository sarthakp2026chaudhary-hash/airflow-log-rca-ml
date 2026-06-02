# Minimal container for running Phase 0 (synthetic log generation).
# Phases 1-4 + Airflow will get their own compose service in later commits.

FROM python:3.12-slim

WORKDIR /app

# Copy package metadata first so dependency installation can be cached.
COPY pyproject.toml ./
COPY src ./src

RUN pip install --no-cache-dir -e ".[datagen]"

# Default command: generate logs into the mounted bucket.
# Run with:
#   docker run --rm -v "$(pwd)/../fake_gcs_bucket:/data" log-rca \
#     log-rca-gen --bucket-root /data
CMD ["log-rca-gen"]
