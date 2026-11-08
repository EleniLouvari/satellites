# syntax=docker/dockerfile:1
FROM python:3.12-slim-bookworm AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib \
    PATH="/opt/venv/bin:${PATH}"

# Geospatial wheels include GDAL/PROJ; only their runtime libraries are needed.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates libexpat1 libgomp1 libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

FROM base AS builder
WORKDIR /build
COPY pyproject.toml README.md ./
COPY src/ ./src/

# Keep build tools and the source checkout outside the final runtime image.
# Binary dependencies avoid compiling GDAL and the scientific stack in this image.
RUN python -m venv /opt/venv \
    && /opt/venv/bin/python -m pip install --no-cache-dir --only-binary=:all: ".[acquisition,database,credentials,interpolation]" \
    && /opt/venv/bin/python -m pip check

FROM base AS runtime
COPY --from=builder /opt/venv /opt/venv

RUN groupadd --gid 10001 satellites \
    && useradd --uid 10001 --gid 10001 --create-home satellites \
    && mkdir /data \
    && chown satellites:satellites /data

USER satellites:satellites
WORKDIR /data

# Import the real runner, including native dependencies, before publishing an image.
RUN python -c "from data_preparation.parcel_stats import prepare_run, run_parcel_stats; import psycopg2, pykrige; from cryptography.fernet import Fernet" \
    && satellites parcel-stats --help

# The CLI records interruption and preserves completed checkpoints on SIGINT.
STOPSIGNAL SIGINT
ENTRYPOINT ["satellites", "parcel-stats"]
CMD ["--help"]
