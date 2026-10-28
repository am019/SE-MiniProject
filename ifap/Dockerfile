# One Dockerfile, one image per service family (NFR-07). Python 3.12, pinned dependencies.
FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PYTHONPATH=/app:/app/common
WORKDIR /app
COPY requirements/base.txt requirements/base.txt
RUN pip install -r requirements/base.txt
COPY common common
COPY services services
COPY config config
COPY web web
COPY data/sample data/sample
RUN useradd --system --uid 10001 ifap && mkdir -p /data && chown ifap /data
USER ifap
ENV IFAP_DATA_DIR=/data
EXPOSE 8000
ARG SERVICE=gateway
ENV SERVICE=${SERVICE}
CMD ["sh", "-c", "exec uvicorn services.${SERVICE}.main:app --host 0.0.0.0 --port 8000 --log-level warning"]

FROM base AS classical
USER root
COPY requirements/classical.txt requirements/classical.txt
RUN pip install -r requirements/classical.txt
USER ifap

FROM base AS dl
USER root
COPY requirements/dl.txt requirements/dl.txt
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements/dl.txt
USER ifap

FROM base AS qml
USER root
COPY requirements/qml.txt requirements/qml.txt
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements/qml.txt
USER ifap
