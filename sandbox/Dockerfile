FROM python:3.12-slim
RUN apt-get update \
    && apt-get install -y --no-install-recommends nodejs npm \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir "pytest>=8,<9" \
    && useradd --uid 10001 --no-create-home --shell /usr/sbin/nologin sandbox
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTEST_ADDOPTS="-p no:cacheprovider" HOME=/tmp
USER 10001
WORKDIR /workspace
