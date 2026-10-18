FROM python:3.12-slim
RUN pip install --no-cache-dir "pytest>=8,<9" \
    && useradd --uid 10001 --no-create-home --shell /usr/sbin/nologin sandbox
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
USER 10001
WORKDIR /workspace
ENTRYPOINT ["python", "-m", "pytest", "-p", "no:cacheprovider"]