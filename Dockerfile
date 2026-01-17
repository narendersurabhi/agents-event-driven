FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY pyproject.toml README.md ./

ENV PYTHONPATH=/app/src

CMD ["python", "-m", "govguard.demo.service"]
