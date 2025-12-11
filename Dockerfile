FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    wget git build-essential \
    && pip install --upgrade pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install poetry
ENV POETRY_VIRTUALENVS_CREATE=false

COPY pyproject.toml poetry.lock* /app/
RUN poetry install --no-root

COPY . .

CMD ["bash"]
