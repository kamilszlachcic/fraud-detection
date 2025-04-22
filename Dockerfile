FROM python:3.11-slim

# Katalog roboczy
WORKDIR /app

# Instalacja minimalnych zależności systemowych (do budowania np. numpy, xgboost)
# Instalacja poetry + build tools + czyszczenie w jednej warstwie
RUN apt-get update && apt-get install -y --no-install-recommends curl build-essential && \
    curl -sSL https://install.python-poetry.org | python3 - && \
    apt-get purge -y build-essential && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /root/.cache /root/.cache/pypoetry


# Ustawienie PATH, żeby poetry było widoczne
ENV PATH="/root/.local/bin:$PATH"

# Skopiuj tylko pliki zależności (by cachować warstwę)
COPY pyproject.toml poetry.lock* /app/

# Wyłącz tworzenie virtualenv w poetry i zainstaluj zależności
RUN poetry config virtualenvs.create false \
 && poetry install --without dev --no-interaction --no-ansi

# Skopiuj resztę kodu źródłowego
COPY . .

EXPOSE 8000

#wyczyść build tools jeśli niepotrzebne w czasie działania
RUN apt-get purge -y build-essential && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /root/.cache /var/lib/apt/lists/*

# Punkt wejścia
CMD ["poetry", "run", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
