# Reproduces the *analysis*, not the measurement.
#
# Every number in the report is derived from the response matrix in
# data/measurements.json.gz, and that derivation is pure arithmetic: it runs
# here identically to anywhere else. The measurement itself deliberately stays
# outside: `profile` and `bench` time model loads and read the OS page cache,
# and on macOS a container is a Linux VM, so timing inside one would distort
# the exact quantities the experiment is about. Reproducing those means
# running `evoroute profile --purge` and `evoroute bench` on the host, against
# a local Ollama.
FROM python:3.11-slim

RUN pip install --no-cache-dir uv

WORKDIR /app
ENV UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1

# Dependencies resolve from the lock file alone, so editing sources does not
# invalidate this layer.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --extra dev

COPY src/ src/
COPY tests/ tests/
COPY data/measurements.json.gz data/
RUN uv sync --frozen --extra dev

ENV PATH="/app/.venv/bin:$PATH" EVOROUTE_DB=/app/results/evoroute.db

# Rebuild the database from the snapshot, then re-derive every result.
CMD ["sh", "-c", "pytest -q && evoroute import-snapshot && evoroute baseline && evoroute stats && evoroute cascade && evoroute figures"]
