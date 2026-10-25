# The optimizer web service (optimizer/web.py): MCP over HTTP at /mcp, JSON at /solve and /formats.
# OR-Tools publishes wheels for this Python, so nothing is compiled.
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt
COPY optimizer ./optimizer

# Hugging Face Spaces runs containers as user 1000 and routes traffic to port 7860; other hosts
# (Render, Cloud Run...) set PORT themselves, which the service reads.
RUN useradd --create-home --uid 1000 app
USER app
ENV HOST=0.0.0.0 PORT=7860 PYTHONUNBUFFERED=1
EXPOSE 7860
CMD ["python", "-m", "optimizer.web"]
