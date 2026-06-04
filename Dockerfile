FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY web_app.py .
COPY static ./static
COPY start.sh .
RUN chmod +x start.sh

ENV WHISPER_CACHE_DIR=/data/whisper-models
ENV OMP_NUM_THREADS=2

EXPOSE 7860

CMD ["./start.sh"]
