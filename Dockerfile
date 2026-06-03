FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY streamlit_app.py .
COPY transcribe_audio.py .
COPY start.sh .
COPY start.sh .
RUN chmod +x start.sh

ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
ENV WHISPER_CACHE_DIR=/app/.whisper-cache

EXPOSE 7860

CMD ["./start.sh"]
