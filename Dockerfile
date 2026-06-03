FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY streamlit_app.py .
COPY transcribe_audio.py .

ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
ENV WHISPER_CACHE_DIR=/app/.whisper-cache

EXPOSE 7860

CMD python -m streamlit run streamlit_app.py --server.address 0.0.0.0 --server.port ${PORT:-7860} --server.enableCORS false --server.enableXsrfProtection false
