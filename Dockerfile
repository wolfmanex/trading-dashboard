# Container for Hugging Face Spaces (Docker SDK) or any Docker host. Spaces expect the app on port 7860
# and run the container as user 1000, so dependencies and the NLTK lexicon are set up for that user.
FROM python:3.12-slim

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH PYTHONUNBUFFERED=1
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt \
    && python -c "import nltk; nltk.download('vader_lexicon', quiet=True)"

COPY --chown=user . .

EXPOSE 7860
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.headless=true", \
     "--server.enableXsrfProtection=false", "--browser.gatherUsageStats=false"]
