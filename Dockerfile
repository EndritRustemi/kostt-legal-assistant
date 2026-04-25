FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p data/laws/kosove data/laws/zrre data/laws/entso-e data/laws/eu \
    data/laws/strategjike data/laws/vendime data/laws/te-tjera

EXPOSE 7860

CMD ["streamlit", "run", "app.py", "--server.port", "7860", "--server.headless", "true", "--browser.gatherUsageStats", "false"]
