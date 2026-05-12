FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p data/laws/kosove data/laws/zrre data/laws/entso-e data/laws/eu \
    data/laws/strategjike data/laws/vendime data/laws/te-tjera

ENV STREAMLIT_SERVER_MAX_UPLOAD_SIZE=200
ENV STREAMLIT_SERVER_MAX_MESSAGE_SIZE=200
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Limit glibc malloc arenas so freed memory is returned to the OS quickly.
# Without this, Python + pyarrow can hold gigabytes of virtual address space
# that cgroup memory accounting counts against the 16 Gi limit.
ENV MALLOC_ARENA_MAX=2
ENV MALLOC_TRIM_THRESHOLD_=131072

EXPOSE 7860

CMD ["streamlit", "run", "app.py", "--server.port", "7860", "--server.address", "0.0.0.0", "--server.fileWatcherType", "none"]
