FROM python:3.12-slim

WORKDIR /workspace

COPY requirements.txt .
RUN pip install --no-cache-dir -q -r requirements.txt
RUN useradd -m sandboxuser
USER sandboxuser
CMD [ "python" ]