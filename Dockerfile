FROM python:3.10

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY model/ model/
COPY utils/ utils/
COPY templates/ templates/
COPY static/ static/

EXPOSE 5000

CMD ["python", "app.py"]