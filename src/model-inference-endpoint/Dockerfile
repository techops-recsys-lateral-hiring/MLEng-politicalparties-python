FROM python:3.9

WORKDIR /usr/src/app

COPY ./embedding-inference-endpoint/requirements.txt .

COPY Model .

RUN pip install --no-cache-dir -r requirements.txt

COPY ./embedding-inference-endpoint .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
