FROM python:3.9-slim


COPY /backend/requirements.txt /backend/
RUN pip install --upgrade pip
RUN apt update && apt install gcc python3-dev git -y --no-install-recommends
RUN pip install --no-cache-dir -r /backend/requirements.txt 

COPY ./artifacts/ /artifacts/
COPY ./src/predict.py /src/
COPY ./src/eda.py /src/
COPY ./config/ /config/
COPY ./ /backend/

WORKDIR /backend

EXPOSE 8000

CMD ["uvicorn", "backend.main:appl","--reload","--host", "0.0.0.0", "--port", "8000"]