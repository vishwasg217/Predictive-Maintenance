FROM python:3.9-slim
WORKDIR /frontend
COPY requirements.txt /frontend/
RUN pip install --upgrade pip
RUN apt update && apt install gcc python3-dev git -y --no-install-recommends
RUN pip install --no-cache-dir -r requirements.txt 
COPY . /frontend

EXPOSE 8501

CMD ["streamlit", "run", "main.py"]