FROM python:3.8-slim AS builder
WORKDIR /app

RUN apt-get update

COPY . /app

RUN pip install packaging==24.0 && pip install ninja==1.13.0
RUN pip install torch==2.0.1 -i https://download.pytorch.org/whl/cu118
RUN pip install flash_attn==2.5.6
RUN pip install -r requirements.txt

CMD ["python", "main.py", "model_name", "input_path", "output_path", "pretrained_model_path", "label_csv_path"]