FROM python:3.8-slim AS builder
WORKDIR /app

RUN apt-get update

COPY . /app
# RUN pip install packaging==24.0 && pip install ninja -f https://pypi.tuna.tsinghua.edu.cn/simple

# RUN pip install torch==2.0.0 -i https://download.pytorch.org/whl/cu118
# RUN pip install triton==2.1.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install torch==2.0.1 -f https://mirrors.aliyun.com/pytorch-wheels/cu118 --timeout 999999999
# RUN pip install numpy==1.24.4 -i https://pypi.tuna.tsinghua.edu.cn/simple

RUN pip install flash_attn-2.5.6+cu118torch2.0cxx11abiFALSE-cp38-cp38-linux_x86_64.whl --timeout 999999999
RUN pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple --timeout 999999999

# # 下载模型权重（替换为实际的模型权重下载链接）
# RUN wget -P model_weights/NT_50M/ <NT_50M_WEIGHT_URL> && \
#     wget -P model_weights/EVO_7B/ <EVO_7B_WEIGHT_URL>



CMD ["python", "main.py", "model_name=xxx", "input_path=xxx", "output_path=xxx"]