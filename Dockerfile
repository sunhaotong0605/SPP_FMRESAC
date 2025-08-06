# 基于官方 Python 镜像
FROM python:3.8-slim

# 设置工作目录
WORKDIR /app

# 安装必要的系统依赖
RUN apt-get update && apt-get install -y git

# 复制项目文件到工作目录
COPY . /app

# 安装 Python 依赖
RUN pip install --no-cache-dir -r requirements.txt

# 下载模型权重（如果模型权重较大，建议使用多阶段构建或在运行时挂载）
RUN mkdir -p model_weights/NT_50M && \
    mkdir -p model_weights/EVO_7B && \
    git clone https://github.com/sunhaotong0605/SPP_FMRESAC.git /app && \
    # 下载 NT_50M 权重
    wget -P model_weights/NT_50M/ <NT_50M_WEIGHT_URL> && \
    # 下载 EVO_7B 权重
    wget -P model_weights/EVO_7B/ <EVO_7B_WEIGHT_URL>

# 设置环境变量
ENV PATH="/app:${PATH}"

# 暴露端口（如果需要）
# EXPOSE 8080

# 设置默认命令
CMD ["python", "main.py"]