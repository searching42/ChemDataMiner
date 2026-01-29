# 使用官方 Python Slim 镜像，体积小
FROM python:3.10-slim

WORKDIR /app

# 安装 RDKit 需要的系统库
RUN apt-get update && apt-get install -y \
    libxrender1 \
    libxext6 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# 复制核心依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 复制源代码
COPY src/ ./src/
COPY main.py .

# 设定入口
ENTRYPOINT ["python", "main.py"]