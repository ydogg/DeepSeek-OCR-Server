# DeepSeek OCR 服务器

这是一个为 DeepSeek OCR 模型提供 OpenAI 兼容 API 的 FastAPI 服务器，支持本地推理和远程API两种模式。

## Docker 镜像

项目提供了两种 Docker 镜像以满足不同部署需求：

### 在线模式镜像
- 仅包含轻量级依赖
- 适用于连接远程 OCR 服务的场景
- 镜像体积小，启动速度快

### 离线模式镜像
- 包含所有必要的 heavy 依赖（torch, transformers, vllm 等）
- 适用于本地 OCR 推理的场景
- 镜像体积大，但可独立运行

## 功能特性

- OpenAI 兼容的 `/v1/chat/completions` 接口
- 简单的 OCR 接口 `/v1/ocr`
- 支持两种运行模式：
  - 本地模式：使用本地 vLLM 运行 DeepSeek OCR 模型
  - 远程模式：通过 OpenAI 兼容 API 调用远程 OCR 服务
- 工作线程池（根据 GPU 内存情况使用 1-2 个工作线程）
- 基于队列的请求处理，实现高效的资源利用
- 每个工作线程在整个服务器生命周期内持有一个模型实例
- 模块化代码组织，便于维护

## 安装

### 方法1：直接安装依赖
1. 安装所需依赖：
```bash
pip install -r server/requirements.txt
```

2. 确保您已准备好 DeepSeek OCR 模型，并在 `config.py` 中进行了配置

### 方法2：使用 Docker 镜像
项目提供了两种 Docker 镜像构建方式：

1. 构建在线模式镜像（轻量级）：
```bash
# 在项目根目录下执行
sudo docker build -f docker/Dockerfile.online -t deepseek-ocr-online .
```

2. 构建离线模式镜像（完整依赖）：
```bash
# 在项目根目录下执行
sudo docker build -f docker/Dockerfile.offline -t deepseek-ocr-offline .
```

## 使用方法

### 启动服务器

#### 方法1：直接运行
```bash
cd server
./start_server.sh
```

或者直接运行：
```bash
python -m server.main
```

服务器将在 `http://0.0.0.0:8001` 启动（地址和端口可在 `server_config.py` 中配置）

#### 方法2：使用 Docker 容器
1. 运行在线模式容器：
```bash
sudo docker run -p 8001:8001 -p 7861:7861 deepseek-ocr-online
```

2. 运行离线模式容器：
```bash
sudo docker run -p 8001:8001 -p 7861:7861 deepseek-ocr-offline
```

容器将启动服务器（端口8001）和Gradio界面（端口7861）

### 运行模式配置

服务器支持两种运行模式，通过 `server_config.py` 中的 `ONLINE_OCR_MODE` 配置项进行切换：

- `ONLINE_OCR_MODE = False`：本地模式，使用本地 vLLM 运行 DeepSeek OCR 模型
- `ONLINE_OCR_MODE = True`：远程模式，通过 OpenAI 兼容 API 调用远程 OCR 服务

在远程模式下，还需要配置：
- `ONLINE_OCR_BASE_URL`：远程 OCR API 的基础 URL
- `ONLINE_OCR_MODEL_NAME`：远程 OCR 模型名称
- `ONLINE_OCR_API_KEY`：远程 OCR API 的访问密钥（如果需要）

### API 接口

#### 1. OpenAI 兼容的对话补全接口

接口地址：`POST /v1/chat/completions`

请求示例：
```json
{
  "model": "deepseek-ocr",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "image_url",
          "image_url": {
            "url": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEASABIAAD/..."
          }
        },
        {
          "type": "text",
          "text": "<|grounding|>Convert the document to markdown."
        }
      ]
    }
  ]
}
```

#### 2. 简单 OCR 接口

接口地址：`POST /v1/ocr`

请求示例（JSON数据）：
```json
{
  "image": "base64_encoded_image_data",
  "prompt": "<image>\n<|grounding|>Convert the document to markdown.",
  "level": "clean"
}
```

#### 3. 健康检查

接口地址：`GET /health`

## 配置

服务器使用与主 DeepSeek OCR 应用相同的配置：
- 模型路径来自 `config.py`
- 提示词来自 `config.py`
- 裁剪模式来自 `config.py`
- 最大并发数来自 `config.py`

服务器还有独立的配置选项，位于 `server_config.py`：
- `MAX_WORKER_THREADS`：工作线程数量（默认值：1，最大值：2）
- `ADDRESS`：服务器监听地址（默认值：0.0.0.0）
- `PORT`：服务器监听端口（默认值：8001）
- `ONLINE_OCR_MODE`：OCR 运行模式（默认值：True，表示远程模式）
- `ONLINE_OCR_BASE_URL`：远程 OCR API 的基础 URL
- `ONLINE_OCR_MODEL_NAME`：远程 OCR 模型名称
- `ONLINE_OCR_API_KEY`：远程 OCR API 的访问密钥

您可以通过修改 `server_config.py` 文件中的配置值来调整服务器设置：
```python
# In server_config.py
MAX_WORKER_THREADS = 2
ADDRESS = "0.0.0.0"
PORT = 8001
ONLINE_OCR_MODE = True
ONLINE_OCR_BASE_URL = "http://localhost:8000/v1"
ONLINE_OCR_MODEL_NAME = "deepseek-ai/DeepSeek-OCR"
ONLINE_OCR_API_KEY = "your-api-key"
```

### Docker 环境变量
在使用 Docker 容器时，可以通过环境变量覆盖配置：
- `ONLINE_OCR_MODE`：设置运行模式（true/false）
- `ONLINE_OCR_BASE_URL`：设置远程 OCR API 基础 URL
- `ONLINE_OCR_MODEL_NAME`：设置远程 OCR 模型名称
- `ONLINE_OCR_API_KEY`：设置远程 OCR API 访问密钥
- `PORT`：设置服务器监听端口

例如：
```bash
sudo docker run -p 8001:8001 -e ONLINE_OCR_MODE=false -e PORT=8001 deepseek-ocr-offline
```

## 实现细节

- 服务器支持两种运行模式，通过统一接口适配本地推理和远程API调用
- 本地模式使用工作线程（根据 GPU 内存情况使用 1-2 个）且每个线程持有一个模型实例
- 请求通过队列排队，由可用的工作线程处理
- 每个工作线程只初始化一次模型，并在整个服务器生命周期内将其保留在内存中
- 图像处理遵循与原始应用相同的流程
- 结果异步返回给客户端
- 远程模式使用 OpenAI Python 客户端库进行 API 调用

### 依赖隔离
为了优化 Docker 镜像大小和内存使用，项目实现了依赖隔离：
- 在线模式入口文件（`main_online.py`）仅导入轻量级依赖
- 离线模式入口文件（`main_offline.py`）导入所有必要的 heavy 依赖
- 通过条件导入和延迟加载避免在在线模式下加载 torch、transformers、vllm 等 heavy 包
- 在线模式 Docker 镜像仅包含 fastapi、uvicorn、openai、Pillow、numpy 等轻量级依赖
- 离线模式 Docker 镜像包含完整的依赖集，包括 torch、transformers、vllm 等

## 代码结构

- `main.py`: 主 FastAPI 应用
- `core/`: 核心处理逻辑
  - `processor.py`: 本地 OCR 处理器实现
  - `online_processor.py`: 远程 OCR 处理器实现
- `schemas/`: 数据模型和模式定义
  - `models.py`: API 请求和响应的 Pydantic 模型
- `config.py`: 服务器配置文件