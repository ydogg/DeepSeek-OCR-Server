# DeepSeek OCR 服务器

这是一个为 DeepSeek OCR 模型提供 OpenAI 兼容 API 的 FastAPI 服务器。
从DeepSeek-OCR项目fork而来，请先访问DeepSeek OCR项目仓库，安装DeepSeek OCR支持模块。

## 功能特性

- OpenAI 兼容的 `/v1/chat/completions` 接口
- 简单的 OCR 接口 `/v1/ocr`
- Gradio Web界面，提供用户友好的OCR体验
- 支持两种运行模式：
  - 本地模式：使用本地 vLLM 运行 DeepSeek OCR 模型
  - 远程模式：通过 OpenAI 兼容 API 调用远程 OCR 服务
- 工作线程池（根据 GPU 内存情况使用 1-2 个工作线程）
- 基于队列的请求处理，实现高效的资源利用
- 每个工作线程在整个服务器生命周期内持有一个模型实例
- 模块化代码组织，便于维护

## 安装

1. 安装所需依赖：
```bash
pip install -r DeepSeek-OCR-vllm/server/requirements.txt
```

2. 安装Gradio界面依赖：
```bash
pip install -r DeepSeek-OCR-vllm/gradio/requirements.txt
```

3. 确保您已准备好 DeepSeek OCR 模型，并在 `DeepSeek-OCR-vllm/config.py` 中进行了配置

## 使用方法

### 启动服务器

```bash
cd DeepSeek-OCR-vllm/server
./start_server.sh
```

或者直接运行：
```bash
cd DeepSeek-OCR-vllm
python -m server.main
```

服务器将在 `http://0.0.0.0:8001` 启动（地址和端口可在 `DeepSeek-OCR-vllm/server/config.py` 中配置）

### 启动Gradio界面

```bash
cd DeepSeek-OCR-vllm/gradio
python app.py
```

Gradio界面将在 `http://0.0.0.0:7861` 启动

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

## Gradio界面功能

Gradio界面提供了一个用户友好的Web界面来使用DeepSeek OCR服务，具有以下功能：

- 上传图片从本地机器
- 配置OCR参数：
  - 自定义提示词
  - 处理级别（raw, md_image, md_text, md_merged）
- 实时查看OCR结果，支持在原始文本和Markdown格式之间切换
- 一键复制OCR结果到剪贴板
- 简单直观的Web界面

### 安装Gradio界面

1. 安装Gradio界面依赖：
```bash
pip install -r DeepSeek-OCR-vllm/gradio/requirements.txt
```

2. 确保DeepSeek OCR服务器正在运行

### 启动Gradio界面

```bash
cd DeepSeek-OCR-vllm/gradio
python app.py
```

Gradio界面将在 `http://0.0.0.0:7861` 启动

### 使用Gradio界面

1. 在浏览器中访问 `http://localhost:7861`

2. 上传图片使用界面

3. 配置OCR参数如果需要：
   - **提示词**: 自定义OCR处理提示词（可选）
   - **处理级别**:
     - `raw`: 原始OCR输出，包含所有标签
     - `md_image`: 清理后的OCR输出，图像标签转换为Markdown
     - `md_text`: 清理后的OCR输出，不包含图像标签
     - `md_merged`: 清理后的OCR输出，包含图像分析（VL模型）

4. 点击"Process Image"发送图片到DeepSeek OCR服务器

5. 在输出框中查看OCR结果，具有以下功能：
   - 使用"Output Format"单选按钮在原始文本和Markdown格式之间切换
   - 在原始文本视图中使用复制按钮将结果复制到剪贴板

### API集成

Gradio界面通过 `/v1/ocr` 端点与DeepSeek OCR服务器通信：

- **端点**: `POST http://localhost:8001/v1/ocr`
- **请求格式**: 包含base64编码图像的JSON
- **参数**:
  - `image`: Base64编码的图像数据
  - `prompt`: OCR提示词（可选）
  - `level`: 处理级别（默认: md_text）

### 依赖要求

- Python 3.8+
- Gradio >= 3.36.0
- Requests >= 2.28.0
- Pillow >= 9.0.0
- 运行中的DeepSeek OCR服务器

## 配置

服务器使用与主 DeepSeek OCR 应用相同的配置：
- 模型路径来自 `DeepSeek-OCR-vllm/config.py`
- 提示词来自 `DeepSeek-OCR-vllm/config.py`
- 裁剪模式来自 `DeepSeek-OCR-vllm/config.py`
- 最大并发数来自 `DeepSeek-OCR-vllm/config.py`

服务器还有独立的配置选项，位于 `DeepSeek-OCR-vllm/server/config.py`：
- `MAX_WORKER_THREADS`：工作线程数量（默认值：1，最大值：2）
- `ADDRESS`：服务器监听地址（默认值：0.0.0.0）
- `PORT`：服务器监听端口（默认值：8001）
- `ONLINE_OCR_MODE`：OCR 运行模式（默认值：True，表示远程模式）
- `ONLINE_OCR_BASE_URL`：远程 OCR API 的基础 URL
- `ONLINE_OCR_MODEL_NAME`：远程 OCR 模型名称
- `ONLINE_OCR_API_KEY`：远程 OCR API 的访问密钥

您可以通过修改 `DeepSeek-OCR-vllm/server/config.py` 文件中的配置值来调整服务器设置：
```python
# In server/config.py
MAX_WORKER_THREADS = 2
ADDRESS = "0.0.0.0"
PORT = 8001
ONLINE_OCR_MODE = True
ONLINE_OCR_BASE_URL = "http://localhost:8000/v1"
ONLINE_OCR_MODEL_NAME = "deepseek-ai/DeepSeek-OCR"
ONLINE_OCR_API_KEY = "your-api-key"
```

## 实现细节

- 服务器支持两种运行模式，通过统一接口适配本地推理和远程API调用
- 本地模式使用工作线程（根据 GPU 内存情况使用 1-2 个）且每个线程持有一个模型实例
- 请求通过队列排队，由可用的工作线程处理
- 每个工作线程只初始化一次模型，并在整个服务器生命周期内将其保留在内存中
- 图像处理遵循与原始应用相同的流程
- 结果异步返回给客户端
- 远程模式使用 OpenAI Python 客户端库进行 API 调用

## 代码结构

- `DeepSeek-OCR-vllm/server/main.py`: 主 FastAPI 应用
- `DeepSeek-OCR-vllm/server/core/`: 核心处理逻辑
  - `processor.py`: 本地 OCR 处理器实现
  - `online_processor.py`: 远程 OCR 处理器实现
- `DeepSeek-OCR-vllm/server/schemas/`: 数据模型和模式定义
  - `models.py`: API 请求和响应的 Pydantic 模型
- `DeepSeek-OCR-vllm/server/config.py`: 服务器配置文件
- `DeepSeek-OCR-vllm/gradio/`: Gradio Web界面
  - `app.py`: Gradio应用主代码
  - `requirements.txt`: Gradio界面依赖
  - `README.md`: Gradio界面使用说明
