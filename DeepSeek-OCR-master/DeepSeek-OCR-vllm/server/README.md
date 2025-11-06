# DeepSeek OCR 服务器

这是一个为 DeepSeek OCR 模型提供 OpenAI 兼容 API 的 FastAPI 服务器。

## 功能特性

- OpenAI 兼容的 `/v1/chat/completions` 接口
- 简单的 OCR 接口 `/v1/images/ocr`
- 工作线程池（根据 GPU 内存情况使用 1-2 个工作线程）
- 基于队列的请求处理，实现高效的资源利用
- 每个工作线程在整个服务器生命周期内持有一个模型实例
- 模块化代码组织，便于维护

## 安装

1. 安装所需依赖：
```bash
pip install -r server/requirements.txt
```

2. 确保您已准备好 DeepSeek OCR 模型，并在 `config.py` 中进行了配置

## 使用方法

### 启动服务器

```bash
cd server
./start_server.sh
```

或者直接运行：
```bash
python -m server.main
```

服务器将在 `http://0.0.0.0:8888` 启动（地址和端口可在 `server/config.py` 中配置）

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
      "content": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEASABIAAD/..."
    }
  ]
}
```

#### 2. 简单 OCR 接口

接口地址：`POST /v1/images/ocr`

请求示例（表单数据）：
```
image: [文件上传]
prompt: "<image>\n<|grounding|>将文档转换为 markdown 格式。"
```

#### 3. 健康检查

接口地址：`GET /health`

## 配置

服务器使用与主 DeepSeek OCR 应用相同的配置：
- 模型路径来自 `config.py`
- 提示词来自 `config.py`
- 裁剪模式来自 `config.py`
- 最大并发数来自 `config.py`

服务器还有独立的配置选项，位于 `server/config.py`：
- `MAX_WORKER_THREADS`：工作线程数量（默认值：1，最大值：2）
- `ADDRESS`：服务器监听地址（默认值：0.0.0.0）
- `PORT`：服务器监听端口（默认值：8888）

您可以通过修改 `server/config.py` 文件中的配置值来调整服务器设置：
```python
# In server/config.py
MAX_WORKER_THREADS = 2
ADDRESS = "0.0.0.0"
PORT = 8888
```

## 实现细节

- 服务器使用工作线程（根据 GPU 内存情况使用 1-2 个）且每个线程持有一个模型实例
- 请求通过队列排队，由可用的工作线程处理
- 每个工作线程只初始化一次模型，并在整个服务器生命周期内将其保留在内存中
- 图像处理遵循与原始应用相同的流程
- 结果异步返回给客户端

## 代码结构

- `main.py`: 主 FastAPI 应用
- `core/`: 核心处理逻辑
  - `processor.py`: 模型工作线程和 OCR 处理器实现
- `schemas/`: 数据模型和模式定义
  - `models.py`: API 请求和响应的 Pydantic 模型
- `models/`: 模型相关工具（未来可能需要）