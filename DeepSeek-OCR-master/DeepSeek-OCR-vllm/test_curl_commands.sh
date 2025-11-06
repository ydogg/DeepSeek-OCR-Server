#!/bin/bash

# 示例:
#   ./test_curl_commands.sh                                    # 使用默认图像路径
#   ./test_curl_commands.sh /path/to/image.jpg                 # 指定图像路径
#   ./test_curl_commands.sh -c                                 # 只显示内容，使用默认图像路径
#   ./test_curl_commands.sh -c /path/to/image.jpg              # 只显示内容，指定图像路径
#   ./test_curl_commands.sh --content-only /path/to/image.jpg  # 只显示内容，指定图像路径

# 只显示content字段
SHOW_CONTENT_ONLY=false
IMAGE_PATH=""

# 解析命令行参数
while [[ $# -gt 0 ]]; do
  case $1 in
    -c|--content-only)
      SHOW_CONTENT_ONLY=true
      shift
      ;;
    -*)
      echo "未知选项: $1"
      echo "用法: $0 [--content-only|-c] [IMAGE_PATH]"
      exit 1
      ;;
    *)
      IMAGE_PATH="$1"
      shift
      ;;
  esac
done


if [ ! -f "$IMAGE_PATH" ]; then
  echo "错误: 图像文件不存在: $IMAGE_PATH"
  exit 1
fi

SERVER_URL="http://localhost:8888"

TEMP_JSON_FILE="./ocr_request.json"
TEMP_RESPONSE_FILE="./ocr_response.json"

IMAGE_BASE64=$(base64 -i "$IMAGE_PATH")

cat > "$TEMP_JSON_FILE" <<EOF
{
  "model": "deepseek-ocr",
  "messages": [
    {
      "role": "user",
      "content": "data:image/jpeg;base64,$IMAGE_BASE64"
    }
  ]
}
EOF

# 使用文件发送请求
if [ "$SHOW_CONTENT_ONLY" = true ]; then
  echo "Showing content only..."
  curl -s --max-time 120 -X POST "$SERVER_URL/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d @"$TEMP_JSON_FILE" > "$TEMP_RESPONSE_FILE"
  
  # 检查响应文件是否存在且不为空
  if [ ! -s "$TEMP_RESPONSE_FILE" ]; then
    echo "错误: 没有收到有效的响应"
    exit 1
  fi
  
  # 使用jq提取content字段
  if command -v jq &> /dev/null; then
    jq -r '.choices[0].message.content' "$TEMP_RESPONSE_FILE"
  else
    echo "jq is not installed. Please install jq to use --content-only option."
    cat "$TEMP_RESPONSE_FILE"
  fi
else
  curl --max-time 120 -X POST "$SERVER_URL/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d @"$TEMP_JSON_FILE"
fi

# 清理临时文件
rm -f "$TEMP_JSON_FILE" "$TEMP_RESPONSE_FILE"
