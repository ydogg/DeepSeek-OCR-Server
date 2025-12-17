# DeepSeek OCR Server Testing

This directory contains several test scripts for the DeepSeek OCR Server:

## Test Scripts

1. `test_openai_endpoint.py` - Tests the OpenAI compatible endpoint
2. `test_openai_comprehensive.py` - Comprehensive test of OpenAI endpoint with different OCR levels
3. `test_ocr_endpoints.py` - Tests the original OCR endpoint and health check
4. `test_comprehensive.py` - Comprehensive test of all endpoints and OCR levels

## Usage

### Prerequisites
- DeepSeek OCR Server running (either online or offline mode)
- Python 3.8+
- Required packages: `requests`, `Pillow`

### Running Tests

1. **Test OpenAI Compatible Endpoint:**
   ```bash
   python test_openai_endpoint.py [image_path]
   ```

2. **Comprehensive OpenAI Endpoint Test:**
   ```bash
   python test_openai_comprehensive.py [image_path]
   ```

3. **Test Original OCR Endpoint:**
   ```bash
   python test_ocr_endpoints.py [image_path]
   ```

4. **Comprehensive Test of All Endpoints:**
   ```bash
   python test_comprehensive.py [image_path]
   ```

If no image path is provided, the scripts will use `test_data/test_image.jpg` by default.

## Endpoints Tested

1. **Health Check**: `GET /health`
2. **OpenAI Compatible**: `POST /v1/chat/completions`
3. **Original OCR**: `POST /v1/ocr`

## OCR Levels

The OCR endpoint supports 4 processing levels:
- `raw` - Raw OCR output with tags
- `md_image` - Cleaned output with image references in Markdown format
- `md_text` - Cleaned text output
- `md_merged` - Full processing with image analysis and VL model enhancement

## Test Results

All tests should return status code 200 for successful execution. The comprehensive test script provides a summary of all test results.