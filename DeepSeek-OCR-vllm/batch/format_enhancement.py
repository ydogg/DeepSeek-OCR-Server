"""
Format enhancement functions for batch processing
"""

import asyncio
import os
from openai import AsyncOpenAI
from config_loader import BATCH_CONFIG


async def enhance_text_format_batch(text_content: str):
    """
    Call the LLM API to enhance text format and quality for batch processing
    """
    try:
        print(f"[Batch Format Enhancement] Starting LLM call for text format enhancement")

        # Check if format enhancement is enabled
        if not hasattr(BATCH_CONFIG, 'format_enhancement_enabled') or not BATCH_CONFIG.format_enhancement_enabled:
            print(f"[Batch Format Enhancement] Format enhancement is disabled, returning original text")
            return text_content

        # Initialize OpenAI client with format enhancement configuration
        client = AsyncOpenAI(
            base_url=BATCH_CONFIG.format_enhancement_base_url,
            api_key=BATCH_CONFIG.format_enhancement_api_key or "sk-test"  # Use test key if none provided
        )

        # Prepare the request with format enhancement prompt
        format_prompt = getattr(BATCH_CONFIG, 'format_enhancement_prompt',
            "请对以下OCR识别的文本内容进行格式优化和增强，修正识别错误，优化文档结构。")

        full_prompt = f"{format_prompt}\n\n```\n{text_content}\n```"

        messages = [
            {
                "role": "user",
                "content": full_prompt
            }
        ]

        model_name = getattr(BATCH_CONFIG, 'format_enhancement_model_name', 'qwen3-coder')
        print(f"[Batch Format Enhancement] Request messages prepared")
        print(f"[Batch Format Enhancement] Model: {model_name}")
        print(f"[Batch Format Enhancement] Original text length: {len(text_content)}")

        # Make the API call using OpenAI client
        base_url = getattr(BATCH_CONFIG, 'format_enhancement_base_url', 'http://localhost:8000/v1')
        print(f"[Batch Format Enhancement] Making API call to {base_url}")
        response = await client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=8192,  # Allow more tokens for format enhancement
            temperature=0.1  # Lower temperature for more consistent formatting
        )

        print(f"[Batch Format Enhancement] API call completed successfully")

        # Process the response
        if response and response.choices:
            enhanced_text = response.choices[0].message.content
            if enhanced_text:
                # Clean up the response - remove any extra formatting
                enhanced_text = enhanced_text.strip()
                # Remove potential code block markers if present
                if enhanced_text.startswith('```'):
                    lines = enhanced_text.split('\n')
                    if len(lines) > 1:
                        enhanced_text = '\n'.join(lines[1:-1]) if enhanced_text.endswith('```') else '\n'.join(lines[1:])
                enhanced_text = enhanced_text.strip()

                print(f"[Batch Format Enhancement] Enhanced text length: {len(enhanced_text)}")
                return enhanced_text
            else:
                print(f"[Batch Format Enhancement] Empty response received")
                return text_content  # Return original if enhancement failed
        else:
            print(f"[Batch Format Enhancement] No response or choices in response")
            return text_content  # Return original if enhancement failed

    except Exception as e:
        print(f"[Batch Format Enhancement] Exception in enhance_text_format_batch: {str(e)}")
        # Add more detailed error information
        import traceback
        print(f"[Batch Format Enhancement] Full traceback: {traceback.format_exc()}")
        print(f"[Batch Format Enhancement] Returning original text due to enhancement failure")
        return text_content  # Return original text if enhancement failed


def enhance_text_format_batch_sync(text_content: str):
    """
    Synchronous wrapper for batch format enhancement
    """
    # Run the async function in the current event loop or create a new one
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If loop is already running, we need to run in a thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, enhance_text_format_batch(text_content))
                return future.result()
        else:
            return loop.run_until_complete(enhance_text_format_batch(text_content))
    except Exception as e:
        print(f"[Batch Format Enhancement] Exception in sync wrapper: {str(e)}")
        return text_content


def enhance_results_batch(results, stage):
    """
    Apply format enhancement to a batch of results based on stage
    """
    if not hasattr(BATCH_CONFIG, 'format_enhancement_enabled') or not BATCH_CONFIG.format_enhancement_enabled:
        print(f"[Batch Format Enhancement] Format enhancement is disabled for stage {stage}")
        return results
    
    # Skip enhancement for raw stage
    if stage == 'raw':
        print(f"[Batch Format Enhancement] Skipping format enhancement for raw stage")
        return results
    
    print(f"[Batch Format Enhancement] Applying format enhancement to {len(results)} results for stage {stage}")
    
    enhanced_results = []
    for idx, result in enumerate(results):
        if result.strip():  # Only enhance non-empty results
            print(f"[Batch Format Enhancement] Enhancing result {idx+1}/{len(results)}")
            enhanced_result = enhance_text_format_batch_sync(result)
            enhanced_results.append(enhanced_result)
        else:
            enhanced_results.append(result)  # Keep empty results as-is
    
    print(f"[Batch Format Enhancement] Format enhancement completed for stage {stage}")
    return enhanced_results