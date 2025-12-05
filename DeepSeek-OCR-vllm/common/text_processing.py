"""
Common text processing functions
Shared between server and batch implementations
"""

import re


def clean_formula(text):
    """Clean formula content by removing specific patterns"""
    formula_pattern = r'\\\[(.*?)\\\]'

    def process_formula(match):
        formula = match.group(1)
        formula = re.sub(r'\\quad\s*\([^)]*\)', '', formula)
        formula = formula.strip()
        return r'\[' + formula + r'\]'

    cleaned_text = re.sub(formula_pattern, process_formula, text)
    return cleaned_text


def re_match(text):
    """Match ref and det tag patterns in the text"""
    pattern = r'(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)'
    matches = re.findall(pattern, text, re.DOTALL)

    matches_image = []
    matches_other = []
    for a_match in matches:
        if '<|ref|>image<|/ref|>' in a_match[0]:
            matches_image.append(a_match[0])
        else:
            matches_other.append(a_match[0])
    return matches, matches_image, matches_other


def convert_image_tags_to_md(content):
    """Convert image tags to Markdown format

    Args:
        content: The text content to process

    Returns:
        str: Content with image tags converted to Markdown format
    """
    # Match ref and det tags
    matches_ref, matches_image, matches_other = re_match(content)

    # Convert image tags to Markdown format
    for idx, a_match_image in enumerate(matches_image):
        # Convert <|ref|>image<|/ref|><|det|>[[40, 191, 960, 920]]<|/det|> to ![](images/ocr_detected_{idx}.jpg)
        content = content.replace(a_match_image, f'![](images/ocr_detected_{idx}.jpg)\n')

    return content


def clean_ref_tags(content, keep_image_tags=False):
    """Clean ref and det tags from the content

    Args:
        content: The text content to clean
        keep_image_tags: If True, keep image tags; if False, remove all tags
    """
    # First clean formula content
    content = clean_formula(content)

    # Match ref and det tags
    matches_ref, matches_image, matches_other = re_match(content)

    if keep_image_tags:
        # Only remove non-image tags, keep image tags
        for idx, a_match_other in enumerate(matches_other):
            content = content.replace(a_match_other, '').replace('\n\n\n\n', '\n\n').replace('\n\n\n', '\n\n').replace('<center>', '').replace('</center>', '')
        # Keep image matches (ref|>image<|/ref|><|det|>...)
    else:
        # Remove all tags (original behavior)
        # Remove matched patterns and clean up extra whitespace and tags
        for idx, a_match_other in enumerate(matches_other):
            content = content.replace(a_match_other, '').replace('\n\n\n\n', '\n\n').replace('\n\n\n', '\n\n').replace('<center>', '').replace('</center>', '')

        # Also remove image matches (ref|>image<|/ref|><|det|>...)
        for idx, a_match_image in enumerate(matches_image):
            content = content.replace(a_match_image, '')

    # Also clean OCR detected image references with ocr_detected_ prefix
    # Use simplified approach that works
    # Find all OCR detected image references using simplified pattern
    ocr_refs = re.findall(r'images/ocr_detected_\d+\.jpg', content)
    for ref in ocr_refs:
        # Remove the full Markdown image reference
        full_ref = f'![]({ref})'
        content = content.replace(full_ref, '')

    return content
