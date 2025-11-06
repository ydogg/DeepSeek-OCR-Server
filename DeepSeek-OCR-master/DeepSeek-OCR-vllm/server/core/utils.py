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
    matches_other = []
    for a_match in matches:
        matches_other.append(a_match[0])
    return matches, matches_other


def clean_ref_tags(content):
    """Clean ref and det tags from the content"""
    # First clean formula content
    content = clean_formula(content)
    
    # Match ref and det tags
    matches_ref, matches_other = re_match(content)
    
    # Remove matched patterns and clean up extra whitespace and tags
    for idx, a_match_other in enumerate(matches_other):
        content = content.replace(a_match_other, '').replace('\n\n\n\n', '\n\n').replace('\n\n\n', '\n\n').replace('<center>', '').replace('</center>', '')
    
    return content