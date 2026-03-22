import re

def mask_emails(text: str) -> tuple[str, int]:
    # Need to avoid double-masking if already masked. The test says:
    # "Some datasets use the string |||EMAIL_ADDRESS||| to represent masked PII."
    # Wait, the instruction might just be "replace emails". 
    # If the text has `|||EMAIL_ADDRESS|||`, it doesn't match the regex for email because of the pipes, 
    # unless part of it matches. But `|||EMAIL_ADDRESS|||` doesn't contain `@`.
    # So a standard regex is fine.
    
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    
    matches = re.findall(email_pattern, text)
    num_masked = len(matches)
    
    text = re.sub(email_pattern, '|||EMAIL_ADDRESS|||', text)
    
    return text, num_masked

def mask_phone_numbers(text: str) -> tuple[str, int]:
    # Formats: 2831823829, (283)-182-3829, (283) 182 3829, 283-182-3829
    # We can use a regex that handles optional area code parens, and optional separators.
    # Note: avoid matching IPv4 addresses or long numbers.
    phone_pattern = r'(?:(?:\(\d{3}\))|(?:\b\d{3}))(?:[- ]?)\d{3}(?:[- ]?)\d{4}\b'
    
    matches = re.findall(phone_pattern, text)
    num_masked = len(matches)
    
    text = re.sub(phone_pattern, '|||PHONE_NUMBER|||', text)
    
    return text, num_masked

def mask_ips(text: str) -> tuple[str, int]:
    # Match valid IPv4 addresses
    ip_pattern = r'\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b'
    
    matches = re.findall(ip_pattern, text)
    num_masked = len(matches)
    
    text = re.sub(ip_pattern, '|||IP_ADDRESS|||', text)
    
    return text, num_masked
