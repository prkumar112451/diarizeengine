import spacy
import re
from spacy.matcher import Matcher

# Load the spaCy model
nlp = spacy.load('en_core_web_sm')

# Helper function to initialize the spaCy matcher
def initialize_matcher(nlp, pattern_words):
    matcher = Matcher(nlp.vocab)
    patterns = [[{"LOWER": word.lower()}] for word in pattern_words]
    for pattern in patterns:
        matcher.add("KEYWORDS", [pattern])
    return matcher

# Set of months in full name and abbreviation (case insensitive)
months_set = {
    "january", "jan", "february", "feb", "march", "mar", "april", "apr",
    "may", "june", "jun", "july", "jul", "august", "aug", "september", "sep", "sept",
    "october", "oct", "november", "nov", "december", "dec"
}

def is_number(token):
    return token.like_num

def is_month(token):
    return token.text.lower() in months_set

def mask_entity_text(text):
    doc = nlp(text)
    masked_text = list(text)
    
    # Keywords for credit card detection
    credit_card_keywords = {'credit', 'card', 'debit', 'payment'}
    
    # Keywords for CVV, expiry detection
    sensitive_keywords = {'cvv', 'expiry', 'expiration'}
    
    segment_id = 0
    segment_word_masking = {}
    
    mask_credit_card_details = False
    
    for i, token in enumerate(doc):
        if token.text == '\n':
            segment_id += 1
            continue
        
        # Search for credit card related words
        if token.text.lower() in credit_card_keywords:
            # Check the next 35 tokens
            next_tokens = doc[i + 1:i + 50]
            numbers_found = 0

            for j, next_token in enumerate(next_tokens):
                if next_token.is_punct or next_token.is_space:
                    continue

                if is_number(next_token):
                    numbers_found += 1
                
                if numbers_found >= 16:
                    for k in range(j + 1):
                        token_to_mask = next_tokens[k]
                        if is_number(token_to_mask):
                            start_char = token_to_mask.idx
                            end_char = start_char + len(token_to_mask.text)
                            masked_text[start_char:end_char] = '*' * (end_char - start_char)
                            if segment_id not in segment_word_masking:
                                segment_word_masking[segment_id] = set()
                            segment_word_masking[segment_id].add(token_to_mask.text)
                    mask_credit_card_details = True
                    break
        
        # If credit card details were masked, look for CVV/expiry keywords
        if mask_credit_card_details and token.text.lower() in sensitive_keywords:
            next_tokens = doc[i + 1:i + 36]
            numbers_to_mask = 0
            for next_token in next_tokens:
                if is_number(next_token) or is_month(next_token):
                    start_char = next_token.idx
                    end_char = start_char + len(next_token.text)
                    masked_text[start_char:end_char] = '*' * (end_char - start_char)
                    if segment_id not in segment_word_masking:
                        segment_word_masking[segment_id] = set()
                    segment_word_masking[segment_id].add(next_token.text)
                    numbers_to_mask += 1
                if numbers_to_mask >= 6:
                    break
    
    return "".join(masked_text), segment_word_masking

# Function to mask words in a segment
def mask_segment_words(segment, words_to_mask):        
    masked_words = []
    for word in segment['words']:
        word_text = word['word'].lower()
                
        for mask_word in words_to_mask:
            if mask_word.lower() in word_text:
                word_text = word_text.replace(mask_word.lower(), '*' * len(mask_word))
                
        word['word'] = word_text
        masked_words.append(word)
    return masked_words


def check_words_in_string(hashset, input_string):
    words = set(input_string.lower().split())
    return not hashset.isdisjoint(words)

# Main function to mask entities in the transcript
def mask_entity(concatenated_text, segments):
    masked_text, segment_word_masking = mask_entity_text(concatenated_text)
        
    # Update segments with masked text and words
    segment_lines = masked_text.split("\n")
    for i, segment in enumerate(segments):
        segment["text"] = segment_lines[i]
        if i in segment_word_masking:
            segment["words"] = mask_segment_words(segment, segment_word_masking[i])
    
    return masked_text

def mask_transcript(segments):
    # Trim the text of each segment
    for segment in segments:
        segment["text"] = segment["text"].strip()
    concatenated_text = "\n".join([segment["text"] for segment in segments]).lower()
    hashset = {"credit", "card", "cvv", "debit"}

    is_sensitive = check_words_in_string(hashset, concatenated_text)
    if is_sensitive:
        masked_text = mask_entity(concatenated_text, segments)
