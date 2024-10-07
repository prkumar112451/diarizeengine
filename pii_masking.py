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

# List of common helper words used in day-to-day English
helper_words = {
    "yeah", "sure", "okay", "right", "cool", "uh", "um", "uh-huh", "hmm", "oh", "ah",
    "well", "so", "like", "just", "actually", "basically", "really", "you know", "I mean",
    "sort of", "kind of", "basically", "literally", "really", "seriously", "totally",
    "obviously", "clearly", "definitely", "exactly", "absolutely", "certainly",
    "definitely", "indeed", "naturally", "undoubtedly", "yes", "no", "can", "you", "repeat", "that", "what's", "what", "ok", "hmm", "oh"
}

# Set of months in full name and abbreviation (case insensitive)
months_set = {
    "january", "jan", "february", "feb", "march", "mar", "april", "apr",
    "may", "june", "jun", "july", "jul", "august", "aug", "september", "sep", "sept",
    "october", "oct", "november", "nov", "december", "dec"
}

def is_number(token):
    return token.like_num

def mask_entity_text(text):
    doc = nlp(text)
    masked_text = list(text)
    
    # Keywords for credit card detection
    credit_card_keywords = {'credit', 'card', 'debit'}
    
    # Keywords for CVV, expiry detection
    sensitive_keywords = {'cvv', 'expiry', 'expiration'}
    
    segment_id = 0
    segment_word_masking = {}
    
    mask_next_numbers = False
    mask_credit_card_details = False
    numbers_found = 0
    current_index = 0

    for i, token in enumerate(doc):
        if token.text == '\n':
            segment_id += 1
            continue
        
        # Search for credit card related words
        if token.text.lower() in credit_card_keywords:
            # Check the next 35 tokens
            next_tokens = doc[i + 1:i + 36]  # Look ahead up to 35 words
            numbers_found = 0

            for j, next_token in enumerate(next_tokens):
                if is_number(next_token):
                    numbers_found += 1
                
                if numbers_found >= 16:
                    # Found at least 16 numbers, mask all the numbers in these 35 words
                    for k in range(j + 1):  # Iterate over the first `j+1` tokens (i.e., within the range where numbers were found)
                        token_to_mask = next_tokens[k]
                        if is_number(token_to_mask):
                            start_char = token_to_mask.idx
                            end_char = start_char + len(token_to_mask.text)
                            masked_text[start_char:end_char] = '*' * (end_char - start_char)
                            if segment_id not in segment_word_masking:
                                segment_word_masking[segment_id] = set()
                            segment_word_masking[segment_id].add(token_to_mask.text)
                    mask_credit_card_details = True  # Trigger that we've masked credit card details
                    break
        
        # If credit card details were masked, look for CVV/expiry keywords
        if mask_credit_card_details and token.text.lower() in sensitive_keywords:
            next_tokens = doc[i + 1:i + 36]  # Check the next 35 tokens to find 6 numbers
            numbers_to_mask = 0
            for next_token in next_tokens:
                if is_number(next_token):
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
