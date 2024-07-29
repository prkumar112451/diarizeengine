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

def mask_entity_text(text, pattern_words, num_mask_count, date_mask_count):
    matcher = initialize_matcher(nlp, pattern_words)
    doc = nlp(text)
    matches = matcher(doc)
    
    masked_text = list(text)
    mask_next_numbers = False
    numbers_count = 0
    dates_count = 0
    
    segment_id = 0
    segment_word_masking = {}

    for token in doc:        
        if token.text == '\n':
            segment_id += 1
            continue
        
        if any(token.text.lower() == word.lower() for word in pattern_words):
            mask_next_numbers = True
            numbers_count = 0
            dates_count = 0
        
        if mask_next_numbers:
            if is_number(token):
                start_char = token.idx
                end_char = start_char + len(token.text)
                masked_text[start_char:end_char] = '*' * (end_char - start_char)
                if segment_id not in segment_word_masking:
                    segment_word_masking[segment_id] = set()
                segment_word_masking[segment_id].add(token.text)
                
                numbers_count += 1
                if numbers_count >= num_mask_count:
                    mask_next_numbers = False
            
            elif token.text.lower() in months_set:
                start_char = token.idx
                end_char = start_char + len(token.text)
                masked_text[start_char:end_char] = '*' * (end_char - start_char)
                
                dates_count += 1
                if dates_count >= date_mask_count:
                    mask_next_numbers = False
            
            if token.pos_ in ["NOUN", "PRON"] and token.text.lower() not in helper_words and dates_count >= date_mask_count and numbers_count >= num_mask_count:
                mask_next_numbers = False
    
    return "".join(masked_text), segment_word_masking

# Function to mask words in a segment
def mask_segment_words(segment, words_to_mask):        
    masked_words = []
    for word in segment['words']:
        word_text = word['word']
        for mask_word in words_to_mask:
            if mask_word in word_text:
                word_text = word_text.replace(mask_word, '*' * len(mask_word))
        word['word'] = word_text
        masked_words.append(word)
    return masked_words


def check_words_in_string(hashset, input_string):
    words = set(input_string.lower().split())
    return not hashset.isdisjoint(words)

# Main function to mask entities in the transcript
def mask_entity(concatenated_text, segments, entity):
    pattern_words = []
    num_mask_count = 16
    date_mask_count = 0

    if entity == 'credit':
        pattern_words = ["credit", "card"]
        num_mask_count = 16
        date_mask_count = 0
    
    if entity == 'cvv':
        pattern_words = ["cvv","digit","digits"]
        num_mask_count = 4
        date_mask_count = 1

    if entity == 'expiry':
        pattern_words = ["expiry","date"]
        num_mask_count = 4
        date_mask_count = 1

    masked_text, segment_word_masking = mask_entity_text(concatenated_text, pattern_words, num_mask_count, date_mask_count)

        
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
    concatenated_text = "\n".join([segment["text"] for segment in segments])
    hashset = {"credit", "card", "cvv"}

    is_sensitive = check_words_in_string(hashset, concatenated_text)

    if is_sensitive:
        masked_text = mask_entity(concatenated_text, segments, 'credit')
        masked_text = mask_entity(masked_text, segments, 'cvv')
        masked_text = mask_entity(masked_text, segments, 'expiry')
