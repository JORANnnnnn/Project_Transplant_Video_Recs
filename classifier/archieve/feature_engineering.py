"""
Text Readability Feature Engineering for YouTube Video Transcripts
This module implements various readability metrics to assess text comprehensibility.
1. Flesch Reading Ease:综合指标，分数越高越容易
2. Flesch-Kincaid Grade Level / SMOG Index / Gunning Fog Index(任选1-2个):提供一个估计的教育年级水平，有助于目标受众定位（例如，针对一般公众的健康信息通常推荐 grade 6-8)
3. Average Word Length (Average syllables per word):衡量词汇的复杂性
4. Long Word Ratio (Proportion of words with >= 3 syllables):直接衡量复杂词汇的密度
5. Lexical Diversity (Type-Token Ratio, TTR):衡量词汇丰富度，过高可能意味着使用了过多生僻词或术语
"""

import re
import math
from typing import List, Dict, Any
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import cmudict

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/cmudict')
except LookupError:
    nltk.download('cmudict')

# Load CMU Pronouncing Dictionary for syllable counting
cmu_dict = cmudict.dict()


def count_syllables(word: str) -> int:
    """
    Count syllables in a word using CMU Pronouncing Dictionary.
    Falls back to simple vowel counting if word not found in dictionary.
    
    Args:
        word: The word to count syllables for
        
    Returns:
        Number of syllables in the word
    """
    word = word.lower().strip()
    
    # Check if word exists in CMU dictionary
    if word in cmu_dict:
        # CMU dict provides phonemes, count vowel sounds
        phonemes = cmu_dict[word][0]  # Take first pronunciation
        return len([p for p in phonemes if p[-1].isdigit()])
    
    # Fallback: count vowel groups
    vowels = 'aeiouy'
    word = re.sub(r'[^a-z]', '', word.lower())
    if not word:
        return 0
    
    # Count vowel groups
    vowel_groups = re.findall(r'[aeiouy]+', word)
    syllable_count = len(vowel_groups)
    
    # Handle silent 'e' at end
    if word.endswith('e') and syllable_count > 1:
        syllable_count -= 1
    
    # Handle words ending with 'le' after consonant
    if word.endswith('le') and len(word) > 2 and word[-3] not in vowels:
        syllable_count += 1
    
    return max(1, syllable_count)


def flesch_reading_ease(text: str) -> float:
    """
    Calculate Flesch Reading Ease score.
    Score ranges from 0-100, higher scores indicate easier reading.
    
    Formula: 206.835 - (1.015 × ASL) - (84.6 × ASW)
    Where ASL = Average Sentence Length, ASW = Average Syllables per Word
    
    Args:
        text: Input text to analyze
        
    Returns:
        Flesch Reading Ease score
    """
    sentences = sent_tokenize(text)
    words = word_tokenize(text.lower())
    
    # Filter out punctuation
    words = [word for word in words if word.isalpha()]
    
    if len(sentences) == 0 or len(words) == 0:
        return 0.0
    
    # Calculate average sentence length
    avg_sentence_length = len(words) / len(sentences)
    
    # Calculate average syllables per word
    total_syllables = sum(count_syllables(word) for word in words)
    avg_syllables_per_word = total_syllables / len(words)
    
    # Flesch Reading Ease formula
    score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
    
    return round(score, 2)


def flesch_kincaid_grade_level(text: str) -> float:
    """
    Calculate Flesch-Kincaid Grade Level.
    Indicates the U.S. grade level needed to understand the text.
    
    Formula: (0.39 × ASL) + (11.8 × ASW) - 15.59
    
    Args:
        text: Input text to analyze
        
    Returns:
        Flesch-Kincaid Grade Level
    """
    sentences = sent_tokenize(text)
    words = word_tokenize(text.lower())
    
    # Filter out punctuation
    words = [word for word in words if word.isalpha()]
    
    if len(sentences) == 0 or len(words) == 0:
        return 0.0
    
    # Calculate average sentence length
    avg_sentence_length = len(words) / len(sentences)
    
    # Calculate average syllables per word
    total_syllables = sum(count_syllables(word) for word in words)
    avg_syllables_per_word = total_syllables / len(words)
    
    # Flesch-Kincaid formula
    grade_level = (0.39 * avg_sentence_length) + (11.8 * avg_syllables_per_word) - 15.59
    
    return round(grade_level, 2)


def smog_index(text: str) -> float:
    """
    Calculate SMOG (Simple Measure of Gobbledygook) Index.
    Estimates years of education needed to understand the text.
    
    Formula: 1.043 × sqrt(polysyllables × 30 / sentences) + 3.1291
    
    Args:
        text: Input text to analyze
        
    Returns:
        SMOG Index score
    """
    sentences = sent_tokenize(text)
    words = word_tokenize(text.lower())
    
    # Filter out punctuation
    words = [word for word in words if word.isalpha()]
    
    if len(sentences) == 0 or len(words) == 0:
        return 0.0
    
    # Count polysyllabic words (3+ syllables)
    polysyllables = sum(1 for word in words if count_syllables(word) >= 3)
    
    # SMOG formula (use exactly 30 sentences or all if less than 30)
    sample_size = min(30, len(sentences))
    smog_score = 1.043 * math.sqrt(polysyllables * 30 / sample_size) + 3.1291
    
    return round(smog_score, 2)


def gunning_fog_index(text: str) -> float:
    """
    Calculate Gunning Fog Index.
    Estimates years of formal education needed to understand the text.
    
    Formula: 0.4 × (ASL + percentage of complex words)
    
    Args:
        text: Input text to analyze
        
    Returns:
        Gunning Fog Index score
    """
    sentences = sent_tokenize(text)
    words = word_tokenize(text.lower())
    
    # Filter out punctuation
    words = [word for word in words if word.isalpha()]
    
    if len(sentences) == 0 or len(words) == 0:
        return 0.0
    
    # Calculate average sentence length
    avg_sentence_length = len(words) / len(sentences)
    
    # Count complex words (3+ syllables, excluding proper nouns and -ed/-ing endings)
    complex_words = 0
    for word in words:
        if count_syllables(word) >= 3:
            # Exclude words ending in -ed or -ing
            if not (word.endswith('ed') or word.endswith('ing')):
                complex_words += 1
    
    # Calculate percentage of complex words
    complex_word_percentage = (complex_words / len(words)) * 100
    
    # Gunning Fog formula
    fog_score = 0.4 * (avg_sentence_length + complex_word_percentage)
    
    return round(fog_score, 2)


def average_word_length(text: str) -> float:
    """
    Calculate average syllables per word.
    
    Args:
        text: Input text to analyze
        
    Returns:
        Average syllables per word
    """
    words = word_tokenize(text.lower())
    words = [word for word in words if word.isalpha()]
    
    if len(words) == 0:
        return 0.0
    
    total_syllables = sum(count_syllables(word) for word in words)
    avg_syllables = total_syllables / len(words)
    
    return round(avg_syllables, 3)


def long_word_ratio(text: str) -> float:
    """
    Calculate proportion of words with 3 or more syllables.
    
    Args:
        text: Input text to analyze
        
    Returns:
        Ratio of long words (0-1)
    """
    words = word_tokenize(text.lower())
    words = [word for word in words if word.isalpha()]
    
    if len(words) == 0:
        return 0.0
    
    long_words = sum(1 for word in words if count_syllables(word) >= 3)
    ratio = long_words / len(words)
    
    return round(ratio, 3)


def lexical_diversity(text: str) -> float:
    """
    Calculate Type-Token Ratio (TTR) - lexical diversity.
    Measures vocabulary richness (unique words / total words).
    
    Args:
        text: Input text to analyze
        
    Returns:
        Type-Token Ratio (0-1)
    """
    words = word_tokenize(text.lower())
    words = [word for word in words if word.isalpha()]
    
    if len(words) == 0:
        return 0.0
    
    unique_words = len(set(words))
    total_words = len(words)
    
    ttr = unique_words / total_words
    
    return round(ttr, 3)


def calculate_readability_features(text: str) -> Dict[str, float]:
    """
    Calculate all readability features for a given text.
    
    Args:
        text: Input text to analyze
        
    Returns:
        Dictionary containing all readability metrics
    """
    features = {
        'flesch_reading_ease': flesch_reading_ease(text),
        'flesch_kincaid_grade_level': flesch_kincaid_grade_level(text),
        'smog_index': smog_index(text),
        'gunning_fog_index': gunning_fog_index(text),
        'average_word_length': average_word_length(text),
        'long_word_ratio': long_word_ratio(text),
        'lexical_diversity': lexical_diversity(text)
    }
    
    return features


def batch_calculate_readability_features(texts: List[str]) -> List[Dict[str, float]]:
    """
    Calculate readability features for a batch of texts.
    
    Args:
        texts: List of input texts to analyze
        
    Returns:
        List of dictionaries containing readability metrics for each text
    """
    results = []
    for text in texts:
        features = calculate_readability_features(text)
        results.append(features)
    
    return results


# Example usage and testing
if __name__ == "__main__":
    # Test with sample text
    sample_text = """
    This is a sample text for testing readability features. 
    It contains various sentence structures and vocabulary levels.
    The purpose is to demonstrate how different metrics work.
    """
    
    features = calculate_readability_features(sample_text)
    print("Readability Features:")
    for metric, value in features.items():
        print(f"{metric}: {value}")
