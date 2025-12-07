"""
Syntactic Complexity Feature Engineering for YouTube Video Transcripts
This module implements syntactic complexity metrics using dependency parsing.
Features align with PEMAT standards for language complexity assessment.

Core Features:
1. Passive Voice Ratio: Detection of passive sentence structures
2. Average Sentence Length: Word count / sentence count
3. Sentence Length Standard Deviation: Variation in sentence length
4. Syntax Tree Depth: Nesting levels of sentence structure
5. Subordinate Clause Density: Proportion of compound and complex sentences
6. Average Dependency Distance: Syntactic distance between words
7. Pronoun Usage Frequency: Clarity of referential expressions
"""

import re
import math
import statistics
from typing import List, Dict, Any, Tuple
from collections import Counter
import spacy
from spacy import displacy
import pandas as pd
import numpy as np

# Load spaCy model (English)
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Please install the English spaCy model: python -m spacy download en_core_web_sm")
    nlp = None


def detect_passive_voice(doc) -> int:
    """
    Detect passive voice constructions in a spaCy document.
    
    Args:
        doc: spaCy document object
        
    Returns:
        Number of passive voice constructions
    """
    passive_count = 0
    
    for token in doc:
        # Check for passive voice indicators
        if (token.dep_ == "auxpass" or  # auxiliary passive
            (token.dep_ == "aux" and token.tag_ == "VBN" and 
             any(child.dep_ == "auxpass" for child in token.children))):
            passive_count += 1
    
    return passive_count


def calculate_passive_voice_ratio(text: str) -> float:
    """
    Calculate the ratio of passive voice constructions to total sentences.
    
    Args:
        text: Input text to analyze
        
    Returns:
        Passive voice ratio (0-1)
    """
    if not nlp:
        return 0.0
    
    doc = nlp(text)
    sentences = list(doc.sents)
    
    if len(sentences) == 0:
        return 0.0
    
    total_passive = sum(detect_passive_voice(sent) for sent in sentences)
    ratio = total_passive / len(sentences)
    
    return round(ratio, 3)


def calculate_average_sentence_length(text: str) -> float:
    """
    Calculate average sentence length in words.
    
    Args:
        text: Input text to analyze
        
    Returns:
        Average sentence length
    """
    if not nlp:
        return 0.0
    
    doc = nlp(text)
    sentences = list(doc.sents)
    
    if len(sentences) == 0:
        return 0.0
    
    sentence_lengths = [len([token for token in sent if not token.is_punct and not token.is_space]) 
                       for sent in sentences]
    
    avg_length = statistics.mean(sentence_lengths)
    return round(avg_length, 2)


def calculate_sentence_length_std(text: str) -> float:
    """
    Calculate standard deviation of sentence lengths.
    
    Args:
        text: Input text to analyze
        
    Returns:
        Standard deviation of sentence lengths
    """
    if not nlp:
        return 0.0
    
    doc = nlp(text)
    sentences = list(doc.sents)
    
    if len(sentences) < 2:
        return 0.0
    
    sentence_lengths = [len([token for token in sent if not token.is_punct and not token.is_space]) 
                       for sent in sentences]
    
    std_dev = statistics.stdev(sentence_lengths)
    return round(std_dev, 2)


def calculate_syntax_tree_depth(doc) -> float:
    """
    Calculate average syntax tree depth for a document.
    
    Args:
        doc: spaCy document object
        
    Returns:
        Average syntax tree depth
    """
    def get_tree_depth(token, visited=None):
        if visited is None:
            visited = set()
        
        if token in visited:
            return 0
        
        visited.add(token)
        
        if not list(token.children):
            return 1
        
        max_child_depth = max(get_tree_depth(child, visited.copy()) for child in token.children)
        return 1 + max_child_depth
    
    sentences = list(doc.sents)
    if not sentences:
        return 0.0
    
    total_depth = 0
    total_sentences = 0
    
    for sent in sentences:
        # Find root of the sentence
        root = None
        for token in sent:
            if token.dep_ == "ROOT":
                root = token
                break
        
        if root:
            depth = get_tree_depth(root)
            total_depth += depth
            total_sentences += 1
    
    if total_sentences == 0:
        return 0.0
    
    avg_depth = total_depth / total_sentences
    return round(avg_depth, 2)


def calculate_subordinate_clause_density(text: str) -> float:
    """
    Calculate the density of subordinate clauses.
    
    Args:
        text: Input text to analyze
        
    Returns:
        Subordinate clause density (0-1)
    """
    if not nlp:
        return 0.0
    
    doc = nlp(text)
    sentences = list(doc.sents)
    
    if len(sentences) == 0:
        return 0.0
    
    subordinate_clauses = 0
    total_clauses = 0
    
    for sent in sentences:
        # Count subordinate conjunctions and relative pronouns
        for token in sent:
            if (token.dep_ in ["mark", "relcl"] or  # subordinating conjunctions, relative clauses
                token.tag_ in ["WDT", "WP", "WP$", "WRB"]):  # wh-words that introduce clauses
                subordinate_clauses += 1
        
        # Count main clauses (approximate)
        main_clauses = len([token for token in sent if token.dep_ == "ROOT"])
        total_clauses += main_clauses + subordinate_clauses
    
    if total_clauses == 0:
        return 0.0
    
    density = subordinate_clauses / total_clauses
    return round(density, 3)


def calculate_average_dependency_distance(text: str) -> float:
    """
    Calculate average dependency distance between syntactically related words.
    
    Args:
        text: Input text to analyze
        
    Returns:
        Average dependency distance
    """
    if not nlp:
        return 0.0
    
    doc = nlp(text)
    total_distance = 0
    total_relations = 0
    
    for token in doc:
        if token.dep_ != "ROOT" and token.head != token:
            # Calculate distance between token and its head
            distance = abs(token.i - token.head.i)
            total_distance += distance
            total_relations += 1
    
    if total_relations == 0:
        return 0.0
    
    avg_distance = total_distance / total_relations
    return round(avg_distance, 2)


def calculate_pronoun_frequency(text: str) -> float:
    """
    Calculate frequency of pronoun usage.
    
    Args:
        text: Input text to analyze
        
    Returns:
        Pronoun frequency (0-1)
    """
    if not nlp:
        return 0.0
    
    doc = nlp(text)
    total_words = len([token for token in doc if not token.is_punct and not token.is_space])
    
    if total_words == 0:
        return 0.0
    
    pronoun_count = len([token for token in doc if token.pos_ == "PRON"])
    frequency = pronoun_count / total_words
    
    return round(frequency, 3)


def calculate_syntactic_complexity_features(text: str) -> Dict[str, float]:
    """
    Calculate all syntactic complexity features for a given text.
    
    Args:
        text: Input text to analyze
        
    Returns:
        Dictionary containing all syntactic complexity metrics
    """
    if not nlp:
        return {
            'passive_voice_ratio': 0.0,
            'average_sentence_length': 0.0,
            'sentence_length_std': 0.0,
            'syntax_tree_depth': 0.0,
            'subordinate_clause_density': 0.0,
            'average_dependency_distance': 0.0,
            'pronoun_frequency': 0.0
        }
    
    doc = nlp(text)
    
    features = {
        'passive_voice_ratio': calculate_passive_voice_ratio(text),
        'average_sentence_length': calculate_average_sentence_length(text),
        'sentence_length_std': calculate_sentence_length_std(text),
        'syntax_tree_depth': calculate_syntax_tree_depth(doc),
        'subordinate_clause_density': calculate_subordinate_clause_density(text),
        'average_dependency_distance': calculate_average_dependency_distance(text),
        'pronoun_frequency': calculate_pronoun_frequency(text)
    }
    
    return features


def batch_calculate_syntactic_features(texts: List[str]) -> List[Dict[str, float]]:
    """
    Calculate syntactic complexity features for a batch of texts.
    
    Args:
        texts: List of input texts to analyze
        
    Returns:
        List of dictionaries containing syntactic complexity metrics for each text
    """
    results = []
    for text in texts:
        features = calculate_syntactic_complexity_features(text)
        results.append(features)
    
    return results


def display_dependency_info(text: str) -> None:
    """
    Display dependency information as text (no visualization).
    
    Args:
        text: Input text to analyze
    """
    if not nlp:
        print("spaCy model not available")
        return
    
    doc = nlp(text)
    
    print("\nDependency Analysis:")
    print("=" * 50)
    print(f"Text: '{text}'")
    print("\nToken Dependencies:")
    print("-" * 50)
    for token in doc:
        print(f"{token.text:15} -> {token.head.text:15} ({token.dep_})")
    
    print("\nSentence Structure:")
    print("-" * 50)
    for sent in doc.sents:
        print(f"Sentence: {sent.text}")
        for token in sent:
            if token.dep_ == "ROOT":
                print(f"  Root: {token.text} ({token.pos_})")
                break


def visualize_dependency_tree(text: str, save_path: str = None) -> None:
    """
    Visualize dependency tree for a sentence.
    
    Args:
        text: Input text to visualize
        save_path: Optional path to save the visualization
    """
    if not nlp:
        print("spaCy model not available for visualization")
        return
    
    doc = nlp(text)
    
    try:
        # Try to create dependency tree visualization with Jupyter display
        displacy.render(doc, style="dep", jupyter=True, options={
            'compact': True,
            'distance': 100,
            'arrow_stroke': 2,
            'arrow_width': 8,
            'arrow_spacing': 20,
            'word_spacing': 45,
            'font': 'Arial'
        })
    except ImportError as e:
        print(f"Jupyter display not available: {e}")
        print("Creating HTML visualization instead...")
        
        # Fallback: create HTML and save to file
        html = displacy.render(doc, style="dep", page=True, options={
            'compact': True,
            'distance': 100,
            'arrow_stroke': 2,
            'arrow_width': 8,
            'arrow_spacing': 20,
            'word_spacing': 45,
            'font': 'Arial'
        })
        
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(html)
            print(f"Visualization saved to: {save_path}")
        else:
            # Save to temporary file
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as f:
                f.write(html)
                print(f"Visualization saved to temporary file: {f.name}")
                print("Open this file in a web browser to view the dependency tree.")
    
    except Exception as e:
        print(f"Error creating visualization: {e}")
        print("Displaying dependency information as text instead...")
        
        # Fallback: display dependency information as text
        print("\nDependency Analysis:")
        print("=" * 50)
        for token in doc:
            print(f"{token.text:15} -> {token.head.text:15} ({token.dep_})")


# Example usage and testing
if __name__ == "__main__":
    # Test with sample text
    sample_text = """
    The implementation of sophisticated pharmacological interventions necessitates comprehensive understanding. 
    These procedures are performed by experienced surgeons who have been trained extensively.
    When the patient arrives, they will be examined thoroughly before any treatment begins.
    """
    
    features = calculate_syntactic_complexity_features(sample_text)
    print("Syntactic Complexity Features:")
    for metric, value in features.items():
        print(f"{metric}: {value}")

