# Integrated Features Documentation

## Overview

The Integrated Feature Extraction System extracts three categories of linguistic and medical features from YouTube transcripts:

1. **Readability Features** - Measures text complexity and accessibility
2. **Medical Term Features** - Analyzes biomedical terminology and entity distributions
3. **Syntactic Features** - Examines grammatical complexity and structure

The system is optimized for batch processing with shared resources and comprehensive error handling.

---

## Models and Dependencies

### Core NLP Libraries

#### 1. **NLTK (Natural Language Toolkit)**
- **Purpose**: Tokenization and syllable counting
- **Resources Used**:
  - `tokenizers/punkt`: Sentence and word tokenization
  - `corpora/cmudict`: CMU Pronouncing Dictionary for accurate syllable counting
- **Auto-download**: Resources are automatically downloaded if missing

#### 2. **spaCy**
- **Model**: `en_core_web_sm` (English small model)
- **Purpose**: Syntactic parsing, dependency analysis, and part-of-speech tagging
- **Installation**: `python -m spacy download en_core_web_sm`
- **Features Extracted**:
  - Dependency parsing
  - Sentence segmentation
  - Part-of-speech tagging
  - Syntactic tree structures

#### 3. **Transformers (Hugging Face)**
- **Model**: `d4data/biomedical-ner-all`
- **Purpose**: Biomedical Named Entity Recognition (NER)
- **Type**: Token classification model
- **Aggregation Strategy**: `simple` (merges adjacent tokens with same label)
- **Memory Requirements**: High (model is large and memory-intensive)
- **Optional**: Can be disabled for faster processing without medical features

### Medical Category Mapping

The system maps raw biomedical NER labels to 11 high-level categories:

| High-Level Category | Raw NER Labels |
|-------------------|----------------|
| **condition** | DISEASE_DISORDER, CLINICAL_EVENT, OUTCOME, HISTORY, FAMILY_HISTORY |
| **symptom** | SIGN_SYMPTOM |
| **therapy** | THERAPEUTIC_PROCEDURE |
| **diagnostic** | DIAGNOSTIC_PROCEDURE, LAB_VALUE |
| **medication** | MEDICATION, ADMINISTRATION, DOSAGE |
| **anatomy** | BIOLOGICAL_STRUCTURE |
| **location** | NONBIOLOGICAL_LOCATION |
| **measurement** | DURATION, FREQUENCY, QUANTITATIVE_CONCEPT, QUALITATIVE_CONCEPT, DATE, TIME, AGE, DISTANCE, AREA, VOLUME, HEIGHT, WEIGHT, MASS |
| **context** | DETAILED_DESCRIPTION, COREFERENCE, SUBJECT, PERSONAL_BACKGROUND, OTHER_ENTITY |
| **activity** | ACTIVITY, OCCUPATION |
| **other** | OTHER_EVENT, COLOR, TEXTURE, SHAPE, SEVERITY |

---

## Feature Definitions and Measurements

### 1. Readability Features

Readability features assess how easy or difficult a text is to read and understand.

#### 1.1 Flesch Reading Ease (`flesch_reading_ease`)
- **Definition**: Measures readability on a 0-100 scale (higher = easier to read)
- **Formula**: `206.835 - (1.015 × ASL) - (84.6 × ASW)`
  - `ASL` = Average Sentence Length (words per sentence)
  - `ASW` = Average Syllables per Word
- **Interpretation**:
  - 90-100: Very easy (5th grade level)
  - 80-89: Easy (6th grade level)
  - 70-79: Fairly easy (7th grade level)
  - 60-69: Standard (8th-9th grade level)
  - 50-59: Fairly difficult (10th-12th grade level)
  - 30-49: Difficult (college level)
  - 0-29: Very difficult (college graduate level)
- **Range**: Typically 0-100 (can go negative for very complex texts)

#### 1.2 Flesch-Kincaid Grade Level (`flesch_kincaid_grade_level`)
- **Definition**: Estimates the U.S. school grade level needed to understand the text
- **Formula**: `(0.39 × ASL) + (11.8 × ASW) - 15.59`
- **Interpretation**: 
  - 0-5: Elementary school
  - 6-8: Middle school
  - 9-12: High school
  - 13+: College/graduate level
- **Range**: Typically 0-20+ (can be negative for very simple texts)

#### 1.3 SMOG Index (`smog_index`)
- **Definition**: Simple Measure of Gobbledygook - estimates years of education needed
- **Formula**: `1.043 × √(polysyllables × 30 / sample_size) + 3.1291`
  - `polysyllables` = Words with 3+ syllables
  - `sample_size` = min(30, number of sentences)
- **Interpretation**: Approximate grade level required
- **Range**: Typically 3-20+

#### 1.4 Gunning Fog Index (`gunning_fog_index`)
- **Definition**: Estimates years of formal education needed to understand text
- **Formula**: `0.4 × (ASL + ComplexWordPercentage)`
  - `ComplexWordPercentage` = Percentage of words with 3+ syllables (excluding words ending in -ed or -ing)
- **Interpretation**: Grade level equivalent
- **Range**: Typically 0-20+

#### 1.5 Average Word Length (`average_word_length`)
- **Definition**: Average number of syllables per word
- **Formula**: `Total Syllables / Total Words`
- **Interpretation**: 
  - < 1.5: Short words (simple vocabulary)
  - 1.5-2.0: Moderate
  - > 2.0: Long words (complex vocabulary)
- **Range**: Typically 1.0-3.0+

#### 1.6 Long Word Ratio (`long_word_ratio`)
- **Definition**: Proportion of words with 3 or more syllables
- **Formula**: `Count(words with ≥3 syllables) / Total Words`
- **Interpretation**: 
  - < 0.1: Few complex words
  - 0.1-0.2: Moderate complexity
  - > 0.2: Many complex words
- **Range**: 0.0-1.0

#### 1.7 Lexical Diversity (`lexical_diversity`)
- **Definition**: Type-Token Ratio (TTR) - measures vocabulary richness
- **Formula**: `Unique Words / Total Words`
- **Interpretation**:
  - < 0.5: Low diversity (repetitive vocabulary)
  - 0.5-0.7: Moderate diversity
  - > 0.7: High diversity (rich vocabulary)
- **Range**: 0.0-1.0

#### Syllable Counting Method
- **Primary**: Uses CMU Pronouncing Dictionary for accurate syllable counts
- **Fallback**: Vowel-based heuristic when word not in dictionary:
  1. Count vowel groups (consecutive vowels)
  2. Subtract 1 if word ends in 'e' (silent e)
  3. Add 1 if word ends in 'le' preceded by consonant
  4. Minimum 1 syllable per word

---

### 2. Medical Term Features

Medical features analyze biomedical terminology extracted from cleaned transcript text using the biomedical NER model.

#### 2.1 Medical Term Density (`medical_density`)
- **Definition**: Ratio of medical entity mentions to the total word count in the cleaned transcript.
- **Formula**: `Medical Entity Mentions / Total Words`
- **Interpretation**:
  - < 0.01: Low medical content
  - 0.01–0.05: Moderate medical content
  - > 0.05: High medical content
- **Range**: 0.0–1.0

#### 2.2 Medical Mention Count (`medical_mention_count`)
- **Definition**: Total number of medical entity mentions, including duplicates.
- **Type**: Integer
- **Interpretation**: Raw count of all biomedical entities detected.

#### 2.3 Medical Unique Count (`medical_unique_count`)
- **Definition**: Number of unique medical terms (case-insensitive).
- **Type**: Integer
- **Interpretation**: Represents the variety of biomedical terminology in the transcript.

#### 2.4 Category Ratio Features (`{category}_ratio`)
- **Definition**: Proportion of medical entities mapped into each high-level biomedical category.
- **Categories**: `condition`, `symptom`, `therapy`, `diagnostic`, `medication`, `anatomy`, `location`, `measurement`, `context`, `activity`, `other`
- **Formula**: `Count(category entities) / medical_mention_count`
- **Range**: 0.0–1.0
- **Sum**: Ratios sum to 1.0 when entities are present.

#### 2.5 Rare Term Ratio – Unique (`rare_term_ratio_unique`)
- **Definition**: Fraction of unique medical terms that appear only once in the transcript.
- **Formula**: `# Unique Terms with Frequency = 1 / medical_unique_count`
- **Interpretation**:
  - High: Diverse, specialized vocabulary
  - Low: Frequently repeated terminology
- **Range**: 0.0–1.0

#### 2.6 Rare Term Ratio – Mentions (`rare_term_ratio_mentions`)
- **Definition**: Fraction of all medical mentions that correspond to rare (frequency-1) terms.
- **Formula**: `# Mentions of Rare Terms / medical_mention_count`
- **Interpretation**: Indicates how much of the medical content comes from non-repeated terms.
- **Range**: 0.0–1.0

#### 2.7 Term Clustering Features

Medical term clustering is computed by dividing the cleaned transcript into **5 equal-length character segments** and assigning each entity by its character offset.

##### 2.7.1 Concentrated (`term_clustering_concentrated`)
- **Definition**: Boolean flag indicating whether medical terms are concentrated in specific transcript regions.
- **Criteria**: `True` if `term_clustering_max_share ≥ 0.30`
- **Interpretation**:
  - `True`: Clear clustering in specific regions
  - `False`: Distributed across the transcript

##### 2.7.2 Gini Coefficient (`term_clustering_gini`)
- **Definition**: Measures inequality in entity distribution across the 5 segments.
- **Range**: 0.0–1.0
- **Interpretation**:
  - 0.0: Evenly distributed
  - > 0.5: Strong clustering

##### 2.7.3 Maximum Share (`term_clustering_max_share`)
- **Definition**: Fraction of all medical mentions occurring in the most entity-dense segment.
- **Formula**: `Max(segment entity count) / medical_mention_count`
- **Range**: 0.0–1.0

#### 2.8 Entity Group Counts (`entity_group_counts`)
- **Definition**: Dictionary mapping raw NER model labels (e.g., `DISEASE_DISORDER`, `MEDICATION`) to their occurrence counts.
- **Type**: Dictionary[str, int]
- **Purpose**: Preserves raw label information for downstream analysis.

---

### 3. Syntactic Features

Syntactic features measure grammatical complexity and sentence structure.

#### 3.1 Passive Voice Ratio (`passive_voice_ratio`)
- **Definition**: Proportion of sentences containing passive voice constructions
- **Detection Method**: 
  - Identifies `auxpass` (passive auxiliary) dependencies
  - Detects auxiliary verbs with past participle tags (`VBN`) and passive children
- **Formula**: `Sentences with Passive Voice / Total Sentences`
- **Interpretation**:
  - < 0.1: Mostly active voice (direct, clear)
  - 0.1-0.3: Moderate passive usage
  - > 0.3: High passive usage (formal, academic style)
- **Range**: 0.0-1.0

#### 3.2 Average Sentence Length (`average_sentence_length`)
- **Definition**: Mean number of non-punctuation, non-space tokens per sentence
- **Formula**: `Σ(Sentence Lengths) / Number of Sentences`
- **Interpretation**:
  - < 10: Very short sentences (simple)
  - 10-20: Moderate length
  - > 20: Long sentences (complex)
- **Range**: Typically 5-50+

#### 3.3 Sentence Length Standard Deviation (`sentence_length_std`)
- **Definition**: Standard deviation of sentence lengths
- **Formula**: Standard deviation of sentence token counts
- **Interpretation**:
  - Low: Consistent sentence length (uniform style)
  - High: Variable sentence length (varied style, may include complex structures)
- **Range**: 0.0+

#### 3.4 Syntax Tree Depth (`syntax_tree_depth`)
- **Definition**: Average depth of dependency parse trees
- **Calculation**: 
  - For each sentence, finds the root token
  - Calculates maximum depth from root to any leaf
  - Averages across all sentences
- **Interpretation**:
  - < 3: Shallow trees (simple structures)
  - 3-5: Moderate depth
  - > 5: Deep trees (complex nested structures)
- **Range**: Typically 2-10+

#### 3.5 Subordinate Clause Density (`subordinate_clause_density`)
- **Definition**: Ratio of subordinate clauses to total clauses
- **Detection Method**: 
  - Identifies subordinating markers (`mark`, `relcl` dependencies)
  - Counts relative pronouns and subordinating conjunctions (`WDT`, `WP`, `WP$`, `WRB` tags)
- **Formula**: `Subordinate Clauses / (Main Clauses + Subordinate Clauses)`
- **Interpretation**:
  - < 0.2: Few subordinate clauses (simple sentences)
  - 0.2-0.4: Moderate subordination
  - > 0.4: High subordination (complex sentences)
- **Range**: 0.0-1.0

#### 3.6 Average Dependency Distance (`average_dependency_distance`)
- **Definition**: Mean linear distance between words and their syntactic heads
- **Formula**: `Mean(|token_index - head_index|)` for all non-root tokens
- **Interpretation**:
  - Low (< 2): Words close to their heads (local dependencies, simpler)
  - Moderate (2-4): Standard dependency patterns
  - High (> 4): Long-distance dependencies (complex structures)
- **Range**: Typically 1.0-10+

#### 3.7 Pronoun Frequency (`pronoun_frequency`)
- **Definition**: Proportion of tokens that are pronouns
- **Formula**: `Pronoun Tokens / Total Non-punctuation Tokens`
- **Interpretation**:
  - < 0.05: Few pronouns (formal, specific)
  - 0.05-0.15: Moderate pronoun usage
  - > 0.15: High pronoun usage (conversational, personal)
- **Range**: 0.0-1.0

---

## Implementation Details

### Processing Pipeline

1. **Text Preprocessing**:
   - Sentence tokenization (NLTK)
   - Word tokenization (NLTK)
   - Filtering of non-alphabetic words for readability metrics

2. **Feature Extraction Order**:
   - Readability features (fastest, no external models)
   - Medical features (slowest, requires NER model)
   - Syntactic features (medium speed, requires spaCy)

3. **Shared Resources**:
   - CMU Dictionary loaded once and reused
   - spaCy model loaded once and reused
   - NER pipeline loaded once and reused
   - Entity counts tracked across documents

### Error Handling

- **Empty Text**: Returns zero-filled feature dictionaries
- **Missing Models**: Returns empty features for that category
- **NER Failures**: Catches exceptions and returns empty medical features
- **SpaCy Failures**: Catches exceptions and returns empty syntactic features

### Performance Considerations

- **Medical NER**: Most memory-intensive and slowest component
  - Can be disabled: `load_medical_ner=False`
- **SpaCy**: Moderate memory usage, faster than NER
  - Can be disabled: `load_spacy=False`
- **Readability**: Fastest, no model loading required
- **Batch Processing**: Pre-tokenization and shared model instances optimize throughput

### Configuration Parameters

- `DEFAULT_CLUSTER_TOKEN_WINDOW = 80`: Token window size for term clustering analysis
- `MEDICAL_CATEGORY_MAP`: Mapping from raw NER labels to high-level categories
- `MEDICAL_CATEGORY_ORDER`: Ordered list of categories for consistent feature output

### Output Format

All features are returned as a single dictionary with:
- Float values rounded to 2-4 decimal places
- Integer counts for mention/unique counts
- Boolean values for concentrated clustering
- Dictionary for entity_group_counts
- Zero-filled defaults when extraction fails

---

## Usage Notes

### Initialization

```python
from integrated_features import FeatureExtractor

# Full initialization (all models)
extractor = FeatureExtractor(load_medical_ner=True, load_spacy=True)

# Fast mode (no medical NER)
extractor = FeatureExtractor(load_medical_ner=False, load_spacy=True)

# Minimal mode (readability only)
extractor = FeatureExtractor(load_medical_ner=False, load_spacy=False)
```

### Feature Extraction

```python
# Extract all features
features = extractor.extract_all_features(
    text="Your transcript text here...",
    include_medical=True,
    include_syntactic=True
)

# Extract specific feature categories
readability = extractor.extract_readability_features(text)
medical = extractor.extract_medical_features(text)
syntactic = extractor.extract_syntactic_features(text)
```

### Model Requirements

- **spaCy**: Requires `en_core_web_sm` model
  - Install: `python -m spacy download en_core_web_sm`
- **Medical NER**: Automatically downloads from Hugging Face
  - Model: `d4data/biomedical-ner-all`
  - Requires internet connection for first download
- **NLTK**: Auto-downloads required resources

### Memory and Performance

- **Medical NER Model**: ~500MB-1GB RAM
- **SpaCy Model**: ~50-100MB RAM
- **Processing Speed**: 
  - Readability: ~1000-5000 words/second
  - Syntactic: ~500-2000 words/second
  - Medical: ~50-200 words/second (depends on entity density)

---

## Feature Summary Table

| Feature Category | Feature Count | Model Required | Speed | Memory Impact |
|-----------------|---------------|----------------|-------|---------------|
| Readability | 7 | None | Fast | Low |
| Medical | 18+ | NER Model | Slow | High |
| Syntactic | 7 | spaCy | Medium | Medium |
| **Total** | **32+** | - | - | - |

---

## References

- **Flesch Reading Ease**: Flesch, R. (1948). "A new readability yardstick." Journal of Applied Psychology.
- **Flesch-Kincaid**: Kincaid, J.P., et al. (1975). "Derivation of new readability formulas." Research Branch Report.
- **SMOG Index**: McLaughlin, G.H. (1969). "SMOG grading: A new readability formula." Journal of Reading.
- **Gunning Fog**: Gunning, R. (1952). "The Technique of Clear Writing." McGraw-Hill.
- **Biomedical NER**: `d4data/biomedical-ner-all` - Hugging Face Model Hub
- **spaCy**: Explosion AI - https://spacy.io/
- **NLTK**: Natural Language Toolkit - https://www.nltk.org/

---

*Last Updated: Based on integrated_features.py implementation*

