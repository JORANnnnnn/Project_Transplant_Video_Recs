"""
Integrated Feature Extraction System for YouTube Transcripts
Optimized for batch processing with shared resources and error handling
"""

import re
import math
import statistics
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Mapping of raw biomedical NER labels to high-level feature groups
MEDICAL_CATEGORY_MAP = {
    "DISEASE_DISORDER": "condition",
    "CLINICAL_EVENT": "condition",
    "OUTCOME": "condition",
    "HISTORY": "condition",
    "FAMILY_HISTORY": "condition",
    
    "SIGN_SYMPTOM": "symptom",
    
    "THERAPEUTIC_PROCEDURE": "therapy",
    
    "DIAGNOSTIC_PROCEDURE": "diagnostic",
    "LAB_VALUE": "diagnostic",
    
    "MEDICATION": "medication",
    "ADMINISTRATION": "medication",
    "DOSAGE": "medication",
    
    "BIOLOGICAL_STRUCTURE": "anatomy",
    
    "NONBIOLOGICAL_LOCATION": "location",
    
    "DURATION": "measurement",
    "FREQUENCY": "measurement",
    "QUANTITATIVE_CONCEPT": "measurement",
    "QUALITATIVE_CONCEPT": "measurement",
    "DATE": "measurement",
    "TIME": "measurement",
    "AGE": "measurement",
    "DISTANCE": "measurement",
    "AREA": "measurement",
    "VOLUME": "measurement",
    "HEIGHT": "measurement",
    "WEIGHT": "measurement",
    "MASS": "measurement",
    
    "DETAILED_DESCRIPTION": "context",
    "COREFERENCE": "context",
    "SUBJECT": "context",
    "PERSONAL_BACKGROUND": "context",
    "OTHER_ENTITY": "context",
    
    "ACTIVITY": "activity",
    "OCCUPATION": "activity",
    
    "NONBIOLOGICAL_LOCATION": "location",
    
    "OTHER_EVENT": "other",
    "COLOR": "other",
    "TEXTURE": "other",
    "SHAPE": "other",
    "SEVERITY": "other",
}

MEDICAL_CATEGORY_ORDER = [
    "condition",
    "symptom",
    "therapy",
    "diagnostic",
    "medication",
    "anatomy",
    "location",
    "measurement",
    "context",
    "activity",
    "other",
]

DEFAULT_CLUSTER_TOKEN_WINDOW = 80

# NLP libraries
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import cmudict
import spacy
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# Initialize NLTK data
for resource in ['tokenizers/punkt', 'corpora/cmudict']:
    try:
        nltk.data.find(resource)
    except LookupError:
        nltk.download(resource.split('/')[-1])


class FeatureExtractor:
    """
    Unified feature extractor with shared resources and batch processing support.
    """
    
    def __init__(self, load_medical_ner: bool = True, load_spacy: bool = True):
        """
        Initialize feature extractor with optional model loading.
        
        Args:
            load_medical_ner: Whether to load medical NER model (memory intensive)
            load_spacy: Whether to load spaCy model
        """
        print("🔧 Initializing Feature Extractor...")
        
        # Shared resources
        self.cmu_dict = cmudict.dict()
        
        # Load spaCy model for syntactic features
        self.nlp = None
        if load_spacy:
            try:
                self.nlp = spacy.load("en_core_web_sm")
                print("✅ Loaded spaCy model")
            except OSError:
                print("⚠️  spaCy model not found. Run: python -m spacy download en_core_web_sm")
        
        # Load medical NER model (optional, memory intensive)
        self.ner_pipeline = None
        self.label_doc_path = Path(__file__).with_name("medical_ner_labels.md")
        self.entity_counts: Counter = Counter()
        self.model_labels: List[str] = []
        
        if load_medical_ner:
            try:
                model_name = "d4data/biomedical-ner-all"
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForTokenClassification.from_pretrained(model_name)
                self.ner_pipeline = pipeline(
                    "ner", 
                    model=model, 
                    tokenizer=tokenizer,
                    aggregation_strategy="simple"
                )
                self._capture_model_labels(model)
                self._write_label_doc()
                print("✅ Loaded Medical NER model")
            except Exception as e:
                print(f"⚠️  Failed to load Medical NER model: {e}")
        
        print("✅ Feature Extractor initialized\n")
    
    # ==================== READABILITY FEATURES ====================
    
    def count_syllables(self, word: str) -> int:
        """Count syllables in a word using CMU dict with fallback."""
        word = word.lower().strip()
        
        if word in self.cmu_dict:
            phonemes = self.cmu_dict[word][0]
            return len([p for p in phonemes if p[-1].isdigit()])
        
        # Fallback: vowel counting
        vowels = 'aeiouy'
        word = re.sub(r'[^a-z]', '', word.lower())
        if not word:
            return 0
        
        vowel_groups = re.findall(r'[aeiouy]+', word)
        syllable_count = len(vowel_groups)
        
        if word.endswith('e') and syllable_count > 1:
            syllable_count -= 1
        
        if word.endswith('le') and len(word) > 2 and word[-3] not in vowels:
            syllable_count += 1
        
        return max(1, syllable_count)
    
    def extract_readability_features(self, text: str, 
                                    words: Optional[List[str]] = None,
                                    sentences: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Extract readability features with optional pre-tokenized input.
        
        Args:
            text: Input text
            words: Pre-tokenized words (optional, for efficiency)
            sentences: Pre-tokenized sentences (optional, for efficiency)
        
        Returns:
            Dictionary of readability metrics
        """
        # Use provided tokens or tokenize
        if sentences is None:
            sentences = sent_tokenize(text)
        if words is None:
            words = word_tokenize(text.lower())
        
        # Filter words
        words = [w for w in words if w.isalpha()]
        
        if not sentences or not words:
            return self._empty_readability_features()
        
        # Calculate metrics
        avg_sentence_length = len(words) / len(sentences)
        total_syllables = sum(self.count_syllables(w) for w in words)
        avg_syllables_per_word = total_syllables / len(words)
        
        # Flesch Reading Ease
        flesch_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
        
        # Flesch-Kincaid Grade Level
        fk_grade = (0.39 * avg_sentence_length) + (11.8 * avg_syllables_per_word) - 15.59
        
        # SMOG Index
        polysyllables = sum(1 for w in words if self.count_syllables(w) >= 3)
        sample_size = min(30, len(sentences))
        smog = 1.043 * math.sqrt(polysyllables * 30 / sample_size) + 3.1291
        
        # Gunning Fog Index
        complex_words = sum(1 for w in words 
                          if self.count_syllables(w) >= 3 
                          and not (w.endswith('ed') or w.endswith('ing')))
        complex_word_pct = (complex_words / len(words)) * 100
        gunning_fog = 0.4 * (avg_sentence_length + complex_word_pct)
        
        # Long word ratio
        long_word_ratio = sum(1 for w in words if self.count_syllables(w) >= 3) / len(words)
        
        # Lexical diversity (TTR)
        lexical_diversity = len(set(words)) / len(words)
        
        return {
            'flesch_reading_ease': round(flesch_score, 2),
            'flesch_kincaid_grade_level': round(fk_grade, 2),
            'smog_index': round(smog, 2),
            'gunning_fog_index': round(gunning_fog, 2),
            'average_word_length': round(avg_syllables_per_word, 3),
            'long_word_ratio': round(long_word_ratio, 3),
            'lexical_diversity': round(lexical_diversity, 3)
        }
    
    def _empty_readability_features(self) -> Dict[str, float]:
        """Return empty readability features."""
        return {
            'flesch_reading_ease': 0.0,
            'flesch_kincaid_grade_level': 0.0,
            'smog_index': 0.0,
            'gunning_fog_index': 0.0,
            'average_word_length': 0.0,
            'long_word_ratio': 0.0,
            'lexical_diversity': 0.0
        }
    
    # ==================== MEDICAL TERM FEATURES ====================
    
    def extract_medical_features(self, text: str, 
                                entities: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Extract medical terminology features.
        
        Args:
            text: Input text
            entities: Pre-extracted entities (optional)
        
        Returns:
            Dictionary of medical features
        """
        # Extract entities if not provided
        if entities is None:
            if self.ner_pipeline is None:
                return self._empty_medical_features()
            try:
                entities = self.ner_pipeline(text)
            except Exception as e:
                print(f"⚠️  NER extraction failed: {e}")
                return self._empty_medical_features()
        
        # Medical term density
        words = re.findall(r"\b\w+(?:'\w+)?\b", text)
        total_words = len(words)
        
        if not entities or total_words == 0:
            return self._empty_medical_features()
        
        # Normalize entity texts
        terms = [self._normalize_entity(e) for e in entities]
        terms = [t for t in terms if t]
        
        mention_count = len(terms)
        unique_count = len(set(t.lower() for t in terms))
        
        # Medical term density
        density = mention_count / total_words if total_words > 0 else 0.0
        
        # Term type distribution
        type_dist = self._term_type_distribution(entities)
        
        # Rare term ratio (document-level)
        rare_ratio = self._rare_term_ratio(entities)
        
        # Term clustering
        clustering = self._term_clustering(text, entities)
        
        # Entity group counts
        entity_group_counts = self._entity_group_counts(entities)
        
        self._update_entity_counts(entities)
        
        return {
            'medical_density': round(density, 4),
            'medical_mention_count': mention_count,
            'medical_unique_count': unique_count,
            **type_dist,
            'rare_term_ratio_unique': round(rare_ratio['ratio_unique'], 3),
            'rare_term_ratio_mentions': round(rare_ratio['ratio_mentions'], 3),
            'term_clustering_concentrated': clustering['concentrated'],
            'term_clustering_gini': round(clustering['gini'], 3),
            'term_clustering_max_share': round(clustering['max_share'], 3),
            'entity_group_counts': entity_group_counts
        }
    
    def _normalize_entity(self, e: Any) -> str:
        """Extract text from entity."""
        if isinstance(e, str):
            return e.strip()
        if isinstance(e, dict):
            return str(e.get('text') or e.get('word') or '').strip()
        return ''
    
    def _term_type_distribution(self, entities: List[Dict]) -> Dict[str, float]:
        """Calculate distribution of medical entity types using observed labels."""
        mapped = []
        for e in entities:
            label = str(e.get("entity_group", "")).upper()
            mapped.append(MEDICAL_CATEGORY_MAP.get(label, "other"))
        
        total = len(mapped)
        if total == 0:
            return {f"{cat}_ratio": 0.0 for cat in MEDICAL_CATEGORY_ORDER}
        
        counts = Counter(mapped)
        return {
            f"{cat}_ratio": round(counts.get(cat, 0) / total, 3)
            for cat in MEDICAL_CATEGORY_ORDER
        }
    
    def _rare_term_ratio(self, entities: List[Dict]) -> Dict[str, float]:
        """Calculate rare term ratio (document frequency = 1)."""
        terms = [self._normalize_entity(e).lower() for e in entities]
        terms = [t for t in terms if t]
        
        if not terms:
            return {'ratio_unique': 0.0, 'ratio_mentions': 0.0}
        
        df = Counter(terms)
        rare_terms = {t for t, c in df.items() if c == 1}
        
        ratio_unique = len(rare_terms) / len(df) if df else 0.0
        ratio_mentions = sum(1 for t in terms if t in rare_terms) / len(terms) if terms else 0.0
        
        return {'ratio_unique': ratio_unique, 'ratio_mentions': ratio_mentions}
    
    def _entity_group_counts(self, entities: List[Dict]) -> Dict[str, int]:
        """Count raw entity_group occurrences for downstream reporting."""
        counts = Counter()
        for e in entities:
            label = str(e.get("entity_group", "")).upper()
            if label:
                counts[label] += 1
        return dict(counts)
    
    def _term_clustering(self, text: str, entities: List[Dict]) -> Dict[str, Any]:
        """Analyze term clustering using fixed-size token windows."""
        tokens = [t for t in word_tokenize(text) if t.strip()]
        if len(tokens) < 2:
            return {'concentrated': False, 'gini': 0.0, 'max_share': 0.0}
        
        chunk_size = DEFAULT_CLUSTER_TOKEN_WINDOW
        if len(tokens) < chunk_size * 2:
            chunk_size = max(1, len(tokens) // 2)
        
        chunks = []
        for i in range(0, len(tokens), chunk_size):
            chunk = " ".join(tokens[i:i + chunk_size]).strip()
            if chunk:
                chunks.append(chunk)
        
        if len(chunks) < 2:
            return {'concentrated': False, 'gini': 0.0, 'max_share': 0.0}

        # Simple approach: count entities per paragraph (by text matching)
        counts = []
        for chunk in chunks:
            chunk_lower = chunk.lower()
            count = sum(1 for e in entities if self._normalize_entity(e).lower() in chunk_lower)
            counts.append(count)
        
        total = sum(counts)
        if total < 5:  # Too few entities to judge
            return {'concentrated': False, 'gini': 0.0, 'max_share': 0.0}
        
        props = [c / total for c in counts]
        max_share = max(props) if props else 0.0
        
        # Gini coefficient
        sorted_counts = sorted(counts)
        n = len(sorted_counts)
        cum = sum(sorted_counts)
        if cum == 0:
            gini = 0.0
        else:
            weighted_sum = sum((i + 1) * x for i, x in enumerate(sorted_counts))
            gini = (n + 1 - 2 * weighted_sum / cum) / n if n > 0 else 0.0
        
        concentrated = max_share >= 0.5 or gini >= 0.5
        
        return {
            'concentrated': concentrated,
            'gini': gini,
            'max_share': max_share
        }
    
    def _empty_medical_features(self) -> Dict[str, Any]:
        """Return empty medical features."""
        return {
            'medical_density': 0.0,
            'medical_mention_count': 0,
            'medical_unique_count': 0,
            **{f"{cat}_ratio": 0.0 for cat in MEDICAL_CATEGORY_ORDER},
            'rare_term_ratio_unique': 0.0,
            'rare_term_ratio_mentions': 0.0,
            'term_clustering_concentrated': False,
            'term_clustering_gini': 0.0,
            'term_clustering_max_share': 0.0,
            'entity_group_counts': {}
        }
    
    def _capture_model_labels(self, model) -> None:
        """Capture label list from model config for documentation."""
        self.model_labels = sorted({label.upper() for label in model.config.id2label.values()})
    
    def _update_entity_counts(self, entities: Optional[List[Dict]]) -> None:
        """Update observed entity-group frequencies and refresh documentation."""
        if not entities:
            return
        for ent in entities:
            label = str(ent.get("entity_group", "")).upper()
            if label:
                self.entity_counts[label] += 1
        self._write_label_doc()
    
    def _write_label_doc(self) -> None:
        """Persist label inventory and current frequency counts."""
        try:
            lines = [
                "# Biomedical NER Labels",
                "",
                "Model: d4data/biomedical-ner-all",
            ]
            if self.model_labels:
                lines.extend(["", "## entity_group values"])
                lines.extend(f"- {label}" for label in self.model_labels)
            if self.entity_counts:
                lines.extend(["", "## Observed entity_group counts", "| entity_group | count |", "| --- | ---: |"])
                for label, count in sorted(self.entity_counts.items(), key=lambda item: item[1], reverse=True):
                    lines.append(f"| {label} | {count} |")
            self.label_doc_path.write_text("\n".join(lines))
        except Exception as e:
            print(f"⚠️  Failed to update label documentation: {e}")
    
    # ==================== SYNTACTIC FEATURES ====================
    
    def extract_syntactic_features(self, text: str, doc=None) -> Dict[str, float]:
        """
        Extract syntactic complexity features.
        
        Args:
            text: Input text
            doc: Pre-processed spaCy doc (optional)
        
        Returns:
            Dictionary of syntactic features
        """
        if self.nlp is None:
            return self._empty_syntactic_features()
        
        try:
            if doc is None:
                doc = self.nlp(text)
            
            sentences = list(doc.sents)
            if not sentences:
                return self._empty_syntactic_features()
            
            # Passive voice ratio
            passive_count = sum(self._detect_passive(sent) for sent in sentences)
            passive_ratio = passive_count / len(sentences)
            
            # Sentence length statistics
            sent_lengths = [len([t for t in sent if not t.is_punct and not t.is_space]) 
                          for sent in sentences]
            avg_sent_len = statistics.mean(sent_lengths)
            std_sent_len = statistics.stdev(sent_lengths) if len(sent_lengths) > 1 else 0.0
            
            # Syntax tree depth
            tree_depth = self._avg_tree_depth(sentences)
            
            # Subordinate clause density
            sub_clause_density = self._subordinate_clause_density(sentences)
            
            # Average dependency distance
            avg_dep_dist = self._avg_dependency_distance(doc)
            
            # Pronoun frequency
            total_words = len([t for t in doc if not t.is_punct and not t.is_space])
            pronoun_count = len([t for t in doc if t.pos_ == "PRON"])
            pronoun_freq = pronoun_count / total_words if total_words > 0 else 0.0
            
            return {
                'passive_voice_ratio': round(passive_ratio, 3),
                'average_sentence_length': round(avg_sent_len, 2),
                'sentence_length_std': round(std_sent_len, 2),
                'syntax_tree_depth': round(tree_depth, 2),
                'subordinate_clause_density': round(sub_clause_density, 3),
                'average_dependency_distance': round(avg_dep_dist, 2),
                'pronoun_frequency': round(pronoun_freq, 3)
            }
        
        except Exception as e:
            print(f"⚠️  Syntactic feature extraction failed: {e}")
            return self._empty_syntactic_features()
    
    def _detect_passive(self, sent) -> int:
        """Detect passive voice constructions."""
        count = 0
        for token in sent:
            if (token.dep_ == "auxpass" or 
                (token.dep_ == "aux" and token.tag_ == "VBN" and 
                 any(child.dep_ == "auxpass" for child in token.children))):
                count += 1
        return count
    
    def _avg_tree_depth(self, sentences) -> float:
        """Calculate average syntax tree depth."""
        def get_depth(token, visited=None):
            if visited is None:
                visited = set()
            if token in visited:
                return 0
            visited.add(token)
            if not list(token.children):
                return 1
            return 1 + max(get_depth(c, visited.copy()) for c in token.children)
        
        depths = []
        for sent in sentences:
            root = next((t for t in sent if t.dep_ == "ROOT"), None)
            if root:
                depths.append(get_depth(root))
        
        return statistics.mean(depths) if depths else 0.0
    
    def _subordinate_clause_density(self, sentences) -> float:
        """Calculate subordinate clause density."""
        sub_clauses = 0
        total_clauses = 0
        
        for sent in sentences:
            for token in sent:
                if token.dep_ in ["mark", "relcl"] or token.tag_ in ["WDT", "WP", "WP$", "WRB"]:
                    sub_clauses += 1
            main_clauses = len([t for t in sent if t.dep_ == "ROOT"])
            total_clauses += main_clauses + sub_clauses
        
        return sub_clauses / total_clauses if total_clauses > 0 else 0.0
    
    def _avg_dependency_distance(self, doc) -> float:
        """Calculate average dependency distance."""
        distances = []
        for token in doc:
            if token.dep_ != "ROOT" and token.head != token:
                distances.append(abs(token.i - token.head.i))
        
        return statistics.mean(distances) if distances else 0.0
    
    def _empty_syntactic_features(self) -> Dict[str, float]:
        """Return empty syntactic features."""
        return {
            'passive_voice_ratio': 0.0,
            'average_sentence_length': 0.0,
            'sentence_length_std': 0.0,
            'syntax_tree_depth': 0.0,
            'subordinate_clause_density': 0.0,
            'average_dependency_distance': 0.0,
            'pronoun_frequency': 0.0
        }
    
    # ==================== UNIFIED EXTRACTION ====================
    
    def extract_all_features(self, text: str, 
                            include_medical: bool = True,
                            include_syntactic: bool = True) -> Dict[str, Any]:
        """
        Extract all features from text with shared preprocessing.
        
        Args:
            text: Input text
            include_medical: Whether to extract medical features
            include_syntactic: Whether to extract syntactic features
        
        Returns:
            Dictionary of all features
        """
        if not text or not isinstance(text, str) or len(text.strip()) < 10:
            return self._empty_all_features()
        
        features = {}
        
        # Pre-tokenize for efficiency
        sentences = sent_tokenize(text)
        words = word_tokenize(text.lower())
        
        # 1. Readability features (fast)
        features.update(self.extract_readability_features(text, words, sentences))
        
        # 2. Medical features (slow, optional)
        if include_medical and self.ner_pipeline is not None:
            try:
                medical_features = self.extract_medical_features(text)
                features.update(medical_features)
            except Exception as e:
                print(f"⚠️  Medical extraction failed: {e}")
                features.update(self._empty_medical_features())
        elif include_medical:
            features.update(self._empty_medical_features())
        
        # 3. Syntactic features (medium speed)
        if include_syntactic and self.nlp is not None:
            try:
                doc = self.nlp(text)
                syntactic_features = self.extract_syntactic_features(text, doc)
                features.update(syntactic_features)
            except Exception as e:
                print(f"⚠️  Syntactic extraction failed: {e}")
                features.update(self._empty_syntactic_features())
        elif include_syntactic:
            features.update(self._empty_syntactic_features())
        
        return features
    
    def _empty_all_features(self) -> Dict[str, Any]:
        """Return all empty features."""
        return {
            **self._empty_readability_features(),
            **self._empty_medical_features(),
            **self._empty_syntactic_features()
        }


# Convenience function
def create_extractor(load_medical_ner: bool = True, 
                    load_spacy: bool = True) -> FeatureExtractor:
    """
    Create and return a FeatureExtractor instance.
    
    Args:
        load_medical_ner: Whether to load medical NER model (memory intensive)
        load_spacy: Whether to load spaCy model
    
    Returns:
        FeatureExtractor instance
    """
    return FeatureExtractor(load_medical_ner=load_medical_ner, load_spacy=load_spacy)
