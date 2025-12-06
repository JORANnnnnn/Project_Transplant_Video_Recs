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

# ==================== MEDICAL TERM CONSTANTS AND HELPERS ====================

# High level medical categories for aggregation
CATEGORIES = [
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

# Keep this alias for compatibility with any code that still references
# MEDICAL_CATEGORY_ORDER
MEDICAL_CATEGORY_ORDER = CATEGORIES

# Simple cleaning rules for YouTube captions
WS = re.compile(r"[ \t\u00A0]+")
LINE_PREFIX = re.compile(r"^\s*(?:>>\s*)+", flags=re.MULTILINE)
TIMESTAMPS = re.compile(r"\b(?:\d{1,2}:)?\d{2}:\d{2}\b")
BRACKETED = re.compile(
    r"\[(?:music|applause|laughter|silence|inaudible)\]",
    flags=re.IGNORECASE,
)


def clean_transcript(text: str) -> str:
    """
    Clean YouTube captions by removing prefixes, timestamps, and bracketed tags.

    This is used only inside the medical feature extractor so that NER
    and density estimates are not confused by caption artifacts.
    """
    if not isinstance(text, str):
        return ""
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    t = LINE_PREFIX.sub("", t)
    t = TIMESTAMPS.sub("", t)
    t = BRACKETED.sub("", t)
    t = WS.sub(" ", t)
    t = re.sub(r"\s*\n\s*", " ", t)
    return t.strip()


def map_raw_label_to_category(raw_label: str) -> str:
    """
    Map a raw NER label (for example entity_group from the HF pipeline)
    to one of the 11 high level medical categories.

    This mapping uses simple substring rules so that it is robust to
    minor changes in label names.
    """
    if not raw_label:
        return "other"

    key = str(raw_label).upper()

    if "CONDITION" in key or "DISEASE" in key:
        return "condition"
    if "SYMPTOM" in key or "SIGN" in key:
        return "symptom"
    if "THERAPY" in key or "TREATMENT" in key or "PROCEDURE" in key:
        return "therapy"
    if "TEST" in key or "EXAM" in key or "DIAGNOSTIC" in key:
        return "diagnostic"
    if "DRUG" in key or "MEDICATION" in key or "CHEM" in key:
        return "medication"
    if "ANATOMY" in key or "BODY" in key or "ORGAN" in key or "STRUCTURE" in key:
        return "anatomy"
    if "LOCATION" in key or "SITE" in key:
        return "location"
    if "MEASUREMENT" in key or "VALUE" in key or "SCORE" in key:
        return "measurement"
    if "CONTEXT" in key:
        return "context"
    if "ACTIVITY" in key or "BEHAVIOR" in key:
        return "activity"

    return "other"


def gini_from_counts(counts: List[int]) -> float:
    """
    Compute the Gini coefficient from a list of non negative counts.

    This is used to measure how concentrated medical entities are
    across segments of the transcript.
    """
    if not counts:
        return 0.0

    total = float(sum(counts))
    if total <= 0.0:
        return 0.0

    sorted_counts = sorted(float(c) for c in counts)
    n = len(sorted_counts)
    cumulative = 0.0
    weighted_sum = 0.0

    for i, x in enumerate(sorted_counts, start=1):
        cumulative += x
        weighted_sum += i * x

    gini = (2.0 * weighted_sum / (n * total)) - (n + 1.0) / n
    # Numerical safety clamp
    if gini < 0.0:
        gini = 0.0
    if gini > 1.0:
        gini = 1.0
    return float(gini)

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
                print("Loaded Medical NER model")
            except Exception as e:
                print(f"⚠️  Failed to load Medical NER model: {e}")
        
        print("Feature Extractor initialized\n")
    
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
        Extract medical terminology features for a single transcript.

        This implementation uses cleaned captions, a robust label-to-category
        mapping, and span-based clustering. It is designed to be compatible with
        the existing pipeline and feature names.
        """
        # If the NER model is not available, return empty defaults
        if self.ner_pipeline is None:
            return self._empty_medical_features()

        # Clean caption text for NER and token based features
        cleaned = clean_transcript(text or "")
        if not cleaned:
            return self._empty_medical_features()

        # Tokenize words on the cleaned text.
        # This word count is used only for medical_density.
        tokens = word_tokenize(cleaned)
        words = [w for w in tokens if re.match(r"\w", w)]
        word_count = len(words)
        if word_count == 0:
            return self._empty_medical_features()

        # Run NER once if entities are not provided
        if entities is None:
            try:
                entities = self.ner_pipeline(cleaned)
            except Exception as e:
                print(f"Warning: NER extraction failed: {e}")
                return self._empty_medical_features()

        if not entities:
            return self._empty_medical_features()

        # Collect term texts, mapped categories, and start positions
        terms: List[str] = []
        categories: List[str] = []
        starts: List[int] = []

        for ent in entities:
            term_text = self._normalize_entity(ent)
            if not term_text:
                continue

            terms.append(term_text)

            raw_label = str(ent.get("entity_group") or ent.get("entity") or "")
            mapped_cat = map_raw_label_to_category(raw_label)
            if mapped_cat not in CATEGORIES:
                mapped_cat = "other"
            categories.append(mapped_cat)

            start_pos = ent.get("start")
            if isinstance(start_pos, int):
                starts.append(start_pos)

        mention_count = len(terms)
        unique_count = len({t.lower() for t in terms}) if terms else 0

        # Start from an empty template so all keys are always present
        result = self._empty_medical_features()
        result["medical_mention_count"] = mention_count
        result["medical_unique_count"] = unique_count

        if mention_count == 0:
            # No entities, but we still return the template with counts set
            return result

        # Medical term density
        result["medical_density"] = mention_count / float(word_count)

        # Category counts and ratios
        cat_counts = Counter(categories)
        for cat in CATEGORIES:
            key = f"{cat}_ratio"
            if mention_count > 0:
                result[key] = cat_counts.get(cat, 0) / float(mention_count)
            else:
                result[key] = 0.0

        # Rare term ratios (within caption)
        if unique_count > 0:
            norm_terms = [t.lower() for t in terms]
            freq = Counter(norm_terms)
            rare_terms = [term for term, c in freq.items() if c == 1]

            rare_unique = len(rare_terms)
            rare_mentions = sum(freq[t] for t in rare_terms)

            result["rare_term_ratio_unique"] = rare_unique / float(unique_count)
            result["rare_term_ratio_mentions"] = rare_mentions / float(mention_count)

        # Term clustering using equal length segments on the cleaned text
        if starts and len(cleaned) > 0:
            n_segments = 5
            seg_counts = [0] * n_segments
            text_len = len(cleaned)

            for s in starts:
                idx = int(s / float(text_len) * n_segments)
                if idx >= n_segments:
                    idx = n_segments - 1
                seg_counts[idx] += 1

            gini = gini_from_counts(seg_counts)
            max_share = max(seg_counts) / float(mention_count) if mention_count > 0 else 0.0
            concentrated = max_share >= 0.30

            result["term_clustering_gini"] = gini
            result["term_clustering_max_share"] = max_share
            result["term_clustering_concentrated"] = concentrated

        # Raw entity group counts for downstream reporting
        entity_group_counts = self._entity_group_counts(entities)
        result["entity_group_counts"] = entity_group_counts

        # Update running label statistics and documentation
        self._update_entity_counts(entities)

        return result

    def _normalize_entity(self, e: Any) -> str:
        """Extract plain text from an entity."""
        if isinstance(e, str):
            return e.strip()
        if isinstance(e, dict):
            return str(e.get("text") or e.get("word") or "").strip()
        return str(e).strip()

    def _entity_group_counts(self, entities: List[Dict]) -> Dict[str, int]:
        """
        Count raw entity_group occurrences for downstream reporting.

        This is used to build the separate entity_counts CSV in the pipeline.
        """
        counts = Counter()
        for e in entities:
            label = str(e.get("entity_group", "")).upper()
            if label:
                counts[label] += 1
        return dict(counts)

    def _empty_medical_features(self) -> Dict[str, Any]:
        """
        Return an empty medical feature dictionary.

        All keys that appear in extract_medical_features are present here
        with neutral defaults. This guarantees a consistent schema.
        """
        base = {
            "medical_density": 0.0,
            "medical_mention_count": 0,
            "medical_unique_count": 0,
            "rare_term_ratio_unique": 0.0,
            "rare_term_ratio_mentions": 0.0,
            "term_clustering_concentrated": False,
            "term_clustering_gini": 0.0,
            "term_clustering_max_share": 0.0,
            "entity_group_counts": {},
        }
        for cat in CATEGORIES:
            base[f"{cat}_ratio"] = 0.0
        return base

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
                lines.extend(
                    [
                        "",
                        "## Observed entity_group counts",
                        "| entity_group | count |",
                        "| --- | ---: |",
                    ]
                )
                for label, count in sorted(
                    self.entity_counts.items(), key=lambda item: item[1], reverse=True
                ):
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
            sent_lengths = [
                len([t for t in sent if not t.is_punct and not t.is_space])
                for sent in sentences
            ]
            avg_sent_len = statistics.mean(sent_lengths)
            std_sent_len = (
                statistics.stdev(sent_lengths) if len(sent_lengths) > 1 else 0.0
            )

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
                "passive_voice_ratio": round(passive_ratio, 3),
                "average_sentence_length": round(avg_sent_len, 2),
                "sentence_length_std": round(std_sent_len, 2),
                "syntax_tree_depth": round(tree_depth, 2),
                "subordinate_clause_density": round(sub_clause_density, 3),
                "average_dependency_distance": round(avg_dep_dist, 2),
                "pronoun_frequency": round(pronoun_freq, 3),
            }

        except Exception as e:
            print(f"⚠️  Syntactic feature extraction failed: {e}")
            return self._empty_syntactic_features()

    def _detect_passive(self, sent) -> int:
        """
        Detect passive voice constructions.
        
        Passive voice could be considered as professional language
        
        """
        count = 0
        for token in sent:
            if (
                token.dep_ == "auxpass"
                or (
                    token.dep_ == "aux"
                    and token.tag_ == "VBN"
                    and any(child.dep_ == "auxpass" for child in token.children)
                )
            ):
                count += 1
        return count

    def _avg_tree_depth(self, sentences) -> float:
        """
        
        Calculate average syntax tree depth.
        
        Synatx tree depth definition:
        The depth of a syntax tree is the maximum number of edges on a path from the root to a leaf node.
        """

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
        """
        
        Calculate subordinate clause density.
        
        Subordinate clause density definition:
        The ratio of subordinate clauses to total clauses.
        
        """
        sub_clauses = 0
        total_clauses = 0

        for sent in sentences:
            for token in sent:
                if token.dep_ in ["mark", "relcl"] or token.tag_ in [
                    "WDT",
                    "WP",
                    "WP$",
                    "WRB",
                ]:
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
            "passive_voice_ratio": 0.0,
            "average_sentence_length": 0.0,
            "sentence_length_std": 0.0,
            "syntax_tree_depth": 0.0,
            "subordinate_clause_density": 0.0,
            "average_dependency_distance": 0.0,
            "pronoun_frequency": 0.0,
        }

    # ==================== UNIFIED EXTRACTION ====================

    def extract_all_features(
        self,
        text: str,
        include_medical: bool = True,
        include_syntactic: bool = True,
    ) -> Dict[str, Any]:
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

        features: Dict[str, Any] = {}

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
            **self._empty_syntactic_features(),
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
