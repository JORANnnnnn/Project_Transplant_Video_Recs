# !pip install transformers torch
# !pip install bioc
# !pip install sentencepiece

# Download Medical terminology
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

model_name = "d4data/biomedical-ner-all"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer,aggregation_strategy="simple")


# Medical Term Density utility
# - Computes: medical entities count / total word count
# - Works with: 
#     * Pre-computed entities (list of strings or list of dicts with 'word'/'text' and optional 'start'/'end'/'entity_group'/'label')
#     * Or with a provided HuggingFace NER pipeline (callable)
#
# This file includes a tiny demo with mocked entities so it runs without external models.

import re
from typing import List, Dict, Any, Optional, Tuple, Union

Word = str
Entity = Union[str, Dict[str, Any]]

WORD_PATTERN = re.compile(r"\b\w+(?:'\w+)?\b", flags=re.UNICODE)

def tokenize_words(text: str) -> List[Word]:
    """Simple, robust word tokenizer that handles apostrophes.
    Counts only alphanumeric word tokens.
    """
    return WORD_PATTERN.findall(text)


def _normalize_entity_text(e: Entity) -> str:
    """Extract a comparable text string from different entity formats."""
    if isinstance(e, str):
        return e.strip()
    # Common HF pipeline keys: 'word' or 'text'
    if 'text' in e:
        return str(e['text']).strip()
    if 'word' in e:
        return str(e['word']).strip()
    # Fallback to repr if an unknown structure
    return str(e).strip()


def entity_counts(entities: Optional[List[Entity]]) -> Tuple[int, int, List[str]]:
    """Return (mention_count, unique_count, normalized_terms_list)."""
    if not entities:
        return 0, 0, []
    terms = [_normalize_entity_text(e) for e in entities if _normalize_entity_text(e)]
    mention_count = len(terms)
    unique_count = len({t.lower() for t in terms})
    return mention_count, unique_count, terms


def medical_term_density(
    text: str,
    entities: Optional[List[Entity]] = None,
    ner_pipeline: Optional[Any] = None,
    count_mode: str = "mentions"  # "mentions" or "unique"
) -> Dict[str, Any]:
    """Compute Medical Term Density = entity_count / total_word_count.
    
    Parameters
    ----------
    text : str
        Input document text.
    entities : Optional[List[Entity]]
        Pre-extracted entities. Each entity can be a string, or a dict containing
        at least 'text' or 'word'. HuggingFace NER outputs typically include
        'word'/'entity_group'/'score'/'start'/'end'.
    ner_pipeline : Optional[Any]
        A callable HuggingFace pipeline for NER. Used if `entities` is None.
    count_mode : str
        "mentions" counts every mention; "unique" counts unique normalized terms.
    
    Returns
    -------
    Dict[str, Any]
        {
            "density": float,
            "total_words": int,
            "entity_count": int,
            "count_mode": "mentions"|"unique",
            "mention_count": int,
            "unique_count": int,
            "terms": List[str]
        }
    """
    words = tokenize_words(text)
    total_words = len(words)

    # Obtain entities if not provided
    computed_entities = entities
    if computed_entities is None and ner_pipeline is not None:
        computed_entities = ner_pipeline(text)
    # If still None, treat as empty
    mention_count, unique_count, terms = entity_counts(computed_entities or [])

    if count_mode not in {"mentions", "unique"}:
        raise ValueError("count_mode must be 'mentions' or 'unique'")

    entity_count = mention_count if count_mode == "mentions" else unique_count
    density = (entity_count / total_words) if total_words > 0 else 0.0

    return {
        "density": density,
        "total_words": total_words,
        "entity_count": entity_count,
        "count_mode": count_mode,
        "mention_count": mention_count,
        "unique_count": unique_count,
        "terms": terms,
    }


from collections import Counter
from typing import List, Dict, Any

def term_type_distribution(entities: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Compute proportion vector for 6 medical entity categories.

    Parameters
    ----------
    entities : list of dict
        Each dict should contain at least {'entity_group': <category_name>}.

    Returns
    -------
    dict
        Proportion of each category among all entities.
        Example:
        {
            'disease_symptom_ratio': 0.33,
            'anatomical_structure_ratio': 0.17,
            'medication_ratio': 0.17,
            'medical_procedure_ratio': 0.17,
            'diagnostic_test_ratio': 0.17,
            'medical_device_ratio': 0.00
        }
    """

    # Canonical mapping for common BioBERT/SciBERT entity labels
    category_map = {
        "DISEASE": "disease_symptom",
        "SYMPTOM": "disease_symptom",
        "ANATOMY": "anatomical_structure",
        "ANATOMICAL": "anatomical_structure",
        "CHEMICAL": "medication",
        "DRUG": "medication",
        "PROCEDURE": "medical_procedure",
        "TEST": "diagnostic_test",
        "DEVICE": "medical_device",
    }

    # Map entities to unified category names
    mapped = []
    for e in entities:
        label = str(e.get("entity_group", "")).upper()
        mapped.append(category_map.get(label, None))

    # Count and compute proportions
    filtered = [m for m in mapped if m is not None]
    total = len(filtered)
    counts = Counter(filtered)
    ratios = {k: counts.get(k, 0) / total if total > 0 else 0.0 for k in [
        "disease_symptom",
        "anatomical_structure",
        "medication",
        "medical_procedure",
        "diagnostic_test",
        "medical_device"
    ]}

    # Add readability aliases
    ratios = {
        "disease_symptom_ratio": ratios["disease_symptom"],
        "anatomical_structure_ratio": ratios["anatomical_structure"],
        "medication_ratio": ratios["medication"],
        "medical_procedure_ratio": ratios["medical_procedure"],
        "diagnostic_test_ratio": ratios["diagnostic_test"],
        "medical_device_ratio": ratios["medical_device"],
    }

    return ratios

# Term Clustering: Are medical terms concentrated in specific paragraphs?
# - Input: text (string), entities (list of dicts with optional 'start'/'end'), or a NER pipeline to compute entities
# - Output: metrics dict with multiple concentration indicators and a boolean "concentrated"
#
# Notes:
# - Paragraphs are separated by two or more newlines, or a single newline followed by a blank line.
# - If entities do not contain character offsets ('start'/'end'), we will approximate by locating the first occurrence of the entity text.
# - Metrics include: counts per paragraph, proportions, HHI, Gini, max share, chi-square vs uniform, variance-to-mean ratio.
#
# Demo at the bottom uses mocked entities so it runs without external models.
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from math import isfinite

PARA_SPLIT = re.compile(r"(?:\r?\n){2,}")  # blank line(s) as paragraph boundaries

@dataclass
class Paragraph:
    text: str
    start: int   # start char index in the full text
    end: int     # end char index in the full text (exclusive)


def split_paragraphs(text: str) -> List[Paragraph]:
    """Split text into paragraphs and return list with char spans."""
    # Find boundaries by splitting on blank lines, but we need global positions
    parts = PARA_SPLIT.split(text)
    paragraphs: List[Paragraph] = []
    cursor = 0
    # Reconstruct by scanning the original text for each part
    for part in parts:
        # Skip leading newlines/spaces at current cursor
        while cursor < len(text) and text[cursor] in "\r\n":
            cursor += 1
        # Find the exact segment in the remaining text
        idx = text.find(part, cursor)
        if idx == -1:
            # Fallback: treat the next chunk of length as the paragraph
            idx = cursor
        start = idx
        end = start + len(part)
        paragraphs.append(Paragraph(part, start, end))
        cursor = end
    return paragraphs


def locate_entity_span(text: str, e: Dict[str, Any]) -> Tuple[int, int]:
    """Return (start, end) offsets for an entity. Prefer provided offsets,
    otherwise approximate by searching the first occurrence of 'text' or 'word'."""
    if "start" in e and "end" in e and isinstance(e["start"], int) and isinstance(e["end"], int):
        return int(e["start"]), int(e["end"])
    candidate = str(e.get("text") or e.get("word") or "").strip()
    if candidate:
        pos = text.find(candidate)
        if pos != -1:
            return pos, pos + len(candidate)
    # Unknown span -> return invalid
    return -1, -1


def counts_per_paragraph(text: str, paragraphs: List[Paragraph], entities: List[Dict[str, Any]]) -> List[int]:
    """Count entities that fall within each paragraph span. Uses start/end if available, else approximate."""
    counts = [0] * len(paragraphs)
    for e in entities:
        s, t = locate_entity_span(text, e)
        if s == -1:
            # Could not place the entity -> skip
            continue
        for i, p in enumerate(paragraphs):
            if s >= p.start and s < p.end:
                counts[i] += 1
                break
    return counts


def proportions(counts: List[int]) -> List[float]:
    total = sum(counts)
    if total == 0:
        return [0.0 for _ in counts]
    return [c / total for c in counts]


def herfindahl_hirschman_index(props: List[float]) -> float:
    return sum(p * p for p in props)


def gini_coefficient(counts: List[int]) -> float:
    """Gini on counts array. Returns 0 for equal distribution, up to 1 for maximum inequality."""
    n = len(counts)
    if n == 0:
        return 0.0
    sorted_counts = sorted(counts)
    total = sum(sorted_counts)
    if total == 0:
        return 0.0
    cum = 0
    weighted_sum = 0
    for i, x in enumerate(sorted_counts, start=1):
        cum += x
        weighted_sum += cum
    # Gini formula for discrete distribution
    gini = (n + 1 - 2 * (weighted_sum / cum)) if cum > 0 else 0.0
    return gini / n if n > 0 else 0.0


def chi_square_uniform(counts: List[int]) -> float:
    """Pearson chi-square statistic vs uniform across paragraphs. Not p-value, only the statistic."""
    n = len(counts)
    total = sum(counts)
    if n == 0 or total == 0:
        return 0.0
    expected = total / n
    stat = 0.0
    for c in counts:
        stat += (c - expected) ** 2 / expected
    return stat


def variance_to_mean_ratio(counts: List[int]) -> float:
    n = len(counts)
    if n == 0:
        return 0.0
    mean = sum(counts) / n
    if mean == 0:
        return 0.0
    var = sum((c - mean) ** 2 for c in counts) / n
    return var / mean


def term_clustering(
    text: str,
    entities: List[Dict[str, Any]],
    min_entities: int = 5,
    gini_threshold: float = 0.5,
    hhi_threshold: float = 0.25,
    max_share_threshold: float = 0.5,
) -> Dict[str, Any]:
    """Compute clustering metrics for entities across paragraphs and a boolean decision.
    
    Parameters
    ----------
    text : str
        Full document text.
    entities : list of dict
        NER entities with optional 'start'/'end' or at least 'text'/'word' keys.
    min_entities : int
        Minimum total entities to make a confident decision. Below this, 'concentrated' may be False.
    gini_threshold : float
        Gini coefficient threshold to call concentration.
    hhi_threshold : float
        Herfindahl-Hirschman threshold to call concentration.
    max_share_threshold : float
        Maximum single paragraph share threshold to call concentration.
    
    Returns
    -------
    dict with metrics and a boolean 'concentrated'.
    """
    paragraphs = split_paragraphs(text)
    if len(paragraphs) == 0:
        return {
            "paragraph_count": 0,
            "counts": [],
            "proportions": [],
            "hhi": 0.0,
            "gini": 0.0,
            "max_share": 0.0,
            "chi_square_uniform": 0.0,
            "vmr": 0.0,
            "total_entities": 0,
            "concentrated": False,
            "paragraph_summaries": [],
        }
    counts = counts_per_paragraph(text, paragraphs, entities)
    props = proportions(counts)
    hhi = herfindahl_hirschman_index(props)
    gini = gini_coefficient(counts)
    max_share = max(props) if props else 0.0
    chi2 = chi_square_uniform(counts)
    vmr = variance_to_mean_ratio(counts)
    total = sum(counts)

    # Decision rule: concentrated if any of the indicators cross thresholds
    # and there are enough entities to be meaningful.
    concentrated = False
    if total >= min_entities:
        if max_share >= max_share_threshold or hhi >= hhi_threshold or gini >= gini_threshold:
            concentrated = True

    # Provide short paragraph summaries
    para_summaries = []
    for p, c, pr in zip(paragraphs, counts, props):
        snippet = re.sub(r"\s+", " ", p.text.strip())[:120]
        para_summaries.append({
            "chars": [p.start, p.end],
            "count": c,
            "share": pr,
            "snippet": snippet
        })

    return {
        "paragraph_count": len(paragraphs),
        "counts": counts,
        "proportions": props,
        "hhi": hhi,
        "gini": gini,
        "max_share": max_share,
        "chi_square_uniform": chi2,
        "vmr": vmr,
        "total_entities": total,
        "concentrated": concentrated,
        "paragraph_summaries": para_summaries,
    }

# Rare Term Ratio: Proportion of low-frequency medical vocabulary
# - Supports two modes:
#   1) 'doc'      -> rarity is defined within the current document: terms with document frequency == 1 among medical entities
#   2) 'external' -> rarity is defined using an external frequency resource (Zipf or per-million-word scale)
#
# Usage:
#   rare_term_ratio(entities, mode='doc')
#   rare_term_ratio(entities, mode='external', external_freqs={'zipf': {'aspirin': 4.5, ...}}, zipf_threshold=3.0)
#
# Returns a dictionary with ratios (over unique terms and mentions), counts, and the rare terms list.
#
# Includes a runnable demo with mock entities and a tiny external frequency dict.
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter
import re

def _ent_text(e: Dict[str, Any]) -> str:
    """Normalize entity text for counting."""
    t = str(e.get("text") or e.get("word") or "").strip()
    # Collapse internal whitespace and lowercase
    t = re.sub(r"\s+", " ", t).lower()
    return t

def _unique_and_mentions(entities: List[Dict[str, Any]]) -> Tuple[List[str], Counter]:
    """Return list of normalized terms and a Counter for document frequency of each term."""
    terms = [_ent_text(e) for e in entities if _ent_text(e)]
    df = Counter(terms)
    uniq_terms = sorted(df.keys())
    return uniq_terms, df

def rare_term_ratio(
    entities: List[Dict[str, Any]],
    mode: str = "doc",
    external_freqs: Optional[Dict[str, Dict[str, float]]] = None,
    zipf_threshold: float = 3.0,     # terms with Zipf < 3.0 are considered rare
    pmw_threshold: float = 1.0       # terms with pmw < 1 per million words are rare
) -> Dict[str, Any]:
    """
    Compute Rare Term Ratio: proportion of low-frequency medical vocabulary.
    
    Parameters
    ----------
    entities : list of dict
        NER entities with 'text' or 'word' keys.
    mode : {'doc', 'external'}
        - 'doc': rare if the term appears only once among the medical entities in the current document.
        - 'external': rare if below thresholds using an external frequency map.
    external_freqs : dict or None
        Optional external frequency resources. Example:
        {
            'zipf': {'aspirin': 4.5, 'troponin': 2.8},
            'pmw':  {'aspirin': 120.0, 'troponin': 0.4}
        }
        You can provide either 'zipf', or 'pmw', or both.
    zipf_threshold : float
        Threshold for Zipf scale rarity. Typical rare cutoff is < 3.0.
    pmw_threshold : float
        Threshold for per-million-words rarity. Typical rare cutoff is < 1.0.
    
    Returns
    -------
    dict
        {
            'ratio_unique': float,     # rare unique terms / total unique terms
            'ratio_mentions': float,   # rare mentions / total mentions
            'rare_unique_count': int,
            'total_unique_terms': int,
            'rare_mentions_count': int,
            'total_mentions': int,
            'rare_terms': List[str],
            'mode': str,
            'criteria': {'zipf_threshold':..., 'pmw_threshold':...}
        }
    """
    uniq_terms, df = _unique_and_mentions(entities)
    total_unique = len(uniq_terms)
    total_mentions = sum(df.values())
    if total_unique == 0 or total_mentions == 0:
        return {
            "ratio_unique": 0.0,
            "ratio_mentions": 0.0,
            "rare_unique_count": 0,
            "total_unique_terms": total_unique,
            "rare_mentions_count": 0,
            "total_mentions": total_mentions,
            "rare_terms": [],
            "mode": mode,
            "criteria": {"zipf_threshold": zipf_threshold, "pmw_threshold": pmw_threshold}
        }

    rare_set = set()

    if mode == "doc":
        # Rare if appears once in the document among medical entities
        for term, c in df.items():
            if c == 1:
                rare_set.add(term)

    elif mode == "external":
        if not external_freqs or not isinstance(external_freqs, dict):
            raise ValueError("external_freqs must be provided for mode='external'.")
        zipf_map = external_freqs.get("zipf", {})
        pmw_map = external_freqs.get("pmw", {})
        for term in uniq_terms:
            is_rare = False
            # Check Zipf first if provided
            if term in zipf_map:
                if zipf_map[term] < zipf_threshold:
                    is_rare = True
            # If no zipf or not rare by zipf, check pmw if available
            if not is_rare and term in pmw_map:
                if pmw_map[term] < pmw_threshold:
                    is_rare = True
            # If neither resource has the term, we can optionally treat as unknown -> rare.
            # Choose a conservative policy: unknown terms are considered rare.
            if term not in zipf_map and term not in pmw_map:
                is_rare = True
            if is_rare:
                rare_set.add(term)
    else:
        raise ValueError("mode must be 'doc' or 'external'")

    # Counts
    rare_unique_count = len(rare_set)
    rare_mentions_count = sum(df[t] for t in rare_set)

    ratio_unique = rare_unique_count / total_unique if total_unique > 0 else 0.0
    ratio_mentions = rare_mentions_count / total_mentions if total_mentions > 0 else 0.0

    return {
        "ratio_unique": ratio_unique,
            "ratio_mentions": ratio_mentions,
            "rare_unique_count": rare_unique_count,
            "total_unique_terms": total_unique,
            "rare_mentions_count": rare_mentions_count,
            "total_mentions": total_mentions,
            "rare_terms": sorted(list(rare_set)),
            "mode": mode,
            "criteria": {"zipf_threshold": zipf_threshold, "pmw_threshold": pmw_threshold}
        }


