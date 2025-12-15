import re
from typing import List, Optional, Dict, Set
import pandas as pd

# ------------------------------
# Dictionaries
# ------------------------------

ORGAN_PATTERNS: Dict[str, str] = {
    r"\bheart\b": "heart",
    r"\bkidney\b": "kidney",
    r"\bliver\b": "liver",
    r"\blung\b": "lung",
    r"\bpancreas\b": "pancreas",
}

DRUGS: Set[str] = {
    "tacrolimus","prograf","cyclosporine","mycophenolate","cellcept","myfortic",
    "prednisone","sirolimus","everolimus","belatacept"
}
DRUG_NORMALIZE = {
    r"\btacro\b": "tacrolimus",
    r"\bprograf\b": "tacrolimus",
    r"\bcellcept\b": "mycophenolate",
    r"\bmyfortic\b": "mycophenolate",
}

SYMPTOMS: Set[str] = {
    "fever","vomiting","nausea","diarrhea","pain","cough","rash","swelling","edema","tremor",
    "headache","fatigue","dizziness","shortness of breath","chills"
}

ACTIONS_CANON = {
    # canonical: pattern list (lowercase)
    "diet": [r"\bwhat (should|to) i eat\b", r"\beat\b", r"\bfoods?\b", r"\bfood restrictions?\b", r"\bdiet\b"],
    "returning to work": [r"\breturn(ing)? to work\b", r"\bgo back to work\b", r"\bcan i work\b", r"\bwork\b"],
    "exercise": [r"\bexercise\b", r"\bworkout\b", r"\bgym\b", r"\bweight(s)?\b"],
    "travel": [r"\btravel\b", r"\bfly\b", r"\bflight\b"],
    "driving": [r"\bdrive|driving\b"],
    "alcohol": [r"\balcohol\b", r"\bdrink(ing)? alcohol\b", r"\bis it safe to drink\b"],
    "smoking": [r"\bsmok(ing|e)\b", r"\bvape|vaping\b"],
    "pregnancy": [r"\bpregnan(t|cy)\b", r"\bconceiv(e|ing)\b"],
    "breastfeeding": [r"\bbreastfeed(ing)?\b"],
    "sports": [r"\bsport(s)?\b", r"\brun(ning)?\b", r"\bswim(ming)?\b"],
    "vaccine": [r"\bvaccin(e|ation)\b", r"\bflu shot\b", r"\bcovid shot\b"],
    "sex": [r"\bsex\b", r"\bsexual activity\b"]
}

LABS: Set[str] = {
    "creatinine","gfr","enzymes","potassium","magnesium",
    "tacrolimus level","tacrolimus levels","level","levels",
    "alt","ast","bilirubin","wbc","platelets"
}

COMPLICATIONS: Set[str] = {
    "infection","rejection","cmv","bk","bk virus","virus","wound","biopsy","pneumonia","uti","sepsis"
}

# Stage and transplant cues
STAGE_PAT = r"(post[\s-]?transplant|pre[\s-]?transplant|\d+\s+(day|days|week|weeks|month|months)\s+(post|after))"
TRANSPLANT_CUE = r"\btransplant\b"

FILLERS = [
    "please help","anyone else","does anyone","thanks in advance","thank you",
    "hi everyone","curious if","just wondering","for now","any tips","any experiences",
    "is this normal","was it really","i m","im","i am","it s","its","tbh","btw","lol"
]

# ------------------------------
# Cleaning
# ------------------------------

def clean_text(s: str) -> str:
    t = (s or "").lower()
    t = re.sub(r"\s+", " ", t)
    for pat, repl in DRUG_NORMALIZE.items():
        t = re.sub(pat, repl, t)
    t = re.sub(r"[^\w\s\-&/']", " ", t)
    for f in FILLERS:
        t = re.sub(r"\b" + re.escape(f) + r"\b", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def sentence_split(text: str) -> List[str]:
    parts = re.split(r"[\.!?;\n]+", text)
    return [p.strip() for p in parts if p.strip()]

# ------------------------------
# Transplant type
# ------------------------------

def detect_transplant_type(text: str) -> str:
    for pat, label in ORGAN_PATTERNS.items():
        if re.search(pat, text):
            return label
    return "general"

def find_organs(text: str) -> List[str]:
    found = []
    for pat, label in ORGAN_PATTERNS.items():
        if re.search(pat, text):
            found.append(label)
    return list(dict.fromkeys(found))

# ------------------------------
# Matchers
# ------------------------------

def find_list_from_set(text: str, vocab: Set[str]) -> List[str]:
    mw = [w for w in vocab if " " in w]
    found = []
    for w in sorted(mw, key=len, reverse=True):
        if re.search(r"\b" + re.escape(w) + r"\b", text):
            found.append(w)
    tokens = re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", text)
    singles = set(vocab) - set(mw)
    for tk in tokens:
        if tk in singles:
            found.append(tk)
    return list(dict.fromkeys(found))

def match_actions(text: str) -> List[str]:
    hits = []
    for canon, patterns in ACTIONS_CANON.items():
        for pat in patterns:
            if re.search(pat, text):
                hits.append(canon)
                break
    return hits

def has_connector(text: str) -> bool:
    return bool(re.search(STAGE_PAT, text)) or bool(re.search(TRANSPLANT_CUE, text))

# ------------------------------
# Candidate builders
# ------------------------------

def join_items(items: List[str], limit: int = 2, conj: str = "and") -> str:
    items = items[:limit]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return f"{items[0]} {conj} {items[1]}"

def build_candidates_from_sentence(sent: str) -> List[str]:
    s = sent.strip()
    if not s:
        return []

    organs = find_organs(s)
    drugs = find_list_from_set(s, DRUGS)
    symptoms = find_list_from_set(s, SYMPTOMS)
    actions = match_actions(s)
    labs = find_list_from_set(s, LABS)
    comps = find_list_from_set(s, COMPLICATIONS)

    # Require some intent signal in sentence
    if not (organs or drugs or symptoms or actions or labs or comps or has_connector(s)):
        return []

    cands: List[str] = []

    # Symptoms + organ
    if symptoms and organs:
        cands.append(f"{join_items(symptoms)} after {organs[0]} transplant")

    # Symptoms + drug
    if symptoms and drugs:
        cands.append(f"{join_items(symptoms)} on {drugs[0]}")

    # Complication + organ
    if comps and organs:
        cands.append(f"{join_items(comps, limit=1)} after {organs[0]} transplant")

    # Labs + organ
    if labs and organs:
        lab = next((x for x in labs if x.endswith("levels") or x.endswith("level")), labs[0])
        lab = re.sub(r"\blevel\b", "levels", lab)
        cands.append(f"{lab} after {organs[0]} transplant")

    # Actions + organ with stage cue
    if actions and organs:
        # Choose preposition from sentence if available
        prep = "after"
        if re.search(r"\bbefore\b", s):
            prep = "before"
        elif re.search(r"\bduring\b", s):
            prep = "during"
        cands.append(f"{actions[0]} {prep} {organs[0]} transplant")

    # Drug generic (no symptoms)
    if drugs and not symptoms:
        if re.search(r"\b(levels?|trough|peak|dose)\b", s):
            cands.append(f"{drugs[0]} levels")
        else:
            cands.append(f"{drugs[0]} side effects")

    # Organ present but nothing specific
    # Do not add "questions after ...". Prefer a mild but specific default:
    if organs and not cands:
        cands.append(f"recovery after {organs[0]} transplant")

    # Clean
    out = []
    for c in cands:
        c = re.sub(r"\s+", " ", c).strip()
        if len(c.split()) >= 2:
            out.append(c)
    return out

def pick_best(candidates: List[str]) -> Optional[str]:
    if not candidates:
        return None
    # Rank: concrete > generic
    order = [
        r".*\b(fever|vomiting|nausea|diarrhea|pain|cough|rash|swelling|tremor|headache|shortness of breath|chills)\b.*after\b.*transplant\b",
        r".*\b(fever|vomiting|nausea|diarrhea|pain|cough|rash|swelling|tremor|headache|shortness of breath|chills)\b.*\bon\b.*",
        r".*\b(infection|rejection|cmv|bk|bk virus|virus|wound|biopsy|uti|sepsis)\b.*after\b.*transplant\b",
        r".*\b(levels?|creatinine|gfr|enzymes|potassium|alt|ast|bilirubin|wbc|platelets)\b.*after\b.*transplant\b",
        r".*\b(diet|exercise|returning to work|travel|driving|alcohol|smoking|pregnancy|breastfeeding|sports|vaccine|sex)\b.*\b(after|before|during)\b.*transplant\b",
        r".*\b(tacrolimus|cyclosporine|mycophenolate|prednisone|sirolimus|everolimus|belatacept)\b.*(levels?|side effects)\b.*",
        r".*\brecovery after\b.*transplant\b"
    ]
    scored = []
    for c in candidates:
        rank = len(order) + 1
        for i, pat in enumerate(order):
            if re.search(pat, c):
                rank = i + 1
                break
        scored.append((rank, len(c.split()), c))
    scored.sort()
    return scored[0][2]

# ------------------------------
# Post-level extraction
# ------------------------------

def extract_keyword(text: str) -> Optional[str]:
    sents = sentence_split(text)
    all_cands: List[str] = []
    for s in sents:
        all_cands.extend(build_candidates_from_sentence(s))
    best = pick_best(all_cands)
    return best

# ------------------------------
# Pipeline
# ------------------------------

def process_posts_df(df: pd.DataFrame) -> pd.DataFrame:
    if not {"title", "body_text"} <= set(df.columns):
        raise ValueError("Input DataFrame must have columns: 'title' and 'body_text'")

    rows = []
    for _, row in df.iterrows():
        merged = f"{(row.get('title') or '').strip()}. {(row.get('body_text') or '').strip()}".strip()
        cleaned = clean_text(merged)

        ttype = detect_transplant_type(cleaned)
        kw = extract_keyword(cleaned)

        # Hard fallback: choose the least-generic possible phrase
        if kw is None:
            if ttype != "general":
                kw = f"recovery after {ttype} transplant"
            else:
                # Prefer a drug-specific default if any drug is mentioned globally
                if any(re.search(r"\b" + d + r"\b", cleaned) for d in DRUGS):
                    # pick the first drug hit
                    d = next(d for d in DRUGS if re.search(r"\b" + d + r"\b", cleaned))
                    kw = f"{d} side effects"
                else:
                    kw = "post-transplant recovery"

        rows.append({"keywords": kw, "transplant_type": ttype})

    return pd.DataFrame(rows, columns=["keywords", "transplant_type"])


# ------------------------------
# Script entry
# ------------------------------
if __name__ == "__main__":
    df = pd.read_csv("reddit_posts.csv")
    df2 = process_posts_df(df)
    print(df2.head(20))
    df2.to_csv("reddit_posts_keywords.csv", index=False)
