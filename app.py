"""
PolyLens v6 — Multilingual Image / Speech / Text Translation System
════════════════════════════════════════════════════════════════════
4 Features, 13 Novelties — all integrated silently into the pipeline.
No extra tabs, no extra buttons.

NOVELTY MAP (what runs inside each feature):
  Live Camera   → N1 N2 N3 N4 N5 N6 N7 N8 N9 N10 N11 N12 N13
  Image Upload  → N1 N2 N3 N5 N7 N9 N10 N11 N12 N13
  Text→Speech   → N2 N3 N5 N7 N10 N12
  Speech→Text   → N2 N3 N5 N7 N10 N12

N1  Context-Preserving OCR Repair        — token-level diff audit
N2  Speech-Aware TTS Normalization       — post-translation, lang-aware, year fix
N3  Real-Time Experimental Validation    — per-stage sub-ms timing, CSV log
N4  Semantic Frame Deduplication         — rolling trigram fingerprint
N5  Script-Aware TTS Voice Routing       — Unicode script → voice profile
N6  Contextual Discourse Memory          — multi-frame paragraph building
N7  Domain-Adaptive Translation Hints    — medical/legal/food/tech/signage
N8  Multi-Frame Voting OCR               — auto 3-frame majority vote (live only)
N9  Reading-Direction & Layout Repair    — RTL detection, column order
N10 Tone/Formality Detection             — register classification
N11 Confidence-Stratified Word Coloring  — per-word OCR confidence tiers
N12 Latency Anomaly Detection            — rolling mean+2σ per stage
N13 Auto-Benchmark Sampling              — silent 3-run sample on every upload
"""

from flask import Flask, render_template, request, jsonify, send_file
import base64, time, csv, os, re, html, unicodedata, statistics, collections

from google.cloud import vision
from google.cloud import translate_v2 as translate
from google.cloud import texttospeech
from google.cloud import speech

app = Flask(__name__)

vision_client    = vision.ImageAnnotatorClient()
translate_client = translate.Client()
tts_client       = texttospeech.TextToSpeechClient()
stt_client       = speech.SpeechClient()

LOG_FILE             = "polylens_log.csv"
CONFIDENCE_THRESHOLD = 0.55

_LOG_COLS = [
    "timestamp","session_id","mode",
    "ocr_latency_ms","stt_latency_ms","translation_latency_ms",
    "tts_latency_ms","total_latency_ms",
    "ocr_confidence","stt_confidence",
    "tokens_repaired","expansions_applied",
    "semantic_similarity","domain_detected",
    "frame_skip_reason","script_family",
    "source_language","target_language",
    "char_count_in","char_count_out","anomaly_flag",
    "vote_frames","benchmark_mean_ms",
]
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE,"w",newline="") as f:
        csv.writer(f).writerow(_LOG_COLS)

_session_counter = 0
def _next_session():
    global _session_counter
    _session_counter += 1
    return f"S{_session_counter:04d}"

def _elapsed(t0): return round((time.time()-t0)*1000,2)


# ══════════════════════════════════════════════════════════════════
# N1 — Context-Preserving OCR Repair
# ══════════════════════════════════════════════════════════════════

_PRESERVE_PATTERNS = [
    (r"\b[pP][hH]\s*\d+[\.,]\d+",                           "chemical_pH"),
    (r"\b[A-Z][a-z]?\d+[A-Z]?[a-z]?\d*\b",                 "chemical_formula"),
    (r"\b\d+(?:\.\d+)?\s*(?:mg|kg|ml|l|g|km|cm|mm|kw|w)\b","quantity_unit"),
    (r"\b\d+\s*%\b",                                         "percentage"),
    (r"\b[A-Z]{1,4}[-/]\d{3,10}\b",                         "product_id"),
    (r"(?:rs\.?|₹|\$|€|£)\s*\d+[\.,]?\d*",                 "price"),
    (r"\b(?:1[0-9]{3}|20[0-9]{2})\b",                       "year"),  # keeps years as digits → N2 fix
]
_OCR_TRANSFORMS = [
    (r"[\"""'']",                               "",           "strip_quotes"),
    (r"[●•■◆▲▼△→←↑↓]",                        " ",          "bullet_to_space"),
    (r"[<>|\\]",                               " ",          "strip_brackets"),
    (r"http\S+|www\S+",                        "",           "remove_url"),
    (r"\b\d{10,}\b",                           "",           "remove_barcode"),
    (r"\b[pP]\s*\.?\s*[hH]\s*(\d+)[,.](\d+)", r"pH \1.\2",  "fix_pH_spacing"),
    (r"(\d+)\s*(?:point|dot)\s*(\d+)",         r"\1.\2",     "fix_decimal_word"),
    (r"(?:rs\.?|₹)\s*(\d+)\s*[.,]\s*(\d+)",   r"₹\1.\2",   "normalise_rupee"),
    (r"#\s*(\d+)",                              r"number \1", "expand_hash_num"),
    (r"\s{2,}",                                " ",          "collapse_whitespace"),
]

def normalize_text_with_diff(raw: str):
    text = html.unescape(raw)
    diffs, pmap = [], {}
    for pat, label in _PRESERVE_PATTERNS:
        for m in re.finditer(pat, text, re.I):
            tok = m.group()
            if tok not in (v[0] for v in pmap.values()):
                key = f"\x00P{len(pmap)}\x00"
                pmap[key] = (tok, label)
                text = text.replace(tok, key, 1)
                diffs.append({"original":tok,"repaired":tok,"rule":label,"type":"preserve"})
    for pat, repl, label in _OCR_TRANSFORMS:
        def _rep(m, _r=repl, _l=label):
            before = m.group()
            after  = m.expand(_r) if _r else ""
            if before.strip() != after.strip():
                diffs.append({"original":before,"repaired":after,"rule":_l,
                              "type":"remove" if after.strip()=="" else "repair"})
            return after
        text = re.sub(pat, _rep, text, flags=re.I)
    for key,(tok,_) in pmap.items():
        text = text.replace(key, tok)
    return text.strip(), diffs

def normalize_text(raw):
    c, _ = normalize_text_with_diff(raw); return c


# ══════════════════════════════════════════════════════════════════
# N11 — Confidence-Stratified Word Coloring
# ══════════════════════════════════════════════════════════════════

def extract_word_confidences(response) -> list:
    result = []
    for page in response.full_text_annotation.pages:
        for block in page.blocks:
            for para in block.paragraphs:
                for word in para.words:
                    wtext = "".join(s.text for s in word.symbols)
                    conf  = word.confidence if word.confidence is not None else 1.0
                    tier  = "high" if conf >= 0.9 else ("mid" if conf >= 0.7 else "low")
                    result.append({"word":wtext,"confidence":round(conf,3),"tier":tier})
    return result


def reconstruct_context(text: str) -> str:
    lines  = re.split(r"[.\n;]", text)
    output, buffer, section = [], [], None
    for line in lines:
        line = line.strip()
        if not line: continue
        lower = line.lower()
        if "set contents"  in lower: section="set";  buffer=[]; continue
        if "free from"     in lower: section="free"; buffer=[]; continue
        if "caution"       in lower or "please be careful" in lower:
            output.append("Caution: please use carefully."); continue
        if "manufactured"  in lower or "customer care" in lower: continue
        if section in ("set","free"): buffer.append(line); continue
        if len(line.split()) >= 4:    output.append(line)
    if buffer and section=="set":
        output.insert(0,"Set contents include "+", ".join(buffer)+".")
    if buffer and section=="free":
        output.append("Free from "+", ".join(buffer)+".")
    return " ".join(output)


# ══════════════════════════════════════════════════════════════════
# N6 — Contextual Discourse Memory
# ══════════════════════════════════════════════════════════════════

_discourse_buffer: list = []
_DISCOURSE_MAX = 5
_DISCOURSE_FLUSH = 8

def _ngram_sim(a, b, n=3):
    def ngs(t): return {t[i:i+n] for i in range(max(0,len(t)-n+1))}
    sa, sb = ngs(a.lower()), ngs(b.lower())
    if not sa or not sb: return 0.0
    return len(sa & sb) / len(sa | sb)

def discourse_update(new_text: str):
    global _discourse_buffer
    words = new_text.strip().split()
    is_extended = False
    if not words:
        return new_text, False, list(_discourse_buffer)
    ends_sent = new_text.rstrip().endswith((".", "!", "?", "।", "。", "？", "！"))
    total_words = sum(len(s.split()) for s in _discourse_buffer)
    if len(words) < 6 and _discourse_buffer and total_words < _DISCOURSE_FLUSH:
        if _discourse_buffer and _ngram_sim(_discourse_buffer[-1], new_text) < 0.7:
            _discourse_buffer.append(new_text)
            is_extended = True
    else:
        _discourse_buffer = [new_text]
    if len(_discourse_buffer) > _DISCOURSE_MAX:
        _discourse_buffer = _discourse_buffer[-_DISCOURSE_MAX:]
    combined = " ".join(_discourse_buffer)
    if ends_sent:
        _discourse_buffer = []
    return combined, is_extended, list(_discourse_buffer)

def reset_discourse():
    global _discourse_buffer
    _discourse_buffer = []


# ══════════════════════════════════════════════════════════════════
# N7 — Domain-Adaptive Translation Hints
# ══════════════════════════════════════════════════════════════════

_DOMAIN_RULES = {
    "medical":   {"kw": r"\b(mg|dosage|tablet|capsule|contraindicated|syrup|injection|prescription|antibiotic|dose|twice daily|side effect|paracetamol|ibuprofen|blood pressure|diabetes|insulin|pharmacy)\b",
                  "hint": "Medical/pharmaceutical context. Preserve drug names and dosage units exactly."},
    "legal":     {"kw": r"\b(hereby|pursuant|clause|agreement|whereas|indemnify|liability|jurisdiction|plaintiff|defendant|affidavit|notary|tribunal|contract|terms and conditions|warranty)\b",
                  "hint": "Legal/contractual document. Use formal legal terminology."},
    "food":      {"kw": r"\b(ingredients|nutrition|calories|protein|carbohydrate|fat|sodium|allergen|gluten|contains|serving size|per 100g|dietary|vegan|organic|preservative|additive)\b",
                  "hint": "Food/nutrition label. Preserve ingredient names and nutritional units."},
    "technical": {"kw": r"\b(voltage|current|watt|ampere|resistance|frequency|circuit|processor|memory|bandwidth|firmware|protocol|API|SDK|configure|installation|specifications|model number|serial)\b",
                  "hint": "Technical/engineering document. Preserve specifications and model numbers."},
    "signage":   {"kw": r"\b(exit|entrance|caution|warning|danger|no smoking|emergency|fire|push|pull|open|closed|out of order|staff only|keep out)\b",
                  "hint": "Public signage. Use clear, imperative translations."},
}

def detect_domain(text: str):
    lower = text.lower()
    for domain, cfg in _DOMAIN_RULES.items():
        if re.search(cfg["kw"], lower, re.I):
            return domain, cfg["hint"]
    return "", ""

def domain_translate(text: str, tgt: str):
    domain_label, hint = detect_domain(text)
    augmented = f"[{hint}]\n{text}" if hint else text
    t0 = time.time()
    r  = translate_client.translate(augmented, target_language=tgt)
    ms = _elapsed(t0)
    raw_trans = r["translatedText"]
    dlang     = r.get("detectedSourceLanguage","")
    translated = re.sub(r"^\s*\[.*?\]\s*","",raw_trans,flags=re.S).strip() or raw_trans
    return translated, ms, dlang, domain_label, hint


# ══════════════════════════════════════════════════════════════════
# N8 — Multi-Frame Voting OCR (silent, integrated in live camera)
#   Called when stableCount reaches threshold — grabs 3 quick frames
#   and votes before final translation.
# ══════════════════════════════════════════════════════════════════

def _vote_words(word_lists: list) -> list:
    if not word_lists: return []
    max_len = max(len(w) for w in word_lists)
    voted   = []
    for i in range(max_len):
        candidates = [wl[i].lower().strip() for wl in word_lists if i < len(wl)]
        if candidates:
            voted.append(collections.Counter(candidates).most_common(1)[0][0])
    return voted

def multi_frame_ocr(frames_b64: list):
    """OCR up to 3 frames, return majority-voted text + per-frame texts + avg conf."""
    per_frame, confs = [], []
    t0 = time.time()
    for b64 in frames_b64[:3]:
        try:
            img_bytes = base64.b64decode(b64.split(",")[1] if "," in b64 else b64)
            resp      = vision_client.document_text_detection(image=vision.Image(content=img_bytes))
            raw       = resp.full_text_annotation.text or ""
            per_frame.append(raw)
            confs.extend([w.confidence for pg in resp.full_text_annotation.pages
                          for bl in pg.blocks for pa in bl.paragraphs
                          for w in pa.words if w.confidence is not None])
        except:
            per_frame.append("")
    ocr_ms   = _elapsed(t0)
    avg_conf = sum(confs)/len(confs) if confs else 1.0
    word_lists = [t.split() for t in per_frame if t.strip()]
    if not word_lists:    return "", per_frame, avg_conf, ocr_ms
    if len(word_lists)==1: return " ".join(word_lists[0]), per_frame, avg_conf, ocr_ms
    return " ".join(_vote_words(word_lists)), per_frame, avg_conf, ocr_ms


# ══════════════════════════════════════════════════════════════════
# N9 — Reading-Direction & Layout Repair
# ══════════════════════════════════════════════════════════════════

_RTL_LANGS = {"ar","he","fa","ur","yi","dv","ps","ug"}

def is_rtl(lang_code: str) -> bool:
    return lang_code.split("-")[0].lower() in _RTL_LANGS

def reconstruct_reading_order(response) -> str:
    blocks_text = []
    for page in response.full_text_annotation.pages:
        page_w = page.width or 1
        for block in page.blocks:
            verts = block.bounding_box.vertices
            if not verts: continue
            xs = [v.x for v in verts]; ys = [v.y for v in verts]
            cx = sum(xs)/len(xs)/page_w; cy = sum(ys)/len(ys)
            btxt = ""
            for para in block.paragraphs:
                for word in para.words:
                    btxt += "".join(s.text for s in word.symbols)+" "
            blocks_text.append((cx, cy, btxt.strip()))
    if not blocks_text:
        return response.full_text_annotation.text or ""
    left  = [b for b in blocks_text if b[0] < 0.5]
    right = [b for b in blocks_text if b[0] >= 0.5]
    if left and right and len(right) >= 2:
        left.sort(key=lambda b:b[1]); right.sort(key=lambda b:b[1])
        ordered = left + right
    else:
        blocks_text.sort(key=lambda b:(b[1],b[0])); ordered = blocks_text
    return " ".join(b[2] for b in ordered if b[2])


# ══════════════════════════════════════════════════════════════════
# N10 — Tone / Formality Detection
# ══════════════════════════════════════════════════════════════════

def detect_formality(text: str):
    t = text.lower()
    formal      = len(re.findall(r"\b(therefore|hereby|pursuant|furthermore|henceforth|respectfully|sincerely|regarding)\b",t))
    informal    = len(re.findall(r"\b(hey|hi|yeah|gonna|wanna|kinda|lol|omg|btw|u|ur|pls)\b",t))
    instruction = len(re.findall(r"\b(please|do not|must|ensure|required|step \d|press|click|open|close|turn)\b",t))
    excl        = text.count("!"); ques = text.count("?")
    scores = {"formal":formal*2,"informal":informal*2+excl,"instructional":instruction*2,"conversational":ques+(1 if len(text.split())<15 else 0)}
    register     = max(scores, key=scores.get)
    total        = sum(scores.values()) or 1
    formal_score = (scores["formal"]+scores["instructional"])/total
    return register, round(formal_score,2)


# ══════════════════════════════════════════════════════════════════
# N2 — Speech-Aware TTS Normalization (post-translation, lang-aware)
#   Years stay as digits through translation so target lang renders
#   them correctly in its own script before TTS expansion.
# ══════════════════════════════════════════════════════════════════

UNIT_MAP = {
    "km":"kilometers","m":"meters","cm":"centimeters","mm":"millimeters",
    "kg":"kilograms","g":"grams","mg":"milligrams","l":"liters","ml":"milliliters",
    "mph":"miles per hour","kph":"kilometers per hour",
    "hz":"hertz","khz":"kilohertz","mhz":"megahertz",
    "gb":"gigabytes","mb":"megabytes","kb":"kilobytes","tb":"terabytes",
    "%":"percent","w":"watts","kw":"kilowatts",
}
ONES=["","one","two","three","four","five","six","seven","eight","nine",
      "ten","eleven","twelve","thirteen","fourteen","fifteen","sixteen",
      "seventeen","eighteen","nineteen"]
TENS=["","","twenty","thirty","forty","fifty","sixty","seventy","eighty","ninety"]

def _n2w(n):
    if n<0:           return "negative "+_n2w(-n)
    if n==0:          return "zero"
    if n<20:          return ONES[n]
    if n<100:         return TENS[n//10]+(" "+ONES[n%10] if n%10 else "")
    if n<1000:        return ONES[n//100]+" hundred"+(" "+_n2w(n%100) if n%100 else "")
    if n<1_000_000:   return _n2w(n//1000)+" thousand"+(" "+_n2w(n%1000) if n%1000 else "")
    if n<1_000_000_000: return _n2w(n//1_000_000)+" million"+(" "+_n2w(n%1_000_000) if n%1_000_000 else "")
    return _n2w(n//1_000_000_000)+" billion"+(" "+_n2w(n%1_000_000_000) if n%1_000_000_000 else "")

_LATIN_LANGS = {"en","fr","de","es","it","pt","nl","sv","tr","id","ms","pl","ro","cs","sk","hr","fi","da","no","hu","vi"}

def speech_normalize_with_audit(text, lang_code="en"):
    audit = []
    def rec(before, after, etype):
        if str(before).strip() != str(after).strip():
            audit.append({"before":str(before).strip(),"after":str(after).strip(),"type":etype})
        return after
    text = re.sub(r"\b[A-Z0-9]{3,}-[A-Z0-9]{3,}-[A-Z0-9]{3,}\b",lambda m:rec(m.group(),"","serial_suppressed"),text)
    text = re.sub(r"\b\d{8,}\b",lambda m:rec(m.group(),"","barcode_suppressed"),text)
    def _cur(m,label): v=_n2w(int(float(m.group(1)))); return rec(m.group(),f"{v} {label}","currency_expanded")
    text = re.sub(r"\$(\d+(?:\.\d+)?)",lambda m:_cur(m,"dollars"),text)
    text = re.sub(r"₹(\d+(?:\.\d+)?)",lambda m:_cur(m,"rupees"),text)
    text = re.sub(r"€(\d+(?:\.\d+)?)",lambda m:_cur(m,"euros"),text)
    upat = r"(\d+(?:\.\d+)?)\s*("+"|".join(re.escape(u) for u in UNIT_MAP)+r")\b"
    def _unit(m):
        try: wn=_n2w(int(float(m.group(1))))
        except: wn=m.group(1)
        return rec(m.group(),f"{wn} {UNIT_MAP.get(m.group(2).lower(),m.group(2))}","unit_expanded")
    text = re.sub(upat,_unit,text,flags=re.I)
    if lang_code.split("-")[0].lower() in _LATIN_LANGS:
        def _int(m):
            try: return rec(m.group(),_n2w(int(m.group())),"number_expanded")
            except: return m.group()
        text = re.sub(r"(?<![A-Za-z\-])\b(\d{1,5})\b(?![A-Za-z\-])",_int,text)
        for sfx,word in [("st","first"),("nd","second"),("rd","third"),("th","th")]:
            text = re.sub(rf"\b(\d+){sfx}\b",lambda m,w=word:rec(m.group(),_n2w(int(m.group(1)))+" "+w,"ordinal"),text)
    for sym,word in [("&"," and "),("+"," plus "),("="," equals "),("@"," at "),("/"," or ")]:
        if sym in text: rec(sym,word.strip(),"symbol_expanded"); text=text.replace(sym,word)
    return re.sub(r"\s+"," ",text).strip(), audit

def speech_normalize(text,lang_code="en"):
    n,_ = speech_normalize_with_audit(text,lang_code); return n


# ══════════════════════════════════════════════════════════════════
# N4 — Semantic Frame Deduplication
# ══════════════════════════════════════════════════════════════════

_semantic_window = []
_WINDOW_SIZE     = 6
_SEM_THRESHOLD   = 0.82

def _ngram_set(text, n=3):
    t = re.sub(r"\s+"," ",text.lower().strip())
    return {t[i:i+n] for i in range(len(t)-n+1)} if len(t)>=n else {t}

def semantic_similarity(text):
    if not _semantic_window: return 0.0
    ng = _ngram_set(text)
    if not ng: return 0.0
    return max(len(ng&p)/len(ng|p) if ng|p else 0.0 for p in _semantic_window)

def register_translation(text):
    global _semantic_window
    _semantic_window.append(_ngram_set(text))
    if len(_semantic_window)>_WINDOW_SIZE: _semantic_window.pop(0)


# ══════════════════════════════════════════════════════════════════
# N5 — Script-Aware TTS Voice Routing
# ══════════════════════════════════════════════════════════════════

_SCRIPT_PROFILES = {
    "Devanagari": {"gender":texttospeech.SsmlVoiceGender.FEMALE, "rate":0.90,"pitch":0.0},
    "Tamil":      {"gender":texttospeech.SsmlVoiceGender.FEMALE, "rate":0.88,"pitch":0.0},
    "Telugu":     {"gender":texttospeech.SsmlVoiceGender.FEMALE, "rate":0.88,"pitch":0.0},
    "Arabic":     {"gender":texttospeech.SsmlVoiceGender.MALE,   "rate":0.85,"pitch":-1.0},
    "CJK":        {"gender":texttospeech.SsmlVoiceGender.FEMALE, "rate":0.95,"pitch":1.0},
    "Cyrillic":   {"gender":texttospeech.SsmlVoiceGender.NEUTRAL,"rate":0.92,"pitch":0.0},
    "Latin":      {"gender":texttospeech.SsmlVoiceGender.NEUTRAL,"rate":1.0, "pitch":0.0},
    "Bengali":    {"gender":texttospeech.SsmlVoiceGender.FEMALE, "rate":0.90,"pitch":0.0},
    "Gurmukhi":   {"gender":texttospeech.SsmlVoiceGender.MALE,   "rate":0.90,"pitch":0.0},
    "Gujarati":   {"gender":texttospeech.SsmlVoiceGender.FEMALE, "rate":0.90,"pitch":0.0},
    "Malayalam":  {"gender":texttospeech.SsmlVoiceGender.FEMALE, "rate":0.88,"pitch":0.0},
    "Kannada":    {"gender":texttospeech.SsmlVoiceGender.FEMALE, "rate":0.88,"pitch":0.0},
    "Korean":     {"gender":texttospeech.SsmlVoiceGender.FEMALE, "rate":0.95,"pitch":1.0},
}

def detect_script(text):
    counts = {}
    for ch in text:
        name = unicodedata.name(ch,"")
        for script,kw in [("Devanagari","DEVANAGARI"),("Tamil","TAMIL"),
                          ("Telugu","TELUGU"),("Arabic","ARABIC"),("CJK","CJK"),
                          ("Cyrillic","CYRILLIC"),("Bengali","BENGALI"),
                          ("Gurmukhi","GURMUKHI"),("Gujarati","GUJARATI"),
                          ("Malayalam","MALAYALAM"),("Kannada","KANNADA"),
                          ("Korean","HANGUL"),("Latin","LATIN")]:
            if kw in name: counts[script]=counts.get(script,0)+1; break
    return max(counts,key=counts.get) if counts else "Latin"

def _tts_with_script(text, lang_code):
    script = detect_script(text)
    p      = _SCRIPT_PROFILES.get(script,_SCRIPT_PROFILES["Latin"])
    t0     = time.time()
    rsp    = tts_client.synthesize_speech(
        input        = texttospeech.SynthesisInput(text=text),
        voice        = texttospeech.VoiceSelectionParams(language_code=lang_code,ssml_gender=p["gender"]),
        audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3,
                                                speaking_rate=p["rate"],pitch=p["pitch"]))
    return rsp.audio_content, _elapsed(t0), script


# ══════════════════════════════════════════════════════════════════
# N12 — Latency Anomaly Detection
# ══════════════════════════════════════════════════════════════════

_lat_history = {"ocr":[],"trans":[],"tts":[],"total":[]}
_LAT_WINDOW  = 30

def check_anomaly(ocr_ms, trans_ms, tts_ms, total_ms) -> dict:
    flags = {}
    for stage, val in [("ocr",ocr_ms),("trans",trans_ms),("tts",tts_ms),("total",total_ms)]:
        hist = _lat_history[stage]
        if len(hist) >= 5:
            mu  = statistics.mean(hist)
            sig = statistics.stdev(hist) if len(hist)>1 else 0
            flags[stage] = bool(val > mu+2*sig and val > 500)
        else:
            flags[stage] = False
        if val > 0:
            hist.append(val)
            if len(hist)>_LAT_WINDOW: hist.pop(0)
    return flags


# ══════════════════════════════════════════════════════════════════
# N13 — Auto-Benchmark Sampling (silent, runs in background)
#   On image upload, runs 2 additional quick OCR-only passes and
#   returns the latency distribution — no extra button needed.
# ══════════════════════════════════════════════════════════════════

def silent_benchmark(image_bytes: bytes, tgt: str, primary_ocr_ms: float) -> dict:
    """Run 2 extra fast OCR passes to build a 3-sample distribution."""
    samples = [primary_ocr_ms]
    try:
        image = vision.Image(content=image_bytes)
        for _ in range(2):
            t = time.time()
            vision_client.document_text_detection(image=image)
            samples.append(_elapsed(t))
    except:
        pass
    if len(samples) < 2:
        return {"samples":samples,"mean":samples[0],"stdev":0,"min":samples[0],"max":samples[0]}
    return {
        "samples": [round(s,2) for s in samples],
        "mean":    round(statistics.mean(samples),2),
        "stdev":   round(statistics.stdev(samples),2),
        "min":     round(min(samples),2),
        "max":     round(max(samples),2),
    }


# ══════════════════════════════════════════════════════════════════
# Shared helpers
# ══════════════════════════════════════════════════════════════════

def _ocr_conf(response):
    scores = [word.confidence
              for page  in response.full_text_annotation.pages
              for block in page.blocks
              for para  in block.paragraphs
              for word  in para.words
              if word.confidence is not None]
    return sum(scores)/len(scores) if scores else 1.0

def _translate(text, tgt):
    t = time.time()
    r = translate_client.translate(text, target_language=tgt)
    return r["translatedText"], _elapsed(t), r.get("detectedSourceLanguage","")

def _log(sid, mode, ocr_ms, stt_ms, trans_ms, tts_ms, total_ms,
         ocr_conf, stt_conf, n_repairs, n_expands, sem_sim,
         domain, skip_reason, script, src, tgt, c_in, c_out,
         anomaly_flag, vote_frames=0, bench_mean=0):
    with open(LOG_FILE,"a",newline="") as f:
        csv.writer(f).writerow([
            round(time.time(),4), sid, mode,
            round(ocr_ms,2), round(stt_ms,2),
            round(trans_ms,2), round(tts_ms,2), round(total_ms,2),
            round(ocr_conf,4), round(stt_conf,4),
            n_repairs, n_expands, round(sem_sim,4), domain,
            skip_reason, script, src, tgt, c_in, c_out,
            anomaly_flag, vote_frames, round(bench_mean,2),
        ])


# ══════════════════════════════════════════════════════════════════
# ROUTES — exactly 4 features
# ══════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template("index.html")


# ── Feature 1: Live Camera ────────────────────────────────────────
@app.route("/process", methods=["POST"])
def process_image():
    """
    Live camera frame pipeline.
    N8 multi-frame voting is triggered automatically when the frontend
    sends multiple frames (frames=[b64,b64,...]) instead of a single image.
    Single-frame mode is the default for every tick.
    """
    t0        = time.time()
    sid       = _next_session()
    body      = request.json
    ui_lang   = body.get("target_lang","en-US")
    tgt       = ui_lang.split("-")[0]
    want_tts  = body.get("want_speech", False)
    use_disc  = body.get("use_discourse", True)

    # N8 — check if frontend sent multiple frames for voting
    frames_list = body.get("frames", [])
    single_img  = body.get("image","")

    vote_count = 0
    if frames_list and len(frames_list) >= 2:
        # Multi-frame voting path (N8)
        voted_text, per_frame_texts, conf, ocr_ms = multi_frame_ocr(frames_list)
        vote_count = len(frames_list)
        raw = voted_text
        # We need a Vision response for word confidences — use first frame
        try:
            img1  = base64.b64decode(frames_list[0].split(",")[1] if "," in frames_list[0] else frames_list[0])
            resp  = vision_client.document_text_detection(image=vision.Image(content=img1))
            word_confs = extract_word_confidences(resp)
        except:
            word_confs = []
    else:
        # Single frame path
        img_bytes = base64.b64decode(single_img.split(",")[1])
        image     = vision.Image(content=img_bytes)
        ocr_t     = time.time()
        resp      = vision_client.document_text_detection(image=image)
        ocr_ms    = _elapsed(ocr_t)
        # N9 — reading-order reconstruction
        raw       = reconstruct_reading_order(resp)
        conf      = _ocr_conf(resp)
        word_confs = extract_word_confidences(resp)

    if not raw:
        return jsonify({"translated":"","skipped":False,"session_id":sid,
                        "sem_similarity":0,"script_family":"","detected_lang":"",
                        "word_confidences":[],"domain":"","formality":"",
                        "discourse_extended":False,"anomalies":{}})

    if not frames_list:
        conf = _ocr_conf(resp)
        if conf < CONFIDENCE_THRESHOLD:
            reason = f"low_confidence:{conf:.2f}"
            _log(sid,"live_camera",ocr_ms,0,0,0,_elapsed(t0),conf,0,0,0,0.0,"",reason,"","auto",tgt,len(raw),0,False)
            return jsonify({
                "translated":"⚠ Low OCR confidence — move camera closer.",
                "confidence":round(conf,2),"latency":_elapsed(t0),
                "ocr_latency":round(ocr_ms,2),"trans_latency":0,"tts_latency":0,
                "raw_text":raw,"cleaned_text":raw,"ocr_diff":[],"tts_audit":[],
                "word_confidences":[],"domain":"","formality":"","formality_score":0,
                "skipped":False,"skip_reason":reason,"session_id":sid,
                "sem_similarity":0,"script_family":"","detected_lang":"",
                "discourse_extended":False,"anomalies":{},"rtl":False,
            })

    # N1 — OCR repair
    cleaned, ocr_diff = normalize_text_with_diff(raw)

    # N6 — Discourse memory
    if use_disc:
        combined, is_extended, buf_snap = discourse_update(cleaned)
    else:
        combined, is_extended, buf_snap = cleaned, False, []

    # N10 — Formality
    formality, formality_score = detect_formality(combined)

    # N7 — Domain-adaptive translation
    translated, trans_ms, detected_lang, domain_label, hint_used = domain_translate(combined, tgt)
    final = reconstruct_context(translated)

    # N4 — Semantic deduplication
    sem_sim = semantic_similarity(final)
    register_translation(final)
    sem_new = sem_sim < _SEM_THRESHOLD

    # N2 + N5 — TTS normalization + script-aware voice
    tts_ms, audio_b64, tts_audit, script_family = 0, None, [], ""
    if want_tts and sem_new:
        normed, tts_audit = speech_normalize_with_audit(final, tgt)
        ab, tts_ms, script_family = _tts_with_script(normed, ui_lang)
        audio_b64 = base64.b64encode(ab).decode()
    else:
        _, tts_audit = speech_normalize_with_audit(final, tgt)
        script_family = detect_script(final)

    if not script_family: script_family = detect_script(final)

    # N12 — Anomaly detection
    anomalies    = check_anomaly(ocr_ms, trans_ms, tts_ms, _elapsed(t0))
    anomaly_flag = any(anomalies.values())

    # N9 — RTL flag for UI
    rtl = is_rtl(tgt)

    total_ms = _elapsed(t0)
    _log(sid,"live_camera",ocr_ms,0,trans_ms,tts_ms,total_ms,
         conf,0,len(ocr_diff),len(tts_audit),sem_sim,
         domain_label,"semantic_dup" if not sem_new else "",
         script_family,detected_lang,tgt,len(raw),len(final),
         anomaly_flag,vote_count,0)

    return jsonify({
        "translated":         final,
        "raw_text":           raw,
        "cleaned_text":       cleaned,
        "discourse_text":     combined,
        "discourse_extended": is_extended,
        "discourse_buffer":   buf_snap,
        "detected_lang":      detected_lang,
        "confidence":         round(conf,2),
        "latency":            round(total_ms,2),
        "ocr_latency":        round(ocr_ms,2),
        "trans_latency":      round(trans_ms,2),
        "tts_latency":        round(tts_ms,2),
        "ocr_diff":           ocr_diff,
        "tts_audit":          tts_audit,
        "word_confidences":   word_confs[:40],
        "domain":             domain_label,
        "domain_hint":        hint_used,
        "formality":          formality,
        "formality_score":    formality_score,
        "audio":              audio_b64,
        "sem_similarity":     round(sem_sim,3),
        "sem_new":            sem_new,
        "script_family":      script_family,
        "rtl":                rtl,
        "anomalies":          anomalies,
        "anomaly_flag":       anomaly_flag,
        "vote_frames":        vote_count,
        "skipped":            False,
        "skip_reason":        "",
        "session_id":         sid,
        "target_lang":        ui_lang,
    })


# ── Feature 2: Image Upload ───────────────────────────────────────
@app.route("/process_upload", methods=["POST"])
def process_upload():
    t0       = time.time()
    sid      = _next_session()
    ui_lang  = request.form.get("target_lang","en-US")
    tgt      = ui_lang.split("-")[0]
    want_tts = request.form.get("want_speech","false").lower()=="true"

    img_bytes = request.files["image"].read()
    image     = vision.Image(content=img_bytes)

    ocr_t  = time.time()
    resp   = vision_client.document_text_detection(image=image)
    ocr_ms = _elapsed(ocr_t)

    # N9 — reading-order reconstruction
    raw = reconstruct_reading_order(resp)
    if not raw:
        return jsonify({"error":"No text detected in image."})

    conf       = _ocr_conf(resp)
    word_confs = extract_word_confidences(resp)

    # N1 — OCR repair
    cleaned, ocr_diff = normalize_text_with_diff(raw)

    # N10 — formality
    formality, formality_score = detect_formality(cleaned)

    # N7 — domain-adaptive translation
    translated, trans_ms, dlang, domain_label, hint = domain_translate(cleaned, tgt)
    final = reconstruct_context(translated)

    # N9 — RTL
    rtl = is_rtl(tgt)

    # N2 + N5 — TTS normalization + script-aware voice
    tts_ms, audio_b64, tts_audit, script_family = 0, None, [], detect_script(final)
    if want_tts:
        normed, tts_audit = speech_normalize_with_audit(final, tgt)
        ab, tts_ms, script_family = _tts_with_script(normed, ui_lang)
        audio_b64 = base64.b64encode(ab).decode()

    # N12 — anomaly
    anomalies    = check_anomaly(ocr_ms, trans_ms, tts_ms, _elapsed(t0))
    anomaly_flag = any(anomalies.values())

    # N13 — silent benchmark (2 extra OCR passes → 3-sample distribution)
    bench = silent_benchmark(img_bytes, tgt, ocr_ms)

    total_ms = _elapsed(t0)
    _log(sid,"image_upload",ocr_ms,0,trans_ms,tts_ms,total_ms,
         conf,0,len(ocr_diff),len(tts_audit),0.0,
         domain_label,"",script_family,dlang,tgt,len(raw),len(final),
         anomaly_flag,0,bench["mean"])

    return jsonify({
        "raw_text":       raw,
        "cleaned_text":   cleaned,
        "translated":     final,
        "detected_lang":  dlang,
        "confidence":     round(conf,2),
        "ocr_latency":    round(ocr_ms,2),
        "trans_latency":  round(trans_ms,2),
        "tts_latency":    round(tts_ms,2),
        "total_latency":  round(total_ms,2),
        "ocr_diff":       ocr_diff,
        "tts_audit":      tts_audit,
        "word_confidences": word_confs[:40],
        "domain":         domain_label,
        "domain_hint":    hint,
        "formality":      formality,
        "formality_score":formality_score,
        "script_family":  script_family,
        "rtl":            rtl,
        "anomalies":      anomalies,
        "anomaly_flag":   anomaly_flag,
        "benchmark":      bench,
        "audio":          audio_b64,
        "session_id":     sid,
    })


# ── Feature 3: Text → Speech ─────────────────────────────────────
@app.route("/text_to_speech", methods=["POST"])
def text_to_speech_route():
    t0       = time.time()
    sid      = _next_session()
    body     = request.json
    text_in  = body.get("text","").strip()
    ui_lang  = body.get("target_lang","en-US")
    tgt      = ui_lang.split("-")[0]
    do_trans = body.get("translate",True)
    if not text_in:
        return jsonify({"error":"No text provided."})

    # N10 — formality
    formality, formality_score = detect_formality(text_in)

    trans_ms, dlang = 0, ""
    if do_trans:
        translated, trans_ms, dlang, domain_label, hint = domain_translate(text_in, tgt)
    else:
        translated = text_in; domain_label = hint = ""

    # N2 + N5
    normed, tts_audit = speech_normalize_with_audit(translated, tgt)
    ab, tts_ms, script_family = _tts_with_script(normed, ui_lang)
    audio_b64 = base64.b64encode(ab).decode()

    # N12
    anomalies    = check_anomaly(0, trans_ms, tts_ms, _elapsed(t0))
    anomaly_flag = any(anomalies.values())

    total_ms = _elapsed(t0)
    _log(sid,"text_to_speech",0,0,trans_ms,tts_ms,total_ms,
         1,0,0,len(tts_audit),0.0,domain_label,"",
         script_family,"user_input",tgt,len(text_in),len(normed),anomaly_flag)

    return jsonify({
        "translated":     translated,
        "normalized":     normed,
        "tts_audit":      tts_audit,
        "script_family":  script_family,
        "detected_lang":  dlang,
        "domain":         domain_label,
        "domain_hint":    hint,
        "formality":      formality,
        "formality_score":formality_score,
        "audio":          audio_b64,
        "trans_latency":  round(trans_ms,2),
        "tts_latency":    round(tts_ms,2),
        "total_latency":  round(total_ms,2),
        "anomalies":      anomalies,
        "session_id":     sid,
    })


# ── Feature 4: Speech → Text ─────────────────────────────────────
@app.route("/speech_to_text", methods=["POST"])
def speech_to_text_route():
    t0       = time.time()
    sid      = _next_session()
    ab64     = request.json.get("audio")
    ui_lang  = request.json.get("target_lang","en-US")
    src_lang = request.json.get("source_lang","en-US")
    tgt      = ui_lang.split("-")[0]
    want_tts = request.json.get("want_speech",False)

    ab     = base64.b64decode(ab64)
    stt_t  = time.time()
    stt_r  = stt_client.recognize(
        config=speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.WEBM_OPUS,
            sample_rate_hertz=48000, language_code=src_lang,
            enable_automatic_punctuation=True, model="latest_long"),
        audio=speech.RecognitionAudio(content=ab))
    stt_ms = _elapsed(stt_t)

    if not stt_r.results:
        return jsonify({"error":"Could not recognise speech — please speak clearly."})

    transcript = " ".join(r.alternatives[0].transcript for r in stt_r.results)
    stt_conf   = stt_r.results[0].alternatives[0].confidence if stt_r.results else 1.0

    # N10 — formality
    formality, formality_score = detect_formality(transcript)

    # N7 — domain-adaptive translation
    translated, trans_ms, dlang, domain_label, hint = domain_translate(transcript, tgt)

    # N2 + N5
    tts_ms, out_b64, tts_audit, script_family = 0, None, [], detect_script(translated)
    if want_tts:
        normed, tts_audit = speech_normalize_with_audit(translated, tgt)
        ab2, tts_ms, script_family = _tts_with_script(normed, ui_lang)
        out_b64 = base64.b64encode(ab2).decode()

    # N12
    anomalies    = check_anomaly(0, stt_ms, tts_ms, _elapsed(t0))
    anomaly_flag = any(anomalies.values())

    total_ms = _elapsed(t0)
    _log(sid,"speech_to_text",0,stt_ms,trans_ms,tts_ms,total_ms,
         0,stt_conf,0,len(tts_audit),0.0,domain_label,"",
         script_family,src_lang,tgt,len(transcript),len(translated),anomaly_flag)

    return jsonify({
        "transcript":     transcript,
        "translated":     translated,
        "confidence":     round(stt_conf,2),
        "stt_latency":    round(stt_ms,2),
        "trans_latency":  round(trans_ms,2),
        "tts_latency":    round(tts_ms,2),
        "total_latency":  round(total_ms,2),
        "tts_audit":      tts_audit,
        "script_family":  script_family,
        "detected_lang":  dlang,
        "domain":         domain_label,
        "domain_hint":    hint,
        "formality":      formality,
        "formality_score":formality_score,
        "anomalies":      anomalies,
        "session_id":     sid,
        "audio":          out_b64,
    })


# ── Utility routes ────────────────────────────────────────────────
@app.route("/reset_discourse", methods=["POST"])
def reset_discourse_route():
    reset_discourse(); return jsonify({"ok":True})

@app.route("/log_stats")
def log_stats():
    if not os.path.exists(LOG_FILE): return jsonify({"count":0})
    with open(LOG_FILE) as f: rows = list(csv.DictReader(f))
    if not rows: return jsonify({"count":0})
    def avg(k):
        v=[float(r[k]) for r in rows if r.get(k) not in ("","None",None)]
        return round(sum(v)/len(v),2) if v else 0
    mc={}
    for r in rows: mc[r.get("mode","?")]=mc.get(r.get("mode","?"),0)+1
    dc={}
    for r in rows:
        d=r.get("domain_detected","") or r.get("domain","")
        if d: dc[d]=dc.get(d,0)+1
    anomaly_count=sum(1 for r in rows if r.get("anomaly_flag") in ("True","1","true"))
    return jsonify({
        "count":len(rows),"avg_total_ms":avg("total_latency_ms"),
        "avg_ocr_ms":avg("ocr_latency_ms"),"avg_stt_ms":avg("stt_latency_ms"),
        "avg_trans_ms":avg("translation_latency_ms"),"avg_tts_ms":avg("tts_latency_ms"),
        "avg_ocr_conf":avg("ocr_confidence"),"avg_stt_conf":avg("stt_confidence"),
        "avg_repairs":avg("tokens_repaired"),"avg_expansions":avg("expansions_applied"),
        "avg_sem_sim":avg("semantic_similarity"),
        "mode_counts":mc,"domain_counts":dc,"anomaly_count":anomaly_count,"recent":rows[-20:],
    })

@app.route("/download_log")
def download_log():
    return send_file(LOG_FILE,as_attachment=True,download_name="polylens_log.csv")

if __name__ == "__main__":
    app.run(debug=True)