"""
Career Development Centre Resume Analyzer
==========================================
Ahmedabad University — CDC Institutional Tool
Production-ready ATS resume analyzer with algorithmic scoring engine.
Deployment target: Streamlit Cloud
Python: 3.10+
"""

import os
import re
import math
import string
import streamlit as st
from collections import Counter
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from groq import Groq, RateLimitError, APIStatusError, APIConnectionError

# ─────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────

MAX_RESUME_CHARS = 4000
MAX_JD_CHARS = 1500
MODEL_ID = "llama-3.1-8b-instant"
MAX_OUTPUT_TOKENS = 2048   # generous limit — score is algorithmic, LLM only does feedback

# ─────────────────────────────────────────
# ENV LOADING
# ─────────────────────────────────────────

load_dotenv()

def get_api_key() -> str:
    try:
        key = st.secrets["GROQ_API_KEY"]
        if key:
            return key
    except (KeyError, FileNotFoundError):
        pass
    key = os.getenv("GROQ_API_KEY", "")
    return key

# ─────────────────────────────────────────
# GROQ CLIENT INIT (cached) — DO NOT MODIFY
# ─────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def get_groq_client(api_key: str) -> Groq:
    return Groq(api_key=api_key)

# ─────────────────────────────────────────
# PDF EXTRACTION — DO NOT MODIFY
# ─────────────────────────────────────────

def extract_pdf_text(uploaded_file) -> str:
    try:
        reader = PdfReader(uploaded_file)
        pages_text = []
        for page in reader.pages:
            content = page.extract_text()
            if content and content.strip():
                pages_text.append(content.strip())
        return "\n".join(pages_text)
    except Exception as e:
        st.warning(f"PDF parsing issue: {e}. Try re-saving the PDF as a standard PDF/A.")
        return ""

# ─────────────────────────────────────────
# TOKEN SAFETY — DO NOT MODIFY
# ─────────────────────────────────────────

def truncate_safely(text: str, max_chars: int, label: str = "input") -> str:
    if len(text) > max_chars:
        st.info(
            f"INFO: {label} truncated to {max_chars} characters "
            f"(was {len(text)}) to stay within processing limits."
        )
        return text[:max_chars]
    return text

# ─────────────────────────────────────────
# PROMPT BUILDER — DO NOT MODIFY
# ─────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a senior career consultant at a university Career Development Centre. "
    "Your job is to provide detailed, structured, and genuinely helpful resume feedback to students and graduates. "
    "IMPORTANT RULES: "
    "Do NOT include an ATS score, numeric score, or any score out of 100 — scoring is handled separately by the system. "
    "Do NOT truncate or cut your response short — write complete, thorough feedback for every section requested. "
    "Be specific — always reference actual content from the resume rather than giving generic advice. "
    "Use markdown formatting with bold headings and bullet points. "
    "Before/after rewrite examples are highly encouraged where relevant. "
    "Your response should feel like written feedback from a real career advisor, not a short automated summary."
)

def build_prompt(mode: str, resume_text: str, job_description: str) -> str:
    jd_section = (
        f"\n\nJob Description:\n{job_description}"
        if job_description.strip()
        else "\n\n(No job description provided — perform general analysis.)"
    )

    if mode == "Quick Scan":
        instructions = (
            "You are reviewing this resume as a career coach. Provide a well-rounded quick scan covering:\n\n"
            "**1. Best-Fit Job Title**\n"
            "State the most suitable role this resume positions the candidate for, and briefly explain why.\n\n"
            "**2. Top 3 Strengths**\n"
            "List the three strongest aspects of this resume with a sentence of explanation for each.\n\n"
            "**3. Top 3 Areas for Improvement**\n"
            "Identify specific weaknesses and give a clear, actionable suggestion to fix each one.\n\n"
            "**4. One Quick Win**\n"
            "Give one single change the candidate can make today that would most improve this resume.\n\n"
            "Be specific — reference actual content from the resume, not generic advice."
        )
    elif mode == "Detailed Analysis":
        instructions = (
            "You are reviewing this resume as a senior career consultant. Provide a comprehensive, detailed analysis:\n\n"
            "**1. Career Profile Summary**\n"
            "Identify the most suitable profession and seniority level. Explain what the resume communicates "
            "about the candidate's background and trajectory.\n\n"
            "**2. Top 5 Strengths**\n"
            "List and explain the five strongest elements. Be specific — cite actual phrases, sections, or "
            "achievements from the resume.\n\n"
            "**3. Top 5 Improvement Areas**\n"
            "For each weakness, provide a concrete, actionable fix. Where possible, show a before/after example.\n\n"
            "**4. Section-by-Section Feedback**\n"
            "Review each major section present (Summary/Objective, Experience, Skills, Education, Projects):\n"
            "- What works well\n"
            "- What is missing or weak\n"
            "- A specific recommendation\n\n"
            "**5. Keyword & Role Alignment**\n"
            "If a job description is provided, assess how well the resume's language matches the role requirements. "
            "Name specific missing terms or concepts.\n\n"
            "**6. Overall Impression**\n"
            "Write 2-3 sentences summarising what a recruiter would think reading this resume for 30 seconds.\n\n"
            "Be thorough and specific. Reference real content from the resume throughout."
        )
    else:  # ATS Optimization
        instructions = (
            "You are an ATS optimization specialist. Provide a detailed optimization report:\n\n"
            "**1. Missing Keywords**\n"
            "List 8-12 specific keywords or phrases this resume is missing, based on the job description "
            "or industry norms. For each keyword, explain where and how to naturally insert it.\n\n"
            "**2. Bullet Point Rewrites**\n"
            "Pick the 4 weakest bullet points from the resume and rewrite each one. Show clearly:\n"
            "- BEFORE: [original]\n"
            "- AFTER: [improved version with action verb + impact + metric]\n\n"
            "**3. Formatting & Parsing Issues**\n"
            "Identify any elements that may cause ATS parsing failures (tables, columns, headers, "
            "graphics, non-standard fonts). Give specific fixes.\n\n"
            "**4. Structure & Section Gaps**\n"
            "Identify any missing standard sections and explain why ATS systems expect them.\n\n"
            "**5. File & Submission Tips**\n"
            "Advise on file format, naming conventions, and submission best practices for ATS systems.\n\n"
            "**6. Priority Action List**\n"
            "End with a numbered list of the top 5 changes to make, ranked by impact.\n\n"
            "Be specific — use examples directly from the resume."
        )

    return (
        f"{instructions}\n\n"
        f"Resume:\n{resume_text}"
        f"{jd_section}"
    )

# ─────────────────────────────────────────
# LLM INFERENCE — DO NOT MODIFY
# ─────────────────────────────────────────

def run_analysis(client: Groq, prompt: str) -> tuple:
    try:
        completion = client.chat.completions.create(
            model=MODEL_ID,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.5,
            max_tokens=MAX_OUTPUT_TOKENS,
            top_p=0.9,
            stream=False,
        )

        choices = getattr(completion, "choices", None)
        if not choices or len(choices) == 0:
            return "The system returned an empty response. Please try again.", False

        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", None) if message else None

        if not content or not content.strip():
            return "The system returned no content. Please try again.", False

        return content.strip(), True

    except RateLimitError:
        return (
            "RATE LIMIT REACHED. Please wait 60 seconds and try again.",
            False,
        )
    except APIConnectionError:
        return (
            "CONNECTION ERROR. Could not reach the analysis service. "
            "Check your internet connection and try again.",
            False,
        )
    except APIStatusError as e:
        if e.status_code == 401:
            return ("INVALID API KEY. Check your configuration.", False)
        elif e.status_code == 503:
            return ("Service temporarily unavailable. Please try again in a minute.", False)
        return f"API error {e.status_code}: {e.message}", False
    except Exception as e:
        return f"Unexpected error: {str(e)}", False

# ═══════════════════════════════════════════════════════════════
# ### NEW FUNCTIONS — ALGORITHMIC ATS SCORING ENGINE
# ═══════════════════════════════════════════════════════════════

# ── Constants ─────────────────────────────────────────────────

STOPWORDS = {
    "a","an","the","and","or","but","in","on","at","to","for","of","with",
    "by","from","as","is","was","are","were","be","been","being","have",
    "has","had","do","does","did","will","would","could","should","may",
    "might","shall","can","need","dare","used","ought","it","its","this",
    "that","these","those","i","me","my","we","our","you","your","he","she",
    "they","them","their","what","which","who","whom","when","where","why",
    "how","all","both","each","few","more","most","other","some","such","no",
    "not","only","same","so","than","too","very","just","about","above","after",
    "also","any","because","before","between","during","here","if","into",
    "make","many","much","new","now","off","out","over","since","still",
    "through","under","until","up","use","while","work","your"
}

IMPACT_VERBS = {
    "developed","designed","implemented","led","optimized","built","improved",
    "engineered","launched","analyzed","automated","created","architected",
    "managed","delivered","established","streamlined","accelerated","reduced",
    "increased","generated","achieved","deployed","integrated","transformed",
    "coordinated","executed","produced","drove","spearheaded","negotiated",
    "mentored","trained","authored","published","presented","secured","saved",
    "expanded","scaled","migrated","refactored","documented","facilitated",
    "collaborated","researched","evaluated","initiated","planned","directed"
}

WEAK_STARTERS = [
    "responsible for",
    "worked on",
    "helped with",
    "assisted with",
    "tasked with",
    "duties included",
    "involved in",
]

RESUME_SECTIONS = {
    "education":    ["education", "academic", "qualification", "degree", "university", "college"],
    "experience":   ["experience", "employment", "work history", "professional background", "career"],
    "skills":       ["skills", "technical skills", "core competencies", "expertise", "proficiencies"],
    "projects":     ["projects", "project", "portfolio", "work samples"],
    "summary":      ["summary", "objective", "profile", "about me", "professional summary"],
}

# ── Helper utilities ──────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, split into tokens, remove stopwords."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = text.split()
    return [t for t in tokens if t not in STOPWORDS and len(t) > 2]


def _compute_tfidf_keywords(jd_text: str, top_n: int = 30) -> list[str]:
    """
    Lightweight single-document TF-IDF approximation.
    Since we have one document, IDF = 1 for all; we rank purely by TF
    after stopword removal — good enough for keyword extraction.
    Returns top_n keywords by frequency.
    """
    tokens = _tokenize(jd_text)
    if not tokens:
        return []
    freq = Counter(tokens)
    # Boost multi-char tokens that look domain-specific (len >= 4)
    ranked = sorted(freq.items(), key=lambda x: (x[1], len(x[0])), reverse=True)
    return [word for word, _ in ranked[:top_n]]


def _get_bullet_lines(resume_text: str) -> list[str]:
    """Extract lines that look like bullet points."""
    lines = resume_text.splitlines()
    bullets = []
    for line in lines:
        stripped = line.strip()
        # Lines starting with common bullet markers or dashes
        if stripped and (
            stripped[0] in ("•", "-", "–", "▪", "·", "*", "◦")
            or (len(stripped) > 3 and stripped[0].islower())
        ):
            bullets.append(stripped.lstrip("•-–▪·*◦ ").strip())
        elif len(stripped) > 20 and stripped[0].isupper() and not stripped.endswith(":"):
            # Could be a sentence-style bullet without marker
            bullets.append(stripped)
    return bullets

# ── Module 1: Keyword Gap Detection (weight 40%) ──────────────

def analyze_keywords(resume_text: str, job_description: str) -> dict:
    """
    Extract JD keywords via TF-IDF approximation.
    Compare against resume tokens.
    Returns matched, missing, and match percentage.
    """
    if not job_description.strip():
        # No JD provided — score neutrally at 60 (can't penalise unfairly)
        return {
            "jd_provided": False,
            "matched": [],
            "missing": [],
            "match_pct": 60.0,
            "score": 60.0,
        }

    jd_keywords = _compute_tfidf_keywords(job_description, top_n=25)
    resume_tokens = set(_tokenize(resume_text))

    matched = [kw for kw in jd_keywords if kw in resume_tokens]
    missing = [kw for kw in jd_keywords if kw not in resume_tokens]
    match_pct = (len(matched) / len(jd_keywords) * 100) if jd_keywords else 0.0

    return {
        "jd_provided": True,
        "matched": matched[:15],   # cap display list
        "missing": missing[:15],
        "match_pct": round(match_pct, 1),
        "score": round(match_pct, 1),
    }

# ── Module 2: Bullet Point Strength Scoring (weight 15%) ──────

def analyze_bullet_strength(resume_text: str) -> dict:
    """
    Detect weak vs strong bullets.
    Weak = starts with phrases like 'Responsible for', 'Worked on'.
    Strong = starts with an action/impact verb.
    """
    bullets = _get_bullet_lines(resume_text)
    if not bullets:
        return {"strong": 0, "weak": 0, "total": 0, "score": 50.0}

    weak_count = 0
    strong_count = 0
    weak_examples = []

    for bullet in bullets:
        lower = bullet.lower()
        is_weak = any(lower.startswith(ws) for ws in WEAK_STARTERS)
        first_word = lower.split()[0] if lower.split() else ""
        is_strong_verb = first_word in IMPACT_VERBS

        if is_weak:
            weak_count += 1
            if len(weak_examples) < 3:
                weak_examples.append(bullet[:80])
        elif is_strong_verb:
            strong_count += 1

    total = len(bullets)
    # Score: strong bullets add, weak subtract
    raw = ((strong_count * 1.0) - (weak_count * 1.5)) / total
    score = max(0.0, min(100.0, 50.0 + raw * 50.0))

    return {
        "strong": strong_count,
        "weak": weak_count,
        "total": total,
        "weak_examples": weak_examples,
        "score": round(score, 1),
    }

# ── Module 3: Impact Verb Detection (weight 10%) ──────────────

def analyze_impact_verbs(resume_text: str) -> dict:
    """Count bullet points starting with predefined impact verbs."""
    bullets = _get_bullet_lines(resume_text)
    if not bullets:
        return {"impact_count": 0, "total": 0, "pct": 0.0, "score": 0.0}

    impact_count = 0
    verbs_found = []
    for bullet in bullets:
        first_word = bullet.lower().split()[0] if bullet.split() else ""
        if first_word in IMPACT_VERBS:
            impact_count += 1
            if first_word not in verbs_found:
                verbs_found.append(first_word)

    pct = (impact_count / len(bullets)) * 100
    # Score maps 0–100% coverage → 0–100
    score = min(100.0, pct * 1.2)  # slight boost since 80% coverage is excellent

    return {
        "impact_count": impact_count,
        "total": len(bullets),
        "pct": round(pct, 1),
        "verbs_found": verbs_found[:10],
        "score": round(score, 1),
    }

# ── Module 4: Quantification Detection (weight 15%) ───────────

def analyze_quantification(resume_text: str) -> dict:
    """Detect bullet points with measurable impact (numbers, %, currency, metrics)."""
    bullets = _get_bullet_lines(resume_text)
    if not bullets:
        return {"quantified": 0, "total": 0, "ratio": 0.0, "score": 0.0}

    # Patterns: percentages, plain numbers ≥2 digits, currency, metric phrases
    quant_patterns = [
        r"\d+%",                          # percentages
        r"\$[\d,]+",                       # currency
        r"\d{2,}",                         # numbers (2+ digits)
        r"(?:increased|reduced|improved|decreased|grew|saved|generated)\s+by",
        r"(?:x\d+|\d+x)",                 # multiplier e.g. 3x
        r"(?:million|billion|thousand)",
    ]
    combined = re.compile("|".join(quant_patterns), re.IGNORECASE)

    quantified = [b for b in bullets if combined.search(b)]
    ratio = len(quantified) / len(bullets)
    score = min(100.0, ratio * 150.0)  # 67% quantification → 100 score

    return {
        "quantified": len(quantified),
        "total": len(bullets),
        "ratio": round(ratio * 100, 1),
        "examples": [q[:80] for q in quantified[:3]],
        "score": round(score, 1),
    }

# ── Module 5: Resume Structure Detection (weight 10%) ─────────

def analyze_structure(resume_text: str) -> dict:
    """Check for presence of standard resume sections."""
    lower_text = resume_text.lower()
    found = {}
    missing = []

    for section, keywords in RESUME_SECTIONS.items():
        detected = any(kw in lower_text for kw in keywords)
        found[section] = detected
        if not detected:
            missing.append(section)

    present_count = sum(1 for v in found.values() if v)
    score = (present_count / len(RESUME_SECTIONS)) * 100

    return {
        "found": found,
        "missing_sections": missing,
        "present_count": present_count,
        "total_sections": len(RESUME_SECTIONS),
        "score": round(score, 1),
    }

# ── Module 6: ATS Formatting Check (weight 10%) ───────────────

def analyze_formatting(resume_text: str) -> dict:
    """
    Penalise ATS-hostile formatting artifacts.
    Rewards clean, text-based resumes.
    """
    penalties = []
    score = 100.0

    # Penalty: vertical bars (table remnants)
    pipe_count = resume_text.count("|")
    if pipe_count > 5:
        score -= 20
        penalties.append(f"Vertical bars detected ({pipe_count}) — likely table remnants that confuse ATS parsers.")

    # Penalty: very short resume (< 200 words)
    word_count = len(resume_text.split())
    if word_count < 200:
        score -= 25
        penalties.append(f"Resume is very short ({word_count} words). ATS systems prefer 400–800 words.")
    elif word_count > 1200:
        score -= 10
        penalties.append(f"Resume may be too long ({word_count} words). Consider condensing to 1–2 pages.")

    # Penalty: excessive special chars (graphics/symbols)
    special_chars = sum(1 for c in resume_text if ord(c) > 127)
    if special_chars > 50:
        score -= 15
        penalties.append("Non-standard characters detected — may indicate graphics or special fonts that ATS cannot parse.")

    # Penalty: very few line breaks (wall of text)
    line_count = resume_text.count("\n")
    if line_count < 10:
        score -= 15
        penalties.append("Low line-break count detected — resume may lack clear section separation.")

    score = max(0.0, min(100.0, score))
    good_signs = []
    if word_count >= 200 and pipe_count <= 5:
        good_signs.append("No major table or pipe formatting detected.")
    if special_chars <= 50:
        good_signs.append("Character encoding appears ATS-compatible.")

    return {
        "penalties": penalties,
        "good_signs": good_signs,
        "word_count": word_count,
        "score": round(score, 1),
    }

# ── Composite Score Calculator ────────────────────────────────

def calculate_real_ats_score(resume_text: str, job_description: str) -> dict:
    """
    Master ATS scoring function.

    Weights:
        Keyword Match      40%
        Bullet Strength    15%
        Impact Verbs       10%
        Quantification     15%
        Structure          10%
        Formatting         10%

    Returns composite score (0–100) and all module results.
    """
    kw    = analyze_keywords(resume_text, job_description)
    bs    = analyze_bullet_strength(resume_text)
    iv    = analyze_impact_verbs(resume_text)
    qnt   = analyze_quantification(resume_text)
    struct= analyze_structure(resume_text)
    fmt   = analyze_formatting(resume_text)

    composite = (
        kw["score"]     * 0.40 +
        bs["score"]     * 0.15 +
        iv["score"]     * 0.10 +
        qnt["score"]    * 0.15 +
        struct["score"] * 0.10 +
        fmt["score"]    * 0.10
    )
    composite = round(min(100.0, max(0.0, composite)), 1)

    return {
        "composite": composite,
        "keyword":      kw,
        "bullet":       bs,
        "impact":       iv,
        "quantification": qnt,
        "structure":    struct,
        "formatting":   fmt,
    }

# ═══════════════════════════════════════════════════════════════
# ### MODIFIED SECTION — Score display helper
# ═══════════════════════════════════════════════════════════════

def score_color(score: float) -> str:
    if score >= 75:
        return "green"
    elif score >= 50:
        return "orange"
    return "red"


def render_ats_breakdown(ats: dict):
    """Render each ATS module's result as structured Streamlit UI."""

    st.markdown("---")

    # ── 1. Keyword Analysis ────────────────────────────────────
    with st.expander("1️⃣ Keyword Analysis (40% weight)", expanded=True):
        kw = ats["keyword"]
        col_a, col_b = st.columns(2)
        col_a.metric("Keyword Match", f"{kw['match_pct']}%")
        col_b.metric("Module Score", f"{kw['score']} / 100")
        st.progress(kw["score"] / 100)

        if not kw["jd_provided"]:
            st.info("No job description provided. Paste a JD for targeted keyword analysis.")
        else:
            if kw["matched"]:
                st.success("✅ **Matched Keywords:** " + ", ".join(f"`{k}`" for k in kw["matched"]))
            if kw["missing"]:
                st.warning("⚠️ **Missing Keywords:** " + ", ".join(f"`{k}`" for k in kw["missing"]))

    # ── 2. Bullet Strength ─────────────────────────────────────
    with st.expander("2️⃣ Bullet Point Strength Analysis (15% weight)", expanded=True):
        bs = ats["bullet"]
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Strong Bullets", bs["strong"])
        col_b.metric("Weak Bullets", bs["weak"])
        col_c.metric("Module Score", f"{bs['score']} / 100")
        st.progress(bs["score"] / 100)
        if bs["weak"] == 0:
            st.success("✅ No weak bullet openers detected.")
        else:
            st.warning(f"⚠️ {bs['weak']} weak bullet(s) found. Avoid starting with: *Responsible for*, *Worked on*, *Helped with*.")
            if bs.get("weak_examples"):
                for ex in bs["weak_examples"]:
                    st.markdown(f"  - _{ex}_")

    # ── 3. Impact Verb Usage ───────────────────────────────────
    with st.expander("3️⃣ Impact Verb Usage (10% weight)", expanded=True):
        iv = ats["impact"]
        col_a, col_b = st.columns(2)
        col_a.metric("Bullets with Impact Verbs", f"{iv['impact_count']} / {iv['total']}")
        col_b.metric("Module Score", f"{iv['score']} / 100")
        st.progress(iv["score"] / 100)
        if iv["verbs_found"]:
            st.success("✅ **Impact verbs used:** " + ", ".join(f"`{v}`" for v in iv["verbs_found"]))
        else:
            st.warning("⚠️ No strong impact verbs detected. Begin bullets with verbs like *Developed*, *Led*, *Optimized*.")

    # ── 4. Quantification Analysis ─────────────────────────────
    with st.expander("4️⃣ Quantification Analysis (15% weight)", expanded=True):
        qnt = ats["quantification"]
        col_a, col_b = st.columns(2)
        col_a.metric("Quantified Bullets", f"{qnt['quantified']} / {qnt['total']}")
        col_b.metric("Module Score", f"{qnt['score']} / 100")
        st.progress(qnt["score"] / 100)
        if qnt["examples"]:
            st.success("✅ **Quantified examples found:**")
            for ex in qnt["examples"]:
                st.markdown(f"  - _{ex}_")
        if qnt["ratio"] < 30:
            st.warning("⚠️ Less than 30% of bullets include measurable outcomes. Add numbers, percentages, or impact metrics.")

    # ── 5. Resume Structure ────────────────────────────────────
    with st.expander("5️⃣ Resume Structure (10% weight)", expanded=True):
        struct = ats["structure"]
        col_a, col_b = st.columns(2)
        col_a.metric("Sections Detected", f"{struct['present_count']} / {struct['total_sections']}")
        col_b.metric("Module Score", f"{struct['score']} / 100")
        st.progress(struct["score"] / 100)
        for section, present in struct["found"].items():
            icon = "✅" if present else "❌"
            st.markdown(f"  {icon} **{section.capitalize()}** section")
        if struct["missing_sections"]:
            st.warning("⚠️ Missing sections: " + ", ".join(s.capitalize() for s in struct["missing_sections"]))

    # ── 6. Formatting Compatibility ────────────────────────────
    with st.expander("6️⃣ Formatting Compatibility (10% weight)", expanded=True):
        fmt = ats["formatting"]
        col_a, col_b = st.columns(2)
        col_a.metric("Word Count", fmt["word_count"])
        col_b.metric("Module Score", f"{fmt['score']} / 100")
        st.progress(fmt["score"] / 100)
        for gs in fmt["good_signs"]:
            st.success(f"✅ {gs}")
        for p in fmt["penalties"]:
            st.warning(f"⚠️ {p}")
        if not fmt["penalties"]:
            st.success("✅ No major formatting issues detected.")


# ═══════════════════════════════════════════════════════════════
# ### UI CHANGES — PAGE CONFIG & BRANDING
# ═══════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="CDC Resume Analyzer — Ahmedabad University",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────
# SIDEBAR — MODIFIED BRANDING
# ─────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🎓 Career Development Centre")
    st.markdown("### Ahmedabad University")
    st.caption("Institutional ATS Resume Analyzer")

    st.divider()

    st.markdown("### How to Use")
    st.markdown(
        "1. Upload your PDF resume\n"
        "2. Optionally paste the target job description\n"
        "3. Choose analysis mode\n"
        "4. Click **Analyze Resume**"
    )

    st.divider()

    st.markdown("### Scoring Breakdown")
    st.markdown(
        "| Factor | Weight |\n"
        "|---|---|\n"
        "| Keyword Match | 40% |\n"
        "| Bullet Strength | 15% |\n"
        "| Quantification | 15% |\n"
        "| Impact Verbs | 10% |\n"
        "| Structure | 10% |\n"
        "| Formatting | 10% |"
    )

    st.divider()

    st.markdown("### Privacy Notice")
    st.markdown(
        "Your resume is processed for analysis purposes. "
        "Do not upload documents containing highly sensitive personal data "
        "(government IDs, financial account numbers, passwords)."
    )

    st.divider()
    st.caption("CDC Resume Analyzer v2.0 · Ahmedabad University")

# ─────────────────────────────────────────
# MAIN UI — MODIFIED BRANDING
# ─────────────────────────────────────────

st.markdown(
    "<h1 style='margin-bottom:0'>🎓 Career Development Centre</h1>"
    "<h3 style='margin-top:4px;color:#555'>Resume Analyzer — Ahmedabad University</h3>",
    unsafe_allow_html=True,
)
st.markdown(
    "Upload your resume to receive an **algorithmic ATS compatibility score** "
    "and structured feedback to strengthen your application."
)

st.divider()

# API key resolution
api_key = get_api_key()
if not api_key:
    st.error(
        "**Service configuration error.**\n\n"
        "Please contact the CDC technical team to resolve this issue."
    )
    st.stop()

client = get_groq_client(api_key)

# ── Inputs ────────────────────────────────────────────────────

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📄 Resume Upload")
    uploaded_file = st.file_uploader(
        "Upload PDF resume",
        type=["pdf"],
        help="Text-based PDFs work best. Scanned image PDFs may not extract correctly.",
    )
    if uploaded_file:
        st.success(f"Uploaded: `{uploaded_file.name}` ({uploaded_file.size / 1024:.1f} KB)")

with col2:
    st.subheader("📋 Job Description")
    job_description = st.text_area(
        "Paste the target job description (optional but recommended)",
        height=180,
        placeholder="Paste the job posting here for targeted keyword analysis...",
        help="Including a job description enables keyword gap analysis and role-specific scoring.",
    )
    st.caption(f"{len(job_description)} / {MAX_JD_CHARS} characters")

st.divider()

# ── Analysis Mode ─────────────────────────────────────────────

st.subheader("Analysis Mode")

mode_descriptions = {
    "Quick Scan": "Fast overview: strengths, improvement areas, and ATS score. Best for a first pass.",
    "Detailed Analysis": "Comprehensive section-by-section feedback with keyword alignment.",
    "ATS Optimization": "Keyword gaps, formatting issues, and specific rewrite suggestions.",
}

analysis_mode = st.radio(
    "Select mode:",
    list(mode_descriptions.keys()),
    horizontal=True,
    label_visibility="collapsed",
)
st.caption(mode_descriptions[analysis_mode])

st.divider()

# ── Analyze Button ────────────────────────────────────────────

analyze_btn = st.button("Analyze Resume", type="primary", use_container_width=True)

if analyze_btn:

    if not uploaded_file:
        st.warning("Please upload a PDF resume before analyzing.")
        st.stop()

    with st.spinner("Reading PDF..."):
        raw_text = extract_pdf_text(uploaded_file)

    if not raw_text.strip():
        st.error(
            "**Could not extract text from this PDF.**\n\n"
            "Common causes:\n"
            "- The PDF is a scanned image (not text-based)\n"
            "- The PDF is password-protected\n"
            "- The PDF uses non-standard encoding\n\n"
            "Try exporting your resume from Word or Google Docs as PDF."
        )
        st.stop()

    resume_text = truncate_safely(raw_text, MAX_RESUME_CHARS, "Resume")
    jd_text = truncate_safely(job_description, MAX_JD_CHARS, "Job description")

    # ── ### MODIFIED SECTION: Compute ATS score BEFORE LLM call ──

    with st.spinner("Computing ATS compatibility score..."):
        ats_results = calculate_real_ats_score(resume_text, jd_text)

    # ── Run LLM for qualitative feedback ──────────────────────

    prompt = build_prompt(analysis_mode, resume_text, jd_text)

    with st.spinner(f"Generating qualitative feedback ({analysis_mode})..."):
        result_text, success = run_analysis(client, prompt)

    # ── Results Header ─────────────────────────────────────────

    st.subheader("📊 Analysis Results")

    # ── ATS Score Widget (algorithmic) ─────────────────────────

    composite = ats_results["composite"]
    color = score_color(composite)

    score_col, gauge_col, _ = st.columns([1, 2, 2])
    with score_col:
        st.metric(label="ATS Compatibility Score", value=f"{composite} / 100")
    with gauge_col:
        st.markdown(f"**Score Band:** :{color}[{'Excellent' if composite >= 75 else 'Needs Improvement' if composite >= 50 else 'Poor'}]")
        st.progress(composite / 100)

    # ── Algorithmic Breakdown ──────────────────────────────────

    render_ats_breakdown(ats_results)

    # ── Qualitative Feedback from analysis engine ──────────────

    st.markdown("---")
    st.subheader("📝 Qualitative Feedback")

    if not success:
        st.error(result_text)
    else:
        st.markdown(result_text)

    # ── Download ───────────────────────────────────────────────

    st.divider()

    download_content = (
        f"CDC Resume Analyzer — Ahmedabad University\n"
        f"{'='*50}\n\n"
        f"ATS Compatibility Score: {composite} / 100\n\n"
        f"Keyword Match:     {ats_results['keyword']['score']} / 100\n"
        f"Bullet Strength:   {ats_results['bullet']['score']} / 100\n"
        f"Impact Verbs:      {ats_results['impact']['score']} / 100\n"
        f"Quantification:    {ats_results['quantification']['score']} / 100\n"
        f"Structure:         {ats_results['structure']['score']} / 100\n"
        f"Formatting:        {ats_results['formatting']['score']} / 100\n\n"
        f"{'='*50}\n\n"
        f"QUALITATIVE FEEDBACK\n\n"
        f"{result_text if success else 'Feedback unavailable.'}\n"
    )

    st.download_button(
        label="⬇️ Download Full Report (TXT)",
        data=download_content,
        file_name=f"cdc_ats_report_{analysis_mode.lower().replace(' ', '_')}.txt",
        mime="text/plain",
    )