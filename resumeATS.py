"""
ResumeATS Free Edition
======================
Production-ready ATS resume analyzer using Groq (LLaMA 3).
Author: Senior AI Architect template
Deployment target: Streamlit Cloud
Python: 3.10+
SDK: groq>=0.9.0
"""

import os
import re
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from groq import Groq, RateLimitError, APIStatusError, APIConnectionError

# ─────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────

MAX_RESUME_CHARS = 4000       # ~1000 tokens — safe for 8192 context model
MAX_JD_CHARS = 1500           # keep job description lean
MODEL_ID = "llama-3.1-8b-instant" # free, fast, stable on Groq as of 2026
MAX_OUTPUT_TOKENS = 800       # enough for structured analysis

# ─────────────────────────────────────────
# ENV LOADING (local dev only)
# ─────────────────────────────────────────

load_dotenv()  # no-op on Streamlit Cloud (uses secrets instead)

def get_api_key() -> str:
    """
    Resolve API key from Streamlit secrets (Cloud) or .env (local).
    Streamlit Cloud: set GROQ_API_KEY in app secrets dashboard.
    Local: set in .env file.
    """
    try:
        key = st.secrets["GROQ_API_KEY"]
        if key:
            return key
    except (KeyError, FileNotFoundError):
        pass

    key = os.getenv("GROQ_API_KEY", "")
    return key

# ─────────────────────────────────────────
# GROQ CLIENT INIT (cached)
# ─────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def get_groq_client(api_key: str) -> Groq:
    """Cache the Groq client across reruns to avoid redundant init."""
    return Groq(api_key=api_key)

# ─────────────────────────────────────────
# PDF EXTRACTION
# ─────────────────────────────────────────

def extract_pdf_text(uploaded_file) -> str:
    """
    Extract plain text from a PDF upload.
    Returns empty string on failure — never raises.
    """
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
# TOKEN SAFETY
# ─────────────────────────────────────────

def truncate_safely(text: str, max_chars: int, label: str = "input") -> str:
    """
    Truncate text with a visible warning if limit is exceeded.
    Uses character count as a proxy for token count (1 token ~= 4 chars).
    """
    if len(text) > max_chars:
        st.info(
            f"INFO: {label} truncated to {max_chars} characters "
            f"(was {len(text)}) to stay within model token limits."
        )
        return text[:max_chars]
    return text

# ─────────────────────────────────────────
# PROMPT BUILDER
# ─────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a professional ATS (Applicant Tracking System) evaluator and career coach. "
    "Provide structured, honest, and actionable feedback. "
    "Always end your response with a line in this exact format: ATS Score: XX/100"
)

def build_prompt(mode: str, resume_text: str, job_description: str) -> str:
    """Build the analysis prompt based on selected mode."""
    jd_section = (
        f"\n\nJob Description:\n{job_description}"
        if job_description.strip()
        else "\n\n(No job description provided — perform general analysis.)"
    )

    if mode == "Quick Scan":
        instructions = (
            "Provide a quick resume scan with:\n"
            "1. Most suitable job title for this resume\n"
            "2. Top 3 strengths (bullet points)\n"
            "3. Top 2 areas for improvement (bullet points)\n"
            "4. ATS Score: XX/100\n\n"
            "Keep response under 300 words."
        )
    elif mode == "Detailed Analysis":
        instructions = (
            "Provide a detailed resume analysis with:\n"
            "1. Most suitable profession\n"
            "2. Top 5 strengths (bullet points)\n"
            "3. Top 3 improvement areas with specific actions\n"
            "4. Section-by-section feedback (Summary, Experience, Skills, Education)\n"
            "5. Keyword alignment with job description (if provided)\n"
            "6. ATS Score: XX/100\n\n"
            "Keep response under 600 words."
        )
    else:  # ATS Optimization
        instructions = (
            "Perform ATS optimization analysis:\n"
            "1. List 5-8 missing keywords the resume should include (based on JD or industry norms)\n"
            "2. Formatting issues that hurt ATS parsing (e.g., tables, headers, graphics)\n"
            "3. Specific rewrite suggestions for weak bullet points (show before/after)\n"
            "4. File format and structure recommendations\n"
            "5. ATS Score: XX/100\n\n"
            "Keep response under 600 words."
        )

    return (
        f"{instructions}\n\n"
        f"Resume:\n{resume_text}"
        f"{jd_section}"
    )

# ─────────────────────────────────────────
# LLM INFERENCE
# ─────────────────────────────────────────

def run_analysis(client: Groq, prompt: str) -> tuple:
    """
    Call Groq API with full error handling.
    Returns (response_text, success_bool).
    Never raises — always returns a usable string.
    """
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
            return "The model returned an empty response. Please try again.", False

        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", None) if message else None

        if not content or not content.strip():
            return "The model returned no content. Please try again.", False

        return content.strip(), True

    except RateLimitError:
        return (
            "RATE LIMIT REACHED. Groq's free tier allows ~14,400 requests/day "
            "and ~30 requests/minute. Please wait 60 seconds and try again.",
            False,
        )
    except APIConnectionError:
        return (
            "CONNECTION ERROR. Could not reach Groq API. "
            "Check your internet connection and try again.",
            False,
        )
    except APIStatusError as e:
        if e.status_code == 401:
            return (
                "INVALID API KEY. Check your GROQ_API_KEY in secrets or .env.",
                False,
            )
        elif e.status_code == 503:
            return (
                "Groq service temporarily unavailable. Please try again in a minute.",
                False,
            )
        return f"API error {e.status_code}: {e.message}", False
    except Exception as e:
        return f"Unexpected error: {str(e)}", False

# ─────────────────────────────────────────
# ATS SCORE EXTRACTOR
# ─────────────────────────────────────────

def extract_ats_score(text: str):
    """
    Parse ATS score from model output using regex.
    Looks for patterns like: ATS Score: 72/100 or Score: 72
    Returns integer 0-100 or None if not found.
    """
    patterns = [
        r"ATS\s+Score[:\s]+(\d{1,3})\s*/\s*100",
        r"Score[:\s]+(\d{1,3})\s*/\s*100",
        r"(\d{1,3})\s*/\s*100",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            score = int(match.group(1))
            if 0 <= score <= 100:
                return score
    return None

def score_color(score: int) -> str:
    """Return color label based on ATS score band."""
    if score >= 75:
        return "green"
    elif score >= 50:
        return "orange"
    return "red"

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────

st.set_page_config(
    page_title="ResumeATS Free Edition",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────

with st.sidebar:
    st.title("📄 ResumeATS")
    st.caption("Free Edition — Powered by LLaMA 3 via Groq")

    st.divider()

    st.markdown("### How it works")
    st.markdown(
        "1. Upload your PDF resume\n"
        "2. Optionally paste a job description\n"
        "3. Choose analysis mode\n"
        "4. Click **Analyze**"
    )

    st.divider()

    st.markdown("### Model Info")
    st.markdown(
        f"**Model:** `{MODEL_ID}`  \n"
        "**Provider:** Groq (free tier)  \n"
        "**Context:** 8,192 tokens  \n"
        "**Rate limit:** ~30 req/min"
    )

    st.divider()

    st.markdown("### Privacy")
    st.markdown(
        "Your resume is sent to Groq's API for analysis. "
        "Do not upload documents with highly sensitive personal data "
        "(SSN, passwords, financial account numbers)."
    )

    st.divider()
    st.caption("v1.0.0 · Built for Streamlit Cloud")

# ─────────────────────────────────────────
# MAIN UI
# ─────────────────────────────────────────

st.title("ResumeATS Free Edition")
st.markdown(
    "**AI-powered resume analysis** using LLaMA 3 (8B) via Groq. "
    "No subscription required."
)

st.divider()

# API key resolution
api_key = get_api_key()
if not api_key:
    st.error(
        "**GROQ_API_KEY not configured.**\n\n"
        "- **Streamlit Cloud:** Go to App Settings → Secrets and add `GROQ_API_KEY = \"gsk_...\"`\n"
        "- **Local:** Create a `.env` file with `GROQ_API_KEY=gsk_...`\n\n"
        "Get a free key at https://console.groq.com/keys"
    )
    st.stop()

client = get_groq_client(api_key)

# Inputs
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("Resume Upload")
    uploaded_file = st.file_uploader(
        "Upload PDF resume",
        type=["pdf"],
        help="Text-based PDFs work best. Scanned image PDFs may not extract correctly.",
    )
    if uploaded_file:
        st.success(f"Uploaded: `{uploaded_file.name}` ({uploaded_file.size / 1024:.1f} KB)")

with col2:
    st.subheader("Job Description")
    job_description = st.text_area(
        "Paste the job description (optional)",
        height=180,
        placeholder="Paste the job posting here for targeted analysis...",
        help="Including a job description enables keyword matching and role-specific scoring.",
    )
    st.caption(f"{len(job_description)} / {MAX_JD_CHARS} characters")

st.divider()

# Analysis Mode
st.subheader("Analysis Mode")

mode_descriptions = {
    "Quick Scan": "Fast overview: strengths, improvements, ATS score. Best for a first pass.",
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

# Analyze Button
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
            "Try exporting your resume from Word/Google Docs as PDF."
        )
        st.stop()

    resume_text = truncate_safely(raw_text, MAX_RESUME_CHARS, "Resume")
    jd_text = truncate_safely(job_description, MAX_JD_CHARS, "Job description")

    prompt = build_prompt(analysis_mode, resume_text, jd_text)

    with st.spinner(f"Analyzing with LLaMA 3 ({analysis_mode})..."):
        result_text, success = run_analysis(client, prompt)

    st.subheader("Analysis Results")

    if not success:
        st.error(result_text)
        st.stop()

    # ATS Score widget
    score = extract_ats_score(result_text)
    if score is not None:
        score_col, _ = st.columns([1, 3])
        with score_col:
            st.metric(label="ATS Compatibility Score", value=f"{score} / 100")
            st.progress(score / 100)

    st.markdown("---")
    st.markdown(result_text)

    st.divider()
    st.download_button(
        label="Download Analysis (TXT)",
        data=result_text,
        file_name=f"ats_analysis_{analysis_mode.lower().replace(' ', '_')}.txt",
        mime="text/plain",
    )