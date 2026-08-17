import os
import sys
import uuid
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from dotenv import load_dotenv

# Ensure the project root and current folder are in Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

load_dotenv()

# Import helper utilities and prompt
from src.helper import (
    download_hugging_face_embeddings,
    check_emergency_keywords,
    format_sources,
    load_pdf_file,
    text_split,
    classify_triage_severity,
    fetch_nearby_hospitals,
    generate_consultation_pdf,
    extract_text_from_pdf,
    generate_medical_report_pdf,
    calculate_bmi_bmr,
    calculate_framingham_cvd_risk,
    calculate_findrisc_diabetes_score,
    calculate_egfr,
    calculate_rag_faithfulness,
    generate_intake_brief_pdf
)
from src.prompt import system_prompt
from langchain_pinecone import Pinecone as PineconeVectorStore
from pinecone import Pinecone as PineconeClient
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate


# ---------------- PAGE CONFIGURATION ---------------- #
st.set_page_config(
    page_title="MediMind AI — Generative AI RAG-Based Clinical Decision Support & Healthcare Platform",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ---------------- CUSTOM CSS FOR MODERN MEDICAL UI ---------------- #
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* Main banner styling */
    .med-hero {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
        border-radius: 16px;
        padding: 24px 30px;
        color: white;
        margin-bottom: 24px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.3);
    }
    
    .med-hero h1 {
        color: #ffffff;
        font-weight: 800;
        font-size: 2.1rem;
        margin-bottom: 6px;
        letter-spacing: -0.5px;
    }
    
    .med-hero p {
        color: #b0bec5;
        font-size: 1.05rem;
        margin-bottom: 0;
    }
    
    .med-badge {
        display: inline-block;
        background: rgba(0, 230, 118, 0.15);
        color: #00e676;
        border: 1px solid rgba(0, 230, 118, 0.3);
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 600;
        margin-right: 8px;
    }
    
    /* Emergency Alert Banner */
    .emergency-banner {
        background: linear-gradient(135deg, #ff1744 0%, #b71c1c 100%);
        color: white;
        padding: 16px 20px;
        border-radius: 12px;
        margin-bottom: 20px;
        box-shadow: 0 4px 20px rgba(255, 23, 68, 0.35);
        animation: pulseEmergency 2s infinite;
    }
    
    @keyframes pulseEmergency {
        0% { transform: scale(1); box-shadow: 0 4px 20px rgba(255, 23, 68, 0.35); }
        50% { transform: scale(1.008); box-shadow: 0 6px 28px rgba(255, 23, 68, 0.55); }
        100% { transform: scale(1); box-shadow: 0 4px 20px rgba(255, 23, 68, 0.35); }
    }
    
    /* Triage Cards */
    .triage-card {
        border-radius: 12px;
        padding: 14px 18px;
        margin: 12px 0 16px 0;
        border-left: 5px solid;
    }
    
    .triage-red {
        background: rgba(255, 23, 68, 0.08);
        border-color: #ff1744;
        color: #ff1744;
    }
    
    .triage-yellow {
        background: rgba(255, 171, 0, 0.08);
        border-color: #ffab00;
        color: #ffab00;
    }
    
    .triage-green {
        background: rgba(0, 200, 83, 0.08);
        border-color: #00c853;
        color: #00c853;
    }
    
    /* Feature Card */
    .feature-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 20px;
        margin-bottom: 16px;
        transition: all 0.3s ease;
    }
    
    .feature-card:hover {
        border-color: rgba(0, 230, 118, 0.3);
        box-shadow: 0 8px 24px -8px rgba(0, 230, 118, 0.15);
    }
    
    /* Source Pill */
    .source-pill {
        display: inline-block;
        background: rgba(33, 150, 243, 0.12);
        color: #42a5f5;
        border: 1px solid rgba(33, 150, 243, 0.25);
        padding: 4px 10px;
        border-radius: 8px;
        font-size: 0.8rem;
        margin: 3px 4px 3px 0;
    }
    
    /* Disclaimer footer */
    .disclaimer-box {
        font-size: 0.82rem;
        color: #78909c;
        border-top: 1px solid rgba(255, 255, 255, 0.08);
        padding-top: 14px;
        margin-top: 24px;
    }
    
    /* Medical Report Analysis Cards */
    .report-card-main {
        background: rgba(15, 32, 39, 0.7);
        border: 1px solid rgba(0, 230, 118, 0.3);
        border-radius: 16px;
        padding: 24px;
        margin-top: 16px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    .report-badge-red {
        display: inline-block;
        background: rgba(255, 23, 68, 0.18);
        color: #ff5252;
        border: 1px solid rgba(255, 23, 68, 0.4);
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.85rem;
        margin-bottom: 12px;
    }
    
    .report-badge-yellow {
        display: inline-block;
        background: rgba(255, 171, 0, 0.18);
        color: #ffd740;
        border: 1px solid rgba(255, 171, 0, 0.4);
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.85rem;
        margin-bottom: 12px;
    }
    
    .report-badge-green {
        display: inline-block;
        background: rgba(0, 230, 118, 0.18);
        color: #69f0ae;
        border: 1px solid rgba(0, 230, 118, 0.4);
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.85rem;
        margin-bottom: 12px;
    }
    
    .step-box {
        background: rgba(255, 255, 255, 0.04);
        border-left: 4px solid #00e676;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 12px;
    }
    
    /* Confidence Badge */
    .confidence-badge {
        display: inline-block;
        background: rgba(103, 58, 183, 0.15);
        color: #b388ff;
        border: 1px solid rgba(179, 136, 255, 0.35);
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-top: 4px;
    }
    
    /* Calculator Stat Box */
    .calc-stat-box {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        margin-bottom: 12px;
    }
    
    .calc-stat-value {
        font-size: 1.8rem;
        font-weight: 800;
        color: #00e676;
        margin: 4px 0;
    }
    
    /* Diet Meal Card */
    .diet-day-box {
        background: rgba(255, 255, 255, 0.03);
        border-left: 4px solid #42a5f5;
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 14px;
    }
    
    /* Rx Prescription Card */
    .rx-box {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 171, 0, 0.25);
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)


# ---------------- SECRETS & API KEY LOADER ---------------- #
def get_api_key(key_name, default=None):
    """Safely retrieves API keys from st.secrets, session_state, or os.environ."""
    # 1. Check Streamlit Cloud secrets
    try:
        if hasattr(st, "secrets") and key_name in st.secrets:
            val = st.secrets[key_name]
            if val and str(val).strip():
                return str(val).strip()
    except Exception:
        pass
    
    # 2. Check session state (manual entry)
    session_key = f"user_{key_name}"
    if session_key in st.session_state and st.session_state[session_key]:
        return str(st.session_state[session_key]).strip()
        
    # 3. Check os.environ (.env file)
    val = os.environ.get(key_name)
    if val and str(val).strip():
        return str(val).strip()
        
    return default


# ---------------- INITIALIZE & CACHE BACKEND SERVICES ---------------- #
@st.cache_resource(show_spinner=False)
def init_services(pinecone_key=None, groq_key=None):
    """Initializes embeddings, vector store, and Groq LLM with caching."""
    if not pinecone_key or not groq_key:
        return None, "Missing PINECONE_API_KEY or GROQ_API_KEY."
        
    try:
        # Safely set environment strings
        os.environ["PINECONE_API_KEY"] = str(pinecone_key)
        os.environ["GROQ_API_KEY"] = str(groq_key)
        
        # 1. Embedding Model
        embeddings = download_hugging_face_embeddings()
        
        # 2. Pinecone Index & Vector Store
        index_name = "medicalbot"
        pc = PineconeClient(api_key=str(pinecone_key))
        index = pc.Index(index_name)
        
        docsearch = PineconeVectorStore(
            index=index,
            embedding=embeddings
        )
        retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})
        
        # 3. LLM (Groq Engine)
        llm = ChatGroq(
            model="openai/gpt-oss-120b", 
            temperature=0.4, 
            max_tokens=600,
            api_key=str(groq_key)
        )
        
        prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{input}"),
            ]
        )
        
        return {
            "retriever": retriever,
            "docsearch": docsearch,
            "llm": llm,
            "prompt": prompt_template,
            "embeddings": embeddings
        }, None
    except Exception as e:
        return None, str(e)


# ---------------- SESSION STATE INITIALIZATION ---------------- #
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! I am your **AI Medical Assistant**. You can describe your symptoms, ask clinical questions, analyze drug interactions, or find nearby medical centers.",
            "sources": [],
            "triage": None,
            "is_emergency": False
        }
    ]

if "reminders" not in st.session_state:
    st.session_state.reminders = [
        {"medicine": "Amoxicillin", "dosage": "500 mg", "time": "08:00 AM", "frequency": "Twice daily after meals"},
        {"medicine": "Vitamin D3", "dosage": "1000 IU", "time": "01:00 PM", "frequency": "Once daily with lunch"}
    ]

if "language" not in st.session_state:
    st.session_state.language = "English"

if "quick_prompt" not in st.session_state:
    st.session_state.quick_prompt = None

if "report_data" not in st.session_state:
    st.session_state.report_data = {
        "filename": None,
        "raw_text": None,
        "analysis": None,
        "pdf_download_path": None
    }

if "report_qa_history" not in st.session_state:
    st.session_state.report_qa_history = []

if "rx_data" not in st.session_state:
    st.session_state.rx_data = {
        "text": None,
        "decoded": None,
        "medications": []
    }

if "diet_data" not in st.session_state:
    st.session_state.diet_data = {
        "condition": None,
        "plan_text": None,
        "pdf_path": None
    }

if "intake_data" not in st.session_state:
    st.session_state.intake_data = {
        "patient_name": "Alex Johnson",
        "age": 42,
        "gender": "Male",
        "complaint": "Persistent fatigue and post-meal dizziness for 3 weeks",
        "vitals": "BP: 138/88 mmHg, HR: 76 bpm, SpO2: 98%, Temp: 98.4°F",
        "meds": "Metformin 500mg (OD), Multivitamin (OD)",
        "lab_anomalies": "Fasting Glucose: 142 mg/dL, HbA1c: 7.4%",
        "pdf_path": None
    }

if "health_trends_records" not in st.session_state:
    st.session_state.health_trends_records = [
        {"Date": "2026-02-15", "HbA1c (%)": 8.6, "Fasting Glucose (mg/dL)": 168, "Total Cholesterol (mg/dL)": 255, "LDL (mg/dL)": 172, "Weight (kg)": 84.5, "Hemoglobin (g/dL)": 11.2},
        {"Date": "2026-05-18", "HbA1c (%)": 7.8, "Fasting Glucose (mg/dL)": 142, "Total Cholesterol (mg/dL)": 228, "LDL (mg/dL)": 145, "Weight (kg)": 82.0, "Hemoglobin (g/dL)": 12.1},
        {"Date": "2026-08-14", "HbA1c (%)": 6.9, "Fasting Glucose (mg/dL)": 118, "Total Cholesterol (mg/dL)": 196, "LDL (mg/dL)": 112, "Weight (kg)": 79.2, "Hemoglobin (g/dL)": 13.4}
    ]


# ---------------- SIDEBAR CONTROLS & DIAGNOSTICS ---------------- #
with st.sidebar:
    st.markdown("### 🩺 MediMind AI Control")
    st.markdown(
        '<span class="med-badge">🟢 System Online</span>'
        '<span class="med-badge">RAG Active</span>',
        unsafe_allow_html=True
    )
    
    st.divider()
    
    # Language Selector
    language_options = [
        "English", "Hindi", "Spanish", "French", "German", 
        "Arabic", "Bengali", "Portuguese", "Russian", "Japanese"
    ]
    selected_lang = st.selectbox(
        "🌐 Response Language",
        options=language_options,
        index=language_options.index(st.session_state.language) if st.session_state.language in language_options else 0
    )
    st.session_state.language = selected_lang
    
    st.divider()
    
    # System Specs
    st.markdown("#### ⚙️ AI Engine Diagnostics")
    st.markdown("""
    - **LLM Model:** `Groq / openai/gpt-oss-120b`
    - **Vector DB:** `Pinecone (medicalbot)`
    - **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`
    - **Dimension:** `384`
    """)
    
    st.divider()
    
    # Emergency Helplines Card
    st.markdown("#### 🚨 Global Emergency Numbers")
    st.markdown("""
    - 🇺🇸 **USA / Canada:** `911`
    - 🇮🇳 **India:** `112 / 108`
    - 🇬🇧 **UK:** `999 / 111`
    - 🇪🇺 **Europe:** `112`
    - 🇦🇺 **Australia:** `000`
    """)
    
    st.divider()
    
    # Clear Chat Button
    if st.button("🗑️ Clear Consultation Chat", use_container_width=True):
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Consultation cleared. How can I assist with your health questions today?",
                "sources": [],
                "triage": None,
                "is_emergency": False
            }
        ]
        st.rerun()
        
    st.markdown("""
    <div class="disclaimer-box">
        ⚠️ <b>Medical Disclaimer:</b><br>
        MediMind AI provides automated medical information for educational & preliminary triage purposes only. Always consult a certified healthcare professional or emergency services for diagnosis and urgent care.
    </div>
    """, unsafe_allow_html=True)


# ---------------- MAIN HERO HEADER ---------------- #
st.markdown("""
<div class="med-hero">
    <h1>🩺 MediMind AI</h1>
    <p>Generative AI RAG-Based Clinical Decision Support & Healthcare Platform</p>
</div>
""", unsafe_allow_html=True)


# ---------------- LOAD BACKEND SERVICES ---------------- #
pinecone_api_key = get_api_key("PINECONE_API_KEY")
groq_api_key = get_api_key("GROQ_API_KEY")

services, error_msg = init_services(pinecone_api_key, groq_api_key)

if error_msg or not services:
    st.markdown("""
    <div class="report-card-main" style="border-color: rgba(255, 171, 0, 0.4);">
        <h3 style="color: #ffd740; margin-top:0;">🔑 API Configuration Required</h3>
        <p style="color: #b0bec5;">To run <b>MediMind AI</b> on Streamlit Community Cloud or locally, your <b>Pinecone API Key</b> and <b>Groq API Key</b> are required.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col_k1, col_k2 = st.columns(2)
    with col_k1:
        st.markdown("#### ⚡ Option 1: Enter Keys Directly Below")
        with st.form("api_key_form"):
            in_groq = st.text_input("Groq API Key", value=groq_api_key or "", type="password", placeholder="gsk_...")
            in_pinecone = st.text_input("Pinecone API Key", value=pinecone_api_key or "", type="password", placeholder="pcsk_...")
            if st.form_submit_button("🚀 Connect & Launch MediMind AI", type="primary", use_container_width=True):
                if in_groq and in_pinecone:
                    st.session_state["user_GROQ_API_KEY"] = in_groq.strip()
                    st.session_state["user_PINECONE_API_KEY"] = in_pinecone.strip()
                    st.rerun()
                else:
                    st.error("Please enter both Groq and Pinecone API keys.")
                    
    with col_k2:
        st.markdown("#### ☁️ Option 2: Streamlit Cloud Settings (Permanent)")
        st.markdown("""
        1. In the bottom right of your Streamlit app, click **"Manage app"** ⚙️ or **Settings** (⋮ top-right).
        2. Click on **Secrets** tab and paste:
        ```toml
        PINECONE_API_KEY = "your_pinecone_key_here"
        GROQ_API_KEY = "your_groq_key_here"
        ```
        3. Click **Save** and the app will reload automatically!
        """)
        
    st.stop()

retriever = services["retriever"]
docsearch = services["docsearch"]
llm = services["llm"]
prompt = services["prompt"]


# ---------------- NAVIGATION TABS ---------------- #
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
    "💬 Clinical Consultation",
    "🔬 Medical Report Analyzer",
    "📸 Prescription Decoder",
    "🥗 AI 7-Day Diet Planner",
    "🧮 Clinical Risk Calculators",
    "📈 Health Trends & Tracker",
    "💊 Drug Interaction Checker",
    "🏥 Clinic & Hospital Finder",
    "📅 Doctor Appointment & Briefs"
])


# ==============================================================================
# TAB 1: MEDICAL CONSULTATION (RAG CHAT WITH FAITHFULNESS METRIC)
# ==============================================================================
with tab1:
    st.markdown("### 💬 Clinical AI Consultation")
    st.caption("Ask questions about symptoms, medications, diseases, treatments, or preventive care with verified RAG literature grounding.")

    # Quick Symptom Chips
    st.markdown("**⚡ Quick Symptom Prompts:**")
    q_col1, q_col2, q_col3, q_col4 = st.columns(4)
    with q_col1:
        if st.button("🤒 Fever & Headache", use_container_width=True):
            st.session_state.quick_prompt = "What are the common causes and home care steps for sudden fever accompanied by a persistent headache?"
    with q_col2:
        if st.button("✨ Skin Rash & Acne Care", use_container_width=True):
            st.session_state.quick_prompt = "What are the effective treatments and daily skincare routines for acne and inflamed skin rash?"
    with q_col3:
        if st.button("🫁 Asthma & Breathing", use_container_width=True):
            st.session_state.quick_prompt = "What are the primary triggers and first-aid measures for an asthma attack?"
    with q_col4:
        if st.button("🦴 Joint Pain & Arthritis", use_container_width=True):
            st.session_state.quick_prompt = "What exercises and dietary measures help reduce chronic knee and joint stiffness?"

    st.markdown("<br>", unsafe_allow_html=True)

    # Render Chat History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="🩺" if msg["role"] == "assistant" else "👤"):
            # Emergency Banner if detected
            if msg.get("is_emergency"):
                st.markdown("""
                <div class="emergency-banner">
                    <h3>🚨 EMERGENCY WARNING DETECTED 🚨</h3>
                    <p><b>This query contains indicators of a life-threatening medical emergency.</b> Please call emergency services (911 / 112) immediately or visit the nearest Emergency Room!</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Triage Badge
            triage = msg.get("triage")
            if triage:
                t_level = triage.get("level", "GREEN")
                css_cls = "triage-red" if t_level == "RED" else ("triage-yellow" if t_level == "YELLOW" else "triage-green")
                st.markdown(f"""
                <div class="triage-card {css_cls}">
                    <b>Triage Level:</b> {triage.get('label', '')}<br>
                    <b>Recommended Action:</b> {triage.get('recommendation', '')}<br>
                    <b>Suggested Specialist:</b> {triage.get('specialist', 'General Physician')}
                </div>
                """, unsafe_allow_html=True)
                
            st.markdown(msg["content"])
            
            # Grounding & Verification Confidence Badge
            conf = msg.get("confidence_badge")
            if conf:
                st.markdown(f"<span class='confidence-badge'>{conf}</span>", unsafe_allow_html=True)
            
            # Document Citations / Sources
            sources = msg.get("sources", [])
            if sources:
                with st.expander("📚 Retrieved Medical Sources & References"):
                    for s in sources:
                        st.markdown(f"<span class='source-pill'>📖 {s}</span>", unsafe_allow_html=True)

    # Handle Input (Chat input or Quick Prompt)
    user_query = st.chat_input("Describe symptoms or ask medical questions...")
    if st.session_state.quick_prompt:
        user_query = st.session_state.quick_prompt
        st.session_state.quick_prompt = None

    if user_query:
        # Display user message immediately
        with st.chat_message("user", avatar="👤"):
            st.markdown(user_query)
            
        with st.chat_message("assistant", avatar="🩺"):
            # 1. Emergency Keyword Guardrail Check
            is_emergency, trigger_keyword = check_emergency_keywords(user_query)
            if is_emergency:
                emergency_msg = (
                    f"🚨 **EMERGENCY WARNING DETECTED** 🚨\n\n"
                    f"Your query contains symptoms associated with a critical emergency (*{trigger_keyword}*). "
                    f"Please call emergency services (**911 / 112**) or seek immediate professional emergency medical care!"
                )
                triage_info = {
                    "level": "RED",
                    "label": "🔴 CRITICAL / EMERGENCY",
                    "recommendation": "Emergency Room (ER) Immediately",
                    "specialist": "Emergency Physician / Cardiologist"
                }
                
                st.markdown("""
                <div class="emergency-banner">
                    <h3>🚨 EMERGENCY WARNING DETECTED 🚨</h3>
                    <p><b>This query contains indicators of a life-threatening medical emergency.</b> Please call emergency services (911 / 112) immediately!</p>
                </div>
                """, unsafe_allow_html=True)
                st.markdown(emergency_msg)
                
                st.session_state.messages.append({
                    "role": "user",
                    "content": user_query
                })
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": emergency_msg,
                    "sources": [],
                    "confidence_badge": "🚨 Emergency Intercept (<10ms)",
                    "is_emergency": True,
                    "triage": triage_info
                })
            else:
                with st.spinner("Analyzing symptoms & retrieving medical literature..."):
                    # 2. Triage Classification
                    triage_info = classify_triage_severity(user_query)
                    
                    # 3. Pinecone Similarity Search
                    retrieved_docs = retriever.invoke(user_query)
                    sources = format_sources(retrieved_docs)
                    context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
                    
                    # 4. Memory Context (Last 3 exchanges)
                    history_list = st.session_state.messages[-6:]
                    history_context = ""
                    if history_list:
                        history_context = "\nRecent Conversation History:\n" + "\n".join(
                            [f"{item['role'].capitalize()}: {item['content']}" for item in history_list if 'content' in item]
                        ) + "\n"
                        
                    # 5. Invoke LLM
                    formatted_prompt = prompt.format_messages(
                        language=st.session_state.language,
                        context=context_text + history_context,
                        input=user_query
                    )
                    
                    response = llm.invoke(formatted_prompt)
                    answer_text = response.content
                    
                    # 6. Compute Real-Time RAG Faithfulness & Verification Score
                    conf_score, conf_badge = calculate_rag_faithfulness(retrieved_docs, answer_text)
                    
                    # Display Triage Badge
                    t_level = triage_info.get("level", "GREEN")
                    css_cls = "triage-red" if t_level == "RED" else ("triage-yellow" if t_level == "YELLOW" else "triage-green")
                    st.markdown(f"""
                    <div class="triage-card {css_cls}">
                        <b>Triage Level:</b> {triage_info.get('label', '')}<br>
                        <b>Recommended Action:</b> {triage_info.get('recommendation', '')}<br>
                        <b>Suggested Specialist:</b> {triage_info.get('specialist', 'General Physician')}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(answer_text)
                    st.markdown(f"<span class='confidence-badge'>{conf_badge}</span>", unsafe_allow_html=True)
                    
                    if sources:
                        with st.expander("📚 Retrieved Medical Sources & References"):
                            for s in sources:
                                st.markdown(f"<span class='source-pill'>📖 {s}</span>", unsafe_allow_html=True)

                    # Update session state
                    st.session_state.messages.append({
                        "role": "user",
                        "content": user_query
                    })
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer_text,
                        "sources": sources,
                        "confidence_badge": conf_badge,
                        "is_emergency": False,
                        "triage": triage_info
                    })


# ==============================================================================
# TAB 2: AI MEDICAL REPORT ANALYZER & DIAGNOSTIC GUIDANCE
# ==============================================================================
with tab2:
    st.markdown("### 🔬 AI Medical Report Analyzer & Diagnostic Guidance")
    st.caption("Upload your medical lab report (PDF) or try a sample test to get an in-depth clinical analysis in simple language, parameter breakdown, diet & lifestyle recommendations, and clear next steps.")

    # Top sample presets for quick testing
    st.markdown("**⚡ Quick Test with Sample Medical Reports:**")
    samp_col1, samp_col2, samp_col3 = st.columns(3)
    
    sample_text_to_load = None
    sample_name_to_load = None
    
    with samp_col1:
        if st.button("🩸 Sample CBC & Anemia Report", use_container_width=True):
            sample_name_to_load = "Sample_CBC_Anemia_Report.pdf"
            sample_text_to_load = """
PATIENT LAB REPORT: COMPLETE BLOOD COUNT (CBC) & IRON PANEL
Patient: John Doe | Age: 38 | Gender: Male | Date: 12-August-2026

TEST RESULTS:
1. Hemoglobin (Hb): 9.4 g/dL [Reference Range: 13.8 - 17.2 g/dL] (LOW - ABNORMAL)
2. Red Blood Cell (RBC) Count: 3.6 million/mcL [Reference Range: 4.5 - 5.9 million/mcL] (LOW)
3. Hematocrit (PCV): 29.5% [Reference Range: 40.7% - 50.3%] (LOW)
4. Mean Corpuscular Volume (MCV): 71.2 fL [Reference Range: 80 - 100 fL] (LOW - Microcytic)
5. Mean Corpuscular Hemoglobin (MCH): 22.0 pg [Reference Range: 27 - 33 pg] (LOW - Hypochromic)
6. White Blood Cell (WBC) Count: 6,400 /mcL [Reference Range: 4,500 - 11,000 /mcL] (NORMAL)
7. Platelet Count: 260,000 /mcL [Reference Range: 150,000 - 450,000 /mcL] (NORMAL)
8. Serum Ferritin: 10.5 ng/mL [Reference Range: 24 - 336 ng/mL] (VERY LOW - Severe Iron Depletion)
9. Total Iron Binding Capacity (TIBC): 440 mcg/dL [Reference Range: 240 - 450 mcg/dL] (HIGH)
"""
    with samp_col2:
        if st.button("🫀 Sample Lipid / Cholesterol Profile", use_container_width=True):
            sample_name_to_load = "Sample_Lipid_Cardio_Report.pdf"
            sample_text_to_load = """
PATIENT LAB REPORT: ADVANCED LIPID & CARDIOVASCULAR PANEL
Patient: Jane Smith | Age: 52 | Gender: Female | Date: 14-August-2026

TEST RESULTS:
1. Total Cholesterol: 265 mg/dL [Reference Range: < 200 mg/dL] (HIGH - ELEVATED)
2. Triglycerides: 240 mg/dL [Reference Range: < 150 mg/dL] (HIGH)
3. LDL Cholesterol (Bad): 172 mg/dL [Reference Range: < 100 mg/dL] (VERY HIGH - High Atherogenic Risk)
4. HDL Cholesterol (Good): 36 mg/dL [Reference Range: > 50 mg/dL for females] (LOW)
5. VLDL Cholesterol: 48 mg/dL [Reference Range: 5 - 40 mg/dL] (HIGH)
6. Total Cholesterol / HDL Ratio: 7.36 [Optimal: < 4.0] (HIGH CARDIOVASCULAR RISK)
7. High-Sensitivity CRP (hs-CRP): 3.8 mg/L [Reference Range: < 1.0 mg/L] (ELEVATED - Vascular Inflammation)
8. Fasting Serum Glucose: 108 mg/dL [Reference Range: 70 - 99 mg/dL] (BORDERLINE IMPAIRED)
"""
    with samp_col3:
        if st.button("🩺 Sample Diabetes & HbA1c Panel", use_container_width=True):
            sample_name_to_load = "Sample_Diabetes_Metabolic_Report.pdf"
            sample_text_to_load = """
PATIENT LAB REPORT: COMPREHENSIVE DIABETIC & METABOLIC PROFILE
Patient: Robert Miller | Age: 47 | Gender: Male | Date: 15-August-2026

TEST RESULTS:
1. Fasting Blood Glucose: 164 mg/dL [Reference Range: 70 - 99 mg/dL] (HIGH - DIABETIC RANGE)
2. Postprandial Blood Glucose (2 hrs): 238 mg/dL [Reference Range: < 140 mg/dL] (VERY HIGH)
3. Glycated Hemoglobin (HbA1c): 8.6% [Reference Range: Normal < 5.7%, Prediabetes 5.7-6.4%, Diabetes >= 6.5%] (HIGH - Poorly Controlled Diabetes)
4. Estimated Average Glucose (eAG): 200 mg/dL (HIGH)
5. Serum Creatinine: 1.1 mg/dL [Reference Range: 0.7 - 1.3 mg/dL] (NORMAL)
6. Estimated GFR (eGFR): 88 mL/min/1.73m2 [Reference Range: >= 90 mL/min] (MILDLY REDUCED)
7. Urine Albumin-to-Creatinine Ratio (UACR): 48 mg/g [Reference Range: < 30 mg/g] (MODERATE MICROALBUMINURIA)
8. Serum Uric Acid: 7.6 mg/dL [Reference Range: 3.5 - 7.2 mg/dL] (HIGH)
"""

    if sample_text_to_load:
        st.session_state.report_data = {
            "filename": sample_name_to_load,
            "raw_text": sample_text_to_load,
            "analysis": None,
            "pdf_download_path": None
        }
        st.session_state.report_qa_history = []
        st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # Upload or Input Section
    col_up1, col_up2 = st.columns([2, 1])
    
    with col_up1:
        uploaded_pdf = st.file_uploader(
            "📁 Drag & Drop Patient Medical Report (PDF)", 
            type=["pdf"],
            key="medical_report_pdf_uploader",
            help="Upload blood tests, pathology reports, metabolic panels, radiology summaries, etc."
        )
        
    with col_up2:
        st.markdown("""
        <div class="feature-card" style="padding: 14px 18px;">
            <h5 style="margin-top:0; color:#00e676;">📋 Supported Reports</h5>
            <p style="margin:0; font-size:0.83rem; color:#b0bec5;">
                • Complete Blood Count (CBC)<br>
                • Lipid Profile / Cholesterol<br>
                • Diabetes & HbA1c Panels<br>
                • Liver (LFT) & Kidney (KFT) Tests<br>
                • Thyroid Profile (TSH/T3/T4)
            </p>
        </div>
        """, unsafe_allow_html=True)

    # If new file uploaded
    if uploaded_pdf is not None:
        upload_folder = os.path.join(current_dir, "uploads")
        os.makedirs(upload_folder, exist_ok=True)
        pdf_path = os.path.join(upload_folder, uploaded_pdf.name)
        
        if st.session_state.report_data.get("filename") != uploaded_pdf.name:
            with open(pdf_path, "wb") as f:
                f.write(uploaded_pdf.getbuffer())
                
            extracted_txt = extract_text_from_pdf(pdf_path)
            st.session_state.report_data = {
                "filename": uploaded_pdf.name,
                "raw_text": extracted_txt,
                "analysis": None,
                "pdf_download_path": None
            }
            st.session_state.report_qa_history = []
            st.success(f"📄 Successfully loaded **{uploaded_pdf.name}**!")

    # Manual text input expander
    with st.expander("✍️ Or Paste Medical Report Text Manually", expanded=False):
        manual_report_text = st.text_area(
            "Paste Lab Values / Clinical Notes:",
            height=120,
            placeholder="e.g. Hemoglobin: 10.1 g/dL, Platelets: 180,000, TSH: 6.8 uIU/mL..."
        )
        if st.button("📥 Load Pasted Text", use_container_width=True):
            if manual_report_text.strip():
                st.session_state.report_data = {
                    "filename": "Manual_Clinical_Notes.txt",
                    "raw_text": manual_report_text.strip(),
                    "analysis": None,
                    "pdf_download_path": None
                }
                st.session_state.report_qa_history = []
                st.success("✅ Pasted medical notes loaded!")
                st.rerun()

    current_report_name = st.session_state.report_data.get("filename")
    current_report_text = st.session_state.report_data.get("raw_text")

    # If a report is loaded, show analyze button & preview
    if current_report_name and current_report_text:
        st.markdown("---")
        c_head1, c_head2 = st.columns([3, 1])
        with c_head1:
            st.markdown(f"#### 📄 Active Report: **`{current_report_name}`**")
        with c_head2:
            if st.button("🔄 Reset / Clear Report", use_container_width=True):
                st.session_state.report_data = {"filename": None, "raw_text": None, "analysis": None, "pdf_download_path": None}
                st.session_state.report_qa_history = []
                st.rerun()

        with st.expander("🔍 View Extracted Raw Report Data", expanded=False):
            st.text(current_report_text[:2000] + ("..." if len(current_report_text) > 2000 else ""))

        # Trigger Deep Analysis Button
        if not st.session_state.report_data.get("analysis"):
            if st.button("🧠 Run Comprehensive AI Clinical Analysis & Next Steps Guidance", type="primary", use_container_width=True):
                with st.spinner(f"Analyzing medical parameters, checking clinical ranges & generating recommendations in {st.session_state.language}..."):
                    analysis_prompt = f"""
You are a highly experienced Chief Medical Officer, Clinical Pathologist, and Patient Health Educator.
A patient has provided the following laboratory / clinical medical report:

--- START MEDICAL REPORT ---
{current_report_text}
--- END MEDICAL REPORT ---

TARGET EXPLANATION LANGUAGE: {st.session_state.language}

Please perform a thorough, compassionate, structured clinical analysis of this report.
Your output must be formatted with the following exact Markdown sections, structured headings, badges, and bullet points:

# 📑 Comprehensive Clinical Medical Report Analysis

### 📋 1. Report Overview & Health Status
- **Detected Test Type / Panel:** (Identify the report type, e.g. Complete Blood Count / Lipid Profile / Renal Function / Thyroid Panel)
- **Patient Context & Date:** (Extract if present)
- **Overall Health Status:** State one of these clearly with emojis:
  * 🔴 **Action Required / Notable Abnormalities Detected** OR
  * 🟡 **Moderate / Borderline Findings - Medical Attention Needed** OR
  * 🟢 **Generally Within Normal Range / Healthy Findings**

---

### 🔍 2. Parameter-by-Parameter Clinical Breakdown
Divide findings strictly into two subsections:

#### ⚠️ Abnormal or Out-of-Range Parameters (High / Low)
For **EVERY** abnormal test value, provide:
- **Parameter Name & Result:** (e.g. **Hemoglobin:** `9.4 g/dL`)
- **Reference Standard Range:** (e.g. Normal: `13.8 - 17.2 g/dL`)
- **Status Indicator:** `HIGH ⬆️` or `LOW ⬇️`
- **What This Means in Simple Everyday Words:** Explain simply why this happens and what effect it has on the patient's body (avoid overwhelming medical jargon).

#### ✅ Normal & Healthy Parameters
- List the key parameters that are comfortably in the safe normal zone to reassure the patient.

---

### 💡 3. Plain Language Executive Summary (आसान भाषा में सारांश)
- Provide a clear, friendly 2-3 paragraph explanation of what this entire report means for the patient in non-technical language. Answer the question: *"How is my overall health based on this report, and what is the primary concern?"*

---

### 🥗 4. Actionable Health & Lifestyle Recommendations
- **🥑 Nutrition & Diet Guidelines:** Specific foods to eat more of, and specific foods or drinks to strictly avoid/minimize based on these exact test results.
- **💧 Hydration & Daily Habits:** Specific daily lifestyle modifications, sleep habits, and stress management.
- **🏃 Physical Activity & Exercise:** Recommended safe exercise routines or physical precautions.

---

### 🧭 5. Next Steps & Doctor Consultation Roadmap
- **👨‍⚕️ Recommended Medical Specialist:** Name the exact specialist doctor they should consult (e.g., *Hematologist, Endocrinologist, Cardiologist, Gastroenterologist, Nephrologist, or General Physician*).
- **🧪 Recommended Follow-up Tests:** Complementary diagnostic tests or re-testing timeline (e.g. *Repeat Serum Ferritin in 6-8 weeks*).
- **⏳ Urgency & Recommended Timeframe:** (e.g. *Schedule consultation within 24-48 hours* / *Within 1-2 weeks*).

---

### ❓ 6. Essential Questions to Ask Your Doctor
List 3-4 specific, impactful questions the patient should bring to their consultation appointment.

---

### 🚨 7. Red Flag Warning Symptoms
List any urgent warning signs or emergency symptoms (e.g. *severe dizziness, shortness of breath, chest pain, fainting*) that mean the patient should seek immediate Emergency Room (ER) care rather than waiting.

---
Ensure the entire response is written clearly, empathetically, and accurately in {st.session_state.language}. Maintain a supportive, professional clinical tone.
"""
                    try:
                        res = llm.invoke(analysis_prompt)
                        st.session_state.report_data["analysis"] = res.content
                        
                        summary_folder = os.path.join(current_dir, "uploads")
                        os.makedirs(summary_folder, exist_ok=True)
                        report_pdf_name = f"AI_Report_Analysis_{st.session_state.session_id[:6]}.pdf"
                        report_pdf_path = os.path.join(summary_folder, report_pdf_name)
                        
                        try:
                            generate_medical_report_pdf(current_report_name, res.content, report_pdf_path)
                            st.session_state.report_data["pdf_download_path"] = report_pdf_path
                        except Exception:
                            st.session_state.report_data["pdf_download_path"] = None
                            
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error during report analysis: {str(e)}")

        # Render Analysis if available
        if st.session_state.report_data.get("analysis"):
            analysis_text = st.session_state.report_data["analysis"]
            
            st.markdown("""
            <div class="report-card-main">
                <div style="display:flex; justify-content:space-between; align-items:center; border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 12px; margin-bottom: 16px;">
                    <div>
                        <h3 style="margin:0; color:#00e676;">📋 AI Clinical Analysis & Guidance</h3>
                        <p style="margin:4px 0 0 0; color:#90a4ae; font-size:0.9rem;">Automated Laboratory Parameter Evaluation & Health Action Plan</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(analysis_text)
            st.markdown("---")
            
            # Action Toolbar: Download PDF & Index to Pinecone
            act_col1, act_col2, act_col3 = st.columns(3)
            
            with act_col1:
                pdf_path = st.session_state.report_data.get("pdf_download_path")
                if pdf_path and os.path.exists(pdf_path):
                    with open(pdf_path, "rb") as f:
                        pdf_bytes = f.read()
                    st.download_button(
                        label="📥 Download AI Analysis (PDF)",
                        data=pdf_bytes,
                        file_name=f"MediMind_Report_Analysis_{current_report_name.replace('.pdf','')}.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                        type="primary"
                    )
                else:
                    st.download_button(
                        label="📥 Download Analysis (Markdown)",
                        data=analysis_text,
                        file_name="Medical_Report_Analysis.md",
                        mime="text/markdown",
                        use_container_width=True
                    )
                    
            with act_col2:
                if st.button("⚡ Index Report into MediMind Memory", use_container_width=True):
                    with st.spinner("Chunking & embedding report into Pinecone vector index..."):
                        try:
                            from langchain_core.documents import Document
                            splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=30)
                            report_doc = Document(
                                page_content=f"PATIENT REPORT ({current_report_name}):\n" + current_report_text,
                                metadata={"source": current_report_name, "page": 1}
                            )
                            chunks = splitter.split_documents([report_doc])
                            docsearch.add_documents(chunks)
                            st.balloons()
                            st.success("✅ Report indexed! The main Clinical Consultation chat in Tab 1 now knows your report context.")
                        except Exception as e:
                            st.error(f"Error indexing to Pinecone: {str(e)}")
                            
            with act_col3:
                if st.button("🔄 Analyze Another Report", use_container_width=True):
                    st.session_state.report_data = {"filename": None, "raw_text": None, "analysis": None, "pdf_download_path": None}
                    st.session_state.report_qa_history = []
                    st.rerun()

            # ---------------- Interactive Follow-up Q&A on the Report ---------------- #
            st.markdown("---")
            st.markdown("### 💬 Ask Questions About This Report")
            st.caption("Have doubts about a specific lab value, dietary restriction, or next step? Ask anything below:")

            # Quick suggestion buttons
            q_btn_col1, q_btn_col2, q_btn_col3 = st.columns(3)
            report_quick_q = None
            with q_btn_col1:
                if st.button("🥦 What specific foods should I eat to improve this?"):
                    report_quick_q = "Based on my uploaded report, what exact foods, groceries, and meal habits should I include in my daily diet to improve these values?"
            with q_btn_col2:
                if st.button("❓ Are any of these values dangerous?"):
                    report_quick_q = "Which of the abnormal values in my report is the most concerning, and is it considered an emergency?"
            with q_btn_col3:
                if st.button("👨‍⚕️ What will my doctor likely prescribe?"):
                    report_quick_q = "What kind of treatments, lifestyle changes, or prescription medications do doctors typically discuss for this specific combination of findings?"

            for qa in st.session_state.report_qa_history:
                with st.chat_message("user", avatar="👤"):
                    st.markdown(qa["question"])
                with st.chat_message("assistant", avatar="🩺"):
                    st.markdown(qa["answer"])

            report_user_input = st.chat_input("Ask a follow-up question regarding your medical report...", key="report_chat_input")
            if report_quick_q:
                report_user_input = report_quick_q

            if report_user_input:
                with st.chat_message("user", avatar="👤"):
                    st.markdown(report_user_input)
                with st.chat_message("assistant", avatar="🩺"):
                    with st.spinner("Analyzing report context to answer your question..."):
                        qa_prompt = f"""
You are an expert AI Physician and Clinical Laboratory Consultant.
The patient has asked a follow-up question regarding their medical report.

--- MEDICAL REPORT DATA ---
{current_report_text}

--- PREVIOUS REPORT ANALYSIS ---
{analysis_text[:1500]}
---

PATIENT'S QUESTION: "{report_user_input}"
LANGUAGE: {st.session_state.language}

Provide a clear, helpful, accurate, and supportive response in {st.session_state.language}.
Keep the explanation easy to understand for a layperson while maintaining medical accuracy.
"""
                        try:
                            qa_res = llm.invoke(qa_prompt)
                            st.markdown(qa_res.content)
                            st.session_state.report_qa_history.append({
                                "question": report_user_input,
                                "answer": qa_res.content
                            })
                        except Exception as e:
                            st.error(f"Error answering question: {str(e)}")


# ==============================================================================
# TAB 3: PRESCRIPTION OCR & MEDICINE DECODER
# ==============================================================================
with tab3:
    st.markdown("### 📸 Prescription OCR & Medicine Decoder")
    st.caption("Upload a doctor's prescription (PDF/Image text) or select a sample prescription to decode handwritten medicines, dosages, timings, food interactions, and auto-sync alarms.")

    st.markdown("**⚡ Quick Test with Sample Prescriptions:**")
    rx_c1, rx_c2, rx_c3 = st.columns(3)
    
    sample_rx_text = None
    with rx_c1:
        if st.button("🫀 Cardiology Prescription", use_container_width=True):
            sample_rx_text = """
DOCTOR'S CLINICAL PRESCRIPTION:
Dr. Sarah Jenkins, MD (Cardiology) - City Heart Institute
Patient: David Clark | Age: 58 | Date: 16-Aug-2026 | Dx: Dyslipidemia & Hypertension

Rx:
1. Tab. Atorvastatin 20 mg - 1 tab (0-0-1) at bedtime after dinner for 90 days.
2. Tab. Metoprolol Tartrate 50 mg - 1 tab (1-0-0) morning after breakfast for 90 days.
3. Tab. Ecosprin (Aspirin) 75 mg - 1 tab (0-1-0) afternoon with lunch for 90 days.
4. Tab. Pantoprazole 40 mg - 1 tab (1-0-0) 30 mins before breakfast for 14 days.

Advice: Low-salt, low-fat diet. Avoid grapefruit juice. Regular BP & Lipid profile monitoring in 3 months.
"""
    with rx_c2:
        if st.button("🩺 Diabetes & Renal Prescription", use_container_width=True):
            sample_rx_text = """
DOCTOR'S CLINICAL PRESCRIPTION:
Dr. Rajan Verma, MD, DM (Endocrinology) - Apex Diabetes Center
Patient: Anita Sharma | Age: 49 | Date: 15-Aug-2026 | Dx: Type 2 Diabetes Mellitus

Rx:
1. Tab. Metformin 1000 mg SR - 1 tab (1-0-1) twice daily immediately with meals (breakfast & dinner).
2. Tab. Glimepiride 1 mg - 1 tab (1-0-0) morning 15 mins before breakfast.
3. Tab. Telmisartan 40 mg - 1 tab (1-0-0) morning after breakfast.
4. Tab. Methylcobalamin (B12) 1500 mcg - 1 tab (0-1-0) once daily after lunch for 30 days.

Advice: Strict diabetic diet. Avoid skipped meals to prevent hypoglycemia. Carry glucose candy.
"""
    with rx_c3:
        if st.button("💊 Respiratory & Infection Rx", use_container_width=True):
            sample_rx_text = """
DOCTOR'S CLINICAL PRESCRIPTION:
Dr. Emily Watson, MD (Internal Medicine) - Metro Clinic
Patient: Michael Brown | Age: 34 | Date: 17-Aug-2026 | Dx: Acute Bronchitis & Sinusitis

Rx:
1. Tab. Augmentin (Amoxicillin + Clavulanate) 625 mg - 1 tab (1-0-1) twice daily after meals for 7 days.
2. Tab. Montelukast + Levocetirizine - 1 tab (0-0-1) at bedtime for 10 days.
3. Tab. Paracetamol 650 mg - 1 tab (1-1-1) thrice daily as needed (SOS) for fever/bodyache.
4. Syp. Ascoril D (Cough Syrup) - 10 ml thrice daily after food for 5 days.

Advice: Steam inhalation twice daily. Drink warm fluids. Complete the full antibiotic course.
"""

    if sample_rx_text:
        st.session_state.rx_data["text"] = sample_rx_text
        st.session_state.rx_data["decoded"] = None
        st.rerun()

    # Manual or file input
    rx_input_text = st.text_area(
        "Paste Prescription Text or Doctor Notes:",
        value=st.session_state.rx_data.get("text") or "",
        height=140,
        placeholder="e.g. Tab. Metformin 500mg (1-0-1) after meals, Tab. Telmisartan 40mg (1-0-0)..."
    )

    if st.button("🧠 Decode Prescription, Check Dosages & Safety", type="primary", use_container_width=True):
        if not rx_input_text.strip():
            st.warning("Please enter or select a prescription to decode.")
        else:
            with st.spinner("Decoding medications, dosages, timings & pharmacological safety..."):
                rx_prompt = f"""
You are an expert Chief Clinical Pharmacologist and Prescription Decoder.
Analyze and decode the following doctor's prescription text:

--- PRESCRIPTION TEXT ---
{rx_input_text}
---

TARGET EXPLANATION LANGUAGE: {st.session_state.language}

Provide a structured, crystal-clear breakdown:
1. 📋 **Decoded Medications Table:**
   - Medicine Name (Brand & Generic)
   - Strength / Dosage (e.g. 500 mg)
   - Exact Timing & Frequency (e.g., Once daily morning after breakfast, Twice daily with meals, Bedtime)
   - Indication / Purpose (What it treats)
   - Duration of Treatment
2. ⚠️ **Key Precautions & Dietary Restrictions:** (e.g. Avoid grapefruit, do not skip meals, avoid alcohol)
3. 💊 **Drug-to-Drug Interaction & Side-Effect Evaluation:** (Check if these prescribed medications interact safely)
4. ⏰ **Recommended Daily Schedule Timeline:** (Morning, Afternoon, Evening, Night)

Format your response cleanly with markdown tables and bullet points in {st.session_state.language}.
"""
                try:
                    res = llm.invoke(rx_prompt)
                    st.session_state.rx_data["decoded"] = res.content
                    st.session_state.rx_data["text"] = rx_input_text
                    st.rerun()
                except Exception as e:
                    st.error(f"Error decoding prescription: {str(e)}")

    if st.session_state.rx_data.get("decoded"):
        st.markdown("---")
        st.markdown("### 📋 Decoded Prescription & Clinical Safety Plan")
        st.markdown(st.session_state.rx_data["decoded"])
        
        st.markdown("---")
        st.markdown("#### ⚡ 1-Click Sync to Medication Dosage Scheduler:")
        
        sync_c1, sync_c2 = st.columns([3, 1])
        with sync_c1:
            st.info("Clicking Sync will automatically add active prescribed medications into your browser Dosage Alarms in Tab 9.")
        with sync_c2:
            if st.button("⏰ Auto-Sync to Med Reminders", type="primary", use_container_width=True):
                # Extract quick mock reminders from prescription text
                lines = rx_input_text.split('\n')
                added_count = 0
                for line in lines:
                    if "Tab." in line or "Cap." in line or "Syp." in line:
                        parts = line.split('-')
                        med_name = parts[0].strip().replace("1.", "").replace("2.", "").replace("3.", "").replace("4.", "").strip()
                        timing = "09:00 AM"
                        if "bedtime" in line.lower() or "0-0-1" in line:
                            timing = "09:00 PM"
                        elif "afternoon" in line.lower() or "0-1-0" in line:
                            timing = "01:00 PM"
                        elif "1-0-1" in line:
                            timing = "08:00 AM & 08:00 PM"
                        
                        st.session_state.reminders.append({
                            "medicine": med_name,
                            "dosage": "As prescribed",
                            "time": timing,
                            "frequency": "Daily as per doctor"
                        })
                        added_count += 1
                
                st.balloons()
                st.success(f"🎉 Successfully synced {added_count} medications to your active Medication Reminders!")


# ==============================================================================
# TAB 4: AI 7-DAY DISEASE-SPECIFIC DIET & MEAL PLANNER
# ==============================================================================
with tab4:
    st.markdown("### 🥗 AI 7-Day Personalized Disease-Specific Diet Planner")
    st.caption("Generate a clinically tailored 7-day nutritional meal schedule (Breakfast, Lunch, Snacks, Dinner) calibrated for chronic health conditions.")

    d_col1, d_col2, d_col3 = st.columns(3)
    with d_col1:
        disease_choice = st.selectbox(
            "Select Clinical Condition:",
            [
                "Type 2 Diabetes (Low GI & Carb Controlled)",
                "Hypertension & High BP (DASH Diet)",
                "High Cholesterol & Cardiovascular Health",
                "High Uric Acid & Gout (Low Purine)",
                "Thyroid Support (Hypothyroidism)",
                "Fatty Liver (NAFLD Recovery)",
                "PCOS / PCOD Insulin Balance",
                "Chronic Kidney Care (Low Sodium/Potassium)",
                "Healthy Weight Loss & Calorie Deficit"
            ]
        )
    with d_col2:
        diet_pref = st.selectbox(
            "Dietary Preference:",
            ["Vegetarian (Indian / Global)", "Non-Vegetarian (Balanced Poultry/Fish)", "Vegan (100% Plant-Based)", "Eggitarian"]
        )
    with d_col3:
        target_cals = st.selectbox(
            "Target Daily Calories:",
            ["1400 - 1600 kcal (Weight Loss)", "1700 - 1900 kcal (Maintenance)", "2000 - 2200 kcal (High Energy)"]
        )

    if st.button("🍳 Generate Personalized 7-Day Clinical Diet Plan", type="primary", use_container_width=True):
        with st.spinner(f"Compiling 7-day personalized clinical meal plan for '{disease_choice}' in {st.session_state.language}..."):
            diet_prompt = f"""
You are an expert Clinical Dietitian and Metabolic Nutritionist.
Create a comprehensive, practical 7-Day Meal Plan tailored specifically for:
- Clinical Condition: {disease_choice}
- Dietary Preference: {diet_pref}
- Caloric Target: {target_cals}
- Target Language: {st.session_state.language}

Structure the plan with:
### 1. 📋 Clinical Dietary Principles for {disease_choice}
(Key nutritional rules, macronutrient ratio, glycemic target)

### 2. 📅 7-Day Complete Meal Timetable (Monday to Sunday)
For each day provide:
- **Early Morning (Empty Stomach):**
- **Breakfast:**
- **Mid-Day Snack:**
- **Lunch:**
- **Evening Tea / Snack:**
- **Dinner:**
- **Bedtime Drink:**

### 3. 🥑 "Foods to Eat Freely" vs 🚫 "Foods to Strictly Avoid"
- Categorized bullet lists of superfoods and forbidden items for this condition.

### 4. 💧 Hydration & Safe Physical Activity Guidelines

Ensure meals use realistic, accessible ingredients and delicious recipes. Written in {st.session_state.language}.
"""
            try:
                diet_res = llm.invoke(diet_prompt)
                st.session_state.diet_data = {
                    "condition": disease_choice,
                    "plan_text": diet_res.content,
                    "pdf_path": None
                }
                st.rerun()
            except Exception as e:
                st.error(f"Error generating diet plan: {str(e)}")

    if st.session_state.diet_data.get("plan_text"):
        st.markdown("---")
        st.markdown(f"### 📋 7-Day Clinical Nutrition Schedule for **{st.session_state.diet_data.get('condition')}**")
        st.markdown(st.session_state.diet_data["plan_text"])
        
        st.download_button(
            label="📥 Download 7-Day Diet Chart (Markdown)",
            data=st.session_state.diet_data["plan_text"],
            file_name=f"7Day_Diet_Plan_{disease_choice[:15].replace(' ','_')}.md",
            mime="text/markdown",
            use_container_width=True
        )


# ==============================================================================
# TAB 5: INTERACTIVE CLINICAL RISK CALCULATORS
# ==============================================================================
with tab5:
    st.markdown("### 🧮 Interactive Clinical Risk Calculators")
    st.caption("Evidence-based clinical calculators for cardiovascular risk, diabetes susceptibility, body composition, and kidney function.")

    calc_tab1, calc_tab2, calc_tab3, calc_tab4 = st.tabs([
        "🫀 Framingham Heart Attack Risk",
        "🩺 FINDRISC Diabetes Risk",
        "⚖️ BMI & BMR Calculator",
        "🫁 eGFR Kidney Health"
    ])

    # Sub-tab 1: Framingham CVD
    with calc_tab1:
        st.markdown("#### 🫀 Framingham 10-Year Cardiovascular Disease (CVD) Risk")
        st.caption("Estimates the 10-year risk of developing coronary heart disease or myocardial infarction.")
        
        c_f1, c_f2, c_f3 = st.columns(3)
        with c_f1:
            f_gender = st.selectbox("Biological Gender", ["Male", "Female"], key="f_gen")
            f_age = st.slider("Age (years)", 20, 79, 48, key="f_age")
        with c_f2:
            f_tchol = st.number_input("Total Cholesterol (mg/dL)", 120, 400, 225, key="f_tc")
            f_hdl = st.number_input("HDL Cholesterol (mg/dL)", 20, 100, 42, key="f_hdl")
        with c_f3:
            f_sbp = st.number_input("Systolic Blood Pressure (mmHg)", 90, 220, 138, key="f_sbp")
            f_smoke = st.checkbox("Current Tobacco / Cigarette Smoker", key="f_smk")
            f_diab = st.checkbox("Diagnosed Diabetes Mellitus", key="f_db")

        if st.button("📊 Calculate 10-Year Heart Risk", type="primary"):
            res_cvd = calculate_framingham_cvd_risk(f_gender, f_age, f_tchol, f_hdl, f_sbp, f_smoke, f_diab)
            
            st.markdown(f"""
            <div class="calc-stat-box">
                <p style="margin:0; color:#90a4ae;">ESTIMATED 10-YEAR CVD RISK</p>
                <div class="calc-stat-value">{res_cvd['risk_percentage']}%</div>
                <h4>{res_cvd['risk_level']}</h4>
                <p style="margin:8px 0 0 0; color:#b0bec5; font-size:0.95rem;"><b>Recommended Clinical Action:</b> {res_cvd['clinical_action']}</p>
            </div>
            """, unsafe_allow_html=True)

    # Sub-tab 2: FINDRISC Diabetes
    with calc_tab2:
        st.markdown("#### 🩺 Finnish Diabetes Risk Score (FINDRISC)")
        st.caption("Clinically validated questionnaire predicting 10-year risk of developing Type 2 Diabetes.")
        
        fd_1, fd_2 = st.columns(2)
        with fd_1:
            fd_age = st.slider("Age", 18, 90, 45, key="fd_age")
            fd_bmi = st.number_input("Body Mass Index (BMI)", 15.0, 50.0, 27.2, key="fd_bmi")
            fd_waist = st.number_input("Waist Circumference (cm)", 60, 150, 96, key="fd_wst")
            fd_act = st.checkbox("Do at least 30 mins of daily physical activity", value=True, key="fd_act")
        with fd_2:
            fd_veg = st.checkbox("Eat vegetables, fruit, or berries every day", value=True, key="fd_veg")
            fd_bp = st.checkbox("Taken medication for High Blood Pressure", value=False, key="fd_bp")
            fd_glu = st.checkbox("Ever found to have high blood sugar (e.g. in pregnancy/illness)", value=False, key="fd_glu")
            fd_fam = st.selectbox("Family history of diabetes:", ["No family history", "2nd Degree (Grandparent/Uncle/Aunt)", "1st Degree (Parent/Sibling)"], key="fd_fam")

        if st.button("📊 Calculate Diabetes Risk Score", type="primary"):
            res_find = calculate_findrisc_diabetes_score(fd_age, fd_bmi, fd_waist, fd_act, fd_veg, fd_bp, fd_glu, fd_fam)
            st.markdown(f"""
            <div class="calc-stat-box">
                <p style="margin:0; color:#90a4ae;">FINDRISC SCORE (0-26)</p>
                <div class="calc-stat-value">{res_find['score']} / 26</div>
                <h4>{res_find['risk_level']}</h4>
                <p style="margin:8px 0 0 0; color:#b0bec5; font-size:0.95rem;"><b>Recommended Prevention:</b> {res_find['clinical_action']}</p>
            </div>
            """, unsafe_allow_html=True)

    # Sub-tab 3: BMI & BMR
    with calc_tab3:
        st.markdown("#### ⚖️ Body Composition, BMR & Caloric Target")
        
        b_c1, b_c2, b_c3 = st.columns(3)
        with b_c1:
            b_gender = st.selectbox("Gender", ["Male", "Female"], key="b_gen")
            b_age = st.number_input("Age", 10, 100, 30, key="b_age")
        with b_c2:
            b_height = st.number_input("Height (cm)", 100, 240, 175, key="b_ht")
            b_weight = st.number_input("Weight (kg)", 30.0, 250.0, 78.0, key="b_wt")
        with b_c3:
            st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
            calc_bmi_btn = st.button("📊 Calculate Body Metrics", type="primary", use_container_width=True)

        if calc_bmi_btn:
            b_res = calculate_bmi_bmr(b_weight, b_height, b_age, b_gender)
            
            res_c1, res_c2, res_c3 = st.columns(3)
            with res_c1:
                st.metric("Body Mass Index (BMI)", f"{b_res['bmi']} kg/m²", b_res['category'])
            with res_c2:
                st.metric("Ideal Healthy Weight", b_res['ibw_range'])
            with res_c3:
                st.metric("Basal Metabolic Rate (BMR)", f"{b_res['bmr']} kcal/day")
                
            st.info(f"💡 **Maintenance Calories:** ~{b_res['maintenance_calories_sedentary']} kcal (Sedentary) | ~{b_res['maintenance_calories_active']} kcal (Active)")

    # Sub-tab 4: eGFR Kidney Health
    with calc_tab4:
        st.markdown("#### 🫁 Estimated Glomerular Filtration Rate (eGFR - CKD-EPI)")
        st.caption("Evaluates kidney filtration capacity and Chronic Kidney Disease staging.")
        
        k_c1, k_c2, k_c3 = st.columns(3)
        with k_c1:
            k_gen = st.selectbox("Gender", ["Male", "Female"], key="k_gen")
        with k_c2:
            k_age = st.number_input("Age (years)", 18, 100, 52, key="k_age")
        with k_c3:
            k_creat = st.number_input("Serum Creatinine (mg/dL)", 0.3, 15.0, 1.2, step=0.1, key="k_cr")

        if st.button("📊 Calculate Kidney Function (eGFR)", type="primary"):
            k_res = calculate_egfr(k_creat, k_age, k_gen)
            st.markdown(f"""
            <div class="calc-stat-box">
                <p style="margin:0; color:#90a4ae;">ESTIMATED GFR (FILTRATION CAPACITY)</p>
                <div class="calc-stat-value">{k_res['egfr']} mL/min/1.73m²</div>
                <h4>{k_res['stage']}</h4>
                <p style="margin:8px 0 0 0; color:#b0bec5; font-size:0.95rem;"><b>Clinical Guidance:</b> {k_res['action']}</p>
            </div>
            """, unsafe_allow_html=True)


# ==============================================================================
# TAB 6: LONGITUDINAL HEALTH TRENDS & MULTI-REPORT TRACKER
# ==============================================================================
with tab6:
    st.markdown("### 📈 Longitudinal Health Trends & Multi-Report Progress Tracker")
    st.caption("Track historical biomarker trajectory across checkup dates with interactive Plotly visual graphs and AI milestone evaluations.")

    df_trends = pd.DataFrame(st.session_state.health_trends_records)

    t_col1, t_col2 = st.columns([2, 1])
    with t_col1:
        st.markdown("#### 📊 Biomarker Progress Visualizer")
        metric_choice = st.selectbox("Select Parameter to Visualize:", ["HbA1c & Fasting Glucose", "Lipid Profile (Total Chol & LDL)", "Hemoglobin & Body Weight"])
        
        if metric_choice == "HbA1c & Fasting Glucose":
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_trends["Date"], y=df_trends["HbA1c (%)"], name="HbA1c (%)", mode="lines+markers", line=dict(color="#ff5252", width=3)))
            fig.add_trace(go.Scatter(x=df_trends["Date"], y=df_trends["Fasting Glucose (mg/dL)"] / 20.0, name="Fasting Glucose (/20 scaled)", mode="lines+markers", line=dict(color="#ffd740", width=2, dash="dash")))
            fig.add_hline(y=5.7, line_dash="dot", line_color="#00e676", annotation_text="Normal HbA1c (<5.7%)")
            fig.update_layout(title="HbA1c & Blood Glucose Trajectory Over Time", template="plotly_dark", height=340)
            st.plotly_chart(fig, use_container_width=True)
            
        elif metric_choice == "Lipid Profile (Total Chol & LDL)":
            fig = go.Figure()
            fig.add_trace(go.Bar(x=df_trends["Date"], y=df_trends["Total Cholesterol (mg/dL)"], name="Total Cholesterol", marker_color="#ff7043"))
            fig.add_trace(go.Bar(x=df_trends["Date"], y=df_trends["LDL (mg/dL)"], name="LDL (Bad Chol)", marker_color="#ab47bc"))
            fig.add_hline(y=200, line_dash="dot", line_color="#00e676", annotation_text="Safe Total Chol (<200 mg/dL)")
            fig.update_layout(barmode="group", title="Lipid Panel Improvement Trends", template="plotly_dark", height=340)
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            fig = px.line(df_trends, x="Date", y=["Hemoglobin (g/dL)", "Weight (kg)"], markers=True, title="Hemoglobin Recovery & Body Weight Tracker", template="plotly_dark", height=340)
            st.plotly_chart(fig, use_container_width=True)

    with t_col2:
        st.markdown("#### ➕ Add New Lab Checkup Record")
        with st.form("add_trend_form", clear_on_submit=True):
            t_date = st.date_input("Checkup Date")
            t_hba1c = st.number_input("HbA1c (%)", 4.0, 15.0, 6.5, step=0.1)
            t_glu = st.number_input("Fasting Glucose (mg/dL)", 50, 400, 110)
            t_tc = st.number_input("Total Cholesterol (mg/dL)", 100, 400, 185)
            t_ldl = st.number_input("LDL (mg/dL)", 40, 250, 105)
            t_wt = st.number_input("Body Weight (kg)", 30.0, 180.0, 78.0)
            t_hb = st.number_input("Hemoglobin (g/dL)", 5.0, 20.0, 13.8)
            
            if st.form_submit_button("💾 Save Checkup Record", type="primary"):
                st.session_state.health_trends_records.append({
                    "Date": str(t_date),
                    "HbA1c (%)": t_hba1c,
                    "Fasting Glucose (mg/dL)": t_glu,
                    "Total Cholesterol (mg/dL)": t_tc,
                    "LDL (mg/dL)": t_ldl,
                    "Weight (kg)": t_wt,
                    "Hemoglobin (g/dL)": t_hb
                })
                st.success("✅ New checkup record logged!")
                st.rerun()

    st.markdown("#### 📋 Historical Health Data Table:")
    st.dataframe(df_trends, use_container_width=True)


# ==============================================================================
# TAB 7: DRUG-TO-DRUG INTERACTION CHECKER
# ==============================================================================
with tab7:
    st.markdown("### 💊 Drug-to-Drug Interaction & Safety Evaluator")
    st.caption("Enter two or more medications to check for pharmacological interactions, adverse risks, and contraindications.")

    col_d1, col_d2 = st.columns([2, 1])
    
    with col_d1:
        drugs_input = st.text_area(
            "Enter Medications (separated by commas):",
            value="Aspirin, Warfarin",
            help="Example: Aspirin, Warfarin or Ibuprofen, Lisinopril, Metformin",
            height=100
        )
        
        # Preset Quick Tests
        st.markdown("**Quick Preset Combinations:**")
        p_col1, p_col2, p_col3 = st.columns(3)
        with p_col1:
            if st.button("🔴 Aspirin + Warfarin"):
                drugs_input = "Aspirin, Warfarin"
        with p_col2:
            if st.button("🟡 Ibuprofen + Lisinopril"):
                drugs_input = "Ibuprofen, Lisinopril"
        with p_col3:
            if st.button("🟠 Metformin + Alcohol"):
                drugs_input = "Metformin, Alcohol"
                
        check_btn = st.button("🔍 Check Drug Interactions", type="primary", use_container_width=True)

    with col_d2:
        st.markdown("""
        <div class="feature-card">
            <h4>💡 Interaction Levels</h4>
            <p><b>🔴 Major:</b> High clinical risk. Avoid combination or seek immediate doctor advice.</p>
            <p><b>🟡 Moderate:</b> Possible side effects or reduced efficacy. Close monitoring needed.</p>
            <p><b>🟢 Minor:</b> Minimal clinical impact. Generally safe under standard dosage.</p>
        </div>
        """, unsafe_allow_html=True)

    if check_btn and drugs_input:
        with st.spinner("Analyzing pharmacological interactions with AI..."):
            prompt_text = (
                f"You are an expert clinical pharmacologist. Analyze the following combination of drugs/substances: '{drugs_input}'.\n\n"
                "Provide a structured evaluation in Markdown with:\n"
                "### 1. Risk Level (🔴 Major / 🟡 Moderate / 🟢 Minor)\n"
                "### 2. Interaction Mechanism & Potential Dangers\n"
                "### 3. Key Symptoms & Warning Signs to Watch\n"
                "### 4. Recommended Clinical Safety Advice\n"
                "Keep the explanation clear, professional, and actionable."
            )
            try:
                res = llm.invoke(prompt_text)
                st.markdown("---")
                st.markdown(res.content)
            except Exception as e:
                st.error(f"Error evaluating drug interaction: {str(e)}")


# ==============================================================================
# TAB 8: HOSPITAL & CLINIC FINDER
# ==============================================================================
with tab8:
    st.markdown("### 🏥 Nearby Hospitals & Healthcare Facilities Locator")
    st.caption("Find emergency centers, specialized hospitals, and pharmacies in any city or district worldwide.")

    col_h1, col_h2 = st.columns([3, 1])
    with col_h1:
        location_input = st.text_input("Enter City, Region, or Zip Code:", value="New York", placeholder="e.g. New York, Mumbai, London, Berlin...")
    with col_h2:
        search_hosp_btn = st.button("📍 Search Facilities", type="primary", use_container_width=True)

    if location_input:
        with st.spinner(f"Locating medical centers in '{location_input}'..."):
            hospitals = fetch_nearby_hospitals(location_input)
            
            st.markdown(f"#### 🏢 Found Facilities in **{location_input}**:")
            
            for h in hospitals:
                name = h.get("name", "Medical Facility")
                address = h.get("address", "Healthcare Zone")
                maps_url = h.get("maps_url", "#")
                
                st.markdown(f"""
                <div class="feature-card">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <h4 style="margin: 0; color: #42a5f5;">🏥 {name}</h4>
                            <p style="margin: 4px 0 0 0; color: #90a4ae; font-size: 0.9rem;">📍 {address}</p>
                        </div>
                        <div>
                            <a href="{maps_url}" target="_blank" style="text-decoration: none;">
                                <button style="background: #00e676; color: #000; border: none; padding: 8px 16px; border-radius: 8px; font-weight: 600; cursor: pointer;">
                                    🗺️ Open Maps
                                </button>
                            </a>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)


# ==============================================================================
# TAB 9: DOCTOR APPOINTMENT SIMULATOR, INTAKE BRIEFS & MED REMINDERS
# ==============================================================================
with tab9:
    st.markdown("### 📅 Doctor Appointment, Intake Briefs & Med Reminders")
    st.caption("Manage medication schedules, simulate clinical doctor bookings, and generate 1-page Pre-Consultation Intake Briefs.")

    sub_t1, sub_t2, sub_t3 = st.tabs([
        "📋 Doctor Intake Brief Generator",
        "⏰ Active Medication Reminders",
        "📄 Consultation PDF Summary Export"
    ])

    # 1. Doctor Intake Brief
    with sub_t1:
        st.markdown("#### 📋 1-Page Pre-Consultation Clinical Intake Brief")
        st.caption("Generates a physician-facing clinical summary of patient complaints, vitals, active medications, and anomalies.")
        
        in_c1, in_c2 = st.columns(2)
        with in_c1:
            p_name = st.text_input("Patient Full Name", value=st.session_state.intake_data["patient_name"])
            p_age = st.number_input("Age", 1, 110, value=st.session_state.intake_data["age"])
            p_gender = st.selectbox("Gender", ["Male", "Female", "Other"], index=0 if st.session_state.intake_data["gender"]=="Male" else 1)
            p_complaint = st.text_area("Chief Health Complaint", value=st.session_state.intake_data["complaint"], height=80)
            
        with in_c2:
            p_vitals = st.text_input("Reported Vitals", value=st.session_state.intake_data["vitals"])
            p_meds = st.text_area("Current Active Medications", value=st.session_state.intake_data["meds"], height=60)
            p_anom = st.text_area("Identified Lab Anomalies", value=st.session_state.intake_data["lab_anomalies"], height=60)

        if st.button("📄 Generate Physician Intake Brief (PDF)", type="primary", use_container_width=True):
            intake_pdf_name = f"Patient_Intake_Brief_{st.session_state.session_id[:6]}.pdf"
            intake_pdf_path = os.path.join(current_dir, "uploads", intake_pdf_name)
            os.makedirs(os.path.join(current_dir, "uploads"), exist_ok=True)
            
            generate_intake_brief_pdf(p_name, p_age, p_gender, p_complaint, p_vitals, p_meds, p_anom, intake_pdf_path)
            st.session_state.intake_data["pdf_path"] = intake_pdf_path
            st.success("✅ Pre-Consultation Clinical Brief successfully compiled!")

        if st.session_state.intake_data.get("pdf_path") and os.path.exists(st.session_state.intake_data["pdf_path"]):
            with open(st.session_state.intake_data["pdf_path"], "rb") as f:
                intake_bytes = f.read()
            st.download_button(
                label="📥 Download Official Physician Intake Brief (PDF)",
                data=intake_bytes,
                file_name="Patient_Clinical_Intake_Brief.pdf",
                mime="application/pdf",
                use_container_width=True
            )

    # 2. Medication Reminders
    with sub_t2:
        st.markdown("#### ⏰ Active Medication Schedule & Alarms")
        
        with st.expander("➕ Add New Medication Reminder", expanded=False):
            with st.form("add_reminder_form_tab9", clear_on_submit=True):
                r_c1, r_c2, r_c3 = st.columns(3)
                with r_c1:
                    med_name = st.text_input("Medicine Name", placeholder="e.g. Metformin")
                with r_c2:
                    med_dosage = st.text_input("Dosage", placeholder="e.g. 500 mg / 1 tablet")
                with r_c3:
                    med_time = st.text_input("Scheduled Time", placeholder="e.g. 09:00 AM")
                    
                med_freq = st.selectbox("Frequency", ["Once daily", "Twice daily", "Three times daily", "Every 8 hours", "As needed (SOS)"])
                
                submit_rem = st.form_submit_button("💾 Save Medication Reminder", type="primary")
                
                if submit_rem and med_name and med_time:
                    st.session_state.reminders.append({
                        "medicine": med_name,
                        "dosage": med_dosage or "1 dose",
                        "time": med_time,
                        "frequency": med_freq
                    })
                    st.success(f"✅ Reminder added for **{med_name}** at **{med_time}**!")
                    st.rerun()

        if not st.session_state.reminders:
            st.info("No active medication reminders configured.")
        else:
            for idx, rem in enumerate(st.session_state.reminders):
                col_r_info, col_r_del = st.columns([5, 1])
                with col_r_info:
                    st.markdown(f"""
                    <div class="feature-card">
                        <div style="display: flex; justify-content: space-between;">
                            <div>
                                <h4 style="margin: 0; color: #ffab00;">💊 {rem['medicine']} ({rem.get('dosage','1 dose')})</h4>
                                <p style="margin: 4px 0 0 0; color: #b0bec5; font-size: 0.9rem;">
                                    ⏰ <b>Time:</b> {rem['time']} &nbsp;|&nbsp; 🔄 <b>Frequency:</b> {rem.get('frequency','Daily')}
                                </p>
                            </div>
                            <div>
                                <span class="med-badge" style="background: rgba(33, 150, 243, 0.15); color: #42a5f5; border-color: rgba(33,150,243,0.3);">Active Alarm</span>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                with col_r_del:
                    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)
                    if st.button("🗑️ Delete", key=f"del_rem_{idx}"):
                        st.session_state.reminders.pop(idx)
                        st.rerun()

    # 3. Consultation PDF
    with sub_t3:
        st.markdown("#### 📄 Export Consultation Summary (PDF)")
        summary_folder = os.path.join(current_dir, "uploads")
        os.makedirs(summary_folder, exist_ok=True)
        pdf_filename = f"consultation_summary_{st.session_state.session_id[:8]}.pdf"
        pdf_filepath = os.path.join(summary_folder, pdf_filename)
        
        export_history = [
            {"role": m.get("role", "system"), "content": m.get("content", "")} 
            for m in st.session_state.messages 
            if m.get("content")
        ]
        
        col_pdf_gen, col_pdf_down = st.columns(2)
        with col_pdf_gen:
            if st.button("🔄 Generate Official PDF Report", type="primary", use_container_width=True):
                with st.spinner("Compiling PDF consultation summary with ReportLab..."):
                    try:
                        generate_consultation_pdf(export_history, pdf_filepath)
                        st.session_state.pdf_ready = True
                        st.success("✅ Consultation PDF successfully created!")
                    except Exception as e:
                        st.error(f"Error generating PDF: {str(e)}")

        if os.path.exists(pdf_filepath):
            with col_pdf_down:
                with open(pdf_filepath, "rb") as pdf_file:
                    pdf_bytes = pdf_file.read()
                    st.download_button(
                        label="📥 Download Consultation Summary PDF",
                        data=pdf_bytes,
                        file_name=f"Medical_Consultation_Summary_{st.session_state.session_id[:8]}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
