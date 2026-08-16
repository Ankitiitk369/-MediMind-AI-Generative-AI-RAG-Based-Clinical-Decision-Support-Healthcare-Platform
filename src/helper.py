from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings


#Extract Data From the PDF File
def load_pdf_file(data):
    loader= DirectoryLoader(data,
                            glob="*.pdf",
                            loader_cls=PyPDFLoader)

    documents=loader.load()

    return documents



#Split the Data into Text Chunks
def text_split(extracted_data):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks=text_splitter.split_documents(extracted_data)
    return text_chunks



#Download the Embeddings from HuggingFace 
def download_hugging_face_embeddings():
    embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')  #this model return 384 dimensions
    return embeddings


# Emergency Guardrail Keywords Detection
EMERGENCY_KEYWORDS = [
    "chest pain", "heart attack", "difficulty breathing", "shortness of breath",
    "stroke", "unconscious", "fainting", "severe bleeding", "heavily bleeding",
    "seizure", "convulsion", "poisoning", "overdose", "suicide", "head injury",
    "anaphylaxis", "choking"
]

def check_emergency_keywords(user_query):
    query_lower = user_query.lower()
    for kw in EMERGENCY_KEYWORDS:
        if kw in query_lower:
            return True, kw
    return False, None


# Format Source References from Retrieved Context Documents
def format_sources(docs):
    sources = []
    seen = set()
    for doc in docs:
        meta = getattr(doc, 'metadata', {})
        source_path = meta.get('source', 'Medical_book.pdf')
        source_name = source_path.replace('\\', '/').split('/')[-1]
        page = meta.get('page', None)
        
        if page is not None:
            # PyPDFLoader pages are 0-indexed; convert to 1-indexed for humans
            ref = f"{source_name} (Page {int(page) + 1})"
        else:
            ref = source_name
            
        if ref not in seen:
            seen.add(ref)
            sources.append(ref)
    return sources


# 1. AI Medical Triage & Specialist Recommendation
def classify_triage_severity(user_query):
    query_lower = user_query.lower()
    
    # Critical Symptoms -> Red
    critical_terms = ["chest pain", "heart attack", "stroke", "difficulty breathing", "shortness of breath", "severe bleeding", "unconscious", "seizure", "choking"]
    for term in critical_terms:
        if term in query_lower:
            return {
                "level": "RED",
                "label": "🔴 CRITICAL / EMERGENCY",
                "recommendation": "Seek immediate emergency room (ER) or call 911/112!",
                "specialist": "Emergency Physician / Cardiologist / Pulmonologist"
            }
            
    # Moderate Symptoms -> Yellow
    moderate_terms = ["fever", "cough", "vomiting", "diarrhea", "fracture", "sprain", "rash", "dizziness", "migraine", "stomach pain", "joint pain"]
    for term in moderate_terms:
        if term in query_lower:
            specialist = "General Physician"
            if "skin" in query_lower or "rash" in query_lower or "acne" in query_lower:
                specialist = "Dermatologist"
            elif "joint" in query_lower or "bone" in query_lower or "fracture" in query_lower:
                specialist = "Orthopedic Specialist"
            elif "stomach" in query_lower or "diarrhea" in query_lower or "vomiting" in query_lower:
                specialist = "Gastroenterologist"
            elif "migraine" in query_lower or "headache" in query_lower:
                specialist = "Neurologist"
            elif "cough" in query_lower or "fever" in query_lower:
                specialist = "General Physician / ENT Specialist"
                
            return {
                "level": "YELLOW",
                "label": "🟡 MODERATE",
                "recommendation": "Schedule an appointment with a doctor within 24-48 hours.",
                "specialist": specialist
            }
            
    # Mild Symptoms -> Green
    return {
        "level": "GREEN",
        "label": "🟢 MILD / GENERAL ENQUIRY",
        "recommendation": "Monitor symptoms at home. Practice good health hygiene and consult a clinic if symptoms worsen.",
        "specialist": "General Practitioner"
    }


# 2. Nearby Hospitals & Pharmacies Locator via OpenStreetMap
import requests

def fetch_nearby_hospitals(location_query):
    try:
        url = "https://nominatim.openstreetmap.org/search"
        headers = {"User-Agent": "MedicalChatbotApp/1.0"}
        params = {
            "q": f"hospitals in {location_query}",
            "format": "json",
            "limit": 5
        }
        res = requests.get(url, headers=headers, params=params, timeout=5)
        if res.status_code == 200:
            data = res.json()
            hospitals = []
            for item in data:
                name = item.get("display_name", "Medical Center")
                lat = item.get("lat")
                lon = item.get("lon")
                maps_url = f"https://www.google.com/maps/search/?api=1&query={lat},{lon}" if lat and lon else "#"
                hospitals.append({
                    "name": name.split(",")[0],
                    "address": ", ".join(name.split(",")[1:3]),
                    "maps_url": maps_url
                })
            if hospitals:
                return hospitals
    except Exception:
        pass
        
    # Fallback response for default searches
    return [
        {"name": f"City General Hospital ({location_query})", "address": f"Central Healthcare Zone, {location_query}", "maps_url": f"https://www.google.com/maps/search/hospitals+in+{location_query}"},
        {"name": f"Apollo Emergency Clinic ({location_query})", "address": f"Main Medical Road, {location_query}", "maps_url": f"https://www.google.com/maps/search/clinics+in+{location_query}"},
        {"name": f"24x7 MedPlus Pharmacy ({location_query})", "address": f"Station Circle, {location_query}", "maps_url": f"https://www.google.com/maps/search/pharmacy+in+{location_query}"}
    ]


# 3. PDF Consultation Summary Generator
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

def generate_consultation_pdf(history_list, output_filepath):
    doc = SimpleDocTemplate(output_filepath, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    title_style = ParagraphStyle('TitleStyle', parent=styles['Heading1'], fontSize=20, textColor=colors.HexColor("#2c3e50"), spaceAfter=15)
    story.append(Paragraph("🩺 Medical Chatbot Consultation Summary", title_style))
    story.append(Paragraph("<b>Date:</b> August 2026 | <b>Patient Record:</b> Digital AI Summary", styles['Normal']))
    story.append(Spacer(1, 15))
    
    # Section Header
    header_style = ParagraphStyle('HeaderStyle', parent=styles['Heading2'], fontSize=14, textColor=colors.HexColor("#2980b9"), spaceAfter=10)
    story.append(Paragraph("Consultation Transcript & Medical Q&A", header_style))
    
    table_data = [["Role", "Query / AI Response"]]
    for item in history_list:
        role = item.get("role", "").capitalize()
        content = item.get("content", "")
        # Shorten or clean content
        table_data.append([role, Paragraph(content, styles['Normal'])])
        
    table = Table(table_data, colWidths=[80, 440])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#3498db")),
        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0,0), (-1,0), 8),
        ('BACKGROUND', (0,1), (-1,-1), colors.HexColor("#f8f9fa")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#bdc3c7")),
    ]))
    
    story.append(table)
    story.append(Spacer(1, 20))
    
    # Disclaimer
    disclaimer_style = ParagraphStyle('DiscStyle', parent=styles['Italic'], fontSize=9, textColor=colors.HexColor("#7f8c8d"))
    story.append(Paragraph("<b>Disclaimer:</b> This document is an AI-generated summary intended for informational reference only. Please share this document with a licensed healthcare practitioner for medical diagnosis and treatment.", disclaimer_style))
    
    doc.build(story)
    return output_filepath