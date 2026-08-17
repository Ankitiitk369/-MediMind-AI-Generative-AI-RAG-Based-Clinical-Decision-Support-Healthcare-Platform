import os
import sys

# If Streamlit is running app.py on Streamlit Cloud, execute streamlit_app.py
if "streamlit" in sys.modules or os.environ.get("STREAMLIT_SERVER_PORT") or os.environ.get("STREAMLIT_RUNTIME") or os.environ.get("STREAMLIT_CONFIG"):
    import streamlit as st
    streamlit_app_path = os.path.join(os.path.dirname(__file__), "streamlit_app.py")
    if os.path.exists(streamlit_app_path):
        with open(streamlit_app_path, "r", encoding="utf-8") as f:
            code = f.read()
        exec(compile(code, streamlit_app_path, "exec"), globals())
        st.stop()

from flask import Flask, render_template, jsonify, request, session, send_file
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
    generate_medical_report_pdf
)
from langchain_pinecone import Pinecone as PineconeVectorStore
from pinecone import Pinecone as PineconeClient
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import system_prompt
from werkzeug.utils import secure_filename
import uuid

app = Flask(__name__)
app.secret_key = "medical_chatbot_secret_key_2026"

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
GROQ_API_KEY = os.environ.get('GROQ_API_KEY')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
if GROQ_API_KEY:
    os.environ["GROQ_API_KEY"] = GROQ_API_KEY

embeddings = download_hugging_face_embeddings()

index_name = "medicalbot"
pc = PineconeClient(api_key=PINECONE_API_KEY)
index = pc.Index(index_name)

docsearch = PineconeVectorStore(
    index=index,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.4, max_tokens=600)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# In-memory stores
chat_history_store = {}
reminders_store = {}


@app.route("/")
def index():
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())
    return render_template('chat.html')


@app.route("/upload", methods=["POST"])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"success": False, "message": "No file part in request"})
    file = request.files['file']
    if file.filename == '':
        return jsonify({"success": False, "message": "No selected file"})
    
    language = request.form.get("language", "English")
    
    if file and file.filename.lower().endswith('.pdf'):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # 1. Extract text and generate comprehensive AI Clinical Analysis
            report_text = extract_text_from_pdf(filepath)
            
            analysis_prompt = (
                f"You are a Chief Medical Officer and Clinical Laboratory Specialist.\n"
                f"Analyze the following medical report in {language}:\n\n"
                f"{report_text}\n\n"
                f"Provide:\n"
                f"1. Report Overview & Health Status\n"
                f"2. Abnormal Parameters Breakdown (Value, Normal Range, Meaning in simple language)\n"
                f"3. Plain Language Summary for the patient\n"
                f"4. Dietary & Lifestyle Recommendations\n"
                f"5. Recommended Specialist & Next Steps\n"
                f"6. Red Flag Warning Symptoms\n"
            )
            analysis_res = llm.invoke(analysis_prompt)
            analysis_text = analysis_res.content

            # 2. Index to Pinecone for RAG retrieval
            docs = load_pdf_file(data=app.config['UPLOAD_FOLDER'])
            chunks = text_split(docs)
            docsearch.add_documents(chunks)
            
            return jsonify({
                "success": True,
                "filename": filename,
                "analysis": analysis_text,
                "message": f"Successfully analyzed and indexed '{filename}'!"
            })
        except Exception as e:
            return jsonify({"success": False, "message": f"Error processing PDF: {str(e)}"})
            
    return jsonify({"success": False, "message": "Only PDF files are supported."})



@app.route("/get", methods=["GET", "POST"])
def chat():
    user_session_id = session.get("session_id", "default_session")
    if user_session_id not in chat_history_store:
        chat_history_store[user_session_id] = []
        
    msg = request.form.get("msg") or (request.json.get("msg") if request.is_json else None)
    language = request.form.get("language") or (request.json.get("language") if request.is_json else "English") or "English"
    
    if not msg:
        return jsonify({"error": "No message provided"}), 400
        
    # 1. Emergency Keyword Guardrail Check
    is_emergency, trigger_keyword = check_emergency_keywords(msg)
    if is_emergency:
        emergency_msg = (
            f"🚨 **EMERGENCY WARNING DETECTED** 🚨\n"
            f"Your query contains symptoms associated with a critical emergency (*{trigger_keyword}*). "
            f"Please call emergency services (911 / 112) or seek immediate professional emergency medical care!"
        )
        return jsonify({
            "answer": emergency_msg,
            "sources": [],
            "is_emergency": True,
            "triage": {"level": "RED", "label": "🔴 CRITICAL / EMERGENCY", "recommendation": "Emergency Room (ER) Immediately", "specialist": "Emergency Physician / Cardiologist"}
        })

    # 2. Triage & Specialist Classification
    triage_info = classify_triage_severity(msg)

    # 3. Similarity Search & Source Retrieval
    retrieved_docs = retriever.invoke(msg)
    sources = format_sources(retrieved_docs)
    
    context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
    
    # 4. Format Memory Context (Last 3 Exchanges)
    history_list = chat_history_store[user_session_id][-6:]
    history_context = ""
    if history_list:
        history_context = "\nRecent Conversation History:\n" + "\n".join([f"{item['role'].capitalize()}: {item['content']}" for item in history_list]) + "\n"

    # 5. Invoke LLM with Prompt
    formatted_prompt = prompt.format_messages(
        language=language,
        context=context_text + history_context,
        input=msg
    )
    
    response = llm.invoke(formatted_prompt)
    answer_text = response.content
    
    # Update Session History
    chat_history_store[user_session_id].append({"role": "user", "content": msg})
    chat_history_store[user_session_id].append({"role": "assistant", "content": answer_text})
    
    return jsonify({
        "answer": answer_text,
        "sources": sources,
        "is_emergency": False,
        "triage": triage_info
    })


# ---------------- NEW REAL-WORLD FEATURES ENDPOINTS ---------------- #

# Feature 2: Drug-to-Drug Interaction Checker
@app.route("/check_drugs", methods=["POST"])
def check_drugs():
    drugs_str = request.form.get("drugs") or (request.json.get("drugs") if request.is_json else "")
    if not drugs_str:
        return jsonify({"success": False, "result": "Please enter at least 2 medicine names."})
        
    prompt_text = (
        f"You are a clinical pharmacology expert. Analyze the following combination of drugs/medications: '{drugs_str}'.\n"
        "Provide a concise evaluation covering:\n"
        "1. Potential Drug Interactions (Major/Moderate/Minor)\n"
        "2. Key Warnings & Side Effects\n"
        "3. Safety Advice\n"
        "Keep response under 4-5 bullet points."
    )
    try:
        response = llm.invoke(prompt_text)
        return jsonify({"success": True, "result": response.content})
    except Exception as e:
        return jsonify({"success": False, "result": f"Error analyzing drug interaction: {str(e)}"})


# Feature 4: Nearby Hospitals & Pharmacies Finder
@app.route("/find_nearby", methods=["POST"])
def find_nearby():
    location = request.form.get("location") or (request.json.get("location") if request.is_json else "New York")
    hospitals = fetch_nearby_hospitals(location)
    return jsonify({"success": True, "location": location, "hospitals": hospitals})


# Feature 5: 1-Click Consultation PDF Download
@app.route("/download_summary", methods=["GET"])
def download_summary():
    user_session_id = session.get("session_id", "default_session")
    history_list = chat_history_store.get(user_session_id, [])
    
    if not history_list:
        history_list = [{"role": "system", "content": "No active conversation transcript found."}]
        
    pdf_filename = f"consultation_summary_{user_session_id[:8]}.pdf"
    pdf_filepath = os.path.join(app.config['UPLOAD_FOLDER'], pdf_filename)
    
    generate_consultation_pdf(history_list, pdf_filepath)
    return send_file(pdf_filepath, as_attachment=True, download_name="Medical_Consultation_Summary.pdf")


# Feature 7: Medication Reminders & Dosage Scheduler
@app.route("/save_reminder", methods=["POST"])
def save_reminder():
    user_session_id = session.get("session_id", "default_session")
    if user_session_id not in reminders_store:
        reminders_store[user_session_id] = []
        
    medicine = request.form.get("medicine") or (request.json.get("medicine") if request.is_json else "")
    time_str = request.form.get("time") or (request.json.get("time") if request.is_json else "")
    dosage = request.form.get("dosage") or (request.json.get("dosage") if request.is_json else "1 tablet")
    
    if medicine and time_str:
        reminder = {"medicine": medicine, "time": time_str, "dosage": dosage}
        reminders_store[user_session_id].append(reminder)
        return jsonify({"success": True, "message": f"Reminder set for {medicine} ({dosage}) at {time_str}!", "reminders": reminders_store[user_session_id]})
        
    return jsonify({"success": False, "message": "Medicine name and time are required."})


@app.route("/list_reminders", methods=["GET"])
def list_reminders():
    user_session_id = session.get("session_id", "default_session")
    reminders = reminders_store.get(user_session_id, [])
    return jsonify({"reminders": reminders})


if __name__ == '__main__':
    if "streamlit" not in sys.modules and not os.environ.get("STREAMLIT_SERVER_PORT"):
        app.run(host="0.0.0.0", port=8080, debug=True)


