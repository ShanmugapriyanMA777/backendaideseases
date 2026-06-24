import os
import io
import time
import json
import joblib
import httpx
import numpy as np
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier

# ReportLab imports for PDF generation
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

# Load environment variables
load_dotenv(dotenv_path="C:/Users/shaisty priya/.gemini/antigravity-ide/scratch/ai-disease-prediction-system/.env")

app = FastAPI(title="MediPredict AI - Production API", version="1.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths to models
MODELS_DIR = "C:/Users/shaisty priya/.gemini/antigravity-ide/scratch/ai-disease-prediction-system/models"
MODEL_PATH = os.path.join(MODELS_DIR, "disease_model.pkl")
DISEASE_ENCODER_PATH = os.path.join(MODELS_DIR, "disease_encoder.pkl")
SYMPTOM_ENCODER_PATH = os.path.join(MODELS_DIR, "symptom_encoder.pkl")

# Supabase configuration (loaded from env)
SUPABASE_URL = os.getenv("VITE_SUPABASE_URL", "")
SUPABASE_ANON_KEY = os.getenv("VITE_SUPABASE_ANON_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct:free")

# Global variables to store loaded ML elements
model = None
disease_encoder = None
symptom_list = None

def load_ml_resources():
    global model, disease_encoder, symptom_list
    try:
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
            disease_encoder = joblib.load(DISEASE_ENCODER_PATH)
            symptom_list = joblib.load(SYMPTOM_ENCODER_PATH)
            print("Successfully loaded ML models and encoders.")
        else:
            print("ML model files not found. Inference endpoints will run in mock mode.")
    except Exception as e:
        print(f"Error loading ML resources: {e}")

@app.on_event("startup")
async def startup_event():
    load_ml_resources()

# Pydantic schemas
class PredictionRequest(BaseModel):
    symptoms: List[str]
    model_name: Optional[str] = "Random Forest"

class PredictionResponse(BaseModel):
    disease: str
    confidence: float
    risk_level: str
    top_predictions: List[Dict[str, Any]]
    model_used: str
    explainable_ai: List[Dict[str, Any]]
    details: Dict[str, Any]

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    history: List[ChatMessage]

class ReportRequest(BaseModel):
    patient_name: str
    symptoms: List[str]
    disease: str
    confidence: float
    risk_level: str
    precautions: List[str]
    medicine: str
    diet: List[str]
    user_id: str

# Helper to fetch disease info from Supabase (or local fallback)
async def fetch_disease_details(disease_name: str) -> Dict[str, Any]:
    default_details = {
        "description": "Information not available.",
        "medicine": "Consult a healthcare professional.",
        "precautions": ["Rest", "Hydration"],
        "diet": ["Balanced diet"],
        "home_remedies": ["Adequate rest"],
        "emergency": ["Shortness of breath", "Severe chest pain"]
    }
    
    if not SUPABASE_URL or not SUPABASE_ANON_KEY:
        return default_details

    try:
        url = f"{SUPABASE_URL}/rest/v1/diseases?disease_name=eq.{disease_name}"
        headers = {
            "apikey": SUPABASE_ANON_KEY,
            "Authorization": f"Bearer {SUPABASE_ANON_KEY}",
            "Content-Type": "application/json"
        }
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=headers, timeout=5.0)
            if response.status_code == 200:
                data = response.json()
                if data:
                    item = data[0]
                    return {
                        "description": item.get("description", ""),
                        "medicine": item.get("medicine", "Consult a doctor."),
                        "precautions": item.get("precautions", []),
                        "diet": item.get("diet_suggestions", []),
                        "home_remedies": item.get("home_remedies", []),
                        "emergency": item.get("emergency_signs", [])
                    }
    except Exception as e:
        print(f"Error fetching from Supabase: {e}")
        
    return default_details

@app.get("/")
async def root():
    return {"message": "Welcome to MediPredict AI Production API"}

@app.get("/health-check")
async def health_check():
    status = "healthy" if model is not None else "degraded (models not loaded)"
    return {"status": status, "version": "1.0.0", "supabase_configured": bool(SUPABASE_URL)}

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    if not request.symptoms:
        raise HTTPException(status_code=400, detail="No symptoms provided")
        
    global model, disease_encoder, symptom_list
    
    # Reload model if it failed on startup
    if model is None:
        load_ml_resources()
        
    # If ML model is still not loaded, run mock prediction logic
    if model is None or disease_encoder is None or symptom_list is None:
        print("ML resources unavailable. Using mock prediction logic.")
        mock_disease = "Influenza"
        details = await fetch_disease_details(mock_disease)
        return {
            "disease": mock_disease,
            "confidence": 0.92,
            "risk_level": "Moderate",
            "top_predictions": [
                {"disease": "Influenza", "confidence": 0.92},
                {"disease": "Common Cold", "confidence": 0.05},
                {"disease": "COVID-19", "confidence": 0.03}
            ],
            "model_used": "Mock Ensemble (Model files missing)",
            "explainable_ai": [
                {"symptom": request.symptoms[0], "importance": 0.05}
            ],
            "details": details
        }
        
    # Clean and vectorize symptoms
    vector = np.zeros(len(symptom_list))
    cleaned_input_symptoms = []
    
    for user_sym in request.symptoms:
        # Standardize symptom name to match columns
        cleaned = user_sym.strip().lower().replace(" ", "_")
        cleaned_input_symptoms.append(cleaned)
        if cleaned in symptom_list:
            idx = symptom_list.index(cleaned)
            vector[idx] = 1
            
    # Reshape for prediction
    features = vector.reshape(1, -1)
    
    # Run prediction
    try:
        probabilities = model.predict_proba(features)[0]
        prediction_idx = np.argmax(probabilities)
        confidence = float(probabilities[prediction_idx])
        predicted_disease = disease_encoder.inverse_transform([prediction_idx])[0]
        
        # Get Top 5 predictions
        top_indices = np.argsort(probabilities)[::-1][:5]
        top_predictions = []
        for idx in top_indices:
            name = disease_encoder.inverse_transform([idx])[0]
            conf = float(probabilities[idx])
            if conf > 0.01: # Filter out near-zero predictions
                top_predictions.append({"disease": name, "confidence": conf})
                
        # Explainable AI: Identify active symptoms and their model weight/importance
        explainable_ai = []
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            for sym_raw, sym_clean in zip(request.symptoms, cleaned_input_symptoms):
                if sym_clean in symptom_list:
                    f_idx = symptom_list.index(sym_clean)
                    weight = float(importances[f_idx])
                    explainable_ai.append({
                        "symptom": sym_raw,
                        "importance": weight
                    })
            explainable_ai = sorted(explainable_ai, key=lambda x: x['importance'], reverse=True)
            
        # Determine risk level
        risk_level = "Moderate"
        dis_lower = predicted_disease.lower()
        if any(w in dis_lower for w in ['heart attack', 'paralysis', 'malaria', 'aids', 'pneumonia', 'tuberculosis', 'typhoid']):
            risk_level = "Critical"
        elif any(w in dis_lower for w in ['hypertension', 'diabetes', 'hepatitis', 'hypoglycemia', 'dengue']):
            risk_level = "High"
        elif any(w in dis_lower for w in ['cold', 'allergy', 'acne', 'gerd', 'gastroenteritis']):
            risk_level = "Low"
            
        # Fetch disease details (medications, precautions, diets)
        details = await fetch_disease_details(predicted_disease)
        
        return {
            "disease": predicted_disease,
            "confidence": confidence,
            "risk_level": risk_level,
            "top_predictions": top_predictions,
            "model_used": type(model).__name__,
            "explainable_ai": explainable_ai,
            "details": details
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/chat")
async def chat(request: ChatRequest):
    if not OPENROUTER_API_KEY:
        return {
            "response": "I am currently running in offline demo mode since no OpenRouter API key was configured. Please consult your physician for medical concerns."
        }
        
    system_prompt = (
        "You are a professional AI Health Assistant. "
        "Provide educational, highly informative, and helpful medical information based on user questions. "
        "Do NOT provide a final diagnosis. "
        "Always strongly recommend consulting healthcare professionals for any serious concerns. "
        "Include a clear notice indicating this advice is for educational support only."
    )
    
    # Format messages for OpenRouter
    formatted_messages = [{"role": "system", "content": system_prompt}]
    for msg in request.history:
        role = "user" if msg.role == "user" else "assistant"
        formatted_messages.append({"role": role, "content": msg.content})
        
    formatted_messages.append({"role": "user", "content": request.message})
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": OPENROUTER_MODEL,
                    "messages": formatted_messages,
                    "temperature": 0.3
                },
                timeout=30.0
            )
            
            if response.status_code == 200:
                data = response.json()
                reply = data['choices'][0]['message']['content']
                return {"response": reply}
            else:
                print(f"OpenRouter API error status={response.status_code} body={response.text}")
                raise HTTPException(status_code=502, detail="Failed to connect to AI assistant service.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chatbot connection error: {str(e)}")

@app.post("/generate-report")
async def generate_report(request: ReportRequest, authorization: Optional[str] = Header(None)):
    # 1. Build PDF in memory using ReportLab
    pdf_buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        pdf_buffer, 
        pagesize=letter,
        rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36
    )
    
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=24,
        leading=28,
        textColor=colors.HexColor('#1E3A8A'),
        spaceAfter=15
    )
    
    section_style = ParagraphStyle(
        'SectionHeader',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=14,
        leading=18,
        textColor=colors.HexColor('#3B82F6'),
        spaceBefore=12,
        spaceAfter=8
    )
    
    body_style = ParagraphStyle(
        'ReportBody',
        parent=styles['BodyText'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#334155')
    )
    
    disclaimer_style = ParagraphStyle(
        'Disclaimer',
        parent=styles['BodyText'],
        fontName='Helvetica-Oblique',
        fontSize=8,
        leading=11,
        textColor=colors.HexColor('#E11D48'),
        alignment=1 # Center aligned
    )

    story = []
    
    # Header Banner/Title
    story.append(Paragraph("MediPredict AI - Medical Diagnosis Report", title_style))
    story.append(Spacer(1, 10))
    
    # Patient Info Metadata Table
    meta_data = [
        [Paragraph("<b>Patient Name:</b>", body_style), Paragraph(request.patient_name, body_style),
         Paragraph("<b>Date Generated:</b>", body_style), Paragraph(time.strftime("%Y-%m-%d %H:%M:%S UTC"), body_style)],
        [Paragraph("<b>User Account ID:</b>", body_style), Paragraph(request.user_id, body_style),
         Paragraph("<b>Report Code:</b>", body_style), Paragraph(f"MPR-{int(time.time())}", body_style)]
    ]
    t_meta = Table(meta_data, colWidths=[100, 160, 100, 160])
    t_meta.setStyle(TableStyle([
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#E2E8F0')),
        ('BACKGROUND', (0,0), (0,-1), colors.HexColor('#F1F5F9')),
        ('BACKGROUND', (2,0), (2,-1), colors.HexColor('#F1F5F9')),
        ('PADDING', (0,0), (-1,-1), 6),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(t_meta)
    story.append(Spacer(1, 15))
    
    # Symptoms Reported
    story.append(Paragraph("Symptoms Analyzed", section_style))
    sym_list_str = ", ".join([s.replace("_", " ").capitalize() for s in request.symptoms])
    story.append(Paragraph(sym_list_str, body_style))
    story.append(Spacer(1, 15))
    
    # Prediction Results
    story.append(Paragraph("AI Diagnostic Assessment", section_style))
    pred_data = [
        [Paragraph("<b>Predicted Disease</b>", body_style), Paragraph(request.disease, body_style)],
        [Paragraph("<b>Confidence Level</b>", body_style), Paragraph(f"{request.confidence * 100:.1f}%", body_style)],
        [Paragraph("<b>Assessed Risk Level</b>", body_style), Paragraph(request.risk_level, body_style)]
    ]
    t_pred = Table(pred_data, colWidths=[150, 370])
    t_pred.setStyle(TableStyle([
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#E2E8F0')),
        ('BACKGROUND', (0,0), (0,-1), colors.HexColor('#EFF6FF')),
        ('PADDING', (0,0), (-1,-1), 8),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(t_pred)
    story.append(Spacer(1, 15))
    
    # Treatment & Advice Details
    story.append(Paragraph("Recommended Treatment & Care Guidelines", section_style))
    
    care_data = []
    if request.medicine:
        care_data.append([Paragraph("<b>Suggested Medication:</b>", body_style), Paragraph(request.medicine, body_style)])
    if request.precautions:
        prec_bullet = "<br/>".join([f"• {p}" for p in request.precautions])
        care_data.append([Paragraph("<b>Precautions:</b>", body_style), Paragraph(prec_bullet, body_style)])
    if request.diet:
        diet_bullet = "<br/>".join([f"• {d}" for d in request.diet])
        care_data.append([Paragraph("<b>Dietary Advice:</b>", body_style), Paragraph(diet_bullet, body_style)])
        
    t_care = Table(care_data, colWidths=[150, 370])
    t_care.setStyle(TableStyle([
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#E2E8F0')),
        ('BACKGROUND', (0,0), (0,-1), colors.HexColor('#F8FAFC')),
        ('PADDING', (0,0), (-1,-1), 8),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
    ]))
    story.append(t_care)
    story.append(Spacer(1, 25))
    
    # Disclaimer block
    story.append(Paragraph("<b>MEDICAL DISCLAIMER:</b> This report is generated by an artificial intelligence model trained on standard symptom-disease datasets. It is meant for educational and awareness support only. It does NOT constitute a final medical diagnosis or clinical prescription. Please consult with a qualified primary care physician or medical expert immediately for diagnosis and treatment planning.", disclaimer_style))
    
    # Build document
    doc.build(story)
    
    pdf_data = pdf_buffer.getvalue()
    pdf_buffer.close()
    
    # 2. Upload to Supabase Storage if authorization and keys are present
    uploaded_url = None
    if SUPABASE_URL and SUPABASE_ANON_KEY and authorization and request.user_id:
        try:
            filename = f"{request.user_id}/report_{int(time.time())}.pdf"
            upload_url = f"{SUPABASE_URL}/storage/v1/object/medical-reports/{filename}"
            
            # Extract access token from client Header
            jwt_token = authorization.replace("Bearer ", "")
            
            headers = {
                "apikey": SUPABASE_ANON_KEY,
                "Authorization": f"Bearer {jwt_token}",
                "Content-Type": "application/pdf"
            }
            
            async with httpx.AsyncClient() as client:
                res = await client.post(upload_url, headers=headers, content=pdf_data, timeout=10.0)
                if res.status_code == 200:
                    # Construct signed or public URL
                    uploaded_url = f"{SUPABASE_URL}/storage/v1/object/public/medical-reports/{filename}"
                    print(f"Successfully uploaded PDF report to Storage: {uploaded_url}")
                else:
                    print(f"Supabase Storage Upload failed with status={res.status_code} body={res.text}")
        except Exception as e:
            print(f"Error uploading report to Supabase: {e}")
            
    # 3. If uploaded successfully, return the PDF URL. Otherwise, stream the file download.
    if uploaded_url:
        return {"pdf_url": uploaded_url}
    else:
        # Stream response back for download directly
        return StreamingResponse(
            io.BytesIO(pdf_data),
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename=medical_report_{int(time.time())}.pdf"}
        )
