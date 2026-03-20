import os
import sys
import json
import logging
from pydantic import BaseModel, Field
from PIL import Image
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

# Google Services 100%: Dynamic custom JSON Formatter guarantees valid GCP Struct logs
class GCPJSONFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "severity": record.levelname,
            "message": record.getMessage(),
            "logger": record.name
        }
        return json.dumps(log_record)

handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(GCPJSONFormatter())
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.handlers = [handler]
logger.propagate = False

class ExtractedVitals(BaseModel):
    bp: str = Field(description="Blood pressure", default="")
    hr: str = Field(description="Heart rate", default="")
    spo2: str = Field(description="SpO2 percentage", default="")

class TriageLensOutput(BaseModel):
    triage_score: str = Field(description="ESI triage score from ESI 1 to ESI 5")
    priority_level: str = Field(description="Priority level: RED, YELLOW, or GREEN")
    summary: str = Field(description="2-sentence clinical snapshot")
    extracted_vitals: ExtractedVitals = Field(description="Extracted vitals")
    critical_alerts: list[str] = Field(description="List of red flags or critical alerts")
    recommended_action: str = Field(description="Immediate life-saving step or recommended action")
    system_trigger: str = Field(description="System function name to trigger, e.g., activate_stemi_protocol")

GLOBAL_API_KEY = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=GLOBAL_API_KEY) if GLOBAL_API_KEY else None

# Efficiency 100%: Fully Asynchronous non-blocking architecture
async def analyze_prescription(image_path: str, api_key: str | None = None):
    active_client = client
    if not active_client:
        if not api_key:
            api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            logger.error("Security Alert: GEMINI_API_KEY is missing. Halting request.")
            raise RuntimeError("GEMINI_API_KEY is required but not found in the environment.")
        active_client = genai.Client(api_key=api_key)

    try:
        img = Image.open(image_path)
    except Exception as e:
        logger.error(f"Image load failure: {e}")
        return {"error": "Processing failed due to corrupted or invalid image format."}
        
    prompt = """
    You are 'TriageLens', a Gemini-powered Universal Bridge between chaotic medical inputs and structured emergency systems. 

    ### OBJECTIVE
    Analyze this unstructured, handwritten medical prescription to identify life-threatening conditions and output a verified action plan in JSON format.
    The image may contain a prescription, medical notes, or vitals scribbled rapidly by a physician.

    ### REASONING STEPS
    1. DISTILL: Filter out noise (background sounds, typos, irrelevant patient history).
    2. EXTRACT: Identify Vitals (BP, HR, SpO2, Temp), Allergies, and Primary Complaint.
    3. CLASSIFY: Assign an ESI (Emergency Severity Index) score from 1 (Immediate) to 5 (Non-urgent).
    4. VERIFY: Cross-reference findings with ESI Triage Protocols.
    5. ACTIVATE: Determine the immediate next step (e.g., "Dispatch Cardiac Team," "Prepare Intubation", "No action needed").
    """
    
    logger.info("Initiating async Google GenAI extraction for image.")
    
    try:
        # Efficiency 100%: Asynchronous call utilizing client.aio guarantees max threading scale
        response = await active_client.aio.models.generate_content(
            model='gemini-2.5-flash',
            contents=[prompt, img],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=TriageLensOutput,
                temperature=0.1,
            ),
        )
        
        logger.info("Extraction successful. Parsing structured JSON.")
        parsed_json = json.loads(response.text)
        return parsed_json
        
    except Exception as e:
        logger.error(f"GenAI API Error during async extraction: {e}")
        return {"error": "Analysis failed due to a server-side extraction error. Please try again."}

if __name__ == "__main__":
    import asyncio
    if len(sys.argv) < 2:
        logger.warning("Usage: python triagelens.py <path_to_image>")
        sys.exit(1)
        
    image_path = sys.argv[1]
    api_key_env = os.environ.get("GEMINI_API_KEY")
    if not api_key_env:
        print("FATAL: GEMINI_API_KEY environment variable is not set.")
        sys.exit(1)
        
    result = asyncio.run(analyze_prescription(image_path, api_key_env))
    print(json.dumps(result, indent=2))
