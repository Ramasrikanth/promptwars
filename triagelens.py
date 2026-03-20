import os
import sys
import json
from pydantic import BaseModel, Field
from PIL import Image
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

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

def analyze_prescription(image_path: str, api_key: str):
    # Initialize the client
    client = genai.Client(api_key=api_key)
    
    # Load the image
    try:
        img = Image.open(image_path)
    except Exception as e:
        print(f"Error loading image: {e}")
        sys.exit(1)
        
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
    
    print(f"Analyzing {image_path} with Gemini...")
    
    try:
        response = client.models.generate_content(
            model='gemini-2.5-pro',
            contents=[prompt, img],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=TriageLensOutput,
                temperature=0.1,
            ),
        )
        
        # The response is a JSON string matching the schema
        print("\n--- TriageLens Extracted Output ---")
        
        # Try parsing and pretty printing
        parsed_json = json.loads(response.text)
        print(json.dumps(parsed_json, indent=2))
        return parsed_json
        
    except Exception as e:
        print(f"\nError during extraction: {e}")
        if hasattr(e, 'response'):
            print(f"Response: {e.response}")
        return {"error": str(e)}

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python triagelens.py <path_to_image>")
        sys.exit(1)
        
    image_path = sys.argv[1]
    
    # User's provided API key
    api_key = os.environ.get("GEMINI_API_KEY", "AIzaSyAs0iicq_hh35pL7BDPiXc2VM80XHo34bA")
    
    analyze_prescription(image_path, api_key)
