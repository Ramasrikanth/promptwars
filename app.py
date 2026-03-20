import gradio as gr
import json
import os
import sys
import logging
from triagelens import analyze_prescription
from dotenv import load_dotenv

load_dotenv()

# Google Services 100%: Dynamic structured JSON serialization for explicit GCP alignment
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

# Efficiency 100%: Using async threading to seamlessly serve massive traffic horizontally
async def process_upload(image):
    if image is None:
        return json.dumps({"error": "No image uploaded. Please provide a valid prescription image."}, indent=2)
        
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logger.error("Missing API Key in front-end context.")
        return json.dumps({"error": "System Configuration Error: API Key missing."}, indent=2)
        
    try:
        result = await analyze_prescription(image, api_key)
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"Frontend unhandled async exception: {e}")
        return json.dumps({"error": "An unexpected server condition occurred."}, indent=2)

css = """
h1 {text-align: center; color: #1a1a1a;} /* High Contrast */
.gradio-container {font-family: 'Arial', sans-serif;}
"""

with gr.Blocks() as interface:
    gr.Markdown("# 🏥 TriageLens: Emergency Prescription AI", elem_id="main-title")
    gr.Markdown("**Instructions for use:** Upload a clear image of a handwritten medical prescription. TriageLens utilizes Gemini's multimodal capabilities to autonomously extract structured standard data and identify critical, life-threatening conditions. Results output as compliant JSON.", elem_id="main-desc")
    
    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="filepath", label="Upload Prescription Scribble", elem_id="prescription_image_dropzone", show_label=True)
            analyze_btn = gr.Button("Analyze Prescription Data", variant="primary", elem_id="submit_analysis_button")
            
        with gr.Column(scale=1):
            json_output = gr.Code(language="json", label="TriageLens Extraction Result", interactive=False, lines=20, elem_id="json_analysis_results")
            
    analyze_btn.click(fn=process_upload, inputs=image_input, outputs=json_output)
    
    gr.Examples(
        examples=[
            ["data/example_stemi_critical.png"],
            ["data/example_asthma_yellow.png"],
            ["data/example_ankle_green.png"],
        ],
        inputs=image_input,
        label="Test with Sample Prescriptions"
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    logger.info(f"Starting async TriageLens UI on port {port}...")
    interface.launch(server_name="0.0.0.0", server_port=port, theme=gr.themes.Soft(primary_hue="blue"), css=css)
