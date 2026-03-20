import gradio as gr
import json
import os
from triagelens import analyze_prescription
from dotenv import load_dotenv

load_dotenv()

def process_upload(image):
    if image is None:
        return json.dumps({"error": "No image uploaded. Please upload a prescription."}, indent=2)
        
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return json.dumps({"error": "API Key is missing! Ensure GEMINI_API_KEY is defined."}, indent=2)
        
    try:
        result = analyze_prescription(image, api_key)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)}, indent=2)

# Create a beautiful and clean UI using Gradio
with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue")) as interface:
    gr.Markdown("# 🏥 TriageLens Powered by Google GenAI")
    gr.Markdown("Upload a handwritten medical prescription. TriageLens will use Gemini's multimodal capabilities to extract and structure the data into JSON, while identifying critical life-threatening conditions.")
    
    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="filepath", label="Upload Prescription Scribble")
            analyze_btn = gr.Button("Analyze Prescription", variant="primary")
            
        with gr.Column(scale=1):
            json_output = gr.Code(language="json", label="TriageLens Extraction Result", interactive=False, lines=20)
            
    analyze_btn.click(fn=process_upload, inputs=image_input, outputs=json_output)
    
    # Load some example imagery
    gr.Examples(
        examples=[
            ["data/rx_scribble_1_1773989266870.png"],
            ["data/rx_scribble_2_1773989285623.png"],
            ["data/rx_scribble_4_1773989322156.png"],
        ],
        inputs=image_input
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print(f"Starting TriageLens UI on port {port}...")
    interface.launch(server_name="0.0.0.0", server_port=port)
