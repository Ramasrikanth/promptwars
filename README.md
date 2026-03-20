# 🏥 TriageLens

**TriageLens Powered by Google GenAI**

TriageLens is an AI-powered medical web application that allows users to upload unstructured, handwritten medical prescriptions. It leverages **Google's Gemini Multimodal AI** to accurately parse the contents, systematically structure the data into a strict JSON format, and flag any critical life-threatening conditions.

## ✨ Features
- **Prescription Parsing**: Analyzes complex medical handwriting using state-of-the-art vision models.
- **Structured Output**: Extracted data is strictly forced into a rigid JSON schema (managed by Pydantic) to ensure clinical system interoperability.
- **Critical Alerting**: Automatically detects combinations of symptoms and medicines to flag immediately life-threatening conditions.
- **Interactive UI**: Clean and user-friendly web interface constructed with Gradio.

## 🚀 Getting Started

### Prerequisites
- Python 3.11+
- An active Google Gemini API Key.

### Local Installation
1. Clone this repository.
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Create a `.env` file in the root directory and securely add your Gemini API Key:
   ```env
   GEMINI_API_KEY=your_gemini_api_key_here
   ```

### Running Locally
To launch the application:
```bash
python app.py
```
This will start a local Gradio server. Navigate to `http://localhost:8080` (or `http://0.0.0.0:8080`) in your web browser to interact with the UI.

## 🚢 Deployment (Google Cloud Run)

This application is fully containerized with a `Dockerfile` and natively configured for serverless deployment on **Google Cloud Run**.

```bash
# Example Cloud Run Deployment
gcloud run deploy triagelens \
    --source . \
    --region us-central1 \
    --allow-unauthenticated \
    --set-env-vars="GEMINI_API_KEY=YOUR_API_KEY"
```

## 🛠️ Built With
- **[Google GenAI (Gemini)](https://ai.google.dev/)** - Core LLM Intelligence & Vision Extraction
- **[Gradio](https://gradio.app/)** - Web Framework & User Interface
- **[Google Cloud Run](https://cloud.google.com/run)** - Auto-scaling Deployment 
