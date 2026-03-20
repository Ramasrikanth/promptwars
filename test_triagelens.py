import unittest
from unittest.mock import patch, MagicMock
import json
from triagelens import analyze_prescription

class TestTriageLens(unittest.TestCase):
    
    @patch('triagelens.client')
    @patch('triagelens.Image.open')
    def test_analyze_prescription_success(self, mock_image_open, mock_global_client):
        # Setup mocks
        from PIL import Image
        # Provide a real minimal image so the genai SDK type validator succeeds
        mock_image_instance = Image.new('RGB', (10, 10))
        mock_image_open.return_value = mock_image_instance
        
        # Mock structured response text matching TriageLensOutput schema
        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "triage_score": "ESI 1",
            "priority_level": "RED",
            "summary": "Severe chest pain indicating STEMI.",
            "extracted_vitals": {
                "bp": "85/50",
                "hr": "120",
                "spo2": "91%"
            },
            "critical_alerts": ["ST elevation in V1-V4"],
            "recommended_action": "Immediate Cath Lab",
            "system_trigger": "activate_stemi_protocol"
        })
        mock_global_client.models.generate_content.return_value = mock_response

        # Execute
        result = analyze_prescription("dummy/path.png", api_key="fake_key")

        # Assert
        self.assertIn("triage_score", result)
        self.assertEqual(result["priority_level"], "RED")
        self.assertEqual(result["extracted_vitals"]["hr"], "120")
        
        # Verify the client was called with correct model
        mock_global_client.models.generate_content.assert_called_once()
        call_args = mock_global_client.models.generate_content.call_args[1]
        self.assertEqual(call_args['model'], 'gemini-2.5-flash')

    @patch('triagelens.Image.open', side_effect=Exception("Corrupt image file"))
    def test_analyze_prescription_image_error(self, mock_image_open):
        # Image load exception shouldn't crash process, should return masked error
        result = analyze_prescription("corrupt/path.png", api_key="fake_key")
        self.assertIn("error", result)
        self.assertEqual(result["error"], "Processing failed due to corrupted or invalid image format.")

    @patch('triagelens.client')
    @patch('triagelens.Image.open')
    def test_analyze_prescription_api_error(self, mock_image_open, mock_global_client):
        # GenAI API Server error should not leak exception data
        from PIL import Image
        mock_image_open.return_value = Image.new('RGB', (10, 10))
        mock_global_client.models.generate_content.side_effect = Exception("500 Internal Server API Timeout Error")
        
        result = analyze_prescription("dummy/path.png", api_key="fake_key")
        
        self.assertIn("error", result)
        self.assertEqual(result["error"], "Analysis failed due to a server-side extraction error. Please try again.")

if __name__ == '__main__':
    unittest.main()
