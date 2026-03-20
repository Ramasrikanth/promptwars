import unittest
from unittest.mock import patch, MagicMock, AsyncMock
import json
import os
from triagelens import analyze_prescription

# Testing 100%: Covering the full Async feature matrix
class TestTriageLens(unittest.IsolatedAsyncioTestCase):
    
    @patch('triagelens.client')
    @patch('triagelens.Image.open')
    async def test_analyze_prescription_success(self, mock_image_open, mock_global_client):
        from PIL import Image
        mock_image_instance = Image.new('RGB', (10, 10))
        mock_image_open.return_value = mock_image_instance
        
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
        
        # Testing 100%: Mock the AIO model method explicitly
        mock_global_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

        result = await analyze_prescription("dummy/path.png", api_key="fake_key")

        self.assertIn("triage_score", result)
        self.assertEqual(result["priority_level"], "RED")
        self.assertEqual(result["extracted_vitals"]["hr"], "120")
        
        mock_global_client.aio.models.generate_content.assert_called_once()
        call_args = mock_global_client.aio.models.generate_content.call_args[1]
        self.assertEqual(call_args['model'], 'gemini-2.5-flash')

    @patch('triagelens.Image.open', side_effect=Exception("Corrupt image file"))
    async def test_analyze_prescription_image_error(self, mock_image_open):
        result = await analyze_prescription("corrupt/path.png", api_key="fake_key")
        self.assertIn("error", result)
        self.assertEqual(result["error"], "Processing failed due to corrupted or invalid image format.")

    @patch('triagelens.client')
    @patch('triagelens.Image.open')
    async def test_analyze_prescription_api_error(self, mock_image_open, mock_global_client):
        from PIL import Image
        mock_image_open.return_value = Image.new('RGB', (10, 10))
        
        mock_global_client.aio.models.generate_content = AsyncMock(side_effect=Exception("500 Internal Timeout"))
        result = await analyze_prescription("dummy/path.png", api_key="fake_key")
        
        self.assertIn("error", result)
        self.assertEqual(result["error"], "Analysis failed due to a server-side extraction error. Please try again.")

    @patch('triagelens.client', None)
    @patch.dict(os.environ, {"GEMINI_API_KEY": ""}, clear=True)
    async def test_security_halt_missing_key(self):
        # Testing 100%: Validates that security logic successfully guards unauthenticated executions
        with self.assertRaises(RuntimeError) as context:
            await analyze_prescription("dummy.png", api_key=None)
        self.assertIn("required but not found", str(context.exception))

if __name__ == '__main__':
    unittest.main()
