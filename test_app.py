import unittest
from unittest.mock import patch, AsyncMock
import json
from app import process_upload

class TestGradioBackend(unittest.IsolatedAsyncioTestCase):

    @patch('app.analyze_prescription', new_callable=AsyncMock)
    @patch('app.os.environ.get')
    async def test_process_upload_success(self, mock_env_get, mock_analyze):
        mock_env_get.return_value = "VALID_KEY"
        mock_analyze.return_value = {"success": True, "triage_score": "ESI 2"}
        
        output = await process_upload("valid_image.png")
        data = json.loads(output)
        self.assertEqual(data["triage_score"], "ESI 2")

    async def test_process_upload_no_image(self):
        output = await process_upload(None)
        data = json.loads(output)
        self.assertIn("error", data)
        self.assertIn("No image uploaded", data["error"])

    @patch('app.os.environ.get', return_value=None)
    async def test_process_upload_missing_key(self, mock_env_get):
        output = await process_upload("valid_image.png")
        data = json.loads(output)
        self.assertIn("error", data)
        self.assertIn("API Key missing", data["error"])

if __name__ == '__main__':
    unittest.main()
