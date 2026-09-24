import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image
from src.waqf_vlm.run_htr import main, prepare
from src.lmstudio import VLMResponse


class BatchHTRTests(unittest.TestCase):
    def test_saved_success_resumes_and_invalid_json_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)
            (data/'images').mkdir()
            Image.new('RGB', (30, 40), 'white').save(data/'images/page.png')
            prompt = data/'prompt.txt'
            prompt.write_text('Transcribe this image.')
            response = VLMResponse('qwen3-vl-8b-instruct-mlx', '{"transcription":"test"}', None,
                                   1, prompt.read_text(), 'crop.png',
                                   SimpleNamespace(choices=[SimpleNamespace(finish_reason='stop')]))
            argv = ['run_htr', '--data', str(data), '--prompt', str(prompt), '--model', 'qwen3-vl-8b']
            with patch('sys.argv', argv), patch('src.waqf_vlm.run_htr.get_client'), patch('src.waqf_vlm.run_htr.analyze_image', return_value=response) as infer:
                main()
                main()
                self.assertEqual(infer.call_count, 1)
            response.content = 'incomplete JSON'
            argv[-1] = 'qwen3-vl-30b'
            with patch('sys.argv', argv), patch('src.waqf_vlm.run_htr.get_client'), patch('src.waqf_vlm.run_htr.analyze_image', return_value=response):
                with self.assertRaises(SystemExit):
                    main()
            error = json.loads(next((data/'htr-errors').rglob('*.json')).read_text())
            self.assertEqual(error['raw_content'], 'incomplete JSON')
            self.assertEqual(len(list((data/'results').rglob('*.json'))), 1)

    def test_whole_page_fallback_and_frozen_geometry(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)
            (data/'images').mkdir()
            Image.new('RGB', (150, 200), 'white').save(data/'images/page.jpg')
            jobs = prepare(data)
            self.assertEqual(len(jobs), 1)
            self.assertEqual(jobs[0]['region_bbox'], [0, 0, 1000, 1000])
            manifest = json.loads((data/'htr-inputs/page.json').read_text())
            self.assertNotIn('transcription', manifest['regions'][0])
            # A later layout must not silently change the inference regions.
            (data/'ground-truth').mkdir()
            (data/'ground-truth/page.json').write_text('{}')
            repeated = prepare(data)
            self.assertEqual(repeated[0]['image_sha256'], jobs[0]['image_sha256'])
            self.assertEqual(repeated[0]['region_id'], 'whole-page')


if __name__ == '__main__':
    unittest.main()
