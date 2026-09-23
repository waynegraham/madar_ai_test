from dataclasses import asdict, replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image, ImageDraw

from src.segmentation import Region
from src.waqf_vlm.segmentation_figure import render_segmentation


class SegmentationFigureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'original.tif'
        Image.new('RGB', (300, 600), '#c4b596').save(self.source)
        self.region = Region('r1', 'main_text', (100, 100, 800, 800), confidence='0.95')
        self.output = self.root / 'assets'

    def render(self, regions=None, **kwargs):
        return render_segmentation(self.source, [self.region] if regions is None else regions,
                                   'Test system', output_dir=self.output, **kwargs)

    def test_deterministic_png_and_immutable_inputs(self):
        original = self.source.read_bytes()
        record = asdict(self.region)
        first = self.render()
        content = first.read_bytes()
        self.assertEqual(first, self.render())
        self.assertEqual(content, first.read_bytes())
        self.assertEqual(original, self.source.read_bytes())
        self.assertEqual(record, asdict(self.region))
        self.assertEqual(first.parent, self.output)
        self.assertEqual(first.suffix, '.png')

    def test_hidden_confidence_does_not_change_figure(self):
        default = self.render()
        changed = self.render([replace(self.region, confidence=0.1)])
        self.assertEqual(default, changed)
        self.assertNotEqual(default, self.render(show_confidence=True))

    def test_manuscript_pixels_and_aspect_ratio_preserved_without_annotations(self):
        result = self.render([])
        with Image.open(result) as image, Image.open(self.source) as source:
            # The image occupies its native 300 x 600 rectangle, with margins added.
            self.assertEqual(image.crop((48, 150, 348, 750)).tobytes(), source.tobytes())

    def test_labels_outside_image_and_review_explicit(self):
        texts = []
        real = ImageDraw.ImageDraw.text
        def capture(draw, position, text, *args, **kwargs):
            texts.append((position, text))
            return real(draw, position, text, *args, **kwargs)
        with patch.object(ImageDraw.ImageDraw, 'text', capture):
            self.render(human_ground_truth=[replace(self.region, id='human')])
        labels = [(pos, text) for pos, text in texts if text.startswith(('P01', 'H01'))]
        self.assertEqual(len(labels), 2)
        self.assertTrue(all(pos[0] > 348 for pos, _ in labels))
        self.assertTrue(any(text == 'Human reviewed' for _, text in texts))
        self.assertFalse(any('confidence' in text for _, text in texts))
        with self.assertRaises(ValueError):
            self.render(reviewed=True, human_ground_truth=[])

    def test_invalid_coordinates_rejected(self):
        with self.assertRaises(ValueError):
            self.render([replace(self.region, bbox=(-1, 0, 100, 100))])


if __name__ == '__main__':
    unittest.main()
