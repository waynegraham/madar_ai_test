from pathlib import Path
import tempfile
import unittest

from PIL import Image

from src.segmentation import Region
from src.images import crop_normalized_bbox
from src.waqf_vlm.experiment import (
    SegmentationSystem, find_disagreements, select_disagreements, illustrate_experiment,
)


class DisagreementTests(unittest.TestCase):
    def systems(self):
        return [
            SegmentationSystem('A', 'a.json', [Region('a', 'main_text', (100, 100, 700, 700))]),
            SegmentationSystem('B', 'b.json', [Region('b', 'stamp_or_seal', (100, 100, 700, 700)),
                                               Region('extra', 'handwritten_annotation', (800, 800, 950, 950))]),
            SegmentationSystem('C', 'c.json', [Region('c', 'archival_mark', (100, 100, 700, 700))]),
        ]

    def test_full_pairwise_list_and_short_selection(self):
        all_items = find_disagreements(self.systems(), {}, limit=None)
        self.assertEqual(len(all_items), 5)  # Three label conflicts and two unmatched cases.
        self.assertEqual(sum(d.kind == 'different_types' for d in all_items), 3)
        self.assertEqual(sum(d.kind == 'unmatched' for d in all_items), 2)
        selected = select_disagreements(all_items, limit=4)
        self.assertEqual(len(selected), 2)  # Spatially duplicate examples suppressed only here.
        self.assertEqual(len(all_items), 5)
        self.assertTrue(all(item in all_items for item in selected))
        self.assertEqual(len({d.identifier for d in all_items}), 5)
        self.assertEqual(select_disagreements(all_items, limit=0), [])

    def test_unmatched_remains_unmatched_despite_overlapping_group(self):
        systems = [SegmentationSystem('A', 'a', [Region('small', 'main_text', (100,100,200,200))]),
                   SegmentationSystem('B', 'b', [Region('large', 'main_text', (0,0,1000,1000))])]
        cases = find_disagreements(systems, {}, limit=None)
        small = next(c for c in cases if c.bbox == (100,100,200,200))
        self.assertEqual(small.observations[0].label, 'main_text')
        self.assertEqual(small.observations[1].label, 'No matched region')
        self.assertIn('main_text', small.observations[1].context)
        self.assertFalse(small.human_review)

    def test_one_to_one_assignment_does_not_reuse_a_box(self):
        box = (100,100,700,700)
        systems = [SegmentationSystem('A', 'a', [Region('a1','main_text',box),Region('a2','main_text',box)]),
                   SegmentationSystem('B', 'b', [Region('b','main_text',box)])]
        cases = find_disagreements(systems, {}, limit=None)
        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0].observations[1].label, 'No matched region')
        self.assertEqual(cases[0].observations[1].overlap, 1)

    def test_references_are_displayed_without_generating_or_adjudicating_differences(self):
        systems = self.systems()
        initial = find_disagreements(systems, {}, limit=None)
        reference = SegmentationSystem('Reviewed', 'truth.xml', [Region('h','main_text',(100,100,700,700))], reviewed=True)
        cases = find_disagreements(systems+[reference], {}, limit=None)
        self.assertEqual(len(cases), len(initial))
        for case in cases:
            self.assertEqual(len(case.observations), 3)
            if case.kind == 'different_types':
                self.assertEqual(case.human_review[0].label, 'main_text')
            else:
                self.assertEqual(case.human_review[0].label, 'No matching reviewed region')

    def test_full_resolution_crop_keeps_original_pixels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root/'original.tif'
            Image.new('RGB',(2400,3000),'beige').save(source)
            original = source.read_bytes()
            output = root/'site'
            (output/'images').mkdir(parents=True)
            all_items = find_disagreements(self.systems(), {}, limit=None)
            illustrate_experiment([], all_items, source, output)
            large = next(c for c in all_items if c.kind == 'different_types')
            with Image.open(output/large.crop) as crop:
                expected = crop_normalized_bbox(source, large.bbox, padding=30)
                self.assertGreater(crop.height,1200)
                self.assertEqual(crop.size,expected.size)
                self.assertEqual(crop.tobytes(),expected.tobytes())
            with Image.open(output/large.crop_preview) as preview:
                self.assertLessEqual(max(preview.size),900)
            self.assertEqual(source.read_bytes(),original)
            identical_crops = [d.crop for d in all_items if d.bbox == large.bbox]
            self.assertEqual(len(set(identical_crops)),1)


if __name__ == '__main__':
    unittest.main()
