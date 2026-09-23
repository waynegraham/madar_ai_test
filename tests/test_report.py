import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from PIL import Image
from lxml import html as html_parser

from src.waqf_vlm.report import build_report, load_report_data
from src.waqf_vlm.experiment import SegmentationSystem, find_disagreements
from src.segmentation import Region


ALTO = '''<alto xmlns="http://www.loc.gov/standards/alto/ns-v4#"><Layout>
<Page WIDTH="100" HEIGHT="100"><PrintSpace><TextBlock ID="dummy">
<TextLine ID="line"><String CONTENT="retained text"/></TextLine>
</TextBlock></PrintSpace></Page></Layout></alto>'''


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root / "data"
        self.data.mkdir()
        self.output = self.root / "site"

    def write(self, path, content):
        destination = self.data / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content, encoding="utf-8")
        return destination

    def test_alto_role_depends_on_explicit_reference_location(self):
        self.write("alto/page.xml", ALTO)
        self.write("ground-truth/alto/page.xml", ALTO)
        report = load_report_data(self.data)
        prediction, reference = report.pages[0].artifacts
        self.assertEqual((prediction.role, prediction.review), ("prediction", "Not yet reviewed"))
        self.assertEqual((reference.role, reference.review), ("reference", "Human reviewed"))
        self.assertEqual(prediction.transcriptions, ["retained text"])

    def test_human_layout_is_not_reviewed_text(self):
        self.write("ground-truth/page.json", json.dumps({"image": "page.tif", "image_width": 100,
            "image_height": 100, "regions": [{"id": "human-1", "type": "main_text", "bbox": [0, 0, 500, 500]}]}))
        page = load_report_data(self.data).pages[0]
        self.assertTrue(page.has_human_layout)
        self.assertFalse(page.has_reviewed_text)
        self.assertEqual(page.artifacts[0].review, "Not yet reviewed")

    def test_build_escapes_text_handles_legacy_confidence_and_preserves_inputs(self):
        record = {"task": "segmentation", "model": "<script>bad</script>", "response": {
            "regions": [{"type": "main_text", "bbox": [0, 0, 500, 500], "confidence": "0.95"}]}}
        self.write("results/page/model/segmentation.json", json.dumps(record))
        self.write("results/page/model/htr.json", json.dumps({"task": "htr", "response": {"transcription": "<b>نص</b>\nثان"}}))
        before = {p: p.read_bytes() for p in self.data.rglob("*") if p.is_file()}
        report = build_report(self.data, self.output)
        html = (self.output / "manuscripts" / report.pages[0].filename).read_text()
        self.assertIn("&lt;script&gt;bad&lt;/script&gt;", html)
        self.assertIn("&lt;b&gt;نص&lt;/b&gt;\nثان", html)
        self.assertIn('dir="auto"', html)
        self.assertIn("Human-corrected transcription: unavailable", html)
        self.assertTrue((self.output / "static/css/site.css").is_file())
        first = (self.output / "index.html").read_bytes()
        build_report(self.data, self.output)
        self.assertEqual(first, (self.output / "index.html").read_bytes())
        self.assertEqual(before, {p: p.read_bytes() for p in self.data.rglob("*") if p.is_file()})

    def test_missing_and_invalid_data_visible(self):
        self.write("results/page/model/broken.json", "not json")
        report = build_report(self.data, self.output)
        self.assertEqual(len(report.issues), 1)
        self.assertIn("Records needing attention", (self.output / "index.html").read_text())
        with self.assertRaises(ValueError):
            load_report_data(self.root / "missing")

    def test_output_protection(self):
        with self.assertRaises(ValueError):
            build_report(self.data, self.data / "generated")
        self.output.mkdir()
        (self.output / "precious.txt").write_text("preserve")
        with self.assertRaises(ValueError):
            build_report(self.data, self.output)
        self.assertEqual((self.output / "precious.txt").read_text(), "preserve")

    def test_empty_report(self):
        build_report(self.data, self.output)
        self.assertIn("No manuscript images", (self.output / "index.html").read_text())

    def test_publication_figures_preserve_source_and_accessible_structure(self):
        image_path = self.data / "images" / "page.tif"
        image_path.parent.mkdir()
        Image.new("RGB", (1000, 2000), "beige").save(image_path)
        original = image_path.read_bytes()
        self.write("alto/page.xml", ALTO)
        report = build_report(self.data, self.output)
        figure = report.pages[0].figures[0]
        self.assertEqual((figure.width, figure.height), (900, 1800))
        with Image.open(self.output / figure.thumbnail) as thumbnail:
            self.assertEqual(thumbnail.size, (320, 640))
        self.assertEqual(original, image_path.read_bytes())
        for path in self.output.rglob("*.html"):
            document = html_parser.fromstring(path.read_text())
            for image in document.xpath('//img'):
                self.assertTrue(image.get('alt'))
                self.assertTrue(image.xpath('ancestor::figure/figcaption'))
                self.assertTrue((path.parent / image.get('src')).is_file())
            for table in document.xpath('//table'):
                self.assertTrue(table.xpath('caption'))
                self.assertTrue(table.xpath('thead/tr/th[@scope="col"]'))
            for details in document.xpath('//details'):
                self.assertTrue(details.xpath('summary'))
            for href in document.xpath('//@href'):
                if href.startswith('#'):
                    self.assertTrue(document.xpath('//*[@id=$target]', target=href[1:]))
                else:
                    self.assertTrue((path.parent / href).is_file())

    def test_unreadable_source_image_is_not_invented(self):
        self.write("images/page.tif", "broken image")
        report = build_report(self.data, self.output)
        self.assertFalse(report.pages[0].figures)
        self.assertIn("display image unavailable", report.issues[0])
        page_html = (self.output / 'manuscripts' / report.pages[0].filename).read_text()
        self.assertIn("Image reproduction unavailable", page_html)

    def test_cli_has_no_inference_imports_or_network(self):
        self.write("results/page/model/segmentation.json", json.dumps({"task": "segmentation", "response": {
            "regions": [{"type": "main_text", "bbox": [100, 100, 900, 900]}]}}))
        (self.data / "images").mkdir()
        Image.new("RGB", (100, 200), "beige").save(self.data / "images/page.tif")
        code = '''import sys
from pathlib import Path
def guard(event, args):
    if event in ("socket.connect", "socket.getaddrinfo"):
        raise AssertionError("Network attempted")
sys.addaudithook(guard)
from src.waqf_vlm.report import main
main(["build", "--data", sys.argv[1], "--output", sys.argv[2]])
assert "src.lmstudio" not in sys.modules
assert "openai" not in sys.modules
'''
        subprocess.run([sys.executable, "-B", "-c", code, str(self.data), str(self.output)], check=True, capture_output=True)

    def test_disagreements_describe_predictions_without_using_reference(self):
        a = Region("a", "main_text", (100, 100, 500, 500))
        b = Region("b", "stamp_or_seal", a.bbox)
        systems = [SegmentationSystem("One", "one", [a]), SegmentationSystem("Two", "two", [b]),
                   SegmentationSystem("Human", "truth", [b], reviewed=True)]
        differences = find_disagreements(systems, {})
        self.assertEqual(len(differences), 1)
        self.assertIn("overlapping region", differences[0].heading)
        self.assertEqual([o.system for o in differences[0].observations], ["One", "Two"])
        self.assertEqual(differences[0].human_review[0].label, "stamp_or_seal")
        self.assertEqual(find_disagreements([systems[0], SegmentationSystem("Same", "same", [a])], {}), [])

    def test_corrected_alto_is_required_for_review_section(self):
        self.write("ground-truth/page.json", json.dumps({"image": "page.tif", "image_width": 100,
            "image_height": 100, "regions": []}))
        report = build_report(self.data, self.output)
        path = self.output / 'manuscripts' / report.pages[0].filename
        self.assertIn('Human review not yet available.', path.read_text())
        self.write('ground-truth/alto/page.xml', ALTO)
        build_report(self.data, self.output)
        self.assertNotIn('Human review not yet available.', path.read_text())
        self.assertIn('Human-corrected ALTO is available', path.read_text())
        self.assertIn('retained text', path.read_text())

    def test_repository_experiment_counts_and_missing_run_metadata(self):
        data = Path(__file__).resolve().parents[1] / 'data'
        page = next(p for p in load_report_data(data).pages if p.id == '1280_AB010309_0005')
        self.assertEqual([s.name for s in page.predictions], ['eScriptorium', 'Qwen3-VL 8B', 'Qwen3-VL 30B'])
        self.assertEqual([len(s.regions) for s in page.predictions], [11, 9, 10])
        self.assertEqual(page.predictions[0].skipped_blocks, 1)
        self.assertEqual([s.counts.get('stamp_or_seal', 0) for s in page.predictions], [0, 3, 3])
        self.assertEqual(page.disagreements[0].bbox, (509, 432, 617, 521))
        self.assertFalse(page.corrected_alto)
        for artifact in page.artifacts:
            if artifact.kind == 'Page layout':
                self.assertIn('1246 × 2048', artifact.details['Input dimensions'])
                self.assertEqual(artifact.details['Quantization'], 'Unavailable — not recorded')


if __name__ == "__main__":
    unittest.main()
