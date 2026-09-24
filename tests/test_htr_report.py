import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.evaluation import htr_metrics
from src.waqf_vlm.htr import load_htr_comparisons
from src.waqf_vlm.report import build_report
from lxml import html


class HTRReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root / 'data'
        self.data.mkdir()
        self.item = {'id': 'human-1', 'type': 'main_text', 'bbox': [100,100,800,800]}
        self.annotation()
        self.result('سَلام\nثان')

    def write(self, path, data):
        path = self.data/path
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(data,ensure_ascii=False),encoding='utf-8')

    def annotation(self):
        self.write('ground-truth/page.json',{'image':'page.tif','image_width':1000,'image_height':1000,'regions':[self.item]})

    def result(self,text,region_id='human-1'):
        self.write('results/page/qwen/htr/result.json',{'task':'htr','model':'qwen3-vl-8b-instruct-mlx',
            'region_id':region_id,'response':{'transcription':text,'language':'Arabic','script':'Arabic'}})

    def load(self):
        issues=[]
        return load_htr_comparisons(self.data,'page',issues),issues

    def test_no_metrics_without_corrected_reference(self):
        self.item['transcription']='unreviewed text'
        self.annotation()
        with patch('src.waqf_vlm.htr.htr_metrics',side_effect=AssertionError('metrics must not run')):
            groups,issues=self.load()
        self.assertFalse(issues)
        self.assertIsNone(groups[0].human)
        self.assertIsNone(groups[0].readings[0].metrics)
        self.assertEqual(groups[0].readings[0].direction,'rtl')

    def test_exact_text_metrics_and_secondary_html(self):
        reference='سلام\nثان'
        self.item.update(transcription=reference,review_status='human_corrected',language='Arabic',script='Arabic')
        self.annotation()
        groups,_=self.load()
        reading=groups[0].readings[0]
        self.assertEqual(reading.text,'سَلام\nثان')
        self.assertEqual(reading.metrics,htr_metrics(reference,reading.text))
        self.assertEqual(reading.metrics.character_errors,1)
        report=build_report(self.data,self.root/'site')
        path=self.root/'site/manuscripts'/report.pages[0].filename
        doc=html.fromstring(path.read_text())
        metrics=doc.xpath('//details[contains(@class,"htr-metrics")]')[0]
        self.assertNotIn('open',metrics.attrib)
        self.assertIn('سَلام\nثان',doc.xpath('//div[contains(@class,"htr-system-reading")]')[0].text_content())
        self.assertTrue(doc.xpath('//div[contains(@class,"htr-transcription") and @dir="rtl"]'))

    def test_corrected_alto_and_machine_alto_same_region(self):
        xml='''<alto xmlns="http://www.loc.gov/standards/alto/ns-v4#"><Layout><Page WIDTH="1000" HEIGHT="1000"><PrintSpace><TextBlock ID="human-1" HPOS="100" VPOS="100" WIDTH="700" HEIGHT="700"><TextLine ID="line"><String CONTENT="سلام"/></TextLine></TextBlock></PrintSpace></Page></Layout></alto>'''
        for prefix in ('ground-truth/alto','alto'):
            path=self.data/prefix/'page.xml'
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(xml)
        groups,issues=self.load()
        self.assertFalse(issues)
        self.assertEqual(len(groups),1)
        self.assertEqual(groups[0].human.text,'سلام')
        self.assertEqual(groups[0].readings[0].system,'eScriptorium')
        self.assertEqual(groups[0].readings[0].metrics.cer,0)

    def test_result_without_human_region_link_is_not_guessed(self):
        self.result('سلام',region_id='missing')
        groups,issues=self.load()
        self.assertEqual(groups,[])
        self.assertIn('no unique human-defined region link',issues[0])

    def test_empty_hypothesis_is_preserved(self):
        self.item.update(transcription='سلام',review_status='human_corrected')
        self.annotation()
        self.result('')
        groups,_=self.load()
        self.assertEqual(groups[0].readings[0].text,'')
        self.assertEqual(groups[0].readings[0].metrics.cer,1)

    def test_frozen_machine_inputs_allow_comparison_before_review(self):
        (self.data/'ground-truth/page.json').unlink()
        self.write('htr-inputs/page.json', {
            'coordinate_system': 'normalized-1000', 'region_source': 'alto/page.xml',
            'regions': [self.item]})
        groups, issues = self.load()
        self.assertFalse(issues)
        self.assertEqual(len(groups), 1)
        self.assertIsNone(groups[0].human)
        self.assertIsNone(groups[0].readings[0].metrics)
        xml = '''<alto xmlns="http://www.loc.gov/standards/alto/ns-v4#"><Layout><Page WIDTH="1000" HEIGHT="1000"><PrintSpace><TextBlock ID="human-1" HPOS="100" VPOS="100" WIDTH="700" HEIGHT="700"><TextLine ID="line"><String CONTENT="سلام"/></TextLine></TextBlock></PrintSpace></Page></Layout></alto>'''
        path = self.data/'ground-truth/alto/page.xml'
        path.parent.mkdir(parents=True)
        path.write_text(xml)
        groups, issues = self.load()
        self.assertFalse(issues)
        self.assertEqual(groups[0].human.text, 'سلام')
        self.assertIsNotNone(groups[0].readings[0].metrics)

    def test_failed_inference_is_visible_without_becoming_a_reading(self):
        self.write('htr-errors/page/qwen/failure.json', {
            'model': 'qwen3-vl-30b-a3b-instruct-mlx', 'region_id': 'human-1'})
        groups, issues = self.load()
        self.assertFalse(issues)
        self.assertEqual(len(groups[0].readings), 1)
        self.assertEqual(groups[0].failures[0]['system'], 'Qwen3-VL 30B')

    def test_real_saved_readings_have_no_accuracy(self):
        data=Path(__file__).resolve().parents[1]/'data'
        issues=[]
        groups=load_htr_comparisons(data,'1280_AB010309_0005',issues)
        # Corpus contents grow as inference runs; legacy orphan readings stay excluded.
        self.assertTrue(all('no unique human-defined region link' in issue for issue in issues))
        self.assertTrue(all(g.human is None for g in groups))
        self.assertTrue(all(r.metrics is None for g in groups for r in g.readings))


if __name__=='__main__':
    unittest.main()
