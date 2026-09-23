from pathlib import Path
import json
import tempfile
import unittest

from src.segmentation import Region
from src.waqf_vlm.experiment import SegmentationSystem
from src.waqf_vlm.report import Artifact, Manuscript, ReportData, load_report_data
from src.waqf_vlm.overview import corpus_overview


class OverviewTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.data=Path(self.temp.name)

    def sample(self, count):
        pages=[]
        for i in range(count):
            region=Region(f'r{i}','stamp_or_seal',(100,100,500,500))
            page=Manuscript(f'p{i}')
            page.segmentations=[SegmentationSystem('Qwen3-VL 8B','result.json',[region]),
                                SegmentationSystem('Human','truth.xml',[region],reviewed=True)]
            page.artifacts=[Artifact('result.json','Page layout','qwen3-vl-8b-instruct-mlx','Not yet reviewed','prediction',details={'Prompt version':'v1'})]
            pages.append(page)
        return ReportData(pages,[])

    def manifest(self,count):
        (self.data/'corpus.json').write_text(json.dumps({'pages':{f'p{i}':{'manuscript_id':f'm{i//2}', 'layout_review_complete':True} for i in range(count)}}))

    def test_small_sample_and_manuscript_grouping_are_dynamic(self):
        sample=self.sample(2)
        unknown=corpus_overview(sample,self.data)
        self.assertEqual(unknown['page_count'],2)
        self.assertIsNone(unknown['manuscript_count'])
        self.assertEqual(unknown['coverage']['Qwen3-VL 8B'],2)
        self.manifest(2)
        summary=corpus_overview(sample,self.data)
        self.assertEqual(summary['manuscript_count'],1)
        self.assertTrue(summary['mapping_complete'])
        self.assertEqual(summary['total_regions'],2) # References excluded.
        self.assertEqual(summary['evaluations'],[])

    def test_review_scope_threshold_and_denominators(self):
        sample=self.sample(5)
        self.assertEqual(corpus_overview(sample,self.data)['evaluations'],[])
        self.manifest(5)
        summary=corpus_overview(sample,self.data)
        row=summary['evaluations'][0]
        self.assertEqual((row['pages'],row['truth_pages'],row['reference_regions'],row['predicted_regions']),(5,5,5,5))
        self.assertEqual((row['precision'],row['recall'],row['f1']),(1,1,1))
        self.assertEqual(corpus_overview(sample,self.data,min_reviewed_pages=6)['evaluations'],[])
        with self.assertRaises(ValueError): corpus_overview(sample,self.data,min_reviewed_pages=0)

    def test_missing_predictions_are_not_perfect_scores(self):
        sample=self.sample(2)
        sample.pages[1].segmentations[0].regions=[]
        self.manifest(2)
        row=corpus_overview(sample,self.data,min_reviewed_pages=2)['evaluations'][0]
        self.assertEqual(row['reference_regions'],2)
        self.assertEqual(row['predicted_regions'],1)
        self.assertEqual(row['recall'],.5)
        self.assertAlmostEqual(row['f1'],2/3)
        sample.pages[0].segmentations[0].regions=[]
        row=corpus_overview(sample,self.data,min_reviewed_pages=2)['evaluations'][0]
        self.assertIsNone(row['precision'])
        self.assertEqual(row['recall'],0)

    def test_duplicate_runs_are_in_inventory_but_not_evaluation(self):
        sample=self.sample(2)
        sample.pages[0].segmentations.append(sample.pages[0].segmentations[0])
        self.manifest(2)
        summary=corpus_overview(sample,self.data,min_reviewed_pages=2)
        self.assertEqual(summary['total_regions'],3)
        self.assertEqual(summary['coverage']['Qwen3-VL 8B'],2)
        self.assertEqual(summary['evaluations'],[])
        self.assertEqual(summary['excluded_ambiguous'],1)

    def test_current_sample_does_not_become_collection_total(self):
        data=Path(__file__).resolve().parents[1]/'data'
        summary=corpus_overview(load_report_data(data),data)
        self.assertEqual(summary['page_count'],6)
        self.assertEqual(summary['coverage'],{'eScriptorium':6,'Qwen3-VL 8B':1,'Qwen3-VL 30B':1})
        self.assertEqual(summary['total_regions'],68)
        self.assertEqual(summary['reviewed_pages'],0)
        self.assertEqual(summary['htr_evaluated'],0)
        self.assertEqual(summary['htr_available'],1)
        self.assertEqual(summary['evaluations'],[])


if __name__=='__main__': unittest.main()
