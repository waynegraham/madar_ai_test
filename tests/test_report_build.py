from pathlib import Path
import hashlib
import json
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
from src.waqf_vlm.report import build_report, load_report_data, serve_report


class BuildTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.data=self.root/'data'
        (self.data/'images').mkdir(parents=True)
        for name in ('one','two'):
            Image.new('RGB',(80,120),'beige').save(self.data/'images'/f'{name}.tif')
        self.output=self.root/'site'

    def snapshot(self):
        return {p.relative_to(self.output).as_posix(): (hashlib.sha256(p.read_bytes()).hexdigest(),p.stat().st_mtime_ns)
                for p in self.output.rglob('*') if p.is_file()}

    def test_full_rebuild_is_identical_in_bytes_and_mtime(self):
        build_report(self.data,self.output)
        before=self.snapshot()
        build_report(self.data,self.output)
        self.assertEqual(before,self.snapshot())

    def test_selected_rebuild_keeps_other_page_and_assets(self):
        build_report(self.data,self.output)
        manifest=json.loads((self.output/'.waqf-report.json').read_text())
        retained=manifest['page_files']['two']
        before=self.snapshot()
        Image.new('RGB',(80,120),'white').save(self.data/'images/one.tif')
        with patch('src.waqf_vlm.report.prepare_figures', wraps=__import__('src.waqf_vlm.report',fromlist=['prepare_figures']).prepare_figures) as render:
            build_report(self.data,self.output,manuscript_id='one')
        self.assertEqual(len(render.call_args_list),1)
        self.assertEqual(render.call_args.args[0].pages[0].id,'one')
        after=self.snapshot()
        self.assertTrue(all(before[p]==after[p] for p in retained))
        self.assertTrue(any(before[p]!=after[p] for p in manifest['page_files']['one']))

    def test_selected_first_build_then_add_page(self):
        build_report(self.data,self.output,manuscript_id='one')
        self.assertEqual(len(list((self.output/'manuscripts').glob('*.html'))),1)
        build_report(self.data,self.output,manuscript_id='two')
        self.assertEqual(len(list((self.output/'manuscripts').glob('*.html'))),2)
        self.assertIn('one',(self.output/'index.html').read_text())
        self.assertIn('two',(self.output/'index.html').read_text())

    def test_unknown_selection_leaves_existing_report_untouched(self):
        build_report(self.data,self.output)
        before=self.snapshot()
        with self.assertRaisesRegex(ValueError,'No manuscript page found'):
            build_report(self.data,self.output,manuscript_id='absent')
        self.assertEqual(before,self.snapshot())

    def test_obsolete_owned_assets_removed_and_unowned_preserved(self):
        build_report(self.data,self.output)
        old=json.loads((self.output/'.waqf-report.json').read_text())['page_files']['two']
        (self.output/'notes.txt').write_text('keep')
        (self.data/'images/two.tif').unlink()
        build_report(self.data,self.output)
        self.assertTrue(all(not (self.output/p).exists() for p in old))
        self.assertEqual((self.output/'notes.txt').read_text(),'keep')

    def test_malformed_htr_and_missing_optional_experiments_do_not_abort(self):
        path=self.data/'results/one/model/htr/bad.json'
        path.parent.mkdir(parents=True)
        path.write_text('[]')
        report=build_report(self.data,self.output)
        self.assertTrue(report.issues)
        self.assertTrue((self.output/'index.html').is_file())

    def test_invalid_manifest_and_missing_server_root(self):
        self.output.mkdir()
        (self.output/'.waqf-report.json').write_text('{broken')
        with self.assertRaisesRegex(ValueError,'Invalid report ownership'):
            build_report(self.data,self.output)
        with self.assertRaisesRegex(ValueError,'build first'):
            serve_report(self.root/'missing')

    def test_failed_render_keeps_previous_output(self):
        build_report(self.data,self.output)
        before=self.snapshot()
        with patch('src.waqf_vlm.report._render_report',side_effect=ValueError('render failed')):
            with self.assertRaisesRegex(ValueError,'render failed'):
                build_report(self.data,self.output)
        self.assertEqual(before,self.snapshot())


if __name__=='__main__': unittest.main()
