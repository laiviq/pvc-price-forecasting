"""Final delivery guards independent of confidential workbook contents."""
import json,tempfile,unittest,sys
from pathlib import Path
from unittest.mock import patch
from pvc_intelligence.final_pipeline import build,main

class FinalDeliveryGuards(unittest.TestCase):
 def test_nonfriday_cutoff_fails_before_data_access(self):
  with self.assertRaisesRegex(ValueError,'Friday'):
   build(Path('absent.xlsb'),Path('absent'),dict(cutoff='2026-07-02'))
 def test_predict_rejects_model_from_another_configuration(self):
  with tempfile.TemporaryDirectory() as d:
   d=Path(d);(d/'config.json').write_text(json.dumps(dict(cutoff='2026-07-03')));(d/'training_receipt.json').write_text(json.dumps(dict(config_sha256='wrong')))
   with patch.object(sys,'argv',['pipeline','predict','--config',str(d/'config.json'),'--model-dir',str(d)]):
    with self.assertRaisesRegex(ValueError,'Model/config mismatch'):main()
 def test_absent_target_fails_before_public_features(self):
  import pandas as pd
  raw=pd.DataFrame(dict(report_date=[46206],выборка=['обучающая'],scenario=['факт'],prod_version=['different'],k=[1.]))
  with patch('pvc_intelligence.final_pipeline.pd.read_excel',return_value=raw):
   with self.assertRaisesRegex(ValueError,'Target absent'):
    build(Path('not_read.xlsb'),Path('absent'),dict(cutoff='2026-07-03',target='target'))
