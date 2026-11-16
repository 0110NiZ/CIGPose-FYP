"""Regression checks for explicit legacy settings and provenance rejection."""
import sys
from pathlib import Path
import unittest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from precision_policy import configure_legacy_precision,assert_legacy_precision,legacy_profile,profile_fingerprint,require_profile

class PrecisionPolicyTests(unittest.TestCase):
    def tearDown(self):
        configure_legacy_precision()

    def test_resets_strict_diagnostic_flags(self):
        torch.backends.cudnn.allow_tf32=False
        torch.backends.cuda.matmul.allow_tf32=True
        torch.backends.cudnn.benchmark=True
        torch.use_deterministic_algorithms(False)
        configure_legacy_precision()
        assert_legacy_precision()
        self.assertTrue(torch.backends.cudnn.allow_tf32)
        self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
        self.assertTrue(torch.are_deterministic_algorithms_enabled())

    def test_rejects_accidental_backend_change(self):
        configure_legacy_precision();torch.backends.cudnn.allow_tf32=False
        with self.assertRaises(RuntimeError):assert_legacy_precision()

    def test_rejects_cpu_autocast(self):
        configure_legacy_precision()
        with torch.autocast('cpu'):
            with self.assertRaises(RuntimeError):assert_legacy_precision()

    def test_profile_cannot_be_mutated_through_return(self):
        p=legacy_profile();p['cudnn_allow_tf32']=False
        self.assertTrue(legacy_profile()['cudnn_allow_tf32'])
        with self.assertRaises(ValueError):require_profile(p,profile_fingerprint())

    def test_rejects_merged_candidate_batches(self):
        import numpy as np
        from build_shared_cache import candidate_from_cache
        configure_legacy_precision()
        with self.assertRaisesRegex(ValueError,'source batches'):
            candidate_from_cache(None,{'features':np.zeros((33,17,256),np.float32)},None,None,'cpu')

    def test_archived_generator_provenance_and_tamper_rejection(self):
        from test_module_a import ModuleATest
        from a_contract import GainLabels
        from shared_cache import hash_file
        fixture=ModuleATest();fixture.setUp()
        try:
            source=fixture.labeldir/'source';source.mkdir()
            script=source/'generator.py';script.write_text('# archived fixture\n')
            fixture.metadata.update(generator_git_commit='',generator_source_snapshot='source/generator.py',
                generator_source_sha256=hash_file(script),precision_profile=legacy_profile(),precision_profile_sha256=profile_fingerprint())
            fixture.write_labels()
            self.assertEqual(len(GainLabels(fixture.labeldir,fixture.cache).ids),3)
            script.write_text('# changed source\n')
            with self.assertRaises(ValueError):GainLabels(fixture.labeldir,fixture.cache)
        finally:fixture.tearDown()

if __name__=='__main__':unittest.main()
