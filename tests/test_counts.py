import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import bootstrap


class BootstrapReviewTests(unittest.TestCase):
    def test_count_width_is_checked(self):
        self.assertEqual(bootstrap.add_counts((1, 2, 3, 4), (5, 6, 7, 8)), (6, 8, 10, 12))
        for left, right in [((1, 2, 3), (1, 2, 3, 4)), ((1, 2, 3, 4, 5), (1, 2, 3, 4))]:
            with self.subTest(left=left), self.assertRaises(ValueError):
                bootstrap.add_counts(left, right)

    def test_holm_empty_and_ties(self):
        self.assertEqual(bootstrap.holm_adjust([]), [])
        self.assertEqual(bootstrap.holm_adjust([0.02, 0.02, 0.1]), [0.06, 0.06, 0.1])
        self.assertEqual(bootstrap.holm_adjust([0.1, 0.02, 0.02]), [0.1, 0.06, 0.06])

    def test_holm_rejects_invalid_probabilities(self):
        for value in [-0.1, 1.1, float('nan'), float('inf')]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                bootstrap.holm_adjust([value])

    def test_paths_reject_empty_or_directory(self):
        for value in [[], '', None, {}]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                bootstrap.resolve_prediction_paths(value)
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "not a file"):
                bootstrap.resolve_prediction_paths(temporary)

    def test_file_columns_are_checked(self):
        with tempfile.TemporaryDirectory() as temporary:
            file = Path(temporary) / 'incomplete.csv'
            file.write_text('seed,date\n42,2020-01-01\n', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, "missing columns"):
                bootstrap.load_predictions(file)

    def test_manifest_paths_checked_before_bootstrap(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'predictions.csv').write_text('', encoding='utf-8')
            rows = [{'name': str(i), 'full': 'predictions.csv', 'baseline': 'predictions.csv'}
                    for i in range(6)]
            rows[-1]['baseline'] = 'missing.csv'
            manifest = root / 'manifest.json'
            manifest.write_text(json.dumps(rows), encoding='utf-8')
            argv = ['bootstrap.py', '--manifest', str(manifest), '--output', str(root / 'out.csv')]
            with patch('sys.argv', argv), patch.object(bootstrap, 'bootstrap_comparison') as comparison:
                with self.assertRaises(FileNotFoundError):
                    bootstrap.main()
                comparison.assert_not_called()

    def test_manifest_names_are_unique(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'predictions.csv').write_text('', encoding='utf-8')
            rows = [{'name': 'same', 'full': 'predictions.csv', 'baseline': 'predictions.csv'}
                    for _ in range(6)]
            with self.assertRaisesRegex(ValueError, "unique"):
                bootstrap.resolve_manifest(rows, root)


if __name__ == '__main__':
    unittest.main()
