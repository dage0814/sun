import contextlib
import io
import unittest
from unittest.mock import patch

import train


class EntryTests(unittest.TestCase):
    def test_invalid_training_lengths(self):
        for option, value in [("--stage2-epochs", "0"), ("--batch-size", "0"),
                              ("--stage1-epochs", "-1"), ("--seed", "-1")]:
            with self.subTest(option=option), patch(
                "sys.argv", ["train.py", "--data", "bundle.pt", option, value]
            ), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                train.parse_args()

    def test_factor_ablation_rejected(self):
        for preset in ("no_gp", "no_ic"):
            with self.subTest(preset=preset), patch(
                "sys.argv", ["train.py", "--data", "bundle.pt", "--ablation", preset]
            ), self.assertRaisesRegex(ValueError, "precomputed features"):
                train.main()

    def test_default_arguments(self):
        with patch("sys.argv", ["train.py", "--data", "bundle.pt"]):
            args = train.parse_args()
        self.assertEqual(args.seed, 42)
        self.assertEqual(args.stage2_epochs, 50)


if __name__ == "__main__":
    unittest.main()
