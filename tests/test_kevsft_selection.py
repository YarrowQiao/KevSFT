"""Offline regression tests for held-out route scoring and checkpoint selection."""
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from kevsft import train

FIXTURE = Path(__file__).resolve().parents[1] / "examples" / "kevsft"


class RoutingMetricsTest(unittest.TestCase):
    def test_fallback_rate_counts_false_fallbacks_among_known_requests(self):
        metrics = train.score_matrix(
            [[2, 0, 1], [0, 0, 0], [2, 1, 3]], ["loans", "events", "other"],
        )
        self.assertEqual(metrics["correct"], 5)
        self.assertEqual(metrics["accuracy"], 5 / 9)
        self.assertEqual(metrics["support"]["events"], 0)
        self.assertIsNone(metrics["recall"]["events"])
        self.assertEqual(metrics["fallback_recall"], 3 / 6)
        self.assertEqual(metrics["fallback_false_positive_rate"], 1 / 3)

    def test_homogeneous_holdouts_return_null_for_undefined_rates(self):
        keys = ["loans", "other"]
        fallback_only = train.score_matrix([[0, 0], [1, 2]], keys)
        self.assertIsNone(fallback_only["fallback_false_positive_rate"])
        known_only = train.score_matrix([[1, 0], [0, 0]], keys)
        self.assertIsNone(known_only["fallback_recall"])


class HeldoutSelectionTest(unittest.TestCase):
    def test_test_set_scores_never_choose_the_winner(self):
        # Epoch 2 beats 1 on the tie-break, 3 ties 2, and 4 loses on accuracy.
        validation = {1: (.8, .4), 2: (.8, .3), 3: (.8, .3), 4: (.7, .1)}
        validation_groups = {e.group for e in train.load_dataset(FIXTURE)[1]["validation"]}
        observations = []
        model = Mock(epoch=0)
        model.named_parameters.return_value = [("head.weight", Mock(requires_grad=True))]
        frozen_meta = SimpleNamespace(
            lora=16, weights_dtype="fp32", special_embeddings=False, base="synthetic-base",
            base_revision=None, head_dim=256, option_isolation=False,
        )

        class Checkpoint:
            full = False
            meta = frozen_meta

            def __init__(self, path):
                self.epoch = 0 if path == "synthetic-base" else int((Path(path) / "epoch").read_text())

            def warm_start(self, model, meta):
                pass

            def load(self, device, options):
                return None, SimpleNamespace(epoch=self.epoch)

            def weights_sha256(self):
                return {}

        def advance_epoch(*_):
            model.epoch += 1
            return 1.0

        def save(directory, candidate, *_):
            directory.mkdir(exist_ok=True)
            (directory / "epoch").write_text(str(candidate.epoch))

        def evaluate(candidate, tokenizer, examples, *args):
            if {e.group for e in examples} == validation_groups:
                observations.append(("validation", candidate.epoch))
                accuracy, ce = validation[candidate.epoch]
            else:
                observations.append(("test", candidate.epoch))
                accuracy, ce = (1., .1) if candidate.epoch == 0 else (.5, .5)
            return {"accuracy": accuracy, "raw_cross_entropy": ce}

        modules = {
            "torch": Mock(),
            "kev.checkpoint": SimpleNamespace(Checkpoint=Checkpoint, LoadOptions=Mock()),
            "kev.model": SimpleNamespace(DecisionModel=Mock(return_value=model), load_tokenizer=Mock()),
        }
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "run"
            args = SimpleNamespace(data=FIXTURE, output=output, model="synthetic-base",
                                   device="cpu", seed=1, lr=.0001, epochs=4,
                                   batch_size=2, max_row_tokens=128)
            with (patch.dict("sys.modules", modules), patch.object(train, "train_epoch", advance_epoch),
                  patch.object(train, "save_candidate", save), patch.object(train, "evaluate", evaluate),
                  redirect_stdout(io.StringIO())):
                train.run(args)
            self.assertEqual(observations, [("validation", 1), ("validation", 2),
                                            ("validation", 3), ("validation", 4),
                                            ("test", 0), ("test", 2)])
            report = json.loads((output / "report.json").read_text())
            self.assertEqual(report["selected_epoch"], 2)
            self.assertEqual(report["test_accuracy_delta"], -.5)
            self.assertEqual((output / "best" / "epoch").read_text(), "2")
