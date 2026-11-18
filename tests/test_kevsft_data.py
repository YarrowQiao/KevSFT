import json
import shutil
import tempfile
import unittest
from pathlib import Path

from kevsft.data import INSTRUCTIONS, load_dataset, load_labels, make_request, render_state

FIXTURE = Path(__file__).resolve().parents[1] / "examples" / "kevsft"


class DatasetTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.data = Path(temp.name) / "dataset"
        shutil.copytree(FIXTURE, self.data)

    def append(self, split, row):
        with (self.data / f"{split}.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row) + "\n")

    def test_options_and_requests_match_the_choice_api(self):
        options, splits = load_dataset(self.data)
        labels = load_labels(self.data)
        request = make_request(labels, "What's the deadline?", ["I borrowed an atlas."])
        question = request["questions"]["route"]
        self.assertEqual(question["instructions"], INSTRUCTIONS)
        self.assertEqual(options, [f"{key}: {text}" for key, text in question["criteria"].items()])
        self.assertEqual(list(labels)[-1], "other")
        self.assertEqual(set(splits), {"train", "validation", "test"})
        self.assertEqual(json.loads(request["state"])["current_user_request"], "What's the deadline?")
        self.assertEqual(render_state("What's the deadline?"), "What's the deadline?")

    def test_holdout_may_omit_routes_seen_in_training(self):
        test_path = self.data / "test.jsonl"
        first = test_path.read_text(encoding="utf-8").splitlines()[0]
        test_path.write_text(first + "\n", encoding="utf-8")
        options, splits = load_dataset(self.data)
        self.assertEqual(len(options), 3)
        self.assertEqual([row.label for row in splits["test"]], [0])

    def test_two_views_of_a_source_must_stay_together(self):
        self.append("test", {"group": "borrow-a", "label": "events", "query": "A new event next winter?"})
        with self.assertRaisesRegex(ValueError, "source group"):
            load_dataset(self.data)

    def test_copied_query_under_another_group_is_detected(self):
        self.append("validation", {"group": "distinct", "label": "loans",
                                   "query": "  CAN  I borrow an audiobook on my card? "})
        with self.assertRaisesRegex(ValueError, "equivalent input"):
            load_dataset(self.data)

    def test_duplicate_key_in_taxonomy_does_not_silently_get_replaced(self):
        (self.data / "labels.json").write_text('{"a":"first","a":"second","other":"rest"}')
        with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
            load_labels(self.data)

    def test_unknown_label_reports_file_and_row_without_query(self):
        self.append("train", {"group": "g", "label": "absent", "query": "This should not appear in the error"})
        with self.assertRaisesRegex(ValueError, r"train.jsonl:[0-9]+") as error:
            load_dataset(self.data)
        self.assertNotIn("This should not appear", str(error.exception))


if __name__ == "__main__":
    unittest.main()
