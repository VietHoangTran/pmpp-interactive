"""Tests for labs/pmpp.py's starter history (work_state, work_detail, starters_missing).

  python3 -m unittest discover -s labs/tests

Builds throwaway labs in a temporary directory; touches nothing in the repository.
"""
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pmpp  # noqa: E402

LAB = "ch99/lab1_demo"


def write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write(text)


class StarterHistory(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.lab = os.path.join(self.root, LAB)
        for d in ("starter", "work"):
            write(os.path.join(self.lab, d, "kernel.cu"), "// v1\n")
            write(os.path.join(self.lab, d, "kernel.metal"), "// metal v1\n")
        self.v1 = pmpp.file_hashes(os.path.join(self.lab, "starter"))

    def tearDown(self):
        shutil.rmtree(self.root)

    def state(self, history):
        return pmpp.work_detail(LAB, self.root, history), pmpp.work_state(LAB, self.root, history)

    def test_identical_to_current_starter(self):
        self.assertEqual(self.state({}), ("current", "untouched"))

    def test_corrected_starter_keeps_old_copy_untouched(self):
        history = {LAB: [{"recorded": "test", "files": self.v1}]}
        write(os.path.join(self.lab, "starter", "kernel.cu"), "// v2, comment corrected\n")
        self.assertEqual(self.state(history), ("earlier", "untouched"))
        # Without the record the same copy reads as edited: the old behaviour.
        self.assertEqual(self.state({}), ("edited", "edited"))

    def test_edited_work_is_edited(self):
        history = {LAB: [{"recorded": "test", "files": self.v1}]}
        write(os.path.join(self.lab, "work", "kernel.cu"), "// v1 plus my kernel\n")
        self.assertEqual(self.state(history), ("edited", "edited"))

    def test_file_present_or_absent_counts(self):
        history = {LAB: [{"recorded": "test", "files": self.v1}]}
        write(os.path.join(self.lab, "work", "launch.mm"), "// extra\n")
        self.assertEqual(self.state(history), ("edited", "edited"))
        os.remove(os.path.join(self.lab, "work", "launch.mm"))
        os.remove(os.path.join(self.lab, "work", "kernel.metal"))
        self.assertEqual(self.state(history), ("edited", "edited"))

    def test_another_labs_history_does_not_count(self):
        history = {"ch98/lab1_other": [{"recorded": "test", "files": self.v1}]}
        write(os.path.join(self.lab, "starter", "kernel.cu"), "// v2\n")
        self.assertEqual(self.state(history), ("edited", "edited"))

    def test_missing_until_recorded(self):
        self.assertEqual(pmpp.starters_missing({}, self.root), [LAB])
        history = {LAB: [{"recorded": "test", "files": self.v1}]}
        self.assertEqual(pmpp.starters_missing(history, self.root), [])
        write(os.path.join(self.lab, "starter", "kernel.cu"), "// v2\n")
        self.assertEqual(pmpp.starters_missing(history, self.root), [LAB])


class RepositoryHistory(unittest.TestCase):
    def test_every_current_starter_is_recorded(self):
        missing = pmpp.starters_missing()
        self.assertEqual(missing, [], "run `python3 labs/pmpp.py starters --record`")

    def test_history_is_well_formed(self):
        h = pmpp.load_starter_history()
        self.assertTrue(h)
        for lab, versions in h.items():
            self.assertTrue(versions, lab)
            for v in versions:
                self.assertEqual(sorted(v["files"]), sorted(pmpp.STUDENT_FILES), lab)
                self.assertIn("recorded", v, lab)


if __name__ == "__main__":
    unittest.main()
