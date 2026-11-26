"""Dependency-free boundary and resume-index tests for P1."""
import unittest
from e5.p1_schedule import explain_factor, sample_order, read_json, ROOT


class ScheduleTests(unittest.TestCase):
    def test_boundaries_and_resume(self):
        weights = [explain_factor(i) for i in range(750)]
        self.assertEqual(weights[0], 0.1)
        self.assertEqual(weights[249], 1.0)
        self.assertEqual(weights[250:], [1.0]*500)
        self.assertTrue(all(a <= b for a,b in zip(weights, weights[1:])))
        for done in (1, 10, 50, 249, 250, 500):
            self.assertEqual(weights[done:], [explain_factor(i) for i in range(done,750)])
        for bad in (-1, 2.5):
            with self.assertRaises(ValueError):
                explain_factor(bad)

    def test_v1_sample_budget(self):
        cfg = read_json(ROOT / "e3/config.json")
        order = sample_order(cfg)
        self.assertEqual(len(order), 750)
        self.assertTrue(all(len(group) == 8 for group in order))
        for epoch in range(3):
            values = [i for batch in order[epoch*250:(epoch+1)*250] for i in batch]
            self.assertEqual(sorted(values), list(range(2000)))


if __name__ == "__main__":
    unittest.main()
