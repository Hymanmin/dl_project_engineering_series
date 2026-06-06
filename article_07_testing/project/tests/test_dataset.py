from __future__ import annotations

import unittest

from core.dataset.cnn_cls.cnn_cls_dataset import CnnClsDataset


class DatasetTest(unittest.TestCase):
    def test_batch_dict_contract(self) -> None:
        dataset = CnnClsDataset({"input_dim": 16, "num_classes": 2}, split="train")
        sample = dataset[0]
        self.assertIn("input", sample)
        self.assertIn("target", sample)
        self.assertEqual(tuple(sample["input"].shape), (16,))


if __name__ == "__main__":
    unittest.main()
