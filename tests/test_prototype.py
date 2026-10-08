from __future__ import annotations

import unittest

import numpy as np

from prototype_core import (
    HORIZONS,
    MODELS,
    SEEDS,
    build_segment_frame,
    load_dataset,
    load_metadata,
    load_summary_metrics,
    predict,
    traffic_class,
)


class PrototypeTests(unittest.TestCase):
    def test_assets_are_consistent(self) -> None:
        nodes, edges = load_metadata()
        self.assertEqual(len(nodes), 88)
        self.assertEqual(len(edges), 86)
        for horizon in HORIZONS:
            data = load_dataset(horizon)
            self.assertEqual(data["x_test"].shape[1:], (12, 88, 8))
            self.assertEqual(data["y_test_raw"].shape[1], 88)
            self.assertTrue(np.isfinite(data["x_test"]).all())
            self.assertTrue(np.isfinite(data["adjacency"]).all())

    def test_summary_contains_every_model_and_horizon(self) -> None:
        summary = load_summary_metrics()
        expected = {(horizon, model) for horizon in HORIZONS for model in MODELS}
        actual = set(zip(summary["horizon_minutes"], summary["model"]))
        self.assertEqual(actual, expected)

    def test_all_checkpoint_families_run_inference(self) -> None:
        data = load_dataset(15)
        x = data["x_test"][0]
        for model_name in MODELS:
            prediction, attention, _ = predict(
                model_name,
                15,
                x,
                ensemble=False,
                seed=SEEDS[0],
                prefer_cuda=False,
            )
            self.assertEqual(prediction.shape, (88,))
            self.assertTrue(np.isfinite(prediction).all())
            if model_name == "a3tgcn":
                self.assertEqual(attention.shape, (88, 12))
                np.testing.assert_allclose(attention.sum(axis=1), 1.0, atol=1e-5)
            else:
                self.assertIsNone(attention)

    def test_segment_classification_is_complete(self) -> None:
        current = np.full(88, 15.0, dtype=np.float32)
        forecast = np.full(88, 16.0, dtype=np.float32)
        frame = build_segment_frame(current, forecast)
        self.assertEqual(len(frame), 88)
        self.assertFalse(frame["traffic_level"].isna().any())

    def test_speed_threshold_classification_uses_kph(self) -> None:
        self.assertEqual(traffic_class(9.9), "Heavy")
        self.assertEqual(traffic_class(10.0), "Slow")
        self.assertEqual(traffic_class(19.9), "Slow")
        self.assertEqual(traffic_class(20.0), "Moderate")
        self.assertEqual(traffic_class(29.9), "Moderate")
        self.assertEqual(traffic_class(30.0), "Free flowing")


if __name__ == "__main__":
    unittest.main()
