"""Rule 5: emotions a language lacks stay out of the loss and the metrics."""
import numpy as np
import torch

from emoldgn.evaluate import macro_f1, track_a
from emoldgn.losses import masked_bce


def test_masked_emotion_cannot_change_the_loss():
    y = torch.zeros(2, 7)
    y[:, 2] = 1
    mask = torch.tensor([[1, 0, 1, 1, 1, 1, 1]] * 2, dtype=torch.float32)
    a = torch.zeros(2, 7, requires_grad=True)
    b = a.detach().clone()
    b[:, 1] = 50.0  # a wild prediction for disgust
    w = torch.ones(7)
    assert masked_bce(a, y, mask, w).item() == masked_bce(b, y, mask, w).item()
    masked_bce(a, y, mask, w).backward()
    assert a.grad[:, 1].abs().sum() == 0


def test_masked_emotion_is_not_averaged_into_macro_f1():
    y = np.array([[1, 0], [0, 1]])
    assert macro_f1(y, y, np.array([1, 1])) == 1.0
    y6 = np.zeros((4, 6), dtype=int)
    y6[:2, 0] = 1
    y6[2:, 2] = 1
    mask = np.array([[1, 0, 1, 1, 1, 1]] * 4, dtype=float)
    out = track_a(y6, y6.copy(), mask, ["eng"] * 4)
    # Five emotions, of which joy, sadness and surprise have no positives and score 0.
    assert abs(out["mean"] - 2 / 5) < 1e-9


def test_mask_broadcast_join_returns_one_row_per_language(spark):
    from pyspark.sql import functions as F

    masks = spark.createDataFrame([("eng", [1, 0, 1, 1, 1, 1]), ("hau", [1] * 6)], ["lang", "mask"])
    rows = spark.createDataFrame([("eng", 1), ("eng", 2), ("hau", 3)], ["lang", "x"])
    assert masks.groupBy("lang").count().filter("count > 1").count() == 0
    joined = rows.join(F.broadcast(masks), "lang")
    assert joined.count() == rows.count()
    assert joined.filter("lang = 'eng'").first()["mask"][1] == 0
