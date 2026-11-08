"""Losses stay finite on the awkward batches."""
import torch

from emoldgn.losses import graph_regulariser, intensity_loss, masked_bce, total_loss

CFG = {"n5": {"lambda_g": 0.1, "mu": 0.01}, "n6": {"lambda_i": 0.5, "alpha": 0.5, "beta": 0.5, "gamma": 1.0}}
FLAGS = {"n2_signed": True, "n5_reg_gate": True, "n6_intensity": True}


def outputs(requires_grad=True):
    return {"logits": torch.randn(4, 7, requires_grad=requires_grad), "int_logits": torch.randn(4, 7, 3, requires_grad=requires_grad),
            "AD": torch.rand(4, 7, 7, requires_grad=requires_grad)}


def test_all_zero_labels():
    b = {"labels": torch.zeros(4, 7), "node_mask": torch.ones(4, 7), "intensity": torch.zeros(4, 6, dtype=torch.long)}
    parts = total_loss(outputs(), b, torch.ones(7), FLAGS, CFG, torch.rand(7, 7))
    assert torch.isfinite(parts["loss"])
    parts["loss"].backward()


def test_rows_lacking_intensity():
    inten = torch.tensor([[-1] * 6, [1, 2, 0, 3, 0, 1], [-1] * 6, [0, 1, 2, 3, 1, 0]])
    out = outputs()
    loss = intensity_loss(out["int_logits"], inten)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(out["int_logits"].grad).all()


def test_batch_with_no_intensity_is_zero_not_nan():
    out = outputs()
    loss = intensity_loss(out["int_logits"], torch.full((4, 6), -1))
    assert loss.item() == 0
    loss.backward()
    assert (out["int_logits"].grad == 0).all()


def test_constant_gold_intensity_has_finite_gradients():
    logits = torch.randn(8, 7, 3, requires_grad=True)
    intensity_loss(logits, torch.ones(8, 6, dtype=torch.long)).backward()
    assert torch.isfinite(logits.grad).all()


def test_identical_predictions_have_finite_gradients():
    logits = torch.zeros(8, 7, 3, requires_grad=True)
    inten = torch.tensor([[0, 1, 2, 3, 0, 1]] * 4 + [[3, 2, 1, 0, 1, 0]] * 4)
    intensity_loss(logits, inten).backward()
    assert torch.isfinite(logits.grad).all()


def test_neutral_row_is_not_part_of_the_intensity_loss():
    a = torch.randn(4, 7, 3)
    b = a.clone()
    b[:, 6] = 99.0
    inten = torch.tensor([[1, 2, 0, 3, 0, 1]] * 4)
    assert intensity_loss(a, inten).item() == intensity_loss(b, inten).item()


def test_ordinal_loss_prefers_the_right_level():
    good = torch.tensor([[[5.0, 5.0, -5.0]] * 7] * 8)  # intensity 2 everywhere
    bad = -good
    inten = torch.full((8, 6), 2)
    inten[0] = 1  # some variance, so the Pearson term is defined
    assert intensity_loss(good, inten) < intensity_loss(bad, inten)


def test_stable_under_half_precision_inputs():
    logits = torch.tensor([[[-20.0, -20.0, -20.0]] * 7] * 4, dtype=torch.float16)
    inten = torch.tensor([[3, 3, 3, 3, 3, 3]] * 4)
    assert torch.isfinite(intensity_loss(logits, inten))


def test_masked_bce_and_regulariser_edges():
    assert torch.isfinite(masked_bce(torch.zeros(2, 7), torch.zeros(2, 7), torch.zeros(2, 7), torch.ones(7)))
    assert graph_regulariser(torch.zeros(3, 7, 7), torch.zeros(7, 7)).item() == 0


def test_b4_loss_ignores_the_neutral_output():
    b = {"labels": torch.zeros(4, 7), "node_mask": torch.ones(4, 7), "intensity": torch.zeros(4, 6, dtype=torch.long)}
    out = outputs()
    other = {k: v.detach().clone() for k, v in out.items()}
    other["logits"][:, 6] = 99.0
    a = total_loss(out, b, torch.ones(7), {}, CFG, None)["loss"].item()
    assert a == total_loss(other, b, torch.ones(7), {}, CFG, None)["loss"].item()
