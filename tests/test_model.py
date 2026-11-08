"""Forward passes for every experiment configuration."""
import pytest
import torch

from emoldgn.ldgn_baseline import build_model
from emoldgn.model import EmoLDGNPlus, norm_adj

from .conftest import FakeEncoder, all_flags, fake_bank

FINAL = {"head": "ldgn", "n1_anchors": True, "n2_signed": True, "n4_hierarchical": True, "n5_reg_gate": True, "n6_intensity": True}


def batch():
    ids = torch.randint(0, 64, (4, 10))
    mask = torch.ones_like(ids)
    mask[0, 7:] = 0
    return ids, mask, torch.tensor([0, 1, 2, 1])


def test_output_shapes_final():
    model = EmoLDGNPlus(FINAL, fake_bank(), FakeEncoder(), hid=512, label_init=torch.randn(7, 768))
    out = model(*batch())
    assert out["logits"].shape == (4, 7)
    assert out["int_logits"].shape == (4, 7, 3)
    assert out["AD"].shape == (4, 7, 7)
    assert out["gate"].shape == (4, 7, 512)
    assert out["att"].shape == (4, 7, 10)


def test_graph_head_size():
    model = EmoLDGNPlus(FINAL, fake_bank(), FakeEncoder(), hid=512, label_init=torch.randn(7, 768))
    assert 3_000_000 < model.graph_head_parameters() < 6_000_000


@pytest.mark.parametrize("flags", all_flags(), ids=lambda f: "-".join(k for k, v in f.items() if v is True) or f["head"])
def test_every_configuration_runs_and_backpropagates(flags):
    model = build_model(flags, fake_bank(), FakeEncoder(), 64, torch.randn(7, 768))
    out = model(*batch())
    assert out["logits"].shape == (4, 7)
    out["logits"].sum().backward()
    assert torch.isfinite(out["logits"]).all()
    assert model.enc.emb.weight.grad is not None


def test_padding_is_ignored_by_attention():
    model = EmoLDGNPlus(FINAL, fake_bank(), FakeEncoder(), hid=32, label_init=torch.randn(7, 768)).eval()
    ids, mask, lang = batch()
    att = model(ids, mask, lang)["att"]
    assert att[0, :, 7:].abs().max() < 1e-6
    assert torch.allclose(att.sum(-1), torch.ones(4, 7), atol=1e-5)


def test_masked_nodes_stay_isolated():
    model = EmoLDGNPlus(FINAL, fake_bank(), FakeEncoder(), hid=32, label_init=torch.randn(7, 768))
    with torch.no_grad():
        model.delta.fill_(1.0)
    out = model(*batch())
    assert out["AD"][0, 1].abs().sum() == 0 and out["AD"][0, :, 1].abs().sum() == 0


def test_norm_adj_handles_empty_rows():
    a = norm_adj(torch.zeros(2, 7, 7))
    assert torch.allclose(a, torch.eye(7).expand(2, 7, 7))


def test_b4_has_no_neutral_edges():
    off = {"head": "ldgn"}
    model = EmoLDGNPlus(off, fake_bank(), FakeEncoder(), hid=32)
    assert model.A_pos[:, -1].abs().sum() == 0 and model.A_pos[:, :, -1].abs().sum() == 0
    assert isinstance(model.Wq, torch.nn.Identity)


def test_save_and_load_roundtrip(tmp_path):
    from emoldgn.model import build_encoder, load_model, save_model

    enc = build_encoder("xlm-roberta-base", tiny=True)
    model = EmoLDGNPlus(FINAL, fake_bank(), enc, hid=16, label_init=torch.randn(7, 32)).eval()
    path = tmp_path / "m.pt"
    save_model(model, path, {"encoder": "xlm-roberta-base", "hid": 16, "tiny_encoder": True})
    loaded, _ = load_model(path)
    ids, mask, lang = batch()
    with torch.no_grad():
        assert torch.allclose(model(ids, mask, lang)["logits"], loaded(ids, mask, lang)["logits"])
