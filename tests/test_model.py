import numpy as np
import torch
from kilix_ml.model.config import Config
from kilix_ml.model.san import SAN
from kilix_ml.runtime.numpy_model import NumpyModel


def test_torch_numpy_and_incremental_parity():
    torch.manual_seed(12)
    torch.set_num_threads(1)
    c = Config(vocab_size=300, width=32, heads=4, kv_heads=2, layers=3, context=64)
    model = SAN(c).eval()
    numpy_model = NumpyModel(c, {k: v.detach().numpy() for k, v in model.state_dict().items()})
    ids = np.array([1, 45, 29, 260, 0, 20, 3])
    with torch.no_grad():
        expected = model(torch.tensor(ids)[None])[0].numpy()
    actual, _ = numpy_model.forward(ids)
    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)
    first, cache = numpy_model.forward(ids[:4])
    rest, cache = numpy_model.forward(ids[4:], cache)
    np.testing.assert_allclose(np.concatenate((first, rest)), expected, atol=1e-5, rtol=1e-5)


def test_assistant_mask_has_no_prompt_loss():
    torch.manual_seed(1)
    model = SAN(Config(vocab_size=32, width=16, heads=2, kv_heads=1, layers=1))
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    labels = torch.tensor([[-100, -100, -100, 4, 5]])
    logits = model(ids)
    expected = torch.nn.functional.cross_entropy(logits[:, 2:4].reshape(-1, 32), ids[:, 3:].reshape(-1))
    torch.testing.assert_close(model(ids, labels), expected)


def test_bfloat16_attention_dtypes_and_backward():
    model = SAN(Config(vocab_size=32, width=16, heads=2, kv_heads=1, layers=2))
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    with torch.autocast('cpu', dtype=torch.bfloat16):
        loss = model(ids, ids)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
