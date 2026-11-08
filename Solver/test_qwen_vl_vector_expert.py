from __future__ import annotations

import torch

from Solver.agent.qwen_vl_vector_expert import GeneratedLoRALinear


def test_generated_lora_backpropagates_to_expert_vector() -> None:
    base = torch.nn.Linear(5, 7, bias=False)
    for parameter in base.parameters():
        parameter.requires_grad_(False)
    layer = GeneratedLoRALinear(base, rank=2, latent_dim=3, seed=17)
    z = torch.nn.Parameter(torch.zeros(3))

    layer.activate(z)
    inputs = torch.randn(4, 5)
    output = layer(inputs)
    torch.testing.assert_close(output, layer.base(inputs), rtol=0, atol=0)
    output.square().mean().backward()

    assert z.grad is not None
    assert torch.isfinite(z.grad).all()
    assert z.grad.norm() > 0
    assert all(parameter.grad is None for parameter in base.parameters())


def test_generated_lora_fixed_mapping_roundtrips() -> None:
    first = GeneratedLoRALinear(torch.nn.Linear(5, 7, bias=False), rank=2, latent_dim=3, seed=23)
    second = GeneratedLoRALinear(torch.nn.Linear(5, 7, bias=False), rank=2, latent_dim=3, seed=99)
    second.load_state_dict(first.state_dict())
    z = torch.randn(3)

    a1, b1 = first.parameters_from_latent(z)
    a2, b2 = second.parameters_from_latent(z)

    torch.testing.assert_close(a1, a2)
    torch.testing.assert_close(b1, b2)
