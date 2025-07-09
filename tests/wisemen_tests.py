import numpy as np
import torch
import pytest
from testing import CorrectedPINN, generate_improved_fluid_points, navier_stokes_residuals_corrected

def get_trained_model():
    # Minimal quick training for test purposes
    model = CorrectedPINN(layers=[2, 16, 16, 3], use_residual=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    fluid_points = generate_improved_fluid_points(500)
    for _ in range(10):
        optimizer.zero_grad()
        pts = torch.tensor(fluid_points, dtype=torch.float32)
        continuity, mom_u, mom_v = navier_stokes_residuals_corrected(model, pts)
        loss = torch.mean(continuity**2) + torch.mean(mom_u**2) + torch.mean(mom_v**2)
        loss.backward()
        optimizer.step()
    return model, fluid_points

def test_model_forward():
    model, _ = get_trained_model()
    x = torch.randn(10, 2)
    out = model(x)
    assert out.shape == (10, 3)
    assert torch.isfinite(out).all()

def test_physics_residuals():
    model, fluid_points = get_trained_model()
    pts = torch.tensor(fluid_points, dtype=torch.float32)
    continuity, mom_u, mom_v = navier_stokes_residuals_corrected(model, pts)
    assert torch.mean(torch.abs(continuity)) < 1.0
    assert torch.mean(torch.abs(mom_u)) < 1.0
    assert torch.mean(torch.abs(mom_v)) < 1.0

def test_gradient_flow():
    model, fluid_points = get_trained_model()
    pts = torch.tensor(fluid_points[:5], dtype=torch.float32, requires_grad=True)
    out = model(pts)
    grad = torch.autograd.grad(out.sum(), model.parameters(), retain_graph=True, allow_unused=True)
    grad_norms = [g.abs().sum().item() for g in grad if g is not None]
    assert all([np.isfinite(g) and g > 0 for g in grad_norms])

def test_serialization():
    model, _ = get_trained_model()
    torch.save(model.state_dict(), 'tmp_model.pth')
    model2 = CorrectedPINN(layers=[2, 16, 16, 3], use_residual=True)
    model2.load_state_dict(torch.load('tmp_model.pth'))
    for p1, p2 in zip(model.parameters(), model2.parameters()):
        assert torch.allclose(p1, p2)

def test_reproducibility():
    torch.manual_seed(42)
    np.random.seed(42)
    model1, _ = get_trained_model()
    torch.manual_seed(42)
    np.random.seed(42)
    model2, _ = get_trained_model()
    params1 = [p.detach().cpu().numpy() for p in model1.parameters()]
    params2 = [p.detach().cpu().numpy() for p in model2.parameters()]
    for a, b in zip(params1, params2):
        assert np.allclose(a, b)

def test_device_consistency():
    if torch.cuda.is_available():
        model, fluid_points = get_trained_model()
        model_cuda = CorrectedPINN(layers=[2, 16, 16, 3], use_residual=True).cuda()
        model_cuda.load_state_dict(model.state_dict())
        x = torch.tensor(fluid_points[:10], dtype=torch.float32)
        out_cpu = model(x)
        out_gpu = model_cuda(x.cuda()).cpu()
        assert np.allclose(out_cpu.detach().numpy(), out_gpu.detach().numpy(), atol=1e-5)

def test_output_range():
    model, _ = get_trained_model()
    x = torch.randn(50, 2)
    out = model(x)
    # Expect velocities and pressure to be finite, not too large
    assert (out.abs() < 1000).all()

if __name__ == "__main__":
    tests = [
        ("Model Forward", test_model_forward),
        ("Physics Residuals", test_physics_residuals),
        ("Gradient Flow", test_gradient_flow),
        ("Serialization", test_serialization),
        ("Reproducibility", test_reproducibility),
        ("Device Consistency", test_device_consistency),
        ("Output Range", test_output_range),
    ]
    passed = 0
    failed = 0
    failed_tests = []
    for name, fn in tests:
        try:
            fn()
            print(f"[PASS] {name}")
            passed += 1
        except Exception as e:
            print(f"[FAIL] {name}: {e}")
            failed += 1
            failed_tests.append(name)
    total = len(tests)
    print("\n--- Wisemen Test Summary ---")
    print(f"Passed: {passed}/{total}")
    print(f"Failed: {failed}/{total}")
    score = 100.0 * passed / total
    print(f"Score: {score:.1f}%")
    if failed_tests:
        print("Failed tests:", ", ".join(failed_tests))
    else:
        print("All wisemen tests passed!")
