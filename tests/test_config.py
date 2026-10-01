from bt25d.config import load_config


def test_inheritance_and_overrides(tmp_path):
    (tmp_path / "base.yaml").write_text("training:\n  rho: 0.25\n  epochs: 10\n")
    (tmp_path / "child.yaml").write_text("base: base.yaml\ntraining:\n  rho: 0.4\n")
    cfg = load_config(str(tmp_path / "child.yaml"), ["training.epochs=3", "loss.head_weights=[1.0, 0.0, 0.0]"])
    assert cfg["training"]["rho"] == 0.4 and cfg["training"]["epochs"] == 3
    assert cfg["loss"]["head_weights"] == [1.0, 0.0, 0.0]
    assert cfg["model"]["n_classes"] == 3


def test_repository_configs_resolve():
    import glob
    import os
    root = os.path.join(os.path.dirname(__file__), "..", "configs")
    files = glob.glob(os.path.join(root, "*.yaml")) + glob.glob(os.path.join(root, "ablations", "*.yaml"))
    assert len(files) >= 20
    for f in files:
        cfg = load_config(f)
        assert cfg["training"]["epochs"] == 10
    assert load_config(os.path.join(root, "final.yaml"))["data"]["train_split"] == "train"
    assert load_config(os.path.join(root, "ablations", "rho_000.yaml"))["training"]["rho"] == 0.0


def test_scientific_notation_is_numeric():
    cfg = load_config(None, ["training.lr=3e-3", "training.eta_min=1E-6"])
    assert cfg["training"]["lr"] == 3e-3 and cfg["training"]["eta_min"] == 1e-6
