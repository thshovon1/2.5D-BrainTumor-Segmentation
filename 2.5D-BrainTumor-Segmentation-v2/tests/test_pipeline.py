"""End-to-end smoke test on synthetic NIfTI data (CPU, about half a minute):
preprocess -> split -> train -> evaluate -> LCC analysis -> seed summary ->
checkpoint check -> prediction to NIfTI -> benchmark -> experiment runner."""

import json
import os
import subprocess
import sys

import nibabel as nib
import numpy as np
import pytest

from synthetic import make_dataset

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def run(*args):
    r = subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, f"{args[0]} failed:\n{r.stdout}\n{r.stderr}"
    return r.stdout


@pytest.fixture(scope="module")
def work(tmp_path_factory):
    w = tmp_path_factory.mktemp("e2e")
    make_dataset(str(w / "raw"), n=12)
    return w


def test_full_pipeline(work):
    w = str(work)
    out = run("scripts/preprocess.py", "--raw-dir", f"{w}/raw", "--out-dir", f"{w}/data", "--size", "32", "--workers", "2")
    assert "12 subjects" in out
    img = np.load(f"{w}/data/BraTS2021_00001/image.npy")
    assert img.shape == (24, 4, 32, 32) and img.dtype == np.float16
    meta = json.load(open(f"{w}/data/BraTS2021_00001/meta.json"))
    assert meta["model_spacing_dhw"] == [1.0, 1.5, 1.5]

    run("scripts/make_split.py", "--data-dir", f"{w}/data", "--splits-dir", f"{w}/splits", "--n-val", "3", "--n-test", "3")
    ids = {s: open(f"{w}/splits/{s}.csv").read().split()[1:] for s in ("train", "val", "test", "dev_train")}
    assert len(ids["train"]) == 6 and len(ids["val"]) == 3 and len(ids["test"]) == 3
    assert not (set(ids["train"]) & set(ids["test"])) and set(ids["test"]) <= set(ids["dev_train"])

    common = ["--device", "cpu", "--set", f"data.data_dir={w}/data", f"data.splits_dir={w}/splits",
              "training.epochs=2", "training.batch_size=8", "training.num_workers=0", "training.lr=3e-3"]
    run("scripts/train.py", "--config", "configs/final.yaml", "--out-dir", f"{w}/runs/s1", "--seed", "1", *common)
    run("scripts/train.py", "--config", "configs/final.yaml", "--out-dir", f"{w}/runs/s2", "--seed", "2", *common)
    s = json.load(open(f"{w}/runs/s1/train_summary.json"))
    assert s["n_train_subjects"] == 6 and s["parameters"] == 683_657 and len(s["threshold"]) == 3
    hist = open(f"{w}/runs/s1/history.csv").read().splitlines()
    assert len(hist) == 3

    ev = ["--device", "cpu", "--data-dir", f"{w}/data", "--splits-dir", f"{w}/splits"]
    run("scripts/evaluate.py", "--checkpoint", f"{w}/runs/s1/best.pth", "--split", "test", "--out-dir", f"{w}/ev/s1", *ev)
    run("scripts/evaluate.py", "--checkpoint", f"{w}/runs/s2/best.pth", "--split", "test", "--out-dir", f"{w}/ev/s2", *ev)
    run("scripts/evaluate.py", "--checkpoint", f"{w}/runs/s1/best.pth", "--split", "val", "--grid", "native",
        "--save-preds", "--out-dir", f"{w}/ev/val", *ev)
    summ = json.load(open(f"{w}/ev/s1/summary.json"))
    assert summ["n_subjects"] == 3 and set(summ["regions"]) == {"WT", "TC", "ET"}
    assert 0 <= summ["regions"]["WT"]["pooled"]["dice"] <= 1
    assert len(os.listdir(f"{w}/ev/val/preds")) == 3

    out = run("scripts/lcc_analysis.py", "--pred-dir", f"{w}/ev/val/preds", "--data-dir", f"{w}/data",
              "--out-dir", f"{w}/ev/lcc")
    lcc = json.load(open(f"{w}/ev/lcc/lcc_summary.json"))
    assert lcc["n"] == 3 and lcc["uni_focal"]["n"] + lcc["multi_focal"]["n"] == 3

    run("scripts/summarize_seeds.py", f"{w}/ev/s1", f"{w}/ev/s2", "--out", f"{w}/ev/table5")
    t5 = json.load(open(f"{w}/ev/table5.json"))
    macro = [r for r in t5["rows"] if r["metric"] == "Macro-Dice"][0]
    assert len(macro["values"]) == 2 and abs(macro["mean"] - np.mean(macro["values"])) < 1e-12

    assert "1/1 checkpoint(s) match" in run("scripts/check_checkpoint.py", f"{w}/runs/s1/best.pth")

    run("scripts/predict.py", "--checkpoint", f"{w}/runs/s1/best.pth", "--inputs", f"{w}/raw/BraTS2021_00003",
        "--out-dir", f"{w}/pred", "--size", "32", "--device", "cpu")
    pred = nib.load(f"{w}/pred/BraTS2021_00003_pred.nii.gz")
    assert pred.shape == (48, 48, 24) and set(np.unique(np.asarray(pred.dataobj))) <= {0, 1, 2, 4}

    run("scripts/benchmark.py", "--size", "32", "--depth", "4", "--repeats", "1", "--device", "cpu",
        "--out", f"{w}/bench.json")
    b = json.load(open(f"{w}/bench.json"))
    assert b["models"]["decoder_free"]["parameters"] == 683_657

    out = run("scripts/run_experiments.py", "--groups", "loss", "rho", "final", "--dry-run", "--runs-dir", f"{w}/r")
    assert out.count("train.py") == 7 + 4 + 3   # the shared proposed run is scheduled once
