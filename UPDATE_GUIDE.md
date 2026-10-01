# Updating the GitHub repository to version 2

This folder contains:

* `2.5D-BrainTumor-Segmentation-v2/`: the complete new repository
* `update_repo.sh`: a script that puts it on GitHub for you

Nothing is pushed until you confirm with `y`. The current version of the repository stays available under the tag `v1.0-single-task`.

## 0. Unzip and install

Unzip everything into one folder, for example `Desktop\github_update`. Open that folder in File Explorer, right-click and choose **Open Git Bash here**. Install the requirements once, if needed (PyTorch is already on your training machine):

```bash
pip install -r 2.5D-BrainTumor-Segmentation-v2/requirements.txt
```

Run every command below in that Git Bash window.

## 1. Prepare your weights and split (before running the script)

The paper's Data availability statement says the trained weights are on GitHub. The code can only reproduce the test results with the exact subject split you used.

**Weights.** Make a folder `my_weights` next to this guide. Put the three final-run checkpoints in it, named:

```
decoder_free_seed2026.pth
decoder_free_seed2027.pth
decoder_free_seed2028.pth
```

If your files are plain state_dicts (saved with `torch.save(model.state_dict(), ...)`), wrap each one with the threshold that was chosen on the validation set for that run. Replace `T` with that threshold:

```bash
python 2.5D-BrainTumor-Segmentation-v2/scripts/package_weights.py --weights path/to/your_seed2026.pth \
    --threshold T --seed 2026 --out my_weights/decoder_free_seed2026.pth
```

**Check the weights against the code:**

```bash
python 2.5D-BrainTumor-Segmentation-v2/scripts/check_checkpoint.py my_weights/*.pth
```

* `strict load: OK`: the weights fit the code. Note the parameter and FLOP counts it prints.
* `strict load: FAILED`: your model differs from the code. Do not push. Send the output and your model code, and the code will be aligned to your model.

**Split.** Make a folder `my_splits` with the subject lists used for your final runs: `dev_train.csv`, `train.csv`, `val.csv` and `test.csv`. Each file has one column with the header `subject_id`. `2.5D-BrainTumor-Segmentation-v2/splits/README.md` describes the files.

## 2. Run the update (Git Bash on Windows)

```bash
WEIGHTS=my_weights SPLITS=my_splits bash update_repo.sh
```

If `python` is not found, put `PYTHON=py` in front of that command.

The script:

1. clones your repository into a new folder;
2. tags the current state as `v1.0-single-task`;
3. replaces the files;
4. runs the tests (about 30 s) and the weights check;
5. lists every change and asks `Commit and push these changes to GitHub? [y/N]`.

Type `y` to push. GitHub may open a browser window to sign in.

## 3. After pushing

* Open https://github.com/thshovon1/2.5D-BrainTumor-Segmentation. The README shows the architecture figure.
* Open the **Actions** tab. The `tests` workflow should turn green within a few minutes.
* Optional: create a release (**Releases → Draft a new release**, tag `v2.0`). If you first link the repository to Zenodo (zenodo.org → GitHub), the release gets a DOI that you can cite in the final paper.

## If something goes wrong

* If you answered anything other than `y`, nothing was pushed. Delete the folder `2.5D-BrainTumor-Segmentation-git` and run the script again.
* If the tests or the weights check fail, the script stops before committing.
