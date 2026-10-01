#!/usr/bin/env bash
# Updates https://github.com/thshovon1/2.5D-BrainTumor-Segmentation to version 2.
#
# Run it in Git Bash (Windows) or a terminal (macOS/Linux) from the folder that
# contains this script and the extracted folder 2.5D-BrainTumor-Segmentation-v2:
#
#     WEIGHTS=my_weights SPLITS=my_splits bash update_repo.sh
#
#   WEIGHTS  folder with the release checkpoints (decoder_free_seed2026.pth, ...)
#   SPLITS   folder with your subject lists (dev_train.csv, train.csv, val.csv, test.csv)
#   PYTHON   python command if it is not "python" (for example PYTHON=py)
#
# The script clones the repository into a new folder, tags the current state as
# v1.0-single-task, replaces the files, runs the tests and the checkpoint check,
# shows the changes and asks before it commits and pushes. Nothing is pushed
# unless you answer y.
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/thshovon1/2.5D-BrainTumor-Segmentation.git}"
NEW_DIR="${NEW_DIR:-2.5D-BrainTumor-Segmentation-v2}"
WORK="2.5D-BrainTumor-Segmentation-git"
PY="${PYTHON:-python}"
TAG="v1.0-single-task"

[ -d "$NEW_DIR/bt25d" ] || { echo "Cannot find the new files in '$NEW_DIR'. Run this script next to that folder."; exit 1; }
[ -e "$WORK" ] && { echo "Folder '$WORK' already exists. Delete or rename it, then run again."; exit 1; }
NEW_DIR="$(cd "$NEW_DIR" && pwd)"
[ -n "${WEIGHTS:-}" ] && WEIGHTS="$(cd "$WEIGHTS" && pwd)"
[ -n "${SPLITS:-}" ] && SPLITS="$(cd "$SPLITS" && pwd)"

echo "== 1/6 Cloning $REPO_URL"
git clone -q "$REPO_URL" "$WORK"
cd "$WORK"

echo "== 2/6 Tagging the current version as $TAG"
if git rev-parse -q --verify "refs/tags/$TAG" >/dev/null; then
  echo "tag $TAG already exists - kept"
else
  git tag -a "$TAG" -m "Single-task WT prototype (version before the revised paper)"
fi

echo "== 3/6 Replacing the files"
git rm -r -q .
cp -R "$NEW_DIR"/. .
if [ -n "${WEIGHTS:-}" ]; then cp "$WEIGHTS"/*.pth weights/; fi
if [ -n "${SPLITS:-}" ]; then cp "$SPLITS"/*.csv splits/; fi

echo "== 4/6 Checking"
"$PY" -m pytest -q
if ls weights/*.pth >/dev/null 2>&1; then
  if ! "$PY" scripts/check_checkpoint.py weights/*.pth; then
    echo "The weights do not match the model code. Nothing was committed or pushed."
    exit 1
  fi
else
  echo "WARNING: weights/ contains no .pth files (the paper promises trained weights)."
fi
ls splits/*.csv >/dev/null 2>&1 || echo "WARNING: splits/ contains no subject lists."

echo "== 5/6 Changes"
git add -A
git status --short
read -r -p "Commit and push these changes to GitHub? [y/N] " ok
if [ "$ok" != "y" ] && [ "$ok" != "Y" ]; then
  echo "Stopped. Nothing was pushed. The prepared repository is in $(pwd)"
  exit 0
fi
git commit -q -F - <<'MSG'
Version 2.0: code for the revised paper

- bt25d package: decoder-free 2.5D network with joint WT/TC/ET heads,
  hybrid loss with deep supervision, balanced slice sampling, metrics
- scripts for preprocessing, subject-level splitting, training,
  full-volume evaluation, LCC analysis, benchmarking, ablations and
  prediction to NIfTI
- configs for every experiment in the paper
- unit tests and an end-to-end test on synthetic data, run by CI
- the single-task prototype moved to legacy/ (also tagged v1.0-single-task)
MSG

echo "== 6/6 Pushing"
git push origin HEAD
git push origin "$TAG"
echo "Done: https://github.com/thshovon1/2.5D-BrainTumor-Segmentation"
