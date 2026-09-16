#!/usr/bin/env bash
# Install the external, self-contained IsaacSim + IsaacLab worker runtime.
#
# Structure and conventions follow UniLab's scripts/tools/setup_isaacsim_env.sh
# (idempotent step markers, install log, EULA acceptance); the version stack is
# pinned to the combination this task's IsaacSim worker was validated against
# (IsaacSim 5.1.0 + IsaacLab 2.3.2.post1 wheels + torch 2.7.0/cu126) and
# installs from wheels instead of an IsaacLab source checkout.
#
# Layout produced:
#   $UNISIM_ISAACSIM_HOME/venv      dedicated Python 3.11 worker venv
#   $UNISIM_ISAACSIM_HOME/.markers  per-step completion markers
#   $UNISIM_ISAACSIM_HOME/install.log
#
# The default root is ~/.cache/unisim/isaacsim, which `simtoolreal-train` /
# `simtoolreal-play` detect automatically (simtoolreal_unilab._isaacsim_worker);
# no environment variable is required after this script succeeds.
#
# Why step 05_importer exists: IsaacLab 2.3.2+ calls
# ImportConfig.set_merge_fixed_ignore_inertia when merging fixed-joint URDF
# children, but every pip wheel of isaacsim (5.1.0.0, the newest published)
# bundles urdf importer extension 2.4.30, whose binding lacks that method — a
# fresh pip install crashes on the first URDF conversion. The newer importer
# (2.4.31) is only published on NVIDIA's extension registry; the step downloads
# it from the registry's public CDN (sha256-pinned) and swaps it over the
# bundled copy. The verify step then fails closed unless the installed binding
# exposes the merge API.
#
# Usage (from the repository root):
#   bash scripts/setup_isaacsim_env.sh
#
# Environment overrides:
#   UNISIM_ISAACSIM_HOME  install root   (default: ~/.cache/unisim/isaacsim)
#   TORCH_VERSION         torch version  (default: 2.7.0)
#   TORCHVISION_VERSION   torchvision    (default: 0.22.0)
#   TORCH_INDEX           torch CUDA index suffix (default: cu128; covers
#                          Blackwell/sm_120 down to older architectures — cu126
#                          wheels lack Blackwell kernels)
#   ISAACSIM_VERSION      isaacsim pip version (default: 5.1.0)
#   ISAACLAB_VERSION      isaaclab pip version (default: 2.3.2.post1)
#   IMPORTER_SPEC         urdf importer registry spec (default: 2.4.31+107.3.3.lx64.r.cp311)
#   IMPORTER_URL          override the importer zip download URL
#   IMPORTER_SHA256       override the importer zip checksum pin
set -euo pipefail

ISAACSIM_HOME=${UNISIM_ISAACSIM_HOME:-${UNILAB_ISAACSIM_HOME:-$HOME/.cache/unisim/isaacsim}}
TORCH_VERSION=${TORCH_VERSION:-2.7.0}
TORCHVISION_VERSION=${TORCHVISION_VERSION:-0.22.0}
TORCH_INDEX=${TORCH_INDEX:-cu128}
ISAACSIM_VERSION=${ISAACSIM_VERSION:-5.1.0}
ISAACLAB_VERSION=${ISAACLAB_VERSION:-2.3.2.post1}
IMPORTER_SPEC=${IMPORTER_SPEC:-2.4.31+107.3.3.lx64.r.cp311}
IMPORTER_URL=${IMPORTER_URL:-"https://d4i3qtqj3r0z5.cloudfront.net/isaacsim.asset.importer.urdf-${IMPORTER_SPEC//+/%2B}.zip"}
IMPORTER_SHA256=${IMPORTER_SHA256:-665cc13d48842ac6d0c203359819c1d5c767a1d920fb8a00179e147ec2fb5201}
PYTHON_VERSION=3.11

VENV_DIR=$ISAACSIM_HOME/venv
MARKER_DIR=$ISAACSIM_HOME/.markers
LOG_FILE=$ISAACSIM_HOME/install.log

log() { echo "[setup_isaacsim_env $(date +%H:%M:%S)] $*"; }

# run_step <marker-name> <command...>
# Skips the command when the marker exists; creates the marker on success.
run_step() {
  local marker=$1; shift
  if [[ -f $MARKER_DIR/$marker ]]; then
    log "SKIP  $marker (already done)"
    return 0
  fi
  log "START $marker"
  "$@"
  touch "$MARKER_DIR/$marker"
  log "DONE  $marker"
}

mkdir -p "$MARKER_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1

log "UNISIM_ISAACSIM_HOME=$ISAACSIM_HOME"
log "isaacsim=$ISAACSIM_VERSION isaaclab=$ISAACLAB_VERSION torch=$TORCH_VERSION+$TORCH_INDEX"
log "urdf importer=$IMPORTER_SPEC"

# Accept the Omniverse Kit EULA non-interactively (install + any Kit launch).
export OMNI_KIT_ACCEPT_EULA=${OMNI_KIT_ACCEPT_EULA:-1}

# ---------------------------------------------------------------------------
# Step 1: Python 3.11 venv (IsaacSim/IsaacLab require Python 3.11).
# ---------------------------------------------------------------------------
step_create_venv() {
  # Drop a leftover broken venv (created without pip by a failed run); venv
  # creation is cheap and no wheel downloads are lost.
  if [[ -d $VENV_DIR && ! -x $VENV_DIR/bin/pip ]]; then
    rm -rf "$VENV_DIR"
  fi
  if command -v uv >/dev/null; then
    uv venv --python "$PYTHON_VERSION" --seed "$VENV_DIR"
  elif command -v "python$PYTHON_VERSION" >/dev/null; then
    "python$PYTHON_VERSION" -m venv "$VENV_DIR"
  else
    echo "Need uv or python$PYTHON_VERSION to create the venv" >&2
    return 1
  fi
  "$VENV_DIR/bin/pip" install --upgrade pip
}
run_step 01_venv step_create_venv

PYTHON="$VENV_DIR/bin/python"
# `python -m pip` keeps working even if the bin/pip shebang breaks mid-move.
pip_cmd() { "$PYTHON" -m pip "$@"; }

# Hold the torch stack across later steps: IsaacLab's [all] extra pulls ML
# deps (e.g. stable-baselines3) that would otherwise upgrade torch to the
# latest PyPI build.  Plain public version pins (no +cuXXX local label —
# local-version constraints break modern pip's resolver) match the wheels
# step 02 installs from the CUDA index.
CONSTRAINT_FILE=$ISAACSIM_HOME/pip-constraints.txt
printf 'torch==%s\ntorchvision==%s\n' "$TORCH_VERSION" "$TORCHVISION_VERSION" > "$CONSTRAINT_FILE"
export PIP_CONSTRAINT="$CONSTRAINT_FILE"

# ---------------------------------------------------------------------------
# Step 2: PyTorch (CUDA wheels; IsaacSim 5.1 is built against this stack).
# pip's HTTP cache (~/.cache/pip) makes re-runs cheap.
# ---------------------------------------------------------------------------
run_step 02_torch \
  pip_cmd install "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION" \
  --index-url "https://download.pytorch.org/whl/$TORCH_INDEX"

# ---------------------------------------------------------------------------
# Step 3: IsaacSim + IsaacLab wheels from the NVIDIA PyPI index (multi-GB;
# cached by pip on re-run). The SDK is installed first so IsaacLab's own
# resolution cannot replace it.
# ---------------------------------------------------------------------------
step_isaacsim() {
  pip_cmd install "isaacsim[all,extscache]==$ISAACSIM_VERSION" \
    --extra-index-url https://pypi.nvidia.com
}
run_step 03_isaacsim step_isaacsim

run_step 04_isaaclab \
  pip_cmd install "isaaclab[all]==$ISAACLAB_VERSION" \
  --extra-index-url https://pypi.nvidia.com

# ---------------------------------------------------------------------------
# Step 4: align the URDF importer extension (see header). Downloads the
# registry zip (resumable), verifies the pinned checksum, and swaps it over
# the pip-bundled 2.4.30 tree inside the venv's isaacsim exts directory.
# ---------------------------------------------------------------------------
IMPORTER_EXT_DIR="$VENV_DIR/lib/python$PYTHON_VERSION/site-packages/isaacsim/exts/isaacsim.asset.importer.urdf"

step_importer() {
  local installed_version
  installed_version=$(grep -m1 '^version' "$IMPORTER_EXT_DIR/config/extension.toml" 2>/dev/null \
    | sed 's/.*"\(.*\)".*/\1/' || true)
  if [[ $installed_version == "${IMPORTER_SPEC%%+*}" ]]; then
    log "importer already at $installed_version; nothing to swap"
    return 0
  fi
  local zip_path=$ISAACSIM_HOME/isaacsim.asset.importer.urdf-$IMPORTER_SPEC.zip
  if [[ ! -f $zip_path ]] || ! echo "$IMPORTER_SHA256  $zip_path" | sha256sum -c --status >/dev/null 2>&1; then
    curl -fL -C - --retry 5 --retry-delay 10 -o "$zip_path" "$IMPORTER_URL"
  fi
  if ! echo "$IMPORTER_SHA256  $zip_path" | sha256sum -c --status; then
    echo "importer zip checksum mismatch for $zip_path" >&2
    return 1
  fi
  local staging=$ISAACSIM_HOME/.importer-stage
  rm -rf "$staging" && mkdir -p "$staging"
  "$PYTHON" -c "import zipfile; zipfile.ZipFile('''$zip_path''').extractall('''$staging''')"
  # The zip holds the extension tree at its root (bin/ config/ isaacsim/ ...).
  if [[ ! -f $staging/config/extension.toml ]]; then
    echo "unexpected importer zip layout (no config/extension.toml at root)" >&2
    return 1
  fi
  rm -rf "${IMPORTER_EXT_DIR}.pre-swap.bak"
  mv "$IMPORTER_EXT_DIR" "${IMPORTER_EXT_DIR}.pre-swap.bak"
  mv "$staging" "$IMPORTER_EXT_DIR"
  local swapped_version
  swapped_version=$(grep -m1 '^version' "$IMPORTER_EXT_DIR/config/extension.toml" | sed 's/.*"\(.*\)".*/\1/')
  if [[ $swapped_version != "${IMPORTER_SPEC%%+*}" ]]; then
    echo "importer swap failed: version is '$swapped_version'" >&2
    return 1
  fi
  log "urdf importer swapped: $installed_version -> $swapped_version (backup at ${IMPORTER_EXT_DIR}.pre-swap.bak)"
}
run_step 05_importer step_importer

# ---------------------------------------------------------------------------
# Step 5: verification (imports + CUDA visibility + pinned versions + the
# importer binding must expose the merge API IsaacLab 2.3.2+ calls).
# ---------------------------------------------------------------------------
step_verify() {
  local urdf_so
  urdf_so=$(find "$IMPORTER_EXT_DIR" -name '_urdf*.so' | head -1)
  if [[ -z $urdf_so ]] || ! grep -aq "set_merge_fixed_ignore_inertia" "$urdf_so"; then
    echo "the installed URDF importer binding lacks set_merge_fixed_ignore_inertia" >&2
    echo "(pip bundles 2.4.30 without it; step 05_importer must have failed)" >&2
    return 1
  fi
  "$PYTHON" - <<EOF
import torch
assert torch.cuda.is_available(), "CUDA not visible to torch"
print("torch", torch.__version__, "cuda", torch.version.cuda)
import isaacsim  # noqa: F401
print("isaacsim import OK")
# Wheel installs do not expose isaaclab.__version__; read the dist metadata.
from importlib.metadata import version
assert version("isaaclab") == "$ISAACLAB_VERSION", version("isaaclab")
print("isaaclab", version("isaaclab"))
EOF
}
run_step 06_verify step_verify

cat <<EOF

[setup_isaacsim_env] 完成。无需配置任何环境变量：
  simtoolreal-train / simtoolreal-play 会自动发现
  $VENV_DIR/bin/python

如需在 unisim 层直接使用（不经 simtoolreal CLI），可选写入 shell rc：
  export UNISIM_ISAACSIM_PYTHON="$VENV_DIR/bin/python"

  venv:     $VENV_DIR
  importer: $IMPORTER_SPEC (registry CDN, checksum-pinned)
  log:      $LOG_FILE

重跑本脚本是安全的：已完成的步骤会被跳过。
首次验证建议跑一次小规模 smoke（见 README「首次成功运行」）。
EOF
