#!/usr/bin/env bash
# Install the external, self-contained IsaacSim + IsaacLab worker runtime.
#
# Structure and conventions follow UniLab's scripts/tools/setup_isaacsim_env.sh
# (idempotent step markers, install log, EULA acceptance); the version stack is
# pinned to the combination this task's IsaacSim worker was validated against
# (IsaacSim 5.1.0 + IsaacLab 2.3.2.post1 pip wheels + torch 2.7.0/cu126) and
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
# Usage (from the repository root):
#   bash scripts/setup_isaacsim_env.sh
#
# Environment overrides:
#   UNISIM_ISAACSIM_HOME  install root   (default: ~/.cache/unisim/isaacsim)
#   TORCH_VERSION         torch version  (default: 2.7.0)
#   TORCHVISION_VERSION   torchvision    (default: 0.22.0)
#   TORCH_INDEX           torch CUDA index suffix (default: cu126)
#   ISAACSIM_VERSION      isaacsim pip version (default: 5.1.0)
#   ISAACLAB_VERSION      isaaclab pip version (default: 2.3.2.post1)
set -euo pipefail

ISAACSIM_HOME=${UNISIM_ISAACSIM_HOME:-${UNILAB_ISAACSIM_HOME:-$HOME/.cache/unisim/isaacsim}}
TORCH_VERSION=${TORCH_VERSION:-2.7.0}
TORCHVISION_VERSION=${TORCHVISION_VERSION:-0.22.0}
TORCH_INDEX=${TORCH_INDEX:-cu126}
ISAACSIM_VERSION=${ISAACSIM_VERSION:-5.1.0}
ISAACLAB_VERSION=${ISAACLAB_VERSION:-2.3.2.post1}
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
# Step 5: verification (imports + CUDA visibility + pinned versions).
# ---------------------------------------------------------------------------
step_verify() {
  "$PYTHON" - <<EOF
import torch
assert torch.cuda.is_available(), "CUDA not visible to torch"
print("torch", torch.__version__, "cuda", torch.version.cuda)
import isaacsim  # noqa: F401
print("isaacsim import OK")
import isaaclab
assert isaaclab.__version__ == "$ISAACLAB_VERSION", isaaclab.__version__
print("isaaclab", isaaclab.__version__)
EOF
}
run_step 05_verify step_verify

cat <<EOF

[setup_isaacsim_env] 完成。无需配置任何环境变量：
  simtoolreal-train / simtoolreal-play 会自动发现
  $VENV_DIR/bin/python

如需在 unisim 层直接使用（不经 simtoolreal CLI），可选写入 shell rc：
  export UNISIM_ISAACSIM_PYTHON="$VENV_DIR/bin/python"

  venv: $VENV_DIR
  log:  $LOG_FILE

重跑本脚本是安全的：已完成的步骤会被跳过。
首次验证建议跑一次小规模 smoke（见 README「首次成功运行」）。
EOF
