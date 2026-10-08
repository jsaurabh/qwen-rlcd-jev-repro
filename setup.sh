#!/bin/sh
set -eu

cd "$(dirname "$0")"
PYTHON=${PYTHON:-python3.11}
MODE=${1:-small}

case "$MODE" in
  small|--small) FULL=0 ;;
  full|--full) FULL=1 ;;
  *) echo "usage: ./setup.sh [--small|--full]" >&2; exit 2 ;;
esac

if [ ! -x .venv/bin/python ]; then
  "$PYTHON" -m venv .venv
fi
.venv/bin/python -m pip install -r requirements.lock.txt

clone_at() {
  url=$1
  directory=$2
  revision=$3
  if [ ! -d "$directory/.git" ]; then
    git clone "$url" "$directory"
  fi
  git -C "$directory" fetch origin "$revision"
  git -C "$directory" checkout --detach "$revision"
}

clone_at https://huggingface.co/harshatheg/Qwen-2.5-1B-RLCD upstream 2af86848be75847ccb3553b0941cc51d6ef7e4e9

.venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
    revision="a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3",
    local_dir="models/qwen-0.5b-4bit",
)
PY

if [ "$FULL" -eq 1 ]; then
  clone_at https://github.com/bespokelabsai/nimble.git reference-nimble f136b3f75721fda4ea961f73993cc50b08488835
  clone_at https://github.com/fstandhartinger/jevbench.git reference-jevbench 5e95f23cbb7be098a9061fea924c4421620ab1a5
  .venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
    revision="8b403126fc14f14cfc99bb4cfa72ecbc129ea677",
    local_dir="models/qwen-1.5b-4bit",
)
PY
fi

mkdir -p runs
echo "Environment ready ($MODE)."
