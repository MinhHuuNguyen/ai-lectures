#!/usr/bin/env bash
set -e

# Cài uv
curl -Ls https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

PROJECT_DIR="venv/common_venv"
VENV_DIR="venv/.venv"

# Đọc python version nếu có
if [ -f "$PROJECT_DIR/.python-version" ]; then
    PYTHON_VERSION=$(cat "$PROJECT_DIR/.python-version")
else
    PYTHON_VERSION=""
fi

# Tạo venv gọn trong venv/.venv
uv venv $VENV_DIR --python $(cat $PROJECT_DIR/.python-version)

# Activate
source $VENV_DIR/bin/activate

# Sync dependencies từ config nằm trong subfolder
if [ -f "$PROJECT_DIR/uv.lock" ]; then
    uv sync --project $PROJECT_DIR
fi
