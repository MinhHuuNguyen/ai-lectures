#!/usr/bin/env bash
set -e

# Cài uv
curl -Ls https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

cd /workspaces/ai-lectures/venv/common_venv
uv sync; uv lock
source /workspaces/ai-lectures/venv/common_venv/bin/activate
cd /workspaces/ai-lectures/

cp 4_deep_learning/notebook/3-generative-ai/.env.example 4_deep_learning/notebook/3-generative-ai/.env
cp 4_deep_learning/notebook/4-ai-agent/.env.example 4_deep_learning/notebook/4-ai-agent/.env
cp 4_deep_learning/notebook/9-retrieval-augmented-generation/.env.example 4_deep_learning/notebook/9-retrieval-augmented-generation/.env
