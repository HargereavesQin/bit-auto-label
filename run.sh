#!/usr/bin/env bash
# ArmorLabel 一键启动：自动激活 conda 环境并运行主程序
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# 本地覆盖（可选）：复制 scripts/env.local.example → scripts/env.local
if [[ -f "$ROOT/scripts/env.local" ]]; then
  # shellcheck source=/dev/null
  source "$ROOT/scripts/env.local"
fi

CONDA_ENV="${CONDA_ENV:-armorlabel}"

_find_conda_sh() {
  local candidate
  for candidate in \
    "${CONDA_SH:-}" \
    "$HOME/anaconda3/etc/profile.d/conda.sh" \
    "$HOME/miniconda3/etc/profile.d/conda.sh" \
    "${CONDA_EXE:+$(dirname "$(dirname "$CONDA_EXE")")/etc/profile.d/conda.sh}"; do
    if [[ -n "$candidate" && -f "$candidate" ]]; then
      echo "$candidate"
      return 0
    fi
  done
  return 1
}

CONDA_SH_PATH="$(_find_conda_sh || true)"
if [[ -z "$CONDA_SH_PATH" ]]; then
  echo "错误: 未找到 conda.sh，请安装 Anaconda/Miniconda 或设置 CONDA_SH / CONDA_EXE" >&2
  exit 1
fi

# shellcheck source=/dev/null
source "$CONDA_SH_PATH"

if ! conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
  echo "错误: conda 环境不存在: $CONDA_ENV" >&2
  echo "可用环境:" >&2
  conda env list >&2
  echo "提示: 环境名区分大小写，本项目默认 armorlabel（全小写）" >&2
  exit 1
fi

conda activate "$CONDA_ENV"
exec python main.py "$@"
