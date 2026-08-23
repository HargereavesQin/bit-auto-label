# -*- coding: utf-8 -*-
"""D6 反序列化安全回归：恶意 .pt 必须被 weights_only 拒绝且零副作用。"""
import os
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from core.safe_torch import torch_load_safe  # noqa: E402


class _Malicious:
    """pickle RCE payload：反序列化即写标记文件。"""

    def __reduce__(self):
        return os.system, (f"type __pwned__ > {os.path.join(os.environ.get('TEMP', '/tmp'), 'pwned_flag.txt')}",)


def _make_malicious_pt(path: Path):
    torch.save(_Malicious(), str(path))  # type: ignore[arg-type]


def test_malicious_pt_rejected_without_side_effect(tmp_path, monkeypatch):
    flag = tmp_path / "pwned_flag.txt"
    payload = tmp_path / "evil.pt"
    # 把 payload 的落点重定向到 tmp_path，避免污染系统 TEMP
    monkeypatch.setenv("TEMP", str(tmp_path))
    _make_malicious_pt(payload)

    monkeypatch.delenv("ARMORLABEL_UNSAFE_LOAD", raising=False)
    with pytest.raises(RuntimeError, match="拒绝加载"):
        torch_load_safe(payload, map_location="cpu")

    assert not flag.exists(), "weights_only 模式下 payload 不得被执行"


def test_malicious_pt_blocked_in_app_loader(tmp_path, monkeypatch):
    """GUI 权重加载入口 load_mlp_conv_pt 必须走安全加载。"""
    payload = tmp_path / "evil_mlp.pt"
    _make_malicious_pt(payload)

    from core.yolo_client import load_mlp_conv_pt

    with pytest.raises(RuntimeError):
        load_mlp_conv_pt(str(payload))


def test_state_dict_roundtrip(tmp_path):
    """正常 state_dict 在安全模式下必须无损加载（不破坏合法功能）。"""
    from core.mlp_conv import MlpConv, NUM_NETWORK_CLASSES

    model = MlpConv(num_classes=NUM_NETWORK_CLASSES, width_mult=1.0)
    sd_path = tmp_path / "ok.pt"
    torch.save(model.state_dict(), str(sd_path))

    from core.yolo_client import load_mlp_conv_pt

    loaded = load_mlp_conv_pt(str(sd_path))
    assert loaded is not None
    got = torch_load_safe(sd_path, map_location="cpu")
    for key in model.state_dict():
        assert torch.equal(got[key], model.state_dict()[key]), f"参数 {key} 加载后不一致"


def test_unsafe_optin_is_explicit_gate(tmp_path, monkeypatch):
    """ARMORLABEL_UNSAFE_LOAD=1 是显式 opt-in 闸门：开启后旧行为可用（用户自担风险）。"""
    payload = tmp_path / "legacy.pt"
    monkeypatch.setenv("TEMP", str(tmp_path))  # 必须先重定向，payload 路径在构造时固化
    _make_malicious_pt(payload)
    monkeypatch.setenv("ARMORLABEL_UNSAFE_LOAD", "1")

    obj = torch_load_safe(payload, map_location="cpu")
    # os.system 经 __reduce__ 执行后 unpickle 产物是其返回值(int)，实例本身即执行证据
    assert (tmp_path / "pwned_flag.txt").exists(), "opt-in 模式应保持旧行为（可加载整对象并执行 reduce）"
    assert isinstance(obj, int)
