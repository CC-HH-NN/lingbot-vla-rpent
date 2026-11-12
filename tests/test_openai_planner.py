"""OpenAI规划器的无网络契约测试。"""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1] / 'src'))
from openai_planner import OpenAIPlanner

def test_missing_key_is_explicit(monkeypatch):
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    try:
        OpenAIPlanner()
    except RuntimeError as exc:
        assert 'OPENAI_API_KEY' in str(exc)
    else:
        raise AssertionError('missing key must fail before network access')
