import asyncio


class _OkClient:
    def __init__(self, out: str):
        self._out = out

    async def generate(self, prompt, system_prompt=None, max_tokens=None, on_token=None, on_complete=None, params=None):
        _ = (system_prompt, max_tokens, on_token, on_complete, params)
        await asyncio.sleep(0)
        return self._out


class _FailClient:
    async def generate(self, *args, **kwargs):
        _ = (args, kwargs)
        raise RuntimeError("out of money")


def test_deepseek_step_failure_is_skipped(monkeypatch):
    from app.pipeline_engine import PipelineStep, run_pipeline

    # Patch model resolution so we can force a DeepSeek failure deterministically.
    def fake_get(mid, config):
        if mid.startswith("deepseek:"):
            return _FailClient(), "DeepSeek"
        if mid == "ollama:step1":
            return _OkClient("draft1"), "step1"
        if mid == "ollama:step3":
            return _OkClient("final"), "step3"
        return _OkClient("x"), "x"

    import app.pipeline_engine as pe

    monkeypatch.setattr(pe, "get_ai_client_for_model_id", fake_get)

    events = []

    async def progress(evt):
        events.append(dict(evt))

    cfg = type("Cfg", (), {})()
    steps = [
        PipelineStep(name="S1", model_id="ollama:step1", system_prompt="", prompt_template="{{input}}"),
        PipelineStep(name="S2", model_id="deepseek:deepseek-chat", system_prompt="", prompt_template="{{input}}"),
        PipelineStep(name="S3", model_id="ollama:step3", system_prompt="", prompt_template="{{input}}"),
    ]
    res = asyncio.run(run_pipeline(config=cfg, steps=steps, vars={"input": "seed"}, progress_callback=progress))

    assert res["outputs"][0] == "draft1"
    # DeepSeek failed -> skipped -> output remains previous input
    assert res["outputs"][1] == "draft1"
    assert res["final"] == "final"
    assert any(e.get("stage") == "skipped" for e in events)

