import asyncio
import json
from pathlib import Path


def test_parse_ashby_url():
    from app.spider import _parse_ashby_url

    parsed = _parse_ashby_url("https://jobs.ashbyhq.com/xbowcareers/9d84dfc6-2e3e-4ea1-b05c-10e50c17985d?src=LinkedIn")
    assert parsed == ("xbowcareers", "9d84dfc6-2e3e-4ea1-b05c-10e50c17985d")
    assert _parse_ashby_url("https://example.com/job/123") is None


def test_match_ashby_job_by_id_fixture():
    from app.spider import _match_ashby_job

    fixture = Path(__file__).parent / "fixtures" / "ashby_job_board.json"
    payload = json.loads(fixture.read_text(encoding="utf-8"))
    job = _match_ashby_job(payload["jobs"], "9d84dfc6-2e3e-4ea1-b05c-10e50c17985d")
    assert job is not None
    assert job["title"] == "Security Engineer"


def test_scrape_job_prefers_ashby_api(monkeypatch):
    from app import spider

    async def fake_fetch(url: str, timeout_ms: int = 30000):
        _ = (url, timeout_ms)
        return {
            "url": "https://jobs.ashbyhq.com/xbowcareers/9d84dfc6-2e3e-4ea1-b05c-10e50c17985d",
            "title": "Security Engineer",
            "description": "Long enough description with requirements and responsibilities.",
            "scrape_source": "ashby_api",
            "scrape_meta": {"method": "public_posting_api", "char_count": 120},
        }

    def fake_sync(url, job_folder, timeout_ms):
        raise AssertionError("browser scrape should not run when ashby api succeeds")

    monkeypatch.setattr(spider, "_fetch_ashby_job", fake_fetch)
    monkeypatch.setattr(spider, "_scrape_job_sync", fake_sync)

    res = asyncio.run(
        spider.scrape_job("https://jobs.ashbyhq.com/xbowcareers/9d84dfc6-2e3e-4ea1-b05c-10e50c17985d")
    )
    assert res["scrape_source"] == "ashby_api"


def test_scrape_job_falls_back_to_browser(monkeypatch):
    from app import spider

    async def fake_fetch(url: str, timeout_ms: int = 30000):
        _ = (url, timeout_ms)
        return None

    def fake_sync(url, job_folder, timeout_ms):
        _ = (job_folder, timeout_ms)
        return {"url": url, "title": "Fallback Role", "description": "A" * 150, "scrape_source": "dom_selector"}

    monkeypatch.setattr(spider, "_fetch_ashby_job", fake_fetch)
    monkeypatch.setattr(spider, "_scrape_job_sync", fake_sync)

    res = asyncio.run(spider.scrape_job("https://example.com/job/123"))
    assert res["scrape_source"] == "dom_selector"


def test_description_confidence_short_is_low():
    from app.spider import _description_confidence

    c = _description_confidence("tiny")
    assert c["confidence"] == "low"
    assert c["char_count"] == 4
