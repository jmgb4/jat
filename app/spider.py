"""Web scraping for job postings using Playwright (headless Chromium)."""

import asyncio
import html
import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

import aiohttp

logger = logging.getLogger(__name__)

try:
    from playwright.sync_api import TimeoutError as PlaywrightTimeout
    from playwright.sync_api import sync_playwright
except ImportError as e:
    sync_playwright = None
    PlaywrightTimeout = Exception  # type: ignore[misc, assignment]

PLAYWRIGHT_NOT_INSTALLED_MSG = (
    "Playwright or Chromium is not installed. From the project folder run: "
    "python -m playwright install chromium (or run: python scripts/jat.py setup)"
)


# ATS domains that need extra JS-render wait time
_SLOW_ATS_DOMAINS = (
    "myworkdayjobs.com",
    "ashbyhq.com",
    "greenhouse.io",
    "lever.co",
    "icims.com",
    "taleo.net",
    "successfactors.com",
    "smartrecruiters.com",
)


def _is_slow_ats(url: str) -> bool:
    url_lower = url.lower()
    return any(domain in url_lower for domain in _SLOW_ATS_DOMAINS)


# Selectors for common job boards, ATS/vendor sites, and generic fallback
JOB_SELECTORS = [
    # Ashby
    "[data-testid='job-description']",
    "[data-qa='job-description']",
    "[class*='ashby-job-posting'] [class*='description']",
    "[class*='ashby'] [class*='description']",
    # LinkedIn
    "[data-job-id] .jobs-description__content",
    ".job-view-layout .jobs-description-content__content",
    ".jobs-box__html-content",
    # Indeed
    "#jobDescriptionText",
    ".jobsearch-JobComponent-description",
    # Workday ATS (data-automation-id attributes)
    "[data-automation-id='jobPostingDescription']",
    "[data-automation-id='jobPostingDescriptionText']",
    "[data-automation-id='job-description']",
    # ATS / vendor pages (Greenhouse, Lever, Workday, etc.)
    "[data-qa='job-description']",
    "[data-qa='job-description-content']",
    ".job-content",
    ".job-description",
    ".job-details",
    ".job-body",
    "#job-details",
    ".content.content--structured",
    "[class*='JobDescription']",
    "[class*='job-description']",
    "[class*='description'] article",
    ".content__body",
    "main",
    "article",
    "#content",
    ".posting-details",
    "body",
]


def _normalise_whitespace(text: str) -> str:
    return re.sub(r"[ \t]+", " ", re.sub(r"\r\n?", "\n", text or "")).strip()


def _trim_description(text: str, max_len: int = 50000) -> str:
    out = _normalise_whitespace(text)
    if len(out) > max_len:
        return out[:max_len] + "\n[... truncated ...]"
    return out


def _description_confidence(text: str) -> dict[str, Any]:
    s = (text or "").strip()
    char_count = len(s)
    keywords = ("requirements", "responsibilities", "experience", "qualifications", "about", "role")
    lowered = s.lower()
    hits = sum(1 for kw in keywords if kw in lowered)
    if char_count >= 1200 and hits >= 2:
        confidence = "high"
    elif char_count >= 500:
        confidence = "medium"
    else:
        confidence = "low"
    return {"char_count": char_count, "keyword_hits": hits, "confidence": confidence}


def _html_to_text(raw_html: str) -> str:
    s = raw_html or ""
    s = re.sub(r"(?i)<br\s*/?>", "\n", s)
    s = re.sub(r"(?i)</p>", "\n\n", s)
    s = re.sub(r"(?i)</li>", "\n", s)
    s = re.sub(r"(?i)<li[^>]*>", "* ", s)
    s = re.sub(r"(?is)<script[^>]*>.*?</script>", " ", s)
    s = re.sub(r"(?is)<style[^>]*>.*?</style>", " ", s)
    s = re.sub(r"(?s)<[^>]+>", " ", s)
    s = html.unescape(s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return _normalise_whitespace(s)


def _parse_ashby_url(url: str) -> tuple[str, str] | None:
    parsed = urlparse(url or "")
    host = (parsed.netloc or "").lower()
    if "jobs.ashbyhq.com" not in host:
        return None
    parts = [p for p in (parsed.path or "").split("/") if p]
    if len(parts) < 2:
        return None
    return parts[0], parts[1]


def _match_ashby_job(jobs: list[dict[str, Any]], job_id: str) -> Optional[dict[str, Any]]:
    target = (job_id or "").strip().lower()
    for job in jobs or []:
        jid = str(job.get("id") or "").strip().lower()
        if jid == target:
            return job
        job_url = str(job.get("jobUrl") or "").strip().lower()
        if job_url.endswith("/" + target):
            return job
    return None


def _extract_ashby_description(job: dict[str, Any]) -> str:
    plain = str(job.get("descriptionPlain") or "").strip()
    if plain:
        return plain
    rich = str(job.get("descriptionHtml") or "").strip()
    if rich:
        return _html_to_text(rich)
    return ""


async def _fetch_ashby_job(url: str, timeout_ms: int = 30000) -> dict[str, Any] | None:
    parsed = _parse_ashby_url(url)
    if not parsed:
        return None
    board, job_id = parsed
    api_url = f"https://api.ashbyhq.com/posting-api/job-board/{board}"
    timeout = aiohttp.ClientTimeout(total=max(5, timeout_ms / 1000))
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(api_url) as resp:
                if resp.status != 200:
                    return None
                payload = await resp.json()
    except Exception as e:
        logger.warning("Ashby API fetch failed for %s: %s", url, e)
        return None

    job = _match_ashby_job(payload.get("jobs") or [], job_id)
    if not job:
        return None
    description = _extract_ashby_description(job)
    if not description.strip():
        return None
    title = str(job.get("title") or "").strip() or url
    quality = _description_confidence(description)
    return {
        "url": url,
        "title": title,
        "description": _trim_description(description),
        "scrape_source": "ashby_api",
        "scrape_meta": {
            **quality,
            "provider": "ashby",
            "method": "public_posting_api",
            "api_url": api_url,
            "attempted_sources": ["ashby_api"],
        },
    }


def _extract_json_ld_description(page) -> str:
    try:
        scripts = page.query_selector_all("script[type='application/ld+json']")
    except Exception:
        scripts = []
    for s in scripts:
        try:
            raw = (s.inner_text() or "").strip()
            if not raw:
                continue
            data = json.loads(raw)
        except Exception:
            continue
        candidates = data if isinstance(data, list) else [data]
        for obj in candidates:
            if not isinstance(obj, dict):
                continue
            desc = str(obj.get("description") or "").strip()
            if desc and len(desc) > 120:
                return _html_to_text(desc)
    return ""


def _scrape_job_sync(url: str, job_folder: Path | None, timeout_ms: int) -> dict[str, Any]:
    """
    Synchronous Playwright scraping.

    We run this in a worker thread to avoid Windows asyncio subprocess limitations
    that can cause Playwright to raise NotImplementedError.
    """
    if sync_playwright is None:
        raise RuntimeError(PLAYWRIGHT_NOT_INSTALLED_MSG)
    if job_folder is not None:
        job_folder = Path(job_folder).resolve()
    title = ""
    description = ""
    source = "none"
    method = "none"
    selector_used = ""
    attempted_sources: list[str] = ["dom_selector", "body_fallback"]

    with sync_playwright() as p:
        try:
            browser = p.chromium.launch(headless=True)
        except Exception as e:
            err = str(e).lower()
            if "executable" in err or "doesn't exist" in err or "browser" in err or "chromium" in err or "playwright" in err:
                raise RuntimeError(PLAYWRIGHT_NOT_INSTALLED_MSG) from e
            raise
        try:
            page = browser.new_page()
            page.set_default_timeout(timeout_ms)
            page.goto(url, wait_until="domcontentloaded")
            try:
                page.wait_for_load_state("load", timeout=8000)
            except Exception:
                pass
            try:
                page.wait_for_load_state("networkidle", timeout=5000)
            except Exception:
                pass

            # For heavy JS/ATS sites give the SPA extra time to render.
            # Workday uses data-automation-id; wait for it specifically then sleep.
            # Other slow-ATS domains (Greenhouse, Lever, iCIMS…) just need the extra sleep.
            if "myworkdayjobs.com" in url.lower():
                try:
                    page.wait_for_selector(
                        "[data-automation-id='jobPostingDescription']",
                        timeout=12000,
                    )
                except Exception:
                    pass
                time.sleep(2.0)
            elif _is_slow_ats(url):
                time.sleep(2.0)
            else:
                time.sleep(0.8)

            # Try to get title from page
            title = page.title() or ""
            if not title or len(title) > 200:
                try:
                    h1 = page.query_selector("h1")
                    if h1:
                        title = (h1.inner_text() or "").strip()[:200] or title
                except Exception:
                    pass

            # Step 1: structured payloads (JSON-LD)
            description = _extract_json_ld_description(page).strip()
            if description and len(description) > 100:
                source = "json_ld"
                method = "script[type='application/ld+json']"
                attempted_sources.insert(0, "json_ld")

            # Try known and generic selectors for description
            if not description:
                for _pass in range(3):
                    for selector in JOB_SELECTORS:
                        try:
                            el = page.query_selector(selector)
                            if el:
                                candidate = (el.inner_text() or "").strip()
                                if candidate and len(candidate) > 100:
                                    description = candidate
                                    selector_used = selector
                                    source = "dom_selector"
                                    method = selector
                                    break
                        except Exception:
                            continue
                    if description:
                        break
                    time.sleep(0.6 + (_pass * 0.4))

            if not description:
                body = page.query_selector("body")
                if body:
                    description = (body.inner_text() or "").strip()
                    source = "body_fallback"
                    method = "body"

            # Limit description size for context
            description = _trim_description(description)
            quality = _description_confidence(description)

            if job_folder:
                job_folder.mkdir(parents=True, exist_ok=True)

                # Save the entire job posting page as HTML.
                try:
                    html_path = job_folder / "job_posting.html"
                    html_path.write_text(page.content(), encoding="utf-8")
                except Exception as e:
                    logger.warning("HTML save failed for %s: %s", url, e)

                # Save the job posting as a PDF for later reference.
                try:
                    pdf_path = job_folder / "job_posting.pdf"
                    page.emulate_media(media="print")
                    page.pdf(
                        path=str(pdf_path),
                        format="A4",
                        margin={"top": "15mm", "bottom": "15mm", "left": "15mm", "right": "15mm"},
                        print_background=True,
                    )
                except Exception as e:
                    logger.warning("PDF save failed for %s: %s", url, e)

        except PlaywrightTimeout:
            raise TimeoutError(f"Scraping timed out for {url}")
        finally:
            browser.close()

    return {
        "url": url,
        "title": title or url,
        "description": description,
        "scrape_source": source or "none",
        "scrape_meta": {
            **_description_confidence(description),
            "method": method or "none",
            "selector": selector_used or None,
            "attempted_sources": attempted_sources,
        },
    }


async def scrape_job(url: str, job_folder: str | Path | None = None, timeout_ms: int = 30000) -> dict[str, Any]:
    """
    Navigate to job URL, extract title and description. Saves HTML and PDF to job_folder if provided.
    Returns dict with url, title, description.
    """
    # Step 0: provider-specific API adapters (more reliable than DOM scraping).
    ashby = await _fetch_ashby_job(url, timeout_ms=timeout_ms)
    if ashby:
        return ashby

    job_folder_path = Path(job_folder) if job_folder else None
    return await asyncio.to_thread(_scrape_job_sync, url, job_folder_path, timeout_ms)
