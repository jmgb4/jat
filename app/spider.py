"""Web scraping for job postings using Playwright (headless Chromium)."""

import asyncio
import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

try:
    from playwright.sync_api import TimeoutError as PlaywrightTimeout
    from playwright.sync_api import sync_playwright
except ImportError as e:
    sync_playwright = None
    PlaywrightTimeout = Exception  # type: ignore[misc, assignment]

PLAYWRIGHT_NOT_INSTALLED_MSG = (
    "Playwright or Chromium is not installed. From the project folder run: "
    ".\\venv\\Scripts\\python.exe -m playwright install chromium"
)


# Selectors for common job boards, ATS/vendor sites, and generic fallback
JOB_SELECTORS = [
    # LinkedIn
    "[data-job-id] .jobs-description__content",
    ".job-view-layout .jobs-description-content__content",
    ".jobs-box__html-content",
    # Indeed
    "#jobDescriptionText",
    ".jobsearch-JobComponent-description",
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
    screenshot_path = None
    title = ""
    description = ""

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
            time.sleep(0.3)

            # Try to get title from page
            title = page.title() or ""
            if not title or len(title) > 200:
                try:
                    h1 = page.query_selector("h1")
                    if h1:
                        title = (h1.inner_text() or "").strip()[:200] or title
                except Exception:
                    pass

            # Try known and generic selectors for description
            for selector in JOB_SELECTORS:
                try:
                    el = page.query_selector(selector)
                    if el:
                        description = (el.inner_text() or "").strip()
                        if description and len(description) > 100:
                            break
                except Exception:
                    continue

            if not description:
                body = page.query_selector("body")
                if body:
                    description = (body.inner_text() or "").strip()

            # Limit description size for context
            if len(description) > 50000:
                description = description[:50000] + "\n[... truncated ...]"

            if job_folder:
                job_folder.mkdir(parents=True, exist_ok=True)
                screenshot_path = job_folder / "screenshot.png"
                page.screenshot(path=str(screenshot_path), full_page=False)
                screenshot_path = str(screenshot_path)

                # Save the entire job posting page as HTML.
                try:
                    html_path = job_folder / "job_posting.html"
                    html_path.write_text(page.content(), encoding="utf-8")
                except Exception as e:
                    logger.warning("HTML save failed for %s: %s", url, e)

                # Save the job posting as a PDF for later reference (after page is fully loaded).
                time.sleep(0.2)
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
        "screenshot_path": screenshot_path,
    }


async def scrape_job(url: str, job_folder: str | Path | None = None, timeout_ms: int = 30000) -> dict[str, Any]:
    """
    Navigate to job URL, extract title and description, optionally save screenshot.
    Returns dict with url, title, description, screenshot_path (or None).
    """
    job_folder_path = Path(job_folder) if job_folder else None
    return await asyncio.to_thread(_scrape_job_sync, url, job_folder_path, timeout_ms)
