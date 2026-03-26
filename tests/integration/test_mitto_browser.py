"""Browser smoke test for the Mitto UI.

Requires a running Mitto frontend.  Deselected by default; opt in with::

    uv run pytest pykoclaw-acp/tests/integration/test_mitto_browser.py -m browser

The base URL defaults to ``http://127.0.0.1:8081`` and can be overridden via
the ``DEV_MITTO_URL`` environment variable (same variable used by the previous
Node/Playwright Test suite).

Uses Python ``playwright.sync_api`` so it runs against the monorepo venv's
Playwright install (1.57.0), which is aligned with the Nix-provided browser
binaries.  This avoids the Node-vs-Nix browser-revision mismatch that caused
the former ``@playwright/test`` runner to fail.
"""

from __future__ import annotations

import os

import pytest
from playwright.sync_api import sync_playwright

pytestmark = pytest.mark.browser

_DEFAULT_URL = "http://127.0.0.1:8081"


def _base_url() -> str:
    return os.environ.get("DEV_MITTO_URL", _DEFAULT_URL)


def test_mitto_page_loads() -> None:
    """Open the Mitto root page and assert the body element is visible."""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        try:
            page = browser.new_page()
            page.set_default_timeout(30_000)
            page.goto(_base_url(), timeout=60_000)
            page.wait_for_selector("body", state="visible", timeout=10_000)
        finally:
            browser.close()
