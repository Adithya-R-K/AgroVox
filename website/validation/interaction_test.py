import os
os.environ["PLAYWRIGHT_BROWSERS_PATH"] = "/opt/pw-browsers"
from playwright.sync_api import sync_playwright

errors = []
with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1440, "height": 900})
    page.on("pageerror", lambda exc: errors.append(str(exc)))
    page.goto("file:///home/claude/site_build_v2/index.html", wait_until="networkidle", timeout=15000)
    page.wait_for_timeout(500)

    print("=== sample chips present? ===", page.query_selector_all("#sampleChips .chip").__len__())
    print("=== pipeline row stages? ===", page.query_selector_all("#pipelineRow .pipe-step").__len__())

    print("\n=== Clicking a sample chip, then Analyze ===")
    page.click("#sampleChips .chip:nth-child(1)")
    page.click("#analyzeBtn")
    page.wait_for_timeout(2500)  # pipeline animation delays

    result_shown = page.eval_on_selector("#resultPanel", "el => el.classList.contains('show')")
    print("resultPanel shown:", result_shown)
    print("resIntent:", page.text_content("#resIntent"))
    print("resAnswer:", (page.text_content("#resAnswer") or "")[:80])
    print("history rows:", page.query_selector_all("#historyTable tbody tr").__len__())

    print("\n=== Page errors during interaction ===")
    print(errors if errors else "(none)")

    page.screenshot(path="screenshot_full.png", full_page=True)
    page.set_viewport_size({"width": 390, "height": 844})
    page.wait_for_timeout(300)
    page.screenshot(path="screenshot_mobile.png", full_page=True)

    # dark mode screenshot
    page.set_viewport_size({"width": 1440, "height": 900})
    page.click("#themeToggle")
    page.wait_for_timeout(300)
    page.screenshot(path="screenshot_dark.png", full_page=False)

    browser.close()
print("\nDone.")
