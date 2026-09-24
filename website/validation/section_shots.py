import os
os.environ["PLAYWRIGHT_BROWSERS_PATH"] = "/opt/pw-browsers"
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1440, "height": 900})
    page.goto("file:///home/claude/site_build_v2/index.html", wait_until="networkidle", timeout=15000)
    page.wait_for_timeout(400)

    for sel, name in [("#modules", "sec_modules"), ("#assistant", "sec_assistant"),
                       ("#dataset", "sec_dataset"), ("#evaluation", "sec_evaluation")]:
        page.locator(sel).scroll_into_view_if_needed()
        page.wait_for_timeout(200)
        page.screenshot(path=f"{name}.png")

    browser.close()
print("done")
