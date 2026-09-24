import os
os.environ["PLAYWRIGHT_BROWSERS_PATH"] = "/opt/pw-browsers"
from playwright.sync_api import sync_playwright

errors = []
console_msgs = []

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1440, "height": 900})
    page.on("console", lambda msg: console_msgs.append(f"[{msg.type}] {msg.text}"))
    page.on("pageerror", lambda exc: errors.append(str(exc)))

    page.goto("file:///home/claude/site_build_v2/index.html", wait_until="networkidle", timeout=15000)
    page.wait_for_timeout(1000)

    print("=== Page errors ===")
    print("\n".join(errors) if errors else "(none)")
    print("\n=== Console (warnings/errors only) ===")
    for m in console_msgs:
        if "error" in m.lower() or "warn" in m.lower() or "fail" in m.lower():
            print(m)

    # Check Chart.js actually rendered something onto the canvases
    chart_canvases = page.query_selector_all("canvas")
    print("\ncanvas elements found:", len(chart_canvases))
    for c in chart_canvases:
        box = c.bounding_box()
        print("  canvas box:", box)

    print("\nmodule cards:", len(page.query_selector_all("#moduleGrid .mcard")))
    print("eval cards  :", len(page.query_selector_all("#evalGrid .ecard")))

    # screenshot full page for visual review
    page.screenshot(path="/home/claude/site_build_v2/screenshot_full.png", full_page=True)
    page.set_viewport_size({"width": 390, "height": 844})
    page.wait_for_timeout(300)
    page.screenshot(path="/home/claude/site_build_v2/screenshot_mobile.png", full_page=True)

    browser.close()

print("\nDone.")
