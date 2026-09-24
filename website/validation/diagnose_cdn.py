import os
os.environ["PLAYWRIGHT_BROWSERS_PATH"] = "/opt/pw-browsers"
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch()

    print("=== Attempt 1: default headless UA, hitting CDN directly ===")
    page = browser.new_page()
    resp_info = {}
    def on_response(resp):
        if "cdnjs" in resp.url:
            resp_info["status"] = resp.status
            resp_info["headers"] = resp.headers
    page.on("response", on_response)
    try:
        r = page.goto("https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.4/chart.umd.min.js", timeout=10000)
        print("direct nav status:", r.status)
        print("headers:", dict(r.headers))
        body = r.text()
        print("body length:", len(body), "starts with:", body[:120])
    except Exception as e:
        print("direct nav failed:", e)
    page.close()

    print("\n=== Attempt 2: spoofed realistic desktop Chrome UA ===")
    context2 = browser.new_context(
        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
    )
    page2 = context2.new_page()
    try:
        r2 = page2.goto("https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.4/chart.umd.min.js", timeout=10000)
        print("spoofed UA nav status:", r2.status)
        body2 = r2.text()
        print("body length:", len(body2), "starts with:", body2[:120])
    except Exception as e:
        print("spoofed UA nav failed:", e)
    context2.close()

    browser.close()
