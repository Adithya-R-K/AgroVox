import os
os.environ["PLAYWRIGHT_BROWSERS_PATH"] = "/opt/pw-browsers"
from playwright.sync_api import sync_playwright

results = []
def check(name, cond):
    results.append((name, bool(cond)))

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1440, "height": 900})
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto("file:///home/claude/site_build_v2/index.html", wait_until="networkidle", timeout=15000)
    page.wait_for_timeout(400)

    # --- Viva mode ---
    page.click("#vivaToggle")
    page.wait_for_timeout(150)
    explain_visible = page.eval_on_selector(".explain-box", "el => getComputedStyle(el).display !== 'none'")
    check("viva mode reveals explain boxes", explain_visible)
    check("viva toggle aria-pressed true", page.get_attribute("#vivaToggle", "aria-pressed") == "true")
    page.click("#vivaToggle")  # toggle back off

    # --- Mobile menu ---
    page.set_viewport_size({"width": 500, "height": 900})
    page.wait_for_timeout(150)
    page.click("#hamburgerBtn")
    page.wait_for_timeout(150)
    check("mobile menu opens", page.eval_on_selector("#mobileMenu", "el => el.classList.contains('open')"))
    page.click("#hamburgerBtn")
    page.wait_for_timeout(150)
    check("mobile menu closes", not page.eval_on_selector("#mobileMenu", "el => el.classList.contains('open')"))
    page.set_viewport_size({"width": 1440, "height": 900})

    # --- Dark mode persistence ---
    page.click("#themeToggle")
    page.wait_for_timeout(150)
    theme_after_click = page.eval_on_selector("html", "el => el.getAttribute('data-theme')")
    page.reload(wait_until="networkidle")
    page.wait_for_timeout(300)
    theme_after_reload = page.eval_on_selector("html", "el => el.getAttribute('data-theme')")
    check("dark mode set after click", theme_after_click == "dark")
    check("dark mode persists after reload", theme_after_reload == "dark")
    # reset to light for rest of test
    page.click("#themeToggle")
    page.wait_for_timeout(150)

    # --- Tabs: dataset group ---
    page.locator("#dataset").scroll_into_view_if_needed()
    page.click("button.tab-btn[data-tab='gazTab']")
    page.wait_for_timeout(100)
    check("gazetteer tab activates", page.eval_on_selector("#gazTab", "el => el.classList.contains('active')"))
    check("qa tab deactivates", not page.eval_on_selector("#qaTab", "el => el.classList.contains('active')"))
    check("gazetteer rows rendered", page.query_selector_all("#gazTable tbody tr").__len__() > 0)

    # --- Full pipeline + feedback + history ---
    page.locator("#assistant").scroll_into_view_if_needed()
    page.fill("#queryInput", "What fertilizer should I use for wheat?")
    page.click("#analyzeBtn")
    page.wait_for_timeout(2600)
    check("result panel shown", page.eval_on_selector("#resultPanel", "el => el.classList.contains('show')"))

    page.click("#fbUp")
    page.wait_for_timeout(100)
    check("feedback thumbs-up picked", page.eval_on_selector("#fbUp", "el => el.classList.contains('picked')"))
    fb_text = page.text_content("#fbThanks")
    check("feedback thanks message shown", "Thanks" in (fb_text or ""))

    page.click("button.tab-btn[data-tab='feedbackTab']")
    page.wait_for_timeout(100)
    feedback_summary = page.text_content("#feedbackSummary")
    check("feedback summary reflects rating", "1 helpful" in (feedback_summary or "") or "helpful" in (feedback_summary or ""))

    page.click("button.tab-btn[data-tab='historyTab']")
    page.wait_for_timeout(100)
    history_rows = page.query_selector_all("#historyTable tbody tr").__len__()
    check("history has entries", history_rows >= 1)

    page.click("#clearHistoryBtn")
    page.wait_for_timeout(100)
    history_after_clear = page.text_content("#historyTable tbody")
    check("clear history works", "No queries yet" in (history_after_clear or ""))

    # --- Speech feature-detection messaging ---
    mic_status = page.text_content("#micStatus")
    check("mic status message present when unsupported or idle", mic_status is not None)

    # --- Second analyze run works after first (no stale state issues) ---
    page.fill("#queryInput", "How do I control whitefly organically?")
    page.click("#analyzeBtn")
    page.wait_for_timeout(2600)
    intent2 = page.text_content("#resIntent")
    check("second pipeline run produces an intent", bool(intent2) and intent2 != "—")

    print("=== Results ===")
    all_pass = True
    for name, ok in results:
        print(("PASS" if ok else "FAIL"), "-", name)
        if not ok: all_pass = False

    print("\n=== Uncaught page errors across whole session ===")
    print(errors if errors else "(none)")
    print("\nALL PASS:", all_pass)

    browser.close()
