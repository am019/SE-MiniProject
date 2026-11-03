"""TC-19 / FR-18: browser test of the dashboard (login -> sample -> experiment -> leaderboard -> predict)."""
import pytest
from playwright.sync_api import expect, sync_playwright

pytestmark = pytest.mark.integration

NO_HSCROLL = "document.documentElement.scrollWidth <= document.documentElement.clientWidth"


def open_page(page, link, heading, settle=True):
    """Click a nav link, wait for that page's heading, and (optionally) for its data requests to finish."""
    page.get_by_role("link", name=link, exact=True).click()
    expect(page.get_by_role("heading", name=heading, exact=True)).to_be_visible(timeout=30_000)
    if settle:
        page.wait_for_load_state("networkidle")


@pytest.mark.parametrize("browser_name", ["chromium", "firefox"])
def test_dashboard_end_to_end(platform, browser_name):
    password = platform.secrets["IFAP_ADMIN_PASSWORD"]
    with sync_playwright() as p:
        browser = getattr(p, browser_name).launch()
        page = browser.new_page(viewport={"width": 1366, "height": 768})
        try:
            page.goto(platform.base)

            # login, then let the Overview page finish loading before navigating away
            page.fill("#u", "admin")
            page.fill("#p", password)
            page.get_by_role("button", name="Sign in").click()
            expect(page.locator("#who")).to_contain_text("admin")
            expect(page.get_by_role("heading", name="Overview", exact=True)).to_be_visible(timeout=30_000)
            page.wait_for_load_state("networkidle")
            assert page.evaluate(NO_HSCROLL), "overview scrolls horizontally at 1366x768"

            # load the synthetic sample
            open_page(page, "Datasets", "Datasets")
            page.locator("button", has_text="sample_index_daily.csv").click(timeout=30_000)
            expect(page.get_by_role("heading", name="Validation report")).to_be_visible(timeout=60_000)

            # run a fast experiment (classical models only)
            open_page(page, "Run experiment", "Run experiment", settle=False)
            page.locator("#go").wait_for(timeout=30_000)
            page.locator("input[value=lstm]").uncheck()
            page.locator("input[value=qml_vqr]").uncheck()
            page.locator("#go").click()
            page.get_by_role("link", name="View leaderboard").click(timeout=180_000)

            # leaderboard, champion and chart
            expect(page.get_by_role("heading", name="Selected champion")).to_be_visible(timeout=30_000)
            expect(page.get_by_text("★ CHAMPION").first).to_be_visible()
            expect(page.get_by_text("Synthetic sample data").first).to_be_visible()
            page.wait_for_load_state("networkidle")
            assert page.evaluate(NO_HSCROLL), "results page scrolls horizontally at 1366x768"

            # prediction with the champion
            open_page(page, "Predict", "Predict with the champion")
            page.get_by_role("button", name="Predict", exact=True).click()
            expect(page.get_by_role("heading", name="Prediction", exact=True)).to_be_visible(timeout=30_000)
        except Exception:
            # show what the browser was looking at when the test failed
            try:
                page.screenshot(path="ui_failure.png")
                print("\n--- PAGE TEXT AT FAILURE ---\n" + page.locator("body").inner_text())
                print("--- screenshot saved as ui_failure.png ---")
            except Exception:
                pass
            raise
        finally:
            browser.close()