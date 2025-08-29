import argparse
import os
import re
import time
import pandas as pd
from bs4 import BeautifulSoup
from urllib.parse import urlparse, parse_qs
from contextlib import contextmanager

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import subprocess

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
# Tell Chrome/Chromium where to write its logs (to the void)
os.environ["CHROME_LOG_FILE"] = "NUL" if os.name == "nt" else "/dev/null"

# Process-wide native stderr sink (keeps our prints on stdout)
_STDERR_DEVNULL = None
def silence_stderr_processwide():
    """
    Redirect fd=2 (native stderr) to null for the lifetime of this process.
    This silences C/C++ logs from Chrome/TFLite that bypass Python logging.
    """
    global _STDERR_DEVNULL
    if _STDERR_DEVNULL is None:
        _STDERR_DEVNULL = open(os.devnull, "w")
        os.dup2(_STDERR_DEVNULL.fileno(), 2)

# -----------------------------
# Tunables (fast timeouts)
# -----------------------------
PAGE_LOAD_TIMEOUT = 8        # seconds for initial navigation
WAIT_TIMEOUT = 5             # seconds for dynamic rows to appear
RETRIES_PER_URL = 0          # extra tries per URL (keep 0 to be strict/fast)
POLITE_DELAY = 0.05          # pause between pages
FLUSH_EVERY = 25             # write partial results every N pages

# -----------------------------
# Native stderr suppressor (kills Chrome/TFLite C++ logs)
# -----------------------------
@contextmanager
def suppress_native_stderr():
    """
    Temporarily redirect the process-level stderr (fd 2) to os.devnull.
    Silences C/C++ logs coming from Chrome/TFLite that bypass Python logging.
    """
    saved_fd = os.dup(2)
    try:
        with open(os.devnull, "w") as devnull:
            os.dup2(devnull.fileno(), 2)
        yield
    finally:
        os.dup2(saved_fd, 2)
        os.close(saved_fd)

# -----------------------------
# Browser
# -----------------------------
def make_driver(headless: bool = True):
    opts = webdriver.ChromeOptions()
    if headless:
        opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--window-size=1280,900")
    opts.page_load_strategy = "eager"

    # Silence Chrome logs
    opts.add_argument("--disable-logging")
    opts.add_argument("--log-level=3")
    opts.add_experimental_option("excludeSwitches", ["enable-logging"])

    # Stop Chrome’s speech/TFLite stack (source of absl/TFLite messages)
    opts.add_argument("--disable-speech-api")
    opts.add_argument("--disable-features=LiveCaption,OnDeviceSpeechRecognition,OptimizationHints")

    #Avoid WebGL/3D entirely to suppress SwiftShader warnings
    opts.add_argument("--disable-3d-apis")
    opts.add_argument("--disable-webgl")
    opts.add_argument("--disable-webgl2")

    # Silence chromedriver output
    try:
        service = Service(ChromeDriverManager().install(), log_output=subprocess.DEVNULL)
    except TypeError:  # older Selenium
        service = Service(ChromeDriverManager().install(), log_path=os.devnull)

    with suppress_native_stderr():
        drv = webdriver.Chrome(service=service, options=opts)

    drv.set_page_load_timeout(PAGE_LOAD_TIMEOUT)
    drv.implicitly_wait(0)  # explicit waits only

    # Block assets we don't need (speeds up slow pages)
    try:
        drv.execute_cdp_cmd("Network.enable", {})
        drv.execute_cdp_cmd("Network.setBlockedURLs", {
            "urls": ["*.png","*.jpg","*.jpeg","*.gif","*.webp","*.svg",
                    "*.mp4","*.webm","*.ogg","*.woff","*.woff2","*.ttf"]
        })
    except Exception:
        pass
    return drv

def infer_gender_from_url(u: str) -> str:
    if "_m_" in u:
        return "Men"
    if "_f_" in u or "_w_" in u:
        return "Women"
    return "Unknown"

def extract_team_and_gender_from_title(soup_obj) -> tuple[str, str]:
    title_h3 = soup_obj.select_one(".panel-heading h3.panel-title")
    if title_h3:
        title_text = title_h3.get_text(strip=True)
        m = re.split(r"\s+(Men's|Women's)\b", title_text, maxsplit=1)
        if len(m) >= 2:
            team_name = m[0].strip()
            gender_word = m[1]
            gender = "Men" if gender_word.startswith("Men") else "Women"
            return team_name, gender
        team_name = re.sub(r"\s*Track\s*&\s*Field.*$", "", title_text).strip()
        return team_name, "Unknown"
    return "Unknown Team", "Unknown"

def extract_division(soup_obj) -> str:
    full_text = soup_obj.get_text(" ", strip=True)
    patt = re.compile(r"(NCAA\s*Division\s*(I{1,3})|NCAA\s*D[123]|NAIA|NJCAA)", re.IGNORECASE)
    m = patt.search(full_text)
    if m:
        found = m.group(0).upper().replace("  ", " ").strip()
        if "NCAA" in found and "DIVISION" in found:
            roman = re.search(r"DIVISION\s*(I{1,3})", found)
            if roman:
                return f"NCAA Division {roman.group(1)}"
        if "NCAA D1" in found:
            return "NCAA Division I"
        if "NCAA D2" in found:
            return "NCAA Division II"
        if "NCAA D3" in found:
            return "NCAA Division III"
        if "NAIA" in found:
            return "NAIA"
        if "NJCAA" in found:
            return "NJCAA"
        return found.title()
    return "Unknown"

def parse_page(url: str, html: str, meta: dict) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")

    # Page context
    gender = meta.get("gender") or infer_gender_from_url(url)
    team_from_title, gender_from_title = extract_team_and_gender_from_title(soup)
    if gender == "Unknown" and gender_from_title != "Unknown":
        gender = gender_from_title
    team_name = meta.get("school") or team_from_title
    division = meta.get("division") or extract_division(soup)

    # For traceability
    q = parse_qs(urlparse(url).query)
    list_hnd = q.get("list_hnd", [""])[0]
    season_hnd = q.get("season_hnd", [""])[0]

    rows = soup.find_all("div", class_="performance-list-row")
    out = []
    for row in rows:
        try:
            parent_event_div = row.find_previous("div", class_="col-lg-12")
            event_name = parent_event_div.find("h3").get_text(strip=True) if parent_event_div else "Unknown Event"

            athlete = row.find("div", class_="col-athlete").get_text(strip=True)
            year_cell = row.find("div", attrs={"data-label": "Year"})
            year = year_cell.get_text(strip=True) if year_cell else ""

            team_div = row.find("div", attrs={"data-label": "Team"})
            team_cell = team_div.get_text(strip=True) if team_div else team_name

            time_div = row.find("div", attrs={"data-label": "Time"})
            mark_div = row.find("div", attrs={"data-label": "Mark"})
            result_div = row.find("div", attrs={"data-label": "Result"})
            score_div = row.find("div", attrs={"data-label": "Score"})
            mark_or_time = ""
            for t in (time_div, mark_div, result_div, score_div):
                if t:
                    mark_or_time = t.get_text(strip=True)
                    break

            wind_div = row.find("div", attrs={"data-label": "Wind"}) or row.find("div", attrs={"data-label": "Wind/Aid"})
            wind = wind_div.get_text(strip=True) if wind_div else ""

            meet = row.find("div", class_="col-meet")
            meet = meet.get_text(strip=True) if meet else ""

            meet_date = row.find("div", attrs={"data-label": "Meet Date"}) or row.find("div", attrs={"data-label": "Date"})
            meet_date = meet_date.get_text(strip=True) if meet_date else ""

            out.append({
                "Division": division,
                "School": team_name,
                "Gender": gender,
                "SeasonLabel": meta.get("season"),
                "SeasonYear": meta.get("year"),
                "ListHnd": list_hnd,
                "SeasonHnd": season_hnd,
                "SourceURL": url,

                "Team": team_cell,
                "Event": event_name,
                "Athlete": athlete,
                "ClassYear": year,
                "MarkOrTime": mark_or_time,
                "Wind": wind,
                "Meet": meet,
                "MeetDate": meet_date,
            })
        except Exception as e:
            print(f"  ⚠️ Row parse error: {e}")
            continue
    return out

def wait_for_rows(driver) -> bool:
    """Return True if rows appear; False if they don't within WAIT_TIMEOUT."""
    try:
        WebDriverWait(driver, WAIT_TIMEOUT).until(
            EC.any_of(
                EC.presence_of_element_located((By.CLASS_NAME, "performance-list-row")),
                EC.presence_of_element_located((By.XPATH, "//*[contains(., 'No performances found')]")),
            )
        )
    except Exception:
        return False
    # Check if rows exist
    return len(driver.find_elements(By.CLASS_NAME, "performance-list-row")) > 0

def scrape_one_url(driver, url: str, meta: dict) -> list[dict]:
    HARD_CAP = 5.0  # seconds
    start = time.monotonic()
    attempts = 0
    while attempts <= RETRIES_PER_URL:
        try:
            # Abort if we've spent the cap already
            if time.monotonic() - start >= HARD_CAP:
                print("  ⏱️ hard cap reached (5s) — skipping")
                return []

            driver.get(url)

            # Don’t let the explicit wait exceed the remaining cap
            remaining = HARD_CAP - (time.monotonic() - start)
            if remaining <= 0:
                print("  ⏱️ hard cap reached (5s) — skipping")
                return []

            # Temporarily override the wait for this call
            ok = False
            try:
                WebDriverWait(driver, min(WAIT_TIMEOUT, remaining)).until(
                    EC.any_of(
                        EC.presence_of_element_located((By.CLASS_NAME, "performance-list-row")),
                        EC.presence_of_element_located((By.XPATH, "//*[contains(., 'No performances found')]")),
                    )
                )
                ok = len(driver.find_elements(By.CLASS_NAME, "performance-list-row")) > 0
            except Exception:
                ok = False

            if not ok:
                # try to explicitly stop loading and bail
                try:
                    driver.execute_cdp_cmd("Page.stopLoading", {})
                except Exception:
                    pass
                print("  ⏱️ timed out/no rows — skipping")
                return []

            return parse_page(url, driver.page_source, meta)

        except Exception as e:
            attempts += 1
            if attempts > RETRIES_PER_URL:
                print(f"  ❌ error after retries: {e}")
                return []
        finally:
            time.sleep(POLITE_DELAY)

def url_has_required_query(u: str) -> bool:
    q = parse_qs(urlparse(u).query)
    return q.get("list_hnd", [""])[0].isdigit() and q.get("season_hnd", [""])[0].isdigit()

# -----------------------------
# Main
# -----------------------------
def main():
    silence_stderr_processwide()
    ap = argparse.ArgumentParser(description="Fast TFRRS scraper that times out quickly and skips slow pages.")
    ap.add_argument("-i", "--input", default="tfrrs_all_ncaa_urls.csv",
                    help="Input CSV with columns: division, school, gender, year, season, url")
    ap.add_argument("-o", "--output", default="tfrrs_performances_fast.csv",
                    help="Output CSV")
    ap.add_argument("--no-headless", action="store_true",
                    help="Show the browser window")
    ap.add_argument("--limit", type=int, default=None,
                    help="Optional limit on number of pages")
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    # normalize column names (allow any case)
    df.columns = [c.lower() for c in df.columns]
    for col in ["division", "school", "gender", "year", "season", "url"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    if args.limit:
        df = df.head(args.limit).copy()

    drv = make_driver(headless=not args.no_headless)

    buffer = []
    total = len(df)
    print(f"🚀 scraping {total} pages (timeout {WAIT_TIMEOUT}s)...")
    for i, row in df.iterrows():
        url = row["url"]
        meta = {
            "division": row["division"],
            "school": row["school"],
            "gender": row["gender"],
            "year": row["year"],
            "season": row["season"],
        }
        print(f"[{i+1}/{total}] {url}")
        for i, row in df.iterrows():
            url = row["url"]
            if not url_has_required_query(url):
                print(f"[{i+1}/{total}] {url}\n  ⚠️ malformed URL (list_hnd/season_hnd) — skipping")
                continue
        buffer.extend(scrape_one_url(drv, url, meta))

        # periodic flush
        if (i + 1) % FLUSH_EVERY == 0 and buffer:
            out_df = pd.DataFrame(buffer)
            write_mode = "a" if os.path.exists(args.output) else "w"
            header = not os.path.exists(args.output)
            out_df.to_csv(args.output, index=False, mode=write_mode, header=header)
            print(f"  💾 flushed {len(buffer)} rows")
            buffer = []

    # final flush
    if buffer:
        out_df = pd.DataFrame(buffer)
        write_mode = "a" if os.path.exists(args.output) else "w"
        header = not os.path.exists(args.output)
        out_df.to_csv(args.output, index=False, mode=write_mode, header=header)
        print(f"  💾 final flush {len(buffer)} rows")

    drv.quit()
    print(f"✅ done. wrote → {args.output}")

if __name__ == "__main__":
    main()
