# ScrapeFromTfrrs_Fast.py
# Scrape TFRRS all_performances pages quickly with a bounded per-URL time.
# Writes a structured .npy file (and periodic checkpoints).
# NOTE: We DO NOT redirect process-wide stderr anymore to avoid silent exits.

import argparse
import os
import re
import time
import random
import traceback
import numpy as np
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
from selenium.common.exceptions import WebDriverException
import subprocess

# -----------------------------
# Env: hush Chrome spam (but DO NOT sink Python stderr)
# -----------------------------
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ["CHROME_LOG_FILE"] = "NUL" if os.name == "nt" else "/dev/null"

# -----------------------------
# Tunables
# -----------------------------
WAIT_TIMEOUT = 5            # seconds for dynamic rows to appear
HARD_CAP_SEC = 6.5          # total budget per URL (nav + wait)
POLITE_DELAY_BASE = 0.15    # base delay between pages
CMD_TIMEOUT = 6             # cap for ANY single WebDriver command (default ~120s)
MAX_CONSEC_ERRORS = 4       # after N consecutive errors, recycle driver
RECYCLE_EVERY = 75          # or recycle periodically regardless of errors
CHECKPOINT_EVERY = 25       # write periodic .npy checkpoints every N pages

LOG_FILE = "scrape_debug.log"

def log(msg: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass

@contextmanager
def suppress_native_stderr():
    """Temporarily redirect native stderr during Chrome driver creation / navigation (only)."""
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

    # Less fingerprinty + quiet logs
    opts.add_argument("--disable-logging")
    opts.add_argument("--log-level=3")
    opts.add_argument("--disable-blink-features=AutomationControlled")
    opts.add_experimental_option("excludeSwitches", ["enable-automation","enable-logging"])
    opts.add_experimental_option("useAutomationExtension", False)

    # Disable Chrome speech / LiveCaption stack
    opts.add_argument("--disable-speech-api")
    opts.add_argument("--disable-features=LiveCaption,OnDeviceSpeechRecognition,OptimizationHints")

    # Avoid 3D warnings
    opts.add_argument("--disable-3d-apis")
    opts.add_argument("--disable-webgl")
    opts.add_argument("--disable-webgl2")

    # Silence chromedriver output
    try:
        service = Service(ChromeDriverManager().install(), log_output=subprocess.DEVNULL)
    except TypeError:
        service = Service(ChromeDriverManager().install(), log_path=os.devnull)

    with suppress_native_stderr():
        drv = webdriver.Chrome(service=service, options=opts)

    # Tighten timeouts on the command channel
    drv.implicitly_wait(0)
    try:
        drv.command_executor.set_timeout(CMD_TIMEOUT)
    except Exception:
        try:
            drv.command_executor._client.timeout = CMD_TIMEOUT
        except Exception:
            try:
                drv.command_executor.timeout = CMD_TIMEOUT
            except Exception:
                pass

    # Block heavy assets (NOT JS)
    try:
        drv.execute_cdp_cmd("Network.enable", {})
        drv.execute_cdp_cmd("Network.setBlockedURLs", {
            "urls": ["*.png","*.jpg","*.jpeg","*.gif","*.webp","*.svg",
                     "*.mp4","*.webm","*.ogg","*.woff","*.woff2","*.ttf"]
        })
    except Exception:
        pass

    # Stealth: UA override + webdriver/languages/plugins
    UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
          "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36")
    try:
        drv.execute_cdp_cmd("Network.setUserAgentOverride", {
            "userAgent": UA, "acceptLanguage": "en-US,en;q=0.9", "platform": "Windows"
        })
        drv.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
            "source": """
            Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
            window.chrome = { runtime: {} };
            Object.defineProperty(navigator, 'languages', {get: () => ['en-US','en']});
            Object.defineProperty(navigator, 'plugins', {get: () => [1,2,3]});
            """
        })
    except Exception:
        pass

    return drv

def recycle_driver(drv, headless=True):
    try:
        drv.quit()
    except Exception:
        pass
    log("♻️ recycling browser")
    return make_driver(headless=headless)

# -----------------------------
# Helpers
# -----------------------------
def infer_gender_from_url(u: str) -> str:
    if "_m_" in u: return "Men"
    if "_f_" in u or "_w_" in u: return "Women"
    return "Unknown"

def extract_team_and_gender_from_title(soup_obj):
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
            if roman: return f"NCAA Division {roman.group(1)}"
        if "NCAA D1" in found: return "NCAA Division I"
        if "NCAA D2" in found: return "NCAA Division II"
        if "NCAA D3" in found: return "NCAA Division III"
        if "NAIA" in found:    return "NAIA"
        if "NJCAA" in found:   return "NJCAA"
        return found.title()
    return "Unknown"

def safe_text(node):
    return node.get_text(strip=True) if node else ""

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
            h3 = parent_event_div.find("h3") if parent_event_div else None
            event_name = safe_text(h3) or "Unknown Event"

            athlete = safe_text(row.find("div", class_="col-athlete"))
            year_cell = row.find("div", attrs={"data-label": "Year"})
            year = safe_text(year_cell)

            team_div = row.find("div", attrs={"data-label": "Team"})
            team_cell = safe_text(team_div) or team_name

            time_div = row.find("div", attrs={"data-label": "Time"})
            mark_div = row.find("div", attrs={"data-label": "Mark"})
            result_div = row.find("div", attrs={"data-label": "Result"})
            score_div = row.find("div", attrs={"data-label": "Score"})
            mark_or_time = ""
            for t in (time_div, mark_div, result_div, score_div):
                s = safe_text(t)
                if s:
                    mark_or_time = s
                    break

            wind_div = (row.find("div", attrs={"data-label": "Wind"}) or
                        row.find("div", attrs={"data-label": "Wind/Aid"}))
            wind = safe_text(wind_div)

            meet = safe_text(row.find("div", class_="col-meet"))

            meet_date_div = (row.find("div", attrs={"data-label": "Meet Date"}) or
                             row.find("div", attrs={"data-label": "Date"}))
            meet_date = safe_text(meet_date_div)

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
            log(f"  ⚠️ Row parse error: {e}")
            continue
    return out

def wait_for_rows(driver) -> bool:
    # Accept either: rows, a known page header, or an explicit empty-state
    try:
        WebDriverWait(driver, WAIT_TIMEOUT).until(
            EC.any_of(
                EC.presence_of_element_located((By.CLASS_NAME, "performance-list-row")),
                EC.presence_of_element_located((By.CSS_SELECTOR, ".panel-heading h3.panel-title")),
                EC.presence_of_element_located((By.XPATH, "//*[contains(., 'No performances found')]")),
            )
        )
    except Exception:
        return False
    return len(driver.find_elements(By.CLASS_NAME, "performance-list-row")) > 0

def cdp_navigate(driver, url: str):
    # Navigate via CDP (non-blocking). If Page isn't enabled, enable it once.
    try:
        driver.execute_cdp_cmd("Page.enable", {})
    except Exception:
        pass
    driver.execute_cdp_cmd("Page.navigate", {"url": url})

def scrape_one_url(driver, url: str, meta: dict) -> list[dict]:
    start = time.monotonic()

    with suppress_native_stderr():
        cdp_navigate(driver, url)

    # Wait bounded by remaining budget
    remaining = HARD_CAP_SEC - (time.monotonic() - start)
    if remaining <= 0:
        log("  ⏱️ hard cap reached — skipping")
        return []

    ok = False
    try:
        WebDriverWait(driver, min(WAIT_TIMEOUT, max(0.5, remaining))).until(
            EC.any_of(
                EC.presence_of_element_located((By.CLASS_NAME, "performance-list-row")),
                EC.presence_of_element_located((By.CSS_SELECTOR, ".panel-heading h3.panel-title")),
                EC.presence_of_element_located((By.XPATH, "//*[contains(., 'No performances found')]")),
            )
        )
        ok = len(driver.find_elements(By.CLASS_NAME, "performance-list-row")) > 0
    except Exception:
        ok = False

    if not ok:
        try:
            driver.execute_cdp_cmd("Page.stopLoading", {})
        except Exception:
            pass
        log("  ⏱️ timed out/no rows — skipping")
        return []

    return parse_page(url, driver.page_source, meta)

def url_has_required_query(u: str) -> bool:
    q = parse_qs(urlparse(u).query)
    return q.get("list_hnd", [""])[0].isdigit() and q.get("season_hnd", [""])[0].isdigit()

# -----------------------------
# Save as NPY (structured array)
# -----------------------------
def save_buffer_as_npy(rows: list[dict], out_path: str):
    if not rows:
        dtype = [
            ("Division", "U1"), ("School", "U1"), ("Gender", "U1"), ("SeasonLabel", "U1"),
            ("SeasonYear", "i4"), ("ListHnd", "i4"), ("SeasonHnd", "i4"), ("SourceURL", "U1"),
            ("Team", "U1"), ("Event", "U1"), ("Athlete", "U1"), ("ClassYear", "U1"),
            ("MarkOrTime", "U1"), ("Wind", "U1"), ("Meet", "U1"), ("MeetDate", "U1"),
        ]
        np.save(out_path, np.empty(0, dtype=dtype))
        log(f"💾 wrote empty NPY → {out_path}")
        return

    df = pd.DataFrame(rows)
    cols = ["Division","School","Gender","SeasonLabel","SeasonYear","ListHnd","SeasonHnd",
            "SourceURL","Team","Event","Athlete","ClassYear","MarkOrTime","Wind","Meet","MeetDate"]
    for c in cols:
        if c not in df.columns:
            df[c] = ""

    str_cols = ["Division","School","Gender","SeasonLabel","SourceURL","Team","Event",
                "Athlete","ClassYear","MarkOrTime","Wind","Meet","MeetDate"]
    str_maxlens = {}
    for c in str_cols:
        s = df[c].astype("string").fillna("")
        df[c] = s
        str_maxlens[c] = max(1, int(s.str.len().max() or 1))

    for c in ["SeasonYear","ListHnd","SeasonHnd"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(-1).astype(np.int32)

    dtype = [(c, f"<U{str_maxlens[c]}") if c in str_cols else (c, "<i4") for c in cols]
    arr = np.empty(len(df), dtype=dtype)
    for c in cols:
        if c in str_cols:
            arr[c] = df[c].to_numpy(dtype=object)
        else:
            arr[c] = df[c].to_numpy(dtype=np.int32)

    np.save(out_path, arr)
    log(f"💾 wrote NPY with {len(arr):,} rows → {out_path}")

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Fast TFRRS scraper (stealthy, bounded per-URL time) → .npy")
    ap.add_argument("-i", "--input", default="tfrrs_all_ncaa_urls.csv",
                    help="Input CSV with columns: division, school, gender, year, season, url")
    ap.add_argument("-o", "--output", default="tfrrs_performances_fast.npy",
                    help="Output .npy filename")
    ap.add_argument("--no-headless", action="store_true",
                    help="Show the browser window")
    ap.add_argument("--limit", type=int, default=None,
                    help="Optional limit on number of pages")
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    df.columns = [c.lower() for c in df.columns]
    for col in ["division", "school", "gender", "year", "season", "url"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    if args.limit:
        df = df.head(args.limit).copy()

    headless = not args.no_headless
    drv = make_driver(headless=headless)

    buffer = []
    consecutive_errors = 0
    total = len(df)
    log(f"🚀 scraping {total} pages (per-URL cap {HARD_CAP_SEC}s)...")

    for i, row in df.iterrows():
        try:
            url = row["url"]
            if not url_has_required_query(url):
                log(f"[{i+1}/{total}] {url}\n  ⚠️ malformed URL (list_hnd/season_hnd) — skipping")
                continue

            meta = {
                "division": row["division"],
                "school": row["school"],
                "gender": row["gender"],
                "year": row["year"],
                "season": row["season"],
            }

            # Recycle periodically to avoid long-lived-session issues
            if (i > 0) and (i % RECYCLE_EVERY == 0):
                drv = recycle_driver(drv, headless=headless)
                consecutive_errors = 0
                time.sleep(0.6 + random.uniform(0.1, 0.4))

            log(f"[{i+1}/{total}] {url}")
            try:
                rows_out = scrape_one_url(drv, url, meta)
                if rows_out:
                    buffer.extend(rows_out)
                # success or benign skip resets error streak
                consecutive_errors = 0
            except WebDriverException as e:
                consecutive_errors += 1
                log(f"  ❌ WebDriver error: {e}; consecutive={consecutive_errors}")
                if consecutive_errors >= MAX_CONSEC_ERRORS:
                    drv = recycle_driver(drv, headless=headless)
                    consecutive_errors = 0
                    time.sleep(1.2 + random.uniform(0.2, 0.6))

            # Periodic checkpoint
            if (i + 1) % CHECKPOINT_EVERY == 0 and buffer:
                ckpt = os.path.splitext(args.output)[0] + f".ckpt_{i+1}.npy"
                save_buffer_as_npy(buffer, ckpt)
                log(f"  🧭 checkpoint saved at page {i+1}")

            # politeness with jitter
            time.sleep(POLITE_DELAY_BASE + random.uniform(0.15, 0.35))

        except Exception as e:
            # Catch-all so unexpected exceptions don't silently kill the script
            consecutive_errors += 1
            log(f"  💥 Unexpected error on index {i}: {e}")
            log(traceback.format_exc())
            if consecutive_errors >= MAX_CONSEC_ERRORS:
                drv = recycle_driver(drv, headless=headless)
                consecutive_errors = 0
                time.sleep(1.0 + random.uniform(0.2, 0.5))
            # continue to the next URL

    try:
        drv.quit()
    except Exception:
        pass

    # Final save
    out_path = args.output
    save_buffer_as_npy(buffer, out_path)
    log("✅ done.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Top-level safety net so you see the error in console + log
        log(f"FATAL: {e}")
        log(traceback.format_exc())
        raise
