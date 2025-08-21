from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import pandas as pd
import re

options = webdriver.ChromeOptions()
options.add_argument("--headless")
options.add_argument("--disable-gpu")
options.add_argument("--no-sandbox")

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

url = "https://www.tfrrs.org/all_performances/IL_college_m_U_of_Chicago.html?list_hnd=5027&season_hnd=681"
driver.get(url)

# Wait for performance rows to render
try:
    WebDriverWait(driver, 15).until(
        EC.presence_of_element_located((By.CLASS_NAME, "performance-list-row"))
    )
    print("✅ Content loaded.")
except Exception:
    print("❌ Timeout waiting for content.")
    driver.quit()
    raise

# Parse the fully rendered page
soup = BeautifulSoup(driver.page_source, "html.parser")
driver.quit()

print("✅ Page fully rendered. Beginning scrape...")

# -------- Helpers to extract Team / Gender / Division --------
def infer_gender_from_url(u: str) -> str:
    if "_m_" in u:
        return "Men"
    if "_f_" in u or "_w_" in u:
        return "Women"
    return "Unknown"

def extract_team_and_gender_from_title(soup_obj) -> tuple[str, str]:
    """
    Title looks like: 'U. of Chicago Men's Track & Field All Performances & Rankings'
    We'll try to parse team name and (as a backup) gender from it.
    """
    title_h3 = soup_obj.select_one(".panel-heading h3.panel-title")
    if title_h3:
        title_text = title_h3.get_text(strip=True)
        # Try to split by " Men's" or " Women's"
        m = re.split(r"\s+(Men's|Women's)\b", title_text, maxsplit=1)
        if len(m) >= 2:
            team_name = m[0].strip()
            gender_word = m[1]
            gender = "Men" if gender_word.startswith("Men") else "Women"
            return team_name, gender
        # If that failed, fall back to trimming the trailing text
        # e.g., remove trailing "Track & Field All Performances & Rankings"
        team_name = re.sub(r"\s*Track\s*&\s*Field.*$", "", title_text).strip()
        return team_name, "Unknown"
    return "Unknown Team", "Unknown"

def extract_division(soup_obj) -> str:
    """
    Try to find 'NCAA Division I/II/III', 'NCAA D1/D2/D3', 'NAIA', or 'NJCAA'
    anywhere in the rendered text.
    """
    full_text = soup_obj.get_text(" ", strip=True)
    # Common patterns
    patt = re.compile(
        r"(NCAA\s*Division\s*(I{1,3})|NCAA\s*D[123]|NAIA|NJCAA)",
        re.IGNORECASE
    )
    m = patt.search(full_text)
    if m:
        # Normalize a bit
        found = m.group(0).upper().replace("  ", " ").strip()
        # Standardize to nice labels
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

# Infer gender from URL first; refine with title if available
gender = infer_gender_from_url(url)
team_from_title, gender_from_title = extract_team_and_gender_from_title(soup)
if gender == "Unknown" and gender_from_title != "Unknown":
    gender = gender_from_title

team_name = team_from_title
division = extract_division(soup)

# -------- Scrape rows --------
all_data = []
rows = soup.find_all("div", class_="performance-list-row")
print(f"Found {len(rows)} performance rows")

for row in rows:
    try:
        # Event name (walk up to event header)
        parent_event_div = row.find_previous("div", class_="col-lg-12")
        event_name = parent_event_div.find("h3").get_text(strip=True) if parent_event_div else "Unknown Event"

        athlete = row.find("div", class_="col-athlete").get_text(strip=True)
        year = row.find("div", attrs={"data-label": "Year"})
        year = year.get_text(strip=True) if year else ""

        # Team (sometimes present as a column on some pages; if missing, use page-level team)
        team_div = row.find("div", attrs={"data-label": "Team"})
        team_cell = team_div.get_text(strip=True) if team_div else team_name

        time_div = row.find("div", attrs={"data-label": "Time"})
        mark_div = row.find("div", attrs={"data-label": "Mark"})
        mark_or_time = ""
        if time_div:
            mark_or_time = time_div.get_text(strip=True)
        elif mark_div:
            mark_or_time = mark_div.get_text(strip=True)

        wind_div = row.find("div", attrs={"data-label": "Wind"})
        wind = wind_div.get_text(strip=True) if wind_div else ""

        meet = row.find("div", class_="col-meet")
        meet = meet.get_text(strip=True) if meet else ""

        meet_date = row.find("div", attrs={"data-label": "Meet Date"})
        meet_date = meet_date.get_text(strip=True) if meet_date else ""

        all_data.append({
            "Team": team_cell,          # per-row if present; else page team
            "Division": division,       # page-level inference
            "Gender": gender,           # URL/title inference
            "Event": event_name,
            "Athlete": athlete,
            "Year": year,
            "Mark/Time": mark_or_time,
            "Wind": wind,
            "Meet": meet,
            "Date": meet_date
        })
    except Exception as e:
        print(f"⚠️ Skipping row due to error: {e}")
        continue

df = pd.DataFrame(all_data)
df.to_csv("tfrrs_performances.csv", index=False)
print("✅ Scrape complete. Here's a preview:")
print(df.head())
