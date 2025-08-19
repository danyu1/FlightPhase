from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import pandas as pd

options = webdriver.ChromeOptions()
options.add_argument("--headless")
options.add_argument("--disable-gpu")
options.add_argument("--no-sandbox")

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

url = "https://www.tfrrs.org/all_performances/IL_college_m_U_of_Chicago.html?list_hnd=5027&season_hnd=681"
driver.get(url)

# ✅ Wait until at least one event block is visible
try:
    WebDriverWait(driver, 15).until(
        EC.presence_of_element_located((By.CLASS_NAME, "performance-list-row"))
    )
    print("✅ Content loaded.")
except:
    print("❌ Timeout waiting for content.")
    driver.quit()
    exit()

# ✅ Now that content is loaded, parse the HTML
soup = BeautifulSoup(driver.page_source, "html.parser")
driver.quit()

print("✅ Page fully rendered. Beginning scrape...")

all_data = []
# Grab all rows directly (skipping event container logic)
rows = soup.find_all("div", class_="performance-list-row")

print(f"Found {len(rows)} performance rows")

for row in rows:
    try:
        # Climb up the DOM to find the event name
        parent_event_div = row.find_previous("div", class_="col-lg-12")
        event_name = parent_event_div.find("h3").get_text(strip=True) if parent_event_div else "Unknown Event"

        athlete = row.find("div", class_="col-athlete").get_text(strip=True)
        year = row.find("div", attrs={"data-label": "Year"}).get_text(strip=True)

        time_div = row.find("div", attrs={"data-label": "Time"})
        mark_div = row.find("div", attrs={"data-label": "Mark"})
        mark_or_time = time_div.get_text(strip=True) if time_div else mark_div.get_text(strip=True)

        wind_div = row.find("div", attrs={"data-label": "Wind"})
        wind = wind_div.get_text(strip=True) if wind_div else ""

        meet = row.find("div", class_="col-meet").get_text(strip=True)
        meet_date = row.find("div", attrs={"data-label": "Meet Date"}).get_text(strip=True)

        all_data.append({
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
