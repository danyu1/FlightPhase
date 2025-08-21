# -*- coding: utf-8 -*-
# Build a CSV of all NCAA DI/DII/DIII schools × (Men/Women) × seasons
# with one column containing the valid All-Performances URL.

import re
import time
import pandas as pd
import requests
from bs4 import BeautifulSoup

# -----------------------------
# Config
# -----------------------------
DIVISION_PAGES = {
    "NCAA Division I": "https://www.tfrrs.org/leagues/49.html",
    "NCAA Division II": "https://www.tfrrs.org/leagues/50.html",
    "NCAA Division III": "https://www.tfrrs.org/leagues/51.html",
}

# Your season handles:
SEASONS = [
    (2025, "Outdoors", 5027, 681),
    (2025, "Indoors",  4874, 661),
    (2024, "Outdoors", 4541, 645),
    (2024, "Indoors",  4466, 627),
    (2023, "Outdoors", 4153, 608),
    (2023, "Indoors",  3909, 584),
    (2022, "Outdoors", 3730, 568),
    (2022, "Indoors",  3501, 548),
    (2021, "Outdoors", 3200, 530),
    (2021, "Indoors",  3167, 519),
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}
REQUEST_TIMEOUT = 20
POLITE_DELAY = 0.3  # seconds between requests

OUT_CSV = "tfrrs_all_ncaa_urls.csv"

# -----------------------------
# Helpers
# -----------------------------
def get_soup(url: str) -> BeautifulSoup:
    r = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return BeautifulSoup(r.text, "html.parser")

def extract_team_links(division_name: str, division_url: str):
    """
    Return list of dicts:
    {
      'division': 'NCAA Division I',
      'team_name': 'Duke',
      'gender': 'm' or 'f',
      'team_url': 'https://www.tfrrs.org/teams/tf/NC_college_m_Duke.html'
    }
    """
    soup = get_soup(division_url)
    out = []

    # The division page has two columns of team links (Men's and Women's).
    # We can simply grab all anchors under the "TEAMS" section, then infer gender from href.
    for a in soup.find_all("a", href=True):
        href = a["href"]
        text = a.get_text(" ", strip=True)
        if not text:
            continue
        # Only take links that go to /teams/tf/STATE_college_g_slug.html
        if not href.startswith("https://") and href.startswith("/"):
            href = "https://www.tfrrs.org" + href

        if "/teams/tf/" in href:
            # Infer gender from path segment: "_m_" or "_f_" (sometimes "_w_" appears; treat as women)
            gender = None
            if "_m_" in href:
                gender = "m"
            elif "_f_" in href or "_w_" in href:
                gender = "f"
            else:
                # skip cross-country or odd links
                continue

            out.append({
                "division": division_name,
                "team_name": text,
                "gender": gender,
                "team_url": href
            })

    # Deduplicate by (division, team_url, gender)
    dedup = {(d["division"], d["team_url"], d["gender"]): d for d in out}
    return list(dedup.values())

def team_to_allperformances_url(team_url: str, list_hnd: int, season_hnd: int) -> str:
    """
    Transform:
      https://www.tfrrs.org/teams/tf/NC_college_m_Duke.html
    ->  https://www.tfrrs.org/all_performances/NC_college_m_Duke.html?list_hnd=5027&season_hnd=681
    """
    # extract the 'STATE_college_g_slug.html' tail
    m = re.search(r"/teams/tf/([A-Z]{2}_college_[mwf]_[^/]+\.html)$", team_url)
    if not m:
        return ""
    tail = m.group(1)
    return f"https://www.tfrrs.org/all_performances/{tail}?list_hnd={list_hnd}&season_hnd={season_hnd}"

# -----------------------------
# Main
# -----------------------------
def main():
    all_team_rows = []
    for div_name, div_url in DIVISION_PAGES.items():
        try:
            teams = extract_team_links(div_name, div_url)
            all_team_rows.extend(teams)
            time.sleep(POLITE_DELAY)
        except Exception as e:
            print(f"Warning: failed to parse {div_name} ({div_url}): {e}")

    # Build rows for every season
    rows = []
    for t in all_team_rows:
        for (year, season_label, list_hnd, season_hnd) in SEASONS:
            url = team_to_allperformances_url(t["team_url"], list_hnd, season_hnd)
            if not url:
                continue
            rows.append({
                "division": t["division"],
                "school": t["team_name"],
                "gender": "Men" if t["gender"] == "m" else "Women",
                "year": year,
                "season": season_label,
                "url": url,
            })

    df = pd.DataFrame(rows).sort_values(
        ["division", "school", "gender", "year", "season"],
        ascending=[True, True, True, False, True]
    ).reset_index(drop=True)

    df.to_csv(OUT_CSV, index=False)
    print(f"✅ Wrote {len(df):,} rows to {OUT_CSV}")
    # Optional preview
    print(df.head(12).to_string(index=False))

if __name__ == "__main__":
    main()
