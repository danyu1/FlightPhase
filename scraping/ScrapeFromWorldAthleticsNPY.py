#!/usr/bin/env python3
# Respectful Playwright scraper for World Athletics "all performances" pages.
# - ProxyJet rotating residential proxies via env vars
# - Concurrency + polite throttling with backoff
# - Saves NumPy structured array
# - Dumps debug HTML when no rows parsed
# - Preflight IP check through proxy

import argparse, asyncio, time, random, re, os, pathlib
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from urllib.parse import urlparse, parse_qs
from playwright.async_api import async_playwright, TimeoutError as PWTimeout

# Tunables
WAIT_TIMEOUT = 10.0          # seconds
HARD_CAP_SEC = 15.0          # seconds
POLITE_DELAY_BASE = 0.4
JITTER = (0.1, 0.3)
CHECKPOINT_EVERY = 1000

def log(msg: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

# -----------------------------
# HTML helpers
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

def parse_page(url: str, html: str, meta: dict) -> List[Dict]:
    soup = BeautifulSoup(html, "html.parser")
    gender = meta.get("gender") or infer_gender_from_url(url)
    team_from_title, gender_from_title = extract_team_and_gender_from_title(soup)
    if gender == "Unknown" and gender_from_title != "Unknown":
        gender = gender_from_title
    team_name = meta.get("school") or team_from_title
    division = meta.get("division") or extract_division(soup)

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
            year = safe_text(row.find("div", attrs={"data-label": "Year"}))
            team_cell = safe_text(row.find("div", attrs={"data-label": "Team"})) or team_name

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
            meet_date = safe_text(row.find("div", attrs={"data-label": "Meet Date"}) or
                                  row.find("div", attrs={"data-label": "Date"}))

            out.append({
                "Division": division, "School": team_name, "Gender": gender,
                "SeasonLabel": meta.get("season"), "SeasonYear": meta.get("year"),
                "ListHnd": list_hnd, "SeasonHnd": season_hnd, "SourceURL": url,
                "Team": team_cell, "Event": event_name, "Athlete": athlete,
                "ClassYear": year, "MarkOrTime": mark_or_time, "Wind": wind,
                "Meet": meet, "MeetDate": meet_date,
            })
        except Exception:
            continue
    return out

def url_has_required_query(u: str) -> bool:
    q = parse_qs(urlparse(u).query)
    return q.get("list_hnd", [""])[0].isdigit() and q.get("season_hnd", [""])[0].isdigit()

# -----------------------------
# Save to NPY
# -----------------------------
def save_buffer_as_npy(rows: List[Dict], out_path: str):
    if not rows:
        dtype = [
            ("Division","U1"),("School","U1"),("Gender","U1"),("SeasonLabel","U1"),
            ("SeasonYear","i4"),("ListHnd","i4"),("SeasonHnd","i4"),("SourceURL","U1"),
            ("Team","U1"),("Event","U1"),("Athlete","U1"),("ClassYear","U1"),
            ("MarkOrTime","U1"),("Wind","U1"),("Meet","U1"),("MeetDate","U1"),
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
    str_cols = [c for c in cols if c not in ("SeasonYear","ListHnd","SeasonHnd")]
    str_maxlens = {c: max(1, int(df[c].astype("string").str.len().max() or 1)) for c in str_cols}
    for c in ["SeasonYear","ListHnd","SeasonHnd"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(-1).astype(np.int32)
    dtype = [(c, f"<U{str_maxlens[c]}") if c in str_cols else (c,"<i4") for c in cols]
    arr = np.empty(len(df), dtype=dtype)
    for c in cols:
        if c in str_cols: arr[c] = df[c].to_numpy(dtype=object)
        else: arr[c] = df[c].to_numpy(dtype=np.int32)
    np.save(out_path, arr)
    log(f"💾 wrote NPY with {len(arr):,} rows → {out_path}")

# -----------------------------
# Job dataclass
# -----------------------------
@dataclass
class Job: idx: int; url: str; meta: dict

# -----------------------------
# Proxy from env
# -----------------------------
def _from_env_proxy() -> dict:
    s = os.getenv("PJ_PROXY", "").strip()
    if s:
        if s.endswith(":"): s = s[:-1]
        parts = s.split(":")
        if len(parts) != 4:
            raise ValueError('PJ_PROXY must be "host:port:user:pass"')
        host, port, user, pwd = parts
    else:
        host = os.getenv("PJ_HOST","").strip()
        port = os.getenv("PJ_PORT","").strip()
        user = os.getenv("PJ_USERNAME","").strip()
        pwd  = os.getenv("PJ_PASSWORD","").strip()
        if not all([host,port,user,pwd]):
            raise ValueError("Missing proxy env vars")
    proto = os.getenv("PJ_PROTO","http").lower()
    return {"server": f"{proto}://{host}:{port}", "username": user, "password": pwd}

# --------------------------
# Fetch one page
# -----------------------------
async def fetch_one(context, job: Job) -> Tuple[str,List[Dict]]:
    async def _route(route, request):
        if request.resource_type in ("image","media","font"):
            return await route.abort()
        return await route.continue_()
    page = await context.new_page()
    await context.route("**/*", _route)
    page.set_default_timeout(int(WAIT_TIMEOUT*1000))
    try:
        async def _do():
            resp = await page.goto(job.url, wait_until="domcontentloaded",
                                   timeout=int(HARD_CAP_SEC*1000))
            status = resp.status if resp else 0
            if status in (403,429): return ("FORBIDDEN",[])
            if 500 <= status < 600: return ("RETRY",[])
            try:
                await page.wait_for_selector(
                    ".performance-list-row, .panel-heading h3.panel-title",
                    timeout=int(WAIT_TIMEOUT*1000)
                )
            except PWTimeout: pass
            html = await page.content()
            rows = parse_page(job.url, html, job.meta)
            if not rows:
                dbg = f"debug_empty_{job.idx+1}.html"
                with open(dbg,"w",encoding="utf-8") as f: f.write(html)
                log(f"🧐 no rows parsed, saved {dbg}")
            return ("OK", rows)
        tag, rows = await asyncio.wait_for(_do(), timeout=HARD_CAP_SEC+1.0)
        return (tag, rows)
    except (PWTimeout, asyncio.TimeoutError):
        return ("TIMEOUT",[])
    except Exception as e:
        log(f"💥 error {e}")
        return ("ERROR",[])
    finally:
        try: await page.close()
        except Exception: pass

# -----------------------------
# Orchestrator
# -----------------------------
async def run(input_csv, output_npy, concurrency, limit,
              headless, storage_state, short_backoff,
              cooldown_after, long_cooldown):
    df = pd.read_csv(input_csv)
    df.columns=[c.lower() for c in df.columns]
    required=["division","school","gender","year","season","url"]
    for c in required:
        if c not in df.columns: raise ValueError(f"Missing column {c}")
    if limit: df=df.head(limit).copy()
    jobs=[Job(i,row["url"],{"division":row["division"],"school":row["school"],
         "gender":row["gender"],"year":row["year"],"season":row["season"]})
          for i,row in df.iterrows() if url_has_required_query(str(row["url"]))]

    results=[]; processed=0
    state_path=pathlib.Path(storage_state)
    proxy=_from_env_proxy()
    log(f"🧩 proxy in use → {proxy['server']} (user={proxy['username']})")

    async with async_playwright() as pw:
        try:
            browser=await pw.chromium.launch(headless=headless,proxy=proxy,
                args=["--disable-3d-apis","--disable-webgl","--disable-webgl2",
                      "--disable-speech-api","--mute-audio",
                      "--disable-features=LiveCaption,OnDeviceSpeechRecognition,OptimizationHints",
                      "--blink-settings=imagesEnabled=false"])
        except Exception as e:
            log(f"❌ failed to launch Chromium with proxy: {e}")
            return

        # Preflight IP check
        try:
            ctx_check=await browser.new_context()
            page_check=await ctx_check.new_page()
            await page_check.goto("https://httpbin.org/ip",timeout=15000)
            ip_json=await page_check.text_content("pre")
            log(f"🌐 external IP via proxy: {ip_json}")
            await ctx_check.close()
        except Exception as e:
            log(f"❓ IP check failed: {e}")

        contexts=[await browser.new_context(viewport={"width":1280,"height":900},
                    storage_state=str(state_path) if state_path.exists() else None)
                  for _ in range(max(1,concurrency))]

        q=asyncio.Queue()
        for j in jobs: q.put_nowait(j)

        async def worker(wid,ctx):
            nonlocal results,processed
            log(f"🚀 worker {wid} start")
            forbidden_hits=0; sleep_mult=1.0
            while not q.empty():
                try: job=q.get_nowait()
                except asyncio.QueueEmpty: break
                log(f"[W{wid}] {job.idx+1}/{len(df)} {job.url}")
                tag,rows=await fetch_one(ctx,job)
                if tag=="OK":
                    if rows: results.extend(rows)
                    forbidden_hits=max(0,forbidden_hits-1)
                    sleep_mult=max(1.0,sleep_mult*0.9)
                elif tag=="FORBIDDEN":
                    forbidden_hits+=1; sleep_mult=min(5.0,sleep_mult*1.5)
                    if forbidden_hits>=cooldown_after:
                        log(f"[W{wid}] ⚠️ many 403/429 → cooling {long_cooldown}s")
                        await asyncio.sleep(long_cooldown); forbidden_hits=0
                    else:
                        log(f"[W{wid}] 403/429 short backoff {short_backoff}s")
                        await asyncio.sleep(short_backoff)
                elif tag in ("RETRY","TIMEOUT","ERROR"):
                    sleep_mult=min(5.0,sleep_mult*1.1)
                    await asyncio.sleep(2+random.uniform(0,2))
                processed+=1
                if CHECKPOINT_EVERY and processed%CHECKPOINT_EVERY==0:
                    tmp=os.path.splitext(output_npy)[0]+f".ckpt_{processed}.npy"
                    save_buffer_as_npy(results,tmp); log(f"🧭 checkpoint {tmp}")
                await asyncio.sleep((POLITE_DELAY_BASE+random.uniform(*JITTER))*sleep_mult)
            log(f"✅ worker {wid} done")

        await asyncio.gather(*(worker(i+1,contexts[i]) for i in range(len(contexts))))
        await browser.close()

    save_buffer_as_npy(results,output_npy)
    log(f"🎯 scraped {len(results):,} rows → {output_npy}")

# -----------------------------
# CLI
# -----------------------------
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("-i","--input",default="tfrrs_all_ncaa_urls.csv")
    ap.add_argument("-o","--output",default="tfrrs_performances_fast.npy")
    ap.add_argument("-c","--concurrency",type=int,default=3)
    ap.add_argument("--limit",type=int,default=None)
    ap.add_argument("--no-headless",action="store_true")
    ap.add_argument("--state",default="tfrrs_storage_state.json")
    ap.add_argument("--short-backoff",type=int,default=12)
    ap.add_argument("--cooldown-after",type=int,default=4)
    ap.add_argument("--long-cooldown",type=int,default=180)
    args=ap.parse_args()
    headless=not args.no_headless
    asyncio.run(run(args.input,args.output,args.concurrency,args.limit,
                    headless,args.state,args.short_backoff,
                    args.cooldown_after,args.long_cooldown))

if __name__=="__main__": main()
