"""
Agent track — finds analyst-type roles whose TITLE doesn't say so.

Runs alongside check_jobs.py, never instead of it. The keyword track
keeps doing what it does; this one picks up what the keyword track
structurally cannot see: "Reporting Specialist", "Program Coordinator",
"Business Operations Associate" and similar.

Pipeline, cheapest step first:
  1. Pull the FULL board per company (no search_keywords pre-filter)
  2. Drop anything already in agent_seen.json          [free]
  3. Drop anything matching agent_filters.json exclude [free]
  4. Stage 1: judge TITLES ONLY, 50 per API call       [~10 calls]
  5. Fetch the JD for survivors only                   [the slow part]
  6. Stage 2: judge full JDs, 5 per API call           [~10 calls]
  7. Telegram the winners

State lives in agent_seen.json so the two tracks never collide.
"""

import json
import os
import re
import sys
import time

import requests
from bs4 import BeautifulSoup

# Reuse every fetcher, parser and helper the keyword track already has.
from check_jobs import (
    HEADERS,
    fetch_company_jobs,
    get_oracle_cloud_parts,
    get_workday_parts,
    load_json,
    save_json,
    send_telegram,
    with_params,
)

COMPANIES_FILE = "companies.json"
AGENT_SEEN_FILE = "agent_seen.json"
AGENT_FILTERS_FILE = "agent_filters.json"

# Gemini free tier: 10 requests/minute. 7s spacing keeps a safe margin.
GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
CALL_SPACING_SECONDS = 7

TITLES_PER_CALL = 50
JDS_PER_CALL = 5

# Hard ceilings so one bad run can't burn the daily quota or spam you.
MAX_STAGE2_JOBS = 120
MAX_ALERTS_PER_RUN = 25

# Per-company fetch ceiling. The agent pulls whole boards, so without
# this a single 19,000-posting employer dominates the run.
MAX_JOBS_PER_COMPANY = 1500

MAX_YEARS_EXPERIENCE = 3.5


# ═══════════════════════════════════════════════════════════════
# Cheap filtering — everything here is free, runs before any API call
# ═══════════════════════════════════════════════════════════════

def load_agent_filters():
    raw = load_json(AGENT_FILTERS_FILE, {})
    agent = raw.get("agent", {})
    terms = [t.lower() for t in agent.get("exclude", [])]
    # A trailing * means prefix match ("phlebotom*" -> phlebotomist).
    # Everything else is anchored at BOTH ends, which matters more than
    # it looks: "\bintern" alone happily matches "Internal Audit", and
    # "\bsr" matches "disaster". Both are roles we want to keep.
    pattern = None
    if terms:
        parts = []
        for t in terms:
            if t.endswith("*"):
                parts.append(re.escape(t[:-1]))
            else:
                parts.append(re.escape(t) + r"\b")
        pattern = re.compile(r"\b(" + "|".join(parts) + r")", re.IGNORECASE)
    return {
        "enabled": agent.get("enabled", True),
        "pattern": pattern,
        "profile": agent.get("profile", ""),
    }


def excluded(title, pattern):
    if not pattern:
        return False
    return bool(pattern.search(title or ""))


# ═══════════════════════════════════════════════════════════════
# JD fetching — one function per platform, mirrors check_jobs.py
# ═══════════════════════════════════════════════════════════════

def html_to_text(html, limit=6000):
    """Strip markup and collapse whitespace. Truncated to keep tokens sane."""
    if not html:
        return ""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer"]):
        tag.decompose()
    text = re.sub(r"\s+", " ", soup.get_text(" ", strip=True))
    return text[:limit]


def jd_workday(company_url, job_id):
    """
    job_id is Workday's externalPath ("/job/Location/Title_R123").
    The cxs endpoint serves the posting detail at the same path.
    """
    parts = get_workday_parts(company_url)
    if not parts:
        return ""
    tenant, dc, site = parts
    api = f"https://{tenant}.{dc}.myworkdayjobs.com/wday/cxs/{tenant}/{site}{job_id}"
    r = requests.get(api, headers={**HEADERS, "Accept": "application/json"}, timeout=30)
    r.raise_for_status()
    info = r.json().get("jobPostingInfo", {})
    return html_to_text(info.get("jobDescription", ""))


def jd_oracle(company_url, job_id):
    parts = get_oracle_cloud_parts(company_url)
    if not parts:
        return ""
    host, site_number = parts
    api = f"https://{host}/hcmRestApi/resources/latest/recruitingCEJobRequisitionDetails"
    params = {
        "expand": "all",
        "onlyData": "true",
        "finder": f"ById;Id={job_id},siteNumber={site_number}",
    }
    r = requests.get(api, params=params, headers=HEADERS, timeout=30)
    r.raise_for_status()
    items = r.json().get("items", [])
    if not items:
        return ""
    return html_to_text(items[0].get("ExternalDescriptionStr", ""))


def jd_generic(job_url):
    """
    Fallback for iCIMS, Greenhouse, Lever, Ashby and custom sites:
    fetch the posting page and strip it. Works wherever the JD is in
    the raw HTML; returns little or nothing on JavaScript-rendered
    pages, which is the same limitation the fallback scraper has.
    """
    r = requests.get(job_url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return html_to_text(r.text)


def fetch_jd(company, job_id, job):
    url = company["url"]
    try:
        if get_workday_parts(url):
            return jd_workday(url, job_id)
        if get_oracle_cloud_parts(url):
            return jd_oracle(url, job_id)
        return jd_generic(job["url"])
    except Exception as e:
        print(f"    [ERROR] JD fetch failed for {job['title']!r}: {e}", file=sys.stderr)
        return ""


# ═══════════════════════════════════════════════════════════════
# Gemini — REST, no SDK dependency
# ═══════════════════════════════════════════════════════════════

def gemini(prompt, api_key, retries=2):
    """
    responseMimeType=application/json makes the model return parseable
    JSON rather than prose wrapped in code fences, which removes the
    single most common failure mode. Retries once on 429 or a parse
    failure; a lost batch costs a few postings, not the run.
    """
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "temperature": 0,
        },
    }
    for attempt in range(retries + 1):
        try:
            r = requests.post(
                GEMINI_URL,
                params={"key": api_key},
                json=body,
                timeout=120,
            )
            if r.status_code == 429:
                wait = 30 * (2 ** attempt)
                print(f"    [WARN] Rate limited, waiting {wait}s", file=sys.stderr)
                time.sleep(wait)
                continue
            r.raise_for_status()
            text = r.json()["candidates"][0]["content"]["parts"][0]["text"]
            return json.loads(text)
        except Exception as e:
            if attempt == retries:
                print(f"    [ERROR] Gemini call failed: {e}", file=sys.stderr)
                return None
            time.sleep(5)
    return None


STAGE1_PROMPT = """You are screening job titles for a data/business analyst job search.

For each title below, decide if the role could PLAUSIBLY involve data analysis
or business analysis work as its core day-to-day, even though the title does
not contain the word "analyst".

Say yes to titles that often hide analytical work, for example:
Reporting Specialist, Business Operations Associate, Program Coordinator,
Insights Associate, Strategy Associate, Revenue Operations, MIS Executive,
Performance Improvement Specialist, Decision Support, Data Steward.

Say no to titles that are clearly something else: engineering build work,
software development, sales quota roles, HR generalist, admin support,
project management with no analytical component, hands-on technical trades.

Be generous at this stage. A wrong "yes" is cheap because the full job
description is read next. A wrong "no" means the role is lost forever.

Titles:
{titles}

Respond ONLY with JSON: {{"keep": [list of the integer ids you would keep]}}"""


STAGE2_PROMPT = """You are evaluating job postings for a data/business analyst job search.

For EACH posting below decide two things independently:

1. is_analytical: Is the CORE day-to-day work data analysis or business
   analysis? Look for things like: writing SQL or queries, building
   dashboards or reports, analysing datasets, requirements gathering,
   process or data modelling, KPI and metrics work, turning data into
   recommendations. Ignore the job title entirely; judge the described work.
   Say false if analysis is only a minor or occasional part of the role.

2. years_required: The minimum years of professional experience the posting
   asks for, as a number. Use 0 if it is entry level, a new graduate role,
   or no experience requirement is stated. If a range is given, use the
   lower bound.

Postings:
{postings}

Respond ONLY with JSON:
{{"results": [{{"id": <int>, "is_analytical": <bool>, "years_required": <number>, "reason": "<one short sentence>"}}]}}"""


def stage1_titles(candidates, api_key):
    """Batch titles, keep the plausible ones. No JD fetching yet."""
    kept = []
    batches = [candidates[i:i + TITLES_PER_CALL]
               for i in range(0, len(candidates), TITLES_PER_CALL)]
    for n, batch in enumerate(batches, 1):
        listing = "\n".join(f"{i}. {c['job']['title']}" for i, c in enumerate(batch))
        print(f"  Stage 1 batch {n}/{len(batches)} ({len(batch)} titles)")
        result = gemini(STAGE1_PROMPT.format(titles=listing), api_key)
        time.sleep(CALL_SPACING_SECONDS)
        if not result:
            continue
        for idx in result.get("keep", []):
            if isinstance(idx, int) and 0 <= idx < len(batch):
                kept.append(batch[idx])
    return kept


def stage2_jds(scored, api_key):
    """Judge full JDs in small batches. Returns the ones that pass."""
    winners = []
    batches = [scored[i:i + JDS_PER_CALL]
               for i in range(0, len(scored), JDS_PER_CALL)]
    for n, batch in enumerate(batches, 1):
        blocks = []
        for i, c in enumerate(batch):
            blocks.append(
                f"--- POSTING {i} ---\n"
                f"Title: {c['job']['title']}\n"
                f"Company: {c['company']['name']}\n"
                f"Description: {c['jd']}\n"
            )
        print(f"  Stage 2 batch {n}/{len(batches)} ({len(batch)} JDs)")
        result = gemini(STAGE2_PROMPT.format(postings="\n".join(blocks)), api_key)
        time.sleep(CALL_SPACING_SECONDS)
        if not result:
            continue
        for row in result.get("results", []):
            idx = row.get("id")
            if not isinstance(idx, int) or not (0 <= idx < len(batch)):
                continue
            if not row.get("is_analytical"):
                continue
            years = row.get("years_required")
            if isinstance(years, (int, float)) and years > MAX_YEARS_EXPERIENCE:
                continue
            c = dict(batch[idx])
            c["reason"] = row.get("reason", "")
            c["years"] = years
            winners.append(c)
    return winners


# ═══════════════════════════════════════════════════════════════

def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("[ERROR] GEMINI_API_KEY not set.", file=sys.stderr)
        sys.exit(1)

    companies = load_json(COMPANIES_FILE, [])
    seen = load_json(AGENT_SEEN_FILE, {})
    cfg = load_agent_filters()

    if not cfg["enabled"]:
        print("Agent track disabled in agent_filters.json.")
        return

    candidates = []
    first_run_companies = []

    for company in companies:
        name = company["name"]
        print(f"Checking {name}...")

        # The agent wants the WHOLE board, not the keyword slice. Strip any
        # ?q= baked into companies.json (CVS and CEVA have one) and pass
        # empty search_keywords so the Workday/Oracle fetchers do one
        # unfiltered pass instead of one pass per keyword.
        wide = dict(company)
        wide["url"] = with_params(company["url"], q=None)

        jobs = fetch_company_jobs(wide, {
            "search_keywords": [],
            "location_enabled": False,
            "roles_enabled": False,
        })
        if jobs is None:
            continue

        if len(jobs) > MAX_JOBS_PER_COMPANY:
            print(f"  [WARN] {len(jobs)} postings, capping at {MAX_JOBS_PER_COMPANY}",
                  file=sys.stderr)
            jobs = dict(list(jobs.items())[:MAX_JOBS_PER_COMPANY])

        prev = set(seen.get(name, {}).keys())
        is_first = name not in seen

        new_ids = [j for j in jobs if j not in prev]
        kept_after_exclude = 0
        for job_id in new_ids:
            job = jobs[job_id]
            if excluded(job.get("title"), cfg["pattern"]):
                continue
            kept_after_exclude += 1
            if not is_first:
                candidates.append({
                    "company": company, "job_id": job_id, "job": job,
                })

        print(f"  {len(jobs)} fetched, {len(new_ids)} new, "
              f"{kept_after_exclude} past exclusions")

        if is_first:
            first_run_companies.append(name)

        seen[name] = jobs

    # Baseline saved before any API spend, so a crash mid-judge doesn't
    # replay the whole board next run.
    save_json(AGENT_SEEN_FILE, seen)

    if first_run_companies:
        print(f"\nFirst run for {len(first_run_companies)} company(s) — "
              f"baselined, no alerts sent for those.")

    if not candidates:
        print("\nNothing new to judge.")
        return

    print(f"\n{len(candidates)} candidate(s) into stage 1.")
    survivors = stage1_titles(candidates, api_key)
    print(f"{len(survivors)} survived stage 1.")

    if len(survivors) > MAX_STAGE2_JOBS:
        print(f"[WARN] Capping stage 2 at {MAX_STAGE2_JOBS}.", file=sys.stderr)
        survivors = survivors[:MAX_STAGE2_JOBS]

    with_jds = []
    for c in survivors:
        jd = fetch_jd(c["company"], c["job_id"], c["job"])
        if len(jd) < 200:
            # Too short to judge — almost always a JS-rendered page or a
            # blocked fetch, not a genuinely tiny posting.
            print(f"    [SKIP] No usable JD: {c['job']['title']}")
            continue
        c["jd"] = jd
        with_jds.append(c)
    print(f"{len(with_jds)} JD(s) fetched.")

    winners = stage2_jds(with_jds, api_key)
    print(f"\n{len(winners)} posting(s) passed.")

    for c in winners[:MAX_ALERTS_PER_RUN]:
        years = c.get("years")
        yrs = f" · {years}y+" if isinstance(years, (int, float)) and years else ""
        where = f"\n📍 {c['job']['location']}" if c["job"].get("location") else ""
        send_telegram(
            f"🤖 <b>{c['company']['name']}</b>{yrs}\n"
            f"{c['job']['title']}{where}\n"
            f"<i>{c.get('reason', '')}</i>\n"
            f"{c['job']['url']}"
        )

    if len(winners) > MAX_ALERTS_PER_RUN:
        send_telegram(
            f"⚠️ {len(winners) - MAX_ALERTS_PER_RUN} more agent matches this run, "
            f"not sent individually. Full list is in the Actions log."
        )


if __name__ == "__main__":
    main()
