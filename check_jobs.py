"""
Job Alert Bot — checks company career pages for new postings.

Supported platforms (reliable, uses structured API):
  - Greenhouse  (boards-api.greenhouse.io)
  - Lever       (api.lever.co)
  - Workday     (*.myworkdayjobs.com, *.myworkdaysite.com)
  - Oracle Cloud / Fusion HCM (*.oraclecloud.com)
  - iCIMS       (*.icims.com)
  - Ashby       (jobs.ashbyhq.com)
  - Amazon      (amazon.jobs)

Also supported, but only when requested explicitly with
"type": "sitemap" in companies.json:
  - XML job sitemaps — for JS-rendered sites that still publish a
    sitemap of their postings (e.g. careers.flowserve.com).

Everything else falls back to an HTML scraper with strict junk
filtering. The fallback is better than nothing but will miss jobs
on JavaScript-rendered sites and may occasionally misidentify
non-job links. The Actions log prints which method was used for
each company so you can tell at a glance what's reliable vs not.
"""

import json
import os
import re
import sys
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse

import requests
from bs4 import BeautifulSoup

COMPANIES_FILE = "companies.json"
SEEN_FILE = "seen_jobs.json"
FILTERS_FILE = "filters.json"

# Ceiling on Telegram messages per run. A company changing its URL, or a
# fetcher that starts seeing postings it previously truncated, can turn a
# routine run into thousands of "new" jobs — enough for Telegram to
# rate-limit the bot. Past this we send one summary instead; the full list
# is always in the Actions log, and state still updates so the next run
# is back to normal.
MAX_ALERTS_PER_RUN = 50
HEADERS = {
    # Keep this reasonably current. iCIMS (and likely others) reject
    # noticeably stale User-Agents with a bare 405 Method Not Allowed,
    # which looks like a broken endpoint rather than a blocked client.
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    # Deliberately no Accept-Encoding: requests advertises exactly what it
    # can decode. Hardcoding "br" without the brotli package installed makes
    # servers send brotli that we then hand to the parser as binary garbage.
    "Connection": "keep-alive",
}

# ── Known junk phrases the fallback scraper should always ignore ──
JUNK_TITLES = {
    "apply now", "see details", "see jobs", "view job", "view jobs",
    "view all jobs", "view job openings", "view all", "search jobs",
    "search", "log in", "log back in!", "sign in", "sign up",
    "job search", "saved jobs", "saved", "careers", "careers home",
    "careers overview", "welcome page", "read more", "learn more",
    "explore", "explore options", "explore benefits", "explore all opportunities",
    "join our talent community", "feel the difference", "ai at work",
    "here", "english", "back", "next", "previous", "home",
    "about us", "about", "contact", "contact us", "privacy", "privacy policy",
    "terms", "terms of use", "cookie policy", "sitemap", "faq",
    "manage application", "your career", "why work here",
    "job categories", "job category", "all categories", "all locations",
    "clear filters", "reset", "subscribe", "newsletter",
    "account security", "application status", "my applications",
    "india jobs", "job alerts", "job alert", "set up job alerts",
    "sign up for job alerts", "create alert", "email me jobs",
    "university & early career", "experienced professionals",
    "our satellite locations", "working at",
}


def load_json(path, default):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return default


def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def with_params(url, **overrides):
    """Return url with the given query params set (added or replaced)."""
    parts = urlparse(url)
    query = parse_qs(parts.query, keep_blank_values=True)
    for key, value in overrides.items():
        if value is None:
            query.pop(key, None)
        else:
            query[key] = [str(value)]
    return urlunparse(parts._replace(query=urlencode(query, doseq=True)))


# ═══════════════════════════════════════════════════════════════
# Filtering — location and role, applied to every company alike
# ═══════════════════════════════════════════════════════════════

US_STATES = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR",
    "california": "CA", "colorado": "CO", "connecticut": "CT", "delaware": "DE",
    "florida": "FL", "georgia": "GA", "hawaii": "HI", "idaho": "ID",
    "illinois": "IL", "indiana": "IN", "iowa": "IA", "kansas": "KS",
    "kentucky": "KY", "louisiana": "LA", "maine": "ME", "maryland": "MD",
    "massachusetts": "MA", "michigan": "MI", "minnesota": "MN",
    "mississippi": "MS", "missouri": "MO", "montana": "MT", "nebraska": "NE",
    "nevada": "NV", "new hampshire": "NH", "new jersey": "NJ",
    "new mexico": "NM", "new york": "NY", "north carolina": "NC",
    "north dakota": "ND", "ohio": "OH", "oklahoma": "OK", "oregon": "OR",
    "pennsylvania": "PA", "rhode island": "RI", "south carolina": "SC",
    "south dakota": "SD", "tennessee": "TN", "texas": "TX", "utah": "UT",
    "vermont": "VT", "virginia": "VA", "washington": "WA",
    "west virginia": "WV", "wisconsin": "WI", "wyoming": "WY",
    "district of columbia": "DC",
}

# State codes only count in a positional context — a bare "IN" or "OR"
# in free text is almost never Indiana or Oregon.
_STATE_CODES = "|".join(sorted(set(US_STATES.values())))
US_PATTERNS = [
    re.compile(r'\b(?:united states|usa|u\.s\.a?\.?)\b', re.I),
    re.compile(r'\bUS-[A-Z]{2}\b'),                      # iCIMS: US-CA-San Francisco
    re.compile(r',\s*(?:%s)\b' % _STATE_CODES),          # "Austin, TX"
    re.compile(r'^\s*(?:%s)\s*[-–]' % _STATE_CODES),     # "NY - Work from home"
    re.compile(r'[-–]\s*(?:%s)\s*$' % _STATE_CODES),     # "Chelmsford-MA"
    re.compile(r'\b(?:%s)\b' % "|".join(US_STATES), re.I),   # "Work At Home-Texas"
]

# Checked only after the US patterns, so "Birmingham, AL" stays a US hit.
NON_US_SIGNALS = [
    "india", "bengaluru", "bangalore", "hyderabad", "mumbai", "pune", "chennai",
    "gurgaon", "gurugram", "noida", "kolkata", "united kingdom", "england",
    "london", "manchester, uk", "edinburgh", "glasgow", "belfast", "dublin",
    "ireland", "france", "paris", "germany", "berlin", "munich", "frankfurt",
    "hamburg", "spain", "madrid", "barcelona", "italy", "milan", "rome",
    "netherlands", "amsterdam", "belgium", "brussels", "luxembourg",
    "switzerland", "zurich", "geneva", "austria", "vienna", "sweden",
    "stockholm", "norway", "oslo", "denmark", "copenhagen", "finland",
    "helsinki", "poland", "warsaw", "krakow", "czechia", "czech republic",
    "prague", "hungary", "budapest", "romania", "bucharest", "portugal",
    "lisbon", "greece", "athens", "turkey", "türkiye", "istanbul", "russia",
    "china", "shanghai", "beijing", "shenzhen", "hong kong", "taiwan",
    "taipei", "japan", "tokyo", "osaka", "korea", "seoul", "singapore",
    "malaysia", "kuala lumpur", "indonesia", "jakarta", "thailand",
    "bangkok", "vietnam", "philippines", "manila", "australia", "sydney",
    "melbourne", "brisbane", "perth", "new zealand", "auckland", "canada",
    "toronto", "vancouver", "montreal", "ottawa", "calgary", "mexico",
    "brazil", "sao paulo", "são paulo", "argentina", "buenos aires", "chile",
    "colombia", "bogota", "peru", "costa rica", "uae", "dubai", "abu dhabi",
    "saudi arabia", "riyadh", "qatar", "doha", "israel", "tel aviv", "egypt",
    "south africa", "johannesburg", "nigeria", "kenya", "morocco",
]


def load_filters():
    raw = load_json(FILTERS_FILE, {})
    loc = raw.get("location", {})
    roles = raw.get("roles", {})
    return {
        "location_enabled": loc.get("enabled", False),
        "countries": {c.upper() for c in loc.get("countries", [])},
        "include_unknown": loc.get("include_unknown", True),
        "roles_enabled": roles.get("enabled", False),
        "include": [s.lower() for s in roles.get("include", [])],
        "exclude": [s.lower() for s in roles.get("exclude", [])],
        "search_keywords": raw.get("search_keywords", []),
    }


def classify_location(text, country_code=None):
    """
    Return "US", "OTHER", or "UNKNOWN".

    Prefers an explicit country code where the platform gives us one
    (Oracle and Amazon do); otherwise reads the human-readable location
    string, checking US signals before foreign ones so that genuinely
    ambiguous city names like Birmingham resolve correctly when the
    state is present.
    """
    if country_code:
        code = country_code.strip().upper()
        if code in ("US", "USA", "UNITED STATES"):
            return "US"
        if code:
            return "OTHER"

    if not text:
        return "UNKNOWN"
    text = text.strip()
    if not text:
        return "UNKNOWN"

    for pattern in US_PATTERNS:
        if pattern.search(text):
            return "US"

    lowered = text.lower()
    for signal in NON_US_SIGNALS:
        if signal in lowered:
            return "OTHER"

    return "UNKNOWN"


def location_allowed(job, filters):
    if not filters["location_enabled"] or not filters["countries"]:
        return True
    verdict = classify_location(job.get("location"), job.get("country_code"))
    if verdict == "UNKNOWN":
        return filters["include_unknown"]
    return verdict in filters["countries"]


def role_allowed(title, filters):
    if not filters["roles_enabled"]:
        return True
    lowered = (title or "").lower()
    if filters["include"] and not any(k in lowered for k in filters["include"]):
        return False
    if any(k in lowered for k in filters["exclude"]):
        return False
    return True


def apply_filters(jobs, filters):
    """Drop jobs that fail the role or location rules. Returns (kept, stats)."""
    kept = {}
    dropped_role = dropped_location = 0
    for job_id, job in jobs.items():
        if not role_allowed(job.get("title"), filters):
            dropped_role += 1
            continue
        if not location_allowed(job, filters):
            dropped_location += 1
            continue
        kept[job_id] = job
    return kept, {"role": dropped_role, "location": dropped_location}


# ═══════════════════════════════════════════════════════════════
# Platform detection — each returns parsed parts or None
# ═══════════════════════════════════════════════════════════════

def get_greenhouse_slug(url):
    m = re.search(r"greenhouse\.io/([a-zA-Z0-9\-_]+)", url)
    return m.group(1) if m else None


def get_lever_slug(url):
    m = re.search(r"lever\.co/([a-zA-Z0-9\-_]+)", url)
    return m.group(1) if m else None


def get_workday_parts(url):
    """
    Workday URLs:
      https://{tenant}.wd{N}.myworkdayjobs.com/{site}
      https://{tenant}.wd{N}.myworkdayjobs.com/en-US/{site}
      https://wd{N}.myworkdaysite.com/recruiting/{tenant}/{site}
    """
    m = re.search(
        r'https?://([a-zA-Z0-9\-]+)\.(wd\d+)\.myworkdayjobs\.com/(?:[a-z]{2}(?:-[A-Z]{2})?/)?([a-zA-Z0-9\-_]+)',
        url
    )
    if m:
        return m.group(1), m.group(2), m.group(3)
    m = re.search(
        r'https?://(wd\d+)\.myworkdaysite\.com/recruiting/([a-zA-Z0-9\-_]+)/([a-zA-Z0-9\-_]+)',
        url
    )
    if m:
        dc, tenant, site = m.group(1), m.group(2), m.group(3)
        return tenant, dc, site
    return None


def get_oracle_cloud_parts(url):
    """Oracle Cloud (Fusion HCM): *.oraclecloud.com with /sites/{SITE}/"""
    m = re.search(r'https?://([a-zA-Z0-9\-]+)\.fa\.(?:[a-zA-Z0-9]+\.)?oraclecloud\.com', url)
    site_m = re.search(r'sites/([A-Za-z0-9_]+)', url)
    if not m or not site_m:
        return None
    host = re.search(r'https?://([a-zA-Z0-9\-\.]+\.oraclecloud\.com)', url).group(1)
    return host, site_m.group(1)


def get_icims_parts(url):
    """
    iCIMS URLs:
      https://{company}.icims.com/jobs/search?ss=1
      https://careers-{company}.icims.com/jobs/search?ss=1
    Returns the base URL (everything before /jobs/) or None.
    """
    if "icims.com" not in url:
        return None
    m = re.search(r'(https?://[a-zA-Z0-9\-\.]+\.icims\.com)', url)
    return m.group(1) if m else None


def get_ashby_slug(url):
    """Ashby: https://jobs.ashbyhq.com/{company}"""
    m = re.search(r'jobs\.ashbyhq\.com/([a-zA-Z0-9\-_]+)', url)
    return m.group(1) if m else None


def is_amazon(url):
    """Amazon runs its own job board with a public search.json endpoint."""
    return bool(re.search(r'(^|\.)amazon\.jobs', urlparse(url).netloc))


# ═══════════════════════════════════════════════════════════════
# Fetchers — one per platform
# ═══════════════════════════════════════════════════════════════

def fetch_greenhouse(slug):
    api = f"https://boards-api.greenhouse.io/v1/boards/{slug}/jobs"
    r = requests.get(api, headers=HEADERS, timeout=20)
    r.raise_for_status()
    jobs = r.json().get("jobs", [])
    return {str(j["id"]): {"title": j["title"], "url": j["absolute_url"],
                           "location": (j.get("location") or {}).get("name")}
            for j in jobs}


def fetch_lever(slug):
    api = f"https://api.lever.co/v0/postings/{slug}?mode=json"
    r = requests.get(api, headers=HEADERS, timeout=20)
    r.raise_for_status()
    jobs = r.json()
    return {j["id"]: {"title": j["text"], "url": j["hostedUrl"],
                      "location": (j.get("categories") or {}).get("location")}
            for j in jobs}


def workday_us_facet(api, country_names):
    """
    Workday exposes a locationCountry facet whose IDs are tenant-specific,
    but they're listed in any search response, so we can look ours up at
    runtime. Not every tenant publishes it (US-only employers generally
    don't) — returns None there and we fall back to filtering on the
    location text.
    """
    r = requests.post(
        api,
        json={"appliedFacets": {}, "limit": 1, "offset": 0, "searchText": ""},
        headers={**HEADERS, "Content-Type": "application/json"},
        timeout=30,
    )
    r.raise_for_status()
    stack = list(r.json().get("facets", []))
    while stack:
        node = stack.pop()
        values = node.get("values") or []
        if node.get("facetParameter") == "locationCountry":
            for v in values:
                if any(n.lower() in v.get("descriptor", "").lower() for n in country_names):
                    return v.get("id")
        stack.extend(v for v in values if isinstance(v, dict) and v.get("values"))
    return None


def _workday_page(api, facets, search_text, offset, limit):
    r = requests.post(
        api,
        json={"appliedFacets": facets, "limit": limit, "offset": offset,
              "searchText": search_text},
        headers={**HEADERS, "Content-Type": "application/json"},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def fetch_workday(tenant, dc, site, base_url, filters=None, max_jobs=4000):
    """
    Workday's cxs endpoint reports `total` only on the first response —
    later pages come back with total=0. Capture it once, or the
    `offset >= total` check trips on page two and silently truncates
    every company to the first 40 postings.

    Rather than pull every posting and filter afterwards, we run one
    search per configured keyword and union the results — CVS Health is
    19,000 postings unfiltered but a few hundred across the analyst
    keywords. A `?q=` on the career-page URL still wins if present.
    """
    filters = filters or {}
    api = f"https://{tenant}.{dc}.myworkdayjobs.com/wday/cxs/{tenant}/{site}/jobs"

    url_q = parse_qs(urlparse(base_url).query).get("q", [""])[0]
    searches = [url_q] if url_q else (filters.get("search_keywords") or [""])

    applied = {}
    if filters.get("location_enabled") and "US" in filters.get("countries", set()):
        facet_id = workday_us_facet(api, ["United States"])
        if facet_id:
            applied["locationCountry"] = [facet_id]

    jobs = {}
    for search_text in searches:
        offset = 0
        limit = 20
        total = None
        while True:
            data = _workday_page(api, applied, search_text, offset, limit)
            postings = data.get("jobPostings", [])
            if not postings:
                break
            if total is None:
                total = data.get("total", 0)
            for p in postings:
                path = p.get("externalPath", "")
                job_id = path or p.get("title", "")
                jobs[job_id] = {
                    "title": p.get("title", "Untitled"),
                    # externalPath is "/job/Location/Title_R123" — relative to
                    # the career *site*, not the host. Without the site segment
                    # every link 404s.
                    "url": f"https://{tenant}.{dc}.myworkdayjobs.com/{site}{path}",
                    "location": p.get("locationsText"),
                }
            offset += limit
            if offset >= total or len(postings) < limit:
                break
            if offset >= max_jobs:
                print(f"  [WARN] Workday: stopped at {max_jobs} of {total} for "
                      f"search {search_text!r}", file=sys.stderr)
                break
    return jobs


def fetch_oracle_cloud(host, site_number, filters=None, max_jobs=4000):
    """
    Oracle Fusion HCM ("Oracle Recruiting Cloud").

    Two things this endpoint is fussy about, both of which silently
    return an empty requisitionList if you get them wrong:
      - `expand` must be present, otherwise the response comes back
        with the search metadata but no jobs attached.
      - The job page is keyed on `Id`, NOT `RequisitionNumber` —
        several tenants (JPMorgan among them) don't populate
        RequisitionNumber at all, which yields .../job/ links.
    """
    filters = filters or {}
    api = f"https://{host}/hcmRestApi/resources/latest/recruitingCEJobRequisitions"
    searches = filters.get("search_keywords") or [None]
    jobs = {}
    for keyword in searches:
        offset = 0
        limit = 200
        total = None
        while True:
            finder = f"findReqs;siteNumber={site_number}"
            if keyword:
                finder += f",keyword={keyword}"
            finder += f",limit={limit},offset={offset},sortBy=POSTING_DATES_DESC"
            params = {
                "onlyData": "true",
                "expand": "requisitionList.secondaryLocations,flexFieldsFacet.values",
                "finder": finder,
            }
            r = requests.get(api, params=params, headers=HEADERS, timeout=30)
            r.raise_for_status()
            items = r.json().get("items", [])
            if not items:
                break
            if total is None:
                total = items[0].get("TotalJobsCount")
            reqs = items[0].get("requisitionList", [])
            if not reqs:
                break
            for job in reqs:
                job_id = job.get("Id") or job.get("RequisitionNumber")
                if not job_id:
                    continue
                jobs[str(job_id)] = {
                    "title": job.get("Title", "Untitled"),
                    "url": f"https://{host}/hcmUI/CandidateExperience/en/sites/{site_number}/job/{job_id}",
                    "location": job.get("PrimaryLocation"),
                    "country_code": job.get("PrimaryLocationCountry"),
                }
            offset += limit
            if len(reqs) < limit:
                break
            if offset >= max_jobs:
                print(f"  [WARN] Oracle Cloud: stopped at {max_jobs} of {total} for "
                      f"keyword {keyword!r}", file=sys.stderr)
                break
    return jobs


def icims_row_location(anchor):
    """
    iCIMS listings carry per-row metadata as <dt>label</dt><dd>value</dd>
    pairs — commonly "Job Locations" (US-CA-San Francisco) and "Country".
    Not every portal enables them; returns {} when absent, which leaves
    the job's location unknown rather than guessing.
    """
    row = anchor
    for _ in range(6):
        row = row.parent
        if row is None:
            return {}
        classes = " ".join(row.get("class") or [])
        if "row" in classes.split() or "iCIMS_JobsTable" in classes:
            break
    else:
        return {}

    found = {}
    for tag in row.find_all("div", class_="iCIMS_JobHeaderTag"):
        label = tag.find("dt")
        value = tag.find("dd")
        if not label or not value:
            continue
        key = label.get_text(strip=True).lower()
        text = value.get_text(" ", strip=True)
        if "location" in key:
            found["location"] = text
        elif "country" in key:
            found["country_code"] = text
    return found


def fetch_icims(base_url, original_url, max_pages=40):
    """
    iCIMS career portals have job links that follow a consistent
    pattern: /jobs/{job_id}/job — much more reliable than the generic
    fallback since we can filter on URL structure instead of guessing
    from link text. Uses the original user-provided URL directly,
    since different iCIMS instances reject constructed search URLs.

    Two portal quirks worth knowing:
      - `in_iframe=1` is required. Without it many instances answer
        the search URL with a bare 405 Method Not Allowed.
      - Results are 20 per page, paged with `pr=0,1,2...`. Paging past
        the end returns 200 with no job links, which is how we stop.
    """
    jobs = {}
    seen_ids = set()
    for page in range(max_pages):
        url = with_params(original_url, ss=1, in_iframe=1, pr=page)
        r = requests.get(url, headers=HEADERS, timeout=20)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")

        page_ids = set()
        for a in soup.find_all("a", href=True):
            href = a["href"]
            # iCIMS job links look like /jobs/1234/job or /jobs/1234/title
            m = re.search(r'/jobs/(\d+)/', href)
            if not m:
                continue
            job_id = m.group(1)
            page_ids.add(job_id)
            if job_id in jobs:  # first link text for this ID wins
                continue
            text = a.get_text(strip=True)
            # Clean up iCIMS's "TitleActual Job Name" prefix pattern
            text = re.sub(r'^(Job\s*)?Title', '', text).strip()
            if not text or len(text) < 4 or text.lower() in JUNK_TITLES:
                continue
            full_url = href if href.startswith("http") else requests.compat.urljoin(base_url, href)
            # Drop in_iframe so the link opens as a normal page in Telegram
            jobs[job_id] = {"title": text, "url": with_params(full_url, in_iframe=None)}
            jobs[job_id].update(icims_row_location(a))

        # Empty page = past the end. A page with no IDs we haven't already
        # seen means the portal ignored `pr` and re-served an earlier page;
        # stop in either case rather than loop to max_pages.
        if not page_ids - seen_ids:
            break
        seen_ids |= page_ids
    else:
        print(f"  [WARN] iCIMS: stopped at max_pages={max_pages} "
              f"({len(jobs)} postings) — listing may be truncated", file=sys.stderr)
    return jobs


def fetch_ashby(slug):
    """Ashby has a proper public JSON API, same category as Greenhouse/Lever."""
    api = f"https://api.ashbyhq.com/posting-api/job-board/{slug}"
    r = requests.get(api, headers=HEADERS, timeout=20)
    r.raise_for_status()
    data = r.json()
    jobs = {}
    for job in data.get("jobs", []):
        job_id = job.get("id", "")
        title = job.get("title", "Untitled")
        job_url = f"https://jobs.ashbyhq.com/{slug}/{job_id}"
        jobs[job_id] = {"title": title, "url": job_url,
                        "location": job.get("location")}
    return jobs


def fetch_amazon(url, filters=None, max_jobs=2000):
    """
    amazon.jobs backs its search page with a public JSON endpoint at
    /en/search.json. Any filters on the URL the user supplied
    (base_query, category, location, ...) are passed straight through,
    which matters here: unfiltered, Amazon returns 10,000+ postings.
    result_limit is capped at 100 by the server — asking for more
    returns a null job list rather than an error.
    """
    filters = filters or {}
    user_params = {k: v[0] for k, v in parse_qs(urlparse(url).query).items()}
    api = "https://www.amazon.jobs/en/search.json"
    searches = ([user_params["base_query"]] if user_params.get("base_query")
                else (filters.get("search_keywords") or [""]))
    jobs = {}
    for keyword in searches:
        offset = 0
        limit = 100
        total = None
        while True:
            params = {**user_params, "base_query": keyword,
                      "result_limit": limit, "offset": offset, "sort": "recent"}
            r = requests.get(api, params=params, headers=HEADERS, timeout=30)
            r.raise_for_status()
            data = r.json()
            batch = data.get("jobs") or []
            if not batch:
                break
            if total is None:
                total = data.get("hits")
            for job in batch:
                job_id = str(job.get("id_icims") or job.get("id") or "")
                if not job_id:
                    continue
                jobs[job_id] = {
                    "title": job.get("title", "Untitled"),
                    "url": "https://www.amazon.jobs" + job.get("job_path", ""),
                    "location": job.get("normalized_location"),
                    "country_code": job.get("country_code"),
                }
            offset += limit
            if len(batch) < limit:
                break
            if offset >= max_jobs:
                print(f"  [WARN] Amazon: stopped at {max_jobs} of {total} for "
                      f"query {keyword!r}", file=sys.stderr)
                break
    return jobs


def fetch_sitemap(url):
    """
    Last resort for JS-rendered career sites that still publish an XML
    sitemap of their postings (Flowserve is the case this was built for).

    Opt in per company with "type": "sitemap" — it can't be detected
    from the URL. Follows a sitemap index one level down into any child
    sitemap whose name mentions jobs. Titles come from the URL slug,
    since a sitemap carries no title field.
    """
    def read(target):
        resp = requests.get(target, headers=HEADERS, timeout=20)
        resp.raise_for_status()
        return BeautifulSoup(resp.text, "xml")

    root = url if url.rstrip("/").endswith(".xml") else requests.compat.urljoin(url, "/sitemap.xml")
    soup = read(root)
    locs = [loc.get_text(strip=True) for loc in soup.find_all("loc")]

    # A sitemap index points at child sitemaps; follow the job ones.
    if soup.find("sitemapindex"):
        children = [u for u in locs if "job" in u.lower()]
        if not children:
            print(f"  [WARN] sitemap: no job sitemap found in index {root}", file=sys.stderr)
            return {}
        locs = []
        for child in children:
            locs += [loc.get_text(strip=True) for loc in read(child).find_all("loc")]

    jobs = {}
    for job_url in locs:
        # e.g. /bridgeville-pa/manual-machinist/2AB5BF82.../job/
        m = re.search(r'/([^/]+)/([^/]+)/([0-9A-Fa-f]{16,})/job/?$', job_url)
        if not m:
            continue
        place_slug, title_slug, job_id = m.group(1), m.group(2), m.group(3)

        # The place slug ends in a region code: "bridgeville-pa" (US) or
        # "queretaro-mex" (not). A 2-letter US state code is the only thing
        # we treat as a positive match; anything else is a foreign posting.
        parts = place_slug.split("-")
        region = parts[-1].upper() if len(parts) > 1 else ""
        location = f"{' '.join(parts[:-1]).title()}, {region}" if region else place_slug.title()
        jobs[job_id] = {
            "title": title_slug.replace("-", " ").title(),
            "url": job_url,
            "location": location,
            "country_code": "US" if region in set(US_STATES.values()) else region,
        }
    return jobs


def is_junk_title(text):
    """Check if a scraped title is likely navigation junk, not a real job."""
    cleaned = text.strip().lower()
    # Exact match against known junk
    if cleaned in JUNK_TITLES:
        return True
    # Starts with known junk prefix
    for junk in JUNK_TITLES:
        if cleaned.startswith(junk):
            return True
    # Too short to be a real job title
    if len(cleaned) < 8:
        return True
    # All one word (real job titles are almost always 2+ words)
    if len(cleaned.split()) < 2:
        return True
    # Contains no letters (just symbols/numbers)
    if not re.search(r'[a-zA-Z]', cleaned):
        return True
    return False


def fetch_generic(url):
    """
    Best-effort fallback for unrecognized career sites.
    Now with much stricter filtering to reduce junk alerts.
    """
    r = requests.get(url, headers=HEADERS, timeout=20)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    jobs = {}
    for a in soup.find_all("a", href=True):
        text = a.get_text(strip=True)
        href = a["href"]
        if not text or len(text) < 8 or len(text) > 120:
            continue
        # Clean up common prefixes iCIMS-style sites add
        text = re.sub(r'^(Job\s*)?Title\s*', '', text).strip()
        if is_junk_title(text):
            continue
        # Must have BOTH a title-like signal AND a link-like signal
        # (old version accepted either/or, which was too loose)
        title_signal = re.search(
            r"\b(analyst|manager|engineer|associate|specialist|coordinator|"
            r"director|lead|intern|consultant|scientist|developer|designer|"
            r"administrator|architect|accountant|advisor|assistant|attorney|"
            r"nurse|physician|technician|supervisor|representative|officer|"
            r"recruiter|paralegal|planner|buyer|auditor|controller)\b",
            text, re.IGNORECASE
        )
        link_signal = re.search(
            r"(job[s]?/|career|position|opening|requisition|req|posting|apply|vacancies)",
            href, re.IGNORECASE
        )
        if title_signal and link_signal:
            full_url = href if href.startswith("http") else requests.compat.urljoin(url, href)
            job_id = full_url
            jobs[job_id] = {"title": text, "url": full_url}
    return jobs


# ═══════════════════════════════════════════════════════════════
# Main router — detects platform, calls the right fetcher
# ═══════════════════════════════════════════════════════════════

def fetch_company_jobs(company, filters=None):
    filters = filters or {}
    url = company["url"]
    name = company["name"]

    gh_slug = get_greenhouse_slug(url)
    lv_slug = get_lever_slug(url)
    wd_parts = get_workday_parts(url)
    oc_parts = get_oracle_cloud_parts(url)
    icims_base = get_icims_parts(url)
    ashby_slug = get_ashby_slug(url)

    try:
        # Explicit override in companies.json wins over URL sniffing
        forced = company.get("type")
        if forced == "sitemap":
            print(f"  [Sitemap] {name}")
            return fetch_sitemap(url)
        elif forced:
            print(f"  [WARN] {name}: unknown type '{forced}' — falling through to "
                  f"auto-detection", file=sys.stderr)

        if is_amazon(url):
            print(f"  [Amazon] {name}")
            return fetch_amazon(url, filters)
        elif gh_slug:
            print(f"  [Greenhouse] {name}")
            return fetch_greenhouse(gh_slug)
        elif lv_slug:
            print(f"  [Lever] {name}")
            return fetch_lever(lv_slug)
        elif wd_parts:
            tenant, dc, site = wd_parts
            print(f"  [Workday] {name} (tenant={tenant}, dc={dc}, site={site})")
            return fetch_workday(tenant, dc, site, url, filters)
        elif oc_parts:
            host, site_number = oc_parts
            print(f"  [Oracle Cloud] {name} (host={host}, site={site_number})")
            return fetch_oracle_cloud(host, site_number, filters)
        elif icims_base:
            print(f"  [iCIMS] {name} (base={icims_base})")
            return fetch_icims(icims_base, url)
        elif ashby_slug:
            print(f"  [Ashby] {name} (slug={ashby_slug})")
            return fetch_ashby(ashby_slug)
        else:
            # Safety net warnings for near-misses
            if "myworkdayjobs.com" in url or "myworkdaysite.com" in url:
                print(f"  [WARN] {name}: looks like Workday but URL wasn't recognized — check URL format", file=sys.stderr)
            elif "oraclecloud.com" in url:
                print(f"  [WARN] {name}: looks like Oracle Cloud but URL wasn't recognized — check URL format", file=sys.stderr)
            elif "icims.com" in url:
                print(f"  [WARN] {name}: looks like iCIMS but URL wasn't recognized — check URL format", file=sys.stderr)
            print(f"  [Fallback scraper] {name} — results may be unreliable")
            return fetch_generic(url)
    except Exception as e:
        print(f"  [ERROR] {name}: {e}", file=sys.stderr)
        return None


def send_telegram(message):
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_ids_raw = os.environ.get("TELEGRAM_CHAT_ID", "")
    chat_ids = [c.strip() for c in chat_ids_raw.split(",") if c.strip()]

    if not token or not chat_ids:
        print("[WARN] Telegram credentials not set — printing instead:")
        print(message)
        return

    api = f"https://api.telegram.org/bot{token}/sendMessage"
    for chat_id in chat_ids:
        try:
            requests.post(api, data={
                "chat_id": chat_id,
                "text": message,
                "parse_mode": "HTML",
                "disable_web_page_preview": False
            }, timeout=20)
        except Exception as e:
            print(f"  [ERROR] Failed to send to chat_id {chat_id}: {e}")


def main():
    companies = load_json(COMPANIES_FILE, [])
    seen = load_json(SEEN_FILE, {})
    filters = load_filters()

    active = []
    if filters["roles_enabled"]:
        active.append(f"roles ({len(filters['include'])} keywords)")
    if filters["location_enabled"]:
        active.append(f"location ({', '.join(sorted(filters['countries']))})")
    print(f"Filters: {' + '.join(active) if active else 'none'}\n")

    pending = []

    for company in companies:
        name = company["name"]
        print(f"Checking {name}...")
        current_jobs = fetch_company_jobs(company, filters)

        if current_jobs is None:
            continue

        fetched = len(current_jobs)
        current_jobs, dropped = apply_filters(current_jobs, filters)
        if dropped["role"] or dropped["location"]:
            print(f"  Filtered {fetched} → {len(current_jobs)} "
                  f"(dropped {dropped['role']} on role, "
                  f"{dropped['location']} on location)")

        prev_ids = set(seen.get(name, {}).keys())
        current_ids = set(current_jobs.keys())
        new_ids = current_ids - prev_ids

        if name not in seen:
            print(f"  First run — recording {len(current_ids)} postings as baseline.")
            seen[name] = current_jobs
            continue

        for job_id in sorted(new_ids):
            job = current_jobs[job_id]
            pending.append((name, job))
            print(f"  NEW: {job['title']}")

        seen[name] = current_jobs

    # Save state before notifying: if Telegram is down or rate-limits us,
    # we'd rather drop alerts once than re-send the same batch every 2 hours.
    save_json(SEEN_FILE, seen)

    for name, job in pending[:MAX_ALERTS_PER_RUN]:
        where = f"\n📍 {job['location']}" if job.get("location") else ""
        send_telegram(f"🆕 <b>{name}</b>\n{job['title']}{where}\n{job['url']}")

    suppressed = len(pending) - MAX_ALERTS_PER_RUN
    if suppressed > 0:
        by_company = {}
        for name, _ in pending[MAX_ALERTS_PER_RUN:]:
            by_company[name] = by_company.get(name, 0) + 1
        breakdown = "\n".join(f"• {n}: {c}" for n, c in sorted(by_company.items()))
        send_telegram(
            f"⚠️ <b>{suppressed} more new postings</b> this run, not sent individually "
            f"to avoid flooding.\n\n{breakdown}\n\nFull list is in the Actions log."
        )
        print(f"\n[WARN] {suppressed} alert(s) suppressed (cap is {MAX_ALERTS_PER_RUN}).",
              file=sys.stderr)

    print(f"\nDone. {len(pending)} new posting(s) found.")


if __name__ == "__main__":
    main()
