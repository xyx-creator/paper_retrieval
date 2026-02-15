import requests
import xml.etree.ElementTree as ET
import time
import os
from config import ARXIV_CATEGORY, INPUT_DIR

def search_arxiv(query_keywords, days=1):
    """
    Search arXiv for papers matching query_keywords within the last N days.
    """
    base_url = "http://export.arxiv.org/api/query"
    
    # Construct query: (cat:cs.CV OR cat:cs.CL) AND (all:keyword1 OR all:keyword2 ...)
    # Note: arXiv API simple query param often works better than complex boolean logic in URL.
    # We will fetch by Category + Date, then filter locally (Phase 1).
    
    # Actually, fetching *everything* from cs.CV for last day is safer, then filtering.
    # Because "all:keyword" might miss synonyms we haven't thought of, 
    # but since we have EXPANDED_KEYWORDS, we can try to construct a broad query if list is short.
    # But URL length limit exists.
    
    # Strategy: Query by Category, sorted by Date. Fetch enough to cover 'days'.
    # Estimate 100 papers per day for active categories like cs.CV
    max_results = max(200, days * 150)
    
    params = {
        "search_query": ARXIV_CATEGORY, # Defined in config
        "sortBy": "submittedDate",
        "sortOrder": "descending",
        "start": 0,
        "max_results": max_results 
    }
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            print(f"Fetching from arXiv: {base_url} (Attempt {attempt+1})")
            response = requests.get(base_url, params=params, timeout=60) # Increased timeout
            
            if response.status_code == 200:
                root = ET.fromstring(response.content)
                namespace = {'atom': 'http://www.w3.org/2005/Atom'}
                
                papers = []
                entries = root.findall('atom:entry', namespace)
                print(f"arXiv returned {len(entries)} entries.")
                
                for entry in entries:
                    published = entry.find('atom:published', namespace).text
                    
                    # Simple date filtering with logging
                    from datetime import datetime
                    pub_dt = datetime.strptime(published, "%Y-%m-%dT%H:%M:%SZ")
                    
                    # Check date but don't stop
                    age_days = (time.time() - pub_dt.timestamp()) / (24 * 3600)
                    if age_days > days:
                         # Optional: Log occasionally if needed, but for now just skip silently to reduce noise
                         # unless we want to debug why everything is skipped.
                         # print(f"Skipping old paper: {age_days:.1f} days old.")
                         continue
                        
                    title = entry.find('atom:title', namespace).text.replace('\n', ' ').strip()
                    summary = entry.find('atom:summary', namespace).text.replace('\n', ' ').strip()
                    id_url = entry.find('atom:id', namespace).text
                    
                    # Get PDF link
                    pdf_link = ""
                    for link in entry.findall('atom:link', namespace):
                        if link.attrib.get('title') == 'pdf':
                            pdf_link = link.attrib.get('href')
                    
                    # Fallback if no explicit pdf link
                    if not pdf_link:
                        pdf_link = id_url.replace("abs", "pdf")
                        
                    papers.append({
                        "title": title,
                        "abstract": summary,
                        "url": pdf_link,
                        "date": published,
                        "venue": "arXiv"
                    })
                    
                return papers
            elif response.status_code in [429, 503]:
                wait_time = (attempt + 1) * 5
                print(f"arXiv Rate Limit/Unavailable ({response.status_code}). Wait {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"Error fetching from arXiv: {response.status_code}")
                return []
        except Exception as e:
            print(f"Exception in search_arxiv (Attempt {attempt+1}): {e}")
            time.sleep(5)
            
    print("Max retries reached for arXiv.")
    return []

def resolve_pdf_url(item):
    """
    Fallback strategy to find a PDF URL.
    Priority:
    1. openAccessPdf.url (Direct S2)
    2. ArXiv ID (Construct URL)
    3. ACL Anthology (Construct URL)
    """
    # 1. Check direct OpenAccess
    if item.get("openAccessPdf") and item["openAccessPdf"].get("url"):
        return item["openAccessPdf"]["url"]
        
    external_ids = item.get("externalIds") or {}
    
    # 2. Check ArXiv
    if "ArXiv" in external_ids:
        arxiv_id = external_ids["ArXiv"]
        return f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        
    # 3. Check ACL
    if "ACL" in external_ids:
        acl_id = external_ids["ACL"]
        return f"https://aclanthology.org/{acl_id}.pdf"
        
    return None

def search_dblp(venue, year):
    """
    Search DBLP for papers in a specific Venue and Year.
    Step 1 of Node 1: Street Sweep.
    """
    base_url = "https://dblp.org/search/publ/api"
    # Construct query: venue:<venue>: year:<year>:
    query = f"venue:{venue}: year:{year}:"
    
    print(f"Searching DBLP for {venue} {year}...")
    
    papers_data = []
    h = 1000 # Max hits per request
    f = 0    # First hit index
    
    while True:
        params = {
            "q": query,
            "format": "json",
            "h": h,
            "f": f
        }
        
        try:
            print(f"  Fetching DBLP offset {f}...")
            response = requests.get(base_url, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                result = data.get("result", {})
                hits = result.get("hits", {})
                
                if not hits or "hit" not in hits:
                    print("  No more hits.")
                    break
                    
                hit_list = hits["hit"]
                if not hit_list:
                    break
                    
                print(f"  Got {len(hit_list)} hits.")
                
                for hit in hit_list:
                    info = hit.get("info", {})
                    title = info.get("title", "").strip()
                    # Remove trailing dot if present
                    if title.endswith("."):
                        title = title[:-1]
                        
                    doi = info.get("doi")
                    
                    if title:
                        papers_data.append({
                            "title": title,
                            "doi": doi,
                            "year": info.get("year"),
                            "venue": info.get("venue")
                        })
                
                # Check if we reached the end
                total = int(hits.get("@total", 0))
                f += h
                if f >= total:
                    break
                    
                # DBLP is generous, but let's be polite
                time.sleep(0.5)
            else:
                print(f"  DBLP Error {response.status_code}")
                break
        except Exception as e:
            print(f"  DBLP Exception: {e}")
            break
            
    print(f"DBLP returned {len(papers_data)} papers.")
    return papers_data

def batch_fetch_s2(papers_data):
    """
    Step 2 of Node 1: Precision Snipe.
    Use S2 Batch API to get details for papers found in DBLP.
    Uses DOIs where available.
    """
    if not papers_data:
        return []
        
    batch_url = "https://api.semanticscholar.org/graph/v1/paper/batch"
    
    from config import S2_API_KEY
    headers = {}
    if S2_API_KEY:
        headers["x-api-key"] = S2_API_KEY
        print("Using S2 API Key.")
    
    # Define batch size inside function
    BATCH_SIZE = 100
        
    # Extract DOIs
    # Priority 1: Use DOI for batch fetch (Fast & Accurate)
    valid_papers_with_doi = [p for p in papers_data if p.get("doi")]
    print(f"Found {len(valid_papers_with_doi)} papers with DOIs out of {len(papers_data)}.")
    
    # Priority 2: Papers without DOI - Search by Title (Slower fallback)
    papers_without_doi = [p for p in papers_data if not p.get("doi")]
    print(f"Found {len(papers_without_doi)} papers WITHOUT DOIs. Attempting title search fallback...")

    enriched_papers = []
    
    # --- Strategy A: Batch Fetch via DOI ---
    if valid_papers_with_doi:
        seen_dois = set()
        unique_papers_doi = []
        for p in valid_papers_with_doi:
            if p["doi"] not in seen_dois:
                seen_dois.add(p["doi"])
                unique_papers_doi.append(p)
                
        print(f"Unique DOIs to fetch: {len(unique_papers_doi)}")
        
        all_dois = [p["doi"] for p in unique_papers_doi]
        
        for i in range(0, len(all_dois), BATCH_SIZE):
            batch_dois = all_dois[i:i+BATCH_SIZE]
            print(f"  [DOI Batch] Fetching {i}-{i+len(batch_dois)}...")
            
            payload = {"ids": batch_dois}
            params_batch = {"fields": "title,abstract,url,year,venue,openAccessPdf,externalIds"}
            
            for attempt in range(3):
                try:
                    response = requests.post(batch_url, json=payload, params=params_batch, headers=headers, timeout=30)
                    if response.status_code == 200:
                        data = response.json()
                        for item in data:
                            if not item: continue
                            pdf_url = resolve_pdf_url(item)
                            
                            if not pdf_url and item.get("externalIds", {}).get("DOI"):
                                pdf_url = f"https://doi.org/{item['externalIds']['DOI']}"
                            
                            enriched_papers.append({
                                "title": item.get("title"), 
                                "abstract": item.get("abstract") or "",
                                "url": pdf_url,
                                "date": str(item.get("year")),
                                "venue": item.get("venue")
                            })
                        break 
                    elif response.status_code == 429:
                        print("  Rate Limit. Waiting 5s...")
                        time.sleep(5)
                    else:
                        print(f"  Batch Error {response.status_code}")
                        break
                except Exception as e:
                    print(f"  Batch Exception: {e}")
                    break

    # --- Strategy B: Title Search Fallback (For recent papers without DOI) ---
    if papers_without_doi:
        # Limit fallback to avoid excessive API calls if list is huge
        MAX_FALLBACK = 50 
        print(f"Fallback: Searching S2 by title for first {MAX_FALLBACK} papers without DOI...")
        
        search_url = "https://api.semanticscholar.org/graph/v1/paper/search"
        
        for i, p in enumerate(papers_without_doi[:MAX_FALLBACK]):
            title = p["title"]
            print(f"  [Title Search] {i+1}/{len(papers_without_doi[:MAX_FALLBACK])}: {title[:30]}...")
            
            params = {
                "query": title,
                "fields": "title,abstract,url,year,venue,openAccessPdf,externalIds",
                "limit": 1
            }
            
            try:
                # Rate limit politeness
                time.sleep(1.0) 
                
                response = requests.get(search_url, params=params, headers=headers, timeout=30)
                if response.status_code == 200:
                    data = response.json()
                    if "data" in data and data["data"]:
                        item = data["data"][0]
                        pdf_url = resolve_pdf_url(item)
                         # If no PDF, try generic DOI link if available
                        if not pdf_url and item.get("externalIds", {}).get("DOI"):
                             pdf_url = f"https://doi.org/{item['externalIds']['DOI']}"

                        enriched_papers.append({
                            "title": item.get("title"),
                            "abstract": item.get("abstract") or "",
                            "url": pdf_url,
                            "date": str(item.get("year")),
                            "venue": item.get("venue")
                        })
                elif response.status_code == 429:
                    print("  Rate Limit. Skipping...")
                    time.sleep(2)
            except Exception as e:
                print(f"  Search Exception: {e}")

    return enriched_papers

def search_semantic_scholar(venue, year):
    """
    Search Semantic Scholar for papers in a specific Venue and Year.
    Uses Batch API where possible (not directly applicable for search, but optimized).
    Actually, S2 Search API doesn't support batch ID lookup directly from query.
    However, if we could get IDs first, we could use POST /graph/v1/paper/batch.
    
    But `paper/search` returns the details directly if fields are requested.
    The issue with `paper/search` is strict rate limiting on public API.
    
    Optimization:
    1. Search for IDs only (lighter?). No, fields=paperId is default.
    2. Then batch fetch details? 
       - Search: `query=...&fields=paperId` -> Get 100 IDs.
       - Batch: `POST /graph/v1/paper/batch` with {"ids": [...], "fields": "title,abstract..."}
       
    This splits 1 heavy search request into:
    - 1 Search Request (maybe lighter/faster?)
    - 1 Batch Request (1 request for up to 500 papers).
    
    This might bypass some timeouts if the search with fields is what's heavy.
    Let's try this 2-step approach.
    """
    base_url = "https://api.semanticscholar.org/graph/v1"
    
    # Step 1: Search for IDs (lighter request) with Pagination
    search_url = f"{base_url}/paper/search"
    # Try both full name and acronym to cover all bases
    # Note: S2 might not support OR in venue field well, but let's try strict full name first as it's more official
    # Or just use "CVPR" and iterate more pages? "CVPR" matched 64 items (mostly workshops).
    # Let's try the full name: "IEEE/CVF Conference on Computer Vision and Pattern Recognition"
    # Actually, S2 uses "Conference on Computer Vision and Pattern Recognition" often.
    # Let's try to query by venue match.
    
    # Heuristic: If venue is "CVPR", expand to include full name variants
    venues_to_search = [venue]
    if venue.upper() == "CVPR":
        venues_to_search.append("IEEE/CVF Conference on Computer Vision and Pattern Recognition")
    
    paper_ids = set()
    MAX_PAPERS_TO_FETCH = 1000 # Safety limit
    
    from config import S2_API_KEY
    headers = {}
    if S2_API_KEY:
        headers["x-api-key"] = S2_API_KEY
        print("Using S2 API Key.")
        
    for v in venues_to_search:
        print(f"Searching venue: {v}...")
        # Quote the venue to handle spaces correctly
        query = f'venue:"{v}" year:{year}'
        
        offset = 0
        limit = 100
        
        while len(paper_ids) < MAX_PAPERS_TO_FETCH:
            params = {
                "query": query,
                "fields": "paperId", 
                "limit": limit,
                "offset": offset
            }
            
            try:
                print(f"  Fetching offset {offset} with query: {query}")
                # Debug request URL
                # print(f"  Request URL: {requests.Request('GET', search_url, params=params).prepare().url}")
                
                response = requests.get(search_url, params=params, headers=headers, timeout=30)
                
                if response.status_code == 200:
                    data = response.json()
                    if "data" not in data or not data["data"]:
                        break # No more results for this venue string
                        
                    new_ids = [p["paperId"] for p in data["data"] if p.get("paperId")]
                    if not new_ids:
                        break
                        
                    before_count = len(paper_ids)
                    paper_ids.update(new_ids)
                    
                    if len(paper_ids) == before_count:
                         # No new IDs added (duplicates?), maybe stop? 
                         # S2 pagination should return next batch. 
                         # If we get same IDs, we are stuck.
                         pass
                    
                    offset += limit
                    time.sleep(1) # Polite delay
                elif response.status_code == 429:
                    print("  Rate Limit. Waiting 5s...")
                    time.sleep(5)
                    # Retry same offset
                    continue 
                else:
                    print(f"  Error {response.status_code}")
                    break
            except Exception as e:
                print(f"  Exception: {e}")
                break
                
    paper_ids = list(paper_ids)
    if not paper_ids:
        print("No papers found in S2 Search.")
        return []

    print(f"Found {len(paper_ids)} unique IDs. Fetching details via Batch API...")

    # Step 2: Batch Fetch Details
    batch_url = f"{base_url}/paper/batch"
    
    # Correct API Usage: 'fields' must be a query parameter, NOT in JSON body for POST /batch?
    # Actually, S2 docs say for POST /paper/batch, fields CAN be in query param OR body?
    # Let's try putting fields in query params to be safe.
    
    # Process in chunks of 100 (S2 Batch Limit is usually 100 or 500)
    BATCH_SIZE = 100
    papers = []
    
    for i in range(0, len(paper_ids), BATCH_SIZE):
        batch_ids = paper_ids[i:i+BATCH_SIZE]
        print(f"  Fetching batch {i}-{i+len(batch_ids)}...")
        
        payload = {"ids": batch_ids}
        params_batch = {"fields": "title,abstract,url,year,venue,openAccessPdf,externalIds"}
        
        for attempt in range(3):
            try:
                response = requests.post(batch_url, json=payload, params=params_batch, headers=headers, timeout=30)
                if response.status_code == 200:
                    data = response.json()
                    for item in data:
                        if not item: continue
                        pdf_url = resolve_pdf_url(item)
                        if not pdf_url: continue
                        
                        papers.append({
                            "title": item.get("title"),
                            "abstract": item.get("abstract") or "",
                            "url": pdf_url,
                            "date": str(item.get("year")),
                            "venue": item.get("venue")
                        })
                    break # Success, move to next batch
                elif response.status_code == 429:
                    time.sleep(5)
                else:
                    print(f"  Batch Error {response.status_code}")
                    break
            except Exception as e:
                print(f"  Batch Exception: {e}")
                break
                
    return papers

def filter_by_keywords(papers, expanded_mandatory, expanded_bonus):
    """
    Phase 1 Filter: Local metadata check with TIERED logic.
    
    Constraint:
    - MUST contain at least one variation from EACH mandatory keyword group.
    - (Bonus keywords are ignored in this hard filter phase, used only for scoring).
    
    Structure of expanded_mandatory:
    {
       "Vision-Language Models": ["Vision-Language Models", "Vision Language Models", ...],
       "Hallucination": ["Hallucination", "Hallucinations", ...]
    }
    """
    filtered = []
    
    # Exclusion terms
    exclusion_terms = ["survey", "review", "benchmark", "comprehensive evaluation", "roadmap"]
    
    for p in papers:
        title_lower = p["title"].lower()
        
        # Exclusion Check
        if any(term in title_lower for term in exclusion_terms):
            continue
            
        text = (p["title"] + " " + p["abstract"]).lower()
        
        # Mandatory Check: AND logic across groups
        all_mandatory_met = True
        
        for root_keyword, variations in expanded_mandatory.items():
            # OR logic within group (e.g. "Hallucination" OR "Hallucinations")
            group_met = False
            for v in variations:
                if v.lower() in text:
                    group_met = True
                    break
            
            if not group_met:
                all_mandatory_met = False
                break
        
        if all_mandatory_met:
            filtered.append(p)
            
    return filtered

def download_pdf(url, save_dir=INPUT_DIR):
    """
    Downloads PDF from URL to save_dir.
    Returns the local file path.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    # Handle Semantic Scholar / ACL URLs that don't end in .pdf or have query params
    # Example: https://ojs.aaai.org/index.php/AAAI/article/view/27918/27858
    # We need a better filename strategy. Use title slug if possible?
    # But download_pdf only takes URL.
    # Let's generate a hash or simple name from URL if it looks weird.
    
    import hashlib
    url_hash = hashlib.md5(url.encode()).hexdigest()[:10]
    
    filename = url.split("/")[-1]
    if "?" in filename:
        filename = filename.split("?")[0]
        
    if not filename.lower().endswith(".pdf"):
        filename = f"{filename}_{url_hash}.pdf"
    
    # Sanitize filename
    filename = "".join([c for c in filename if c.isalpha() or c.isdigit() or c in (' ', '.', '_', '-')]).strip()
    save_path = os.path.join(save_dir, filename)
    
    if os.path.exists(save_path):
        # Check if empty (failed download)
        if os.path.getsize(save_path) > 1024:
             print(f"File already exists: {save_path}")
             return save_path
        else:
             print(f"File exists but small/empty, re-downloading: {save_path}")
             os.remove(save_path)
        
    try:
        print(f"Downloading {url}...")
        # Add headers to mimic browser to avoid 403 on some sites (AAAI, etc)
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers, stream=True, timeout=30)
        
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            
            # Verify download size
            if os.path.getsize(save_path) < 1000: # Less than 1KB is likely an error page
                print(f"Warning: Downloaded file too small ({os.path.getsize(save_path)} bytes). Likely an error page.")
                # Don't delete immediately, maybe inspect? Or treat as failure.
                return None
                
            return save_path
        else:
            print(f"Failed to download {url}: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return None
