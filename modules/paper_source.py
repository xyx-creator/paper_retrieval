from __future__ import annotations

import asyncio
import hashlib
import os
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from html import unescape
from typing import Any, Dict, List, Optional, Sequence
from urllib.parse import urljoin

import httpx

from config import ARXIV_CATEGORY, INPUT_DIR, S2_API_KEY


async def search_arxiv(query_keywords: Sequence[str], days: int = 1) -> List[Dict[str, Any]]:
    """
    Search arXiv for papers matching query_keywords within the last N days.
    """
    # Kept for API compatibility. Current strategy is category + date window.
    _ = query_keywords

    base_url = "https://export.arxiv.org/api/query"
    max_results = max(200, days * 150)
    params = {
        "search_query": ARXIV_CATEGORY,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
        "start": 0,
        "max_results": max_results,
    }

    max_retries = 3
    for attempt in range(max_retries):
        try:
            print(f"Fetching from arXiv: {base_url} (Attempt {attempt + 1})")
            async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
                response = await client.get(base_url, params=params)

            if response.status_code == 200:
                root = ET.fromstring(response.content)
                namespace = {"atom": "http://www.w3.org/2005/Atom"}

                papers: List[Dict[str, Any]] = []
                entries = root.findall("atom:entry", namespace)
                print(f"arXiv returned {len(entries)} entries.")

                for entry in entries:
                    published_node = entry.find("atom:published", namespace)
                    if published_node is None or not published_node.text:
                        continue

                    published = published_node.text
                    pub_dt = datetime.strptime(published, "%Y-%m-%dT%H:%M:%SZ")
                    age_days = (time.time() - pub_dt.timestamp()) / (24 * 3600)
                    if age_days > days:
                        continue

                    title_node = entry.find("atom:title", namespace)
                    summary_node = entry.find("atom:summary", namespace)
                    id_node = entry.find("atom:id", namespace)
                    if title_node is None or summary_node is None or id_node is None:
                        continue

                    title = title_node.text.replace("\n", " ").strip()
                    summary = summary_node.text.replace("\n", " ").strip()
                    id_url = id_node.text

                    pdf_link = ""
                    for link in entry.findall("atom:link", namespace):
                        if link.attrib.get("title") == "pdf":
                            pdf_link = link.attrib.get("href", "")
                            break

                    if not pdf_link:
                        pdf_link = id_url.replace("abs", "pdf")

                    papers.append(
                        {
                            "title": title,
                            "abstract": summary,
                            "url": pdf_link,
                            "date": published,
                            "venue": "arXiv",
                        }
                    )
                return papers

            if response.status_code in {429, 503}:
                wait_time = (attempt + 1) * 5
                print(
                    f"arXiv Rate Limit/Unavailable ({response.status_code}). "
                    f"Wait {wait_time}s..."
                )
                await asyncio.sleep(wait_time)
                continue

            print(f"Error fetching from arXiv: {response.status_code}")
            return []
        except Exception as exc:
            print(f"Exception in search_arxiv (Attempt {attempt + 1}): {exc}")
            await asyncio.sleep(5)

    print("Max retries reached for arXiv.")
    return []


def resolve_pdf_url(item: Dict[str, Any]) -> Optional[str]:
    """
    Fallback strategy to find a PDF URL.
    Priority:
    1. openAccessPdf.url
    2. ArXiv ID
    3. ACL Anthology
    """
    if item.get("openAccessPdf") and item["openAccessPdf"].get("url"):
        return item["openAccessPdf"]["url"]

    external_ids = item.get("externalIds") or {}
    if "ArXiv" in external_ids:
        return f"https://arxiv.org/pdf/{external_ids['ArXiv']}.pdf"
    if "ACL" in external_ids:
        return f"https://aclanthology.org/{external_ids['ACL']}.pdf"
    return None


async def search_dblp(venue: str, year: int) -> List[Dict[str, Any]]:
    """
    Search DBLP for papers in a specific venue and year.
    """
    base_url = "https://dblp.org/search/publ/api"
    query = f"venue:{venue}: year:{year}:"

    print(f"Searching DBLP for {venue} {year}...")
    papers_data: List[Dict[str, Any]] = []

    h = 1000
    f = 0
    async with httpx.AsyncClient(timeout=30.0) as client:
        while True:
            params = {
                "q": query,
                "format": "json",
                "h": h,
                "f": f,
            }

            try:
                print(f"  Fetching DBLP offset {f}...")
                response = await client.get(base_url, params=params)

                if response.status_code != 200:
                    print(f"  DBLP Error {response.status_code}")
                    break

                data = response.json()
                result = data.get("result", {})
                hits = result.get("hits", {})

                if not hits or "hit" not in hits:
                    print("  No more hits.")
                    break

                hit_list = hits.get("hit") or []
                if isinstance(hit_list, dict):
                    hit_list = [hit_list]
                if not hit_list:
                    break

                print(f"  Got {len(hit_list)} hits.")
                for hit in hit_list:
                    info = hit.get("info", {})
                    title = str(info.get("title", "")).strip()
                    if title.endswith("."):
                        title = title[:-1]
                    doi = info.get("doi")

                    if title:
                        papers_data.append(
                            {
                                "title": title,
                                "doi": doi,
                                "year": info.get("year"),
                                "venue": info.get("venue"),
                            }
                        )

                total = int(hits.get("@total", 0))
                f += h
                if f >= total:
                    break

                await asyncio.sleep(0.5)
            except Exception as exc:
                print(f"  DBLP Exception: {exc}")
                break

    print(f"DBLP returned {len(papers_data)} papers.")
    return papers_data


async def batch_fetch_s2(papers_data: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Use Semantic Scholar batch + fallback title search to enrich papers.
    """
    if not papers_data:
        return []

    batch_url = "https://api.semanticscholar.org/graph/v1/paper/batch"
    search_url = "https://api.semanticscholar.org/graph/v1/paper/search"

    headers: Dict[str, str] = {}
    if S2_API_KEY:
        headers["x-api-key"] = S2_API_KEY
        print("Using S2 API Key.")

    batch_size = 100
    valid_papers_with_doi = [p for p in papers_data if p.get("doi")]
    papers_without_doi = [p for p in papers_data if not p.get("doi")]

    print(
        f"Found {len(valid_papers_with_doi)} papers with DOIs "
        f"out of {len(papers_data)}."
    )
    print(
        f"Found {len(papers_without_doi)} papers WITHOUT DOIs. "
        "Attempting title search fallback..."
    )

    enriched_papers: List[Dict[str, Any]] = []

    async with httpx.AsyncClient(timeout=30.0) as client:
        if valid_papers_with_doi:
            seen_dois = set()
            unique_papers_doi = []
            for paper in valid_papers_with_doi:
                doi = paper["doi"]
                if doi in seen_dois:
                    continue
                seen_dois.add(doi)
                unique_papers_doi.append(paper)

            all_dois = [p["doi"] for p in unique_papers_doi]
            print(f"Unique DOIs to fetch: {len(all_dois)}")

            for i in range(0, len(all_dois), batch_size):
                batch_dois = all_dois[i : i + batch_size]
                print(f"  [DOI Batch] Fetching {i}-{i + len(batch_dois)}...")

                payload = {"ids": batch_dois}
                params_batch = {
                    "fields": "title,abstract,url,year,venue,openAccessPdf,externalIds"
                }

                for _attempt in range(3):
                    try:
                        response = await client.post(
                            batch_url,
                            json=payload,
                            params=params_batch,
                            headers=headers,
                        )
                        if response.status_code == 200:
                            data = response.json()
                            for item in data:
                                if not item:
                                    continue
                                pdf_url = resolve_pdf_url(item)
                                if (
                                    not pdf_url
                                    and item.get("externalIds", {}).get("DOI")
                                ):
                                    pdf_url = (
                                        f"https://doi.org/{item['externalIds']['DOI']}"
                                    )

                                enriched_papers.append(
                                    {
                                        "title": item.get("title"),
                                        "abstract": item.get("abstract") or "",
                                        "url": pdf_url,
                                        "date": str(item.get("year")),
                                        "venue": item.get("venue"),
                                    }
                                )
                            break

                        if response.status_code == 429:
                            print("  Rate Limit. Waiting 5s...")
                            await asyncio.sleep(5)
                            continue

                        print(f"  Batch Error {response.status_code}")
                        break
                    except Exception as exc:
                        print(f"  Batch Exception: {exc}")
                        break

        if papers_without_doi:
            max_fallback = 50
            fallback_papers = papers_without_doi[:max_fallback]
            print(
                "Fallback: Searching S2 by title for first "
                f"{max_fallback} papers without DOI..."
            )

            for i, paper in enumerate(fallback_papers):
                title = str(paper.get("title", ""))
                print(
                    f"  [Title Search] {i + 1}/{len(fallback_papers)}: "
                    f"{title[:30]}..."
                )

                params = {
                    "query": title,
                    "fields": "title,abstract,url,year,venue,openAccessPdf,externalIds",
                    "limit": 1,
                }

                try:
                    await asyncio.sleep(1.0)
                    response = await client.get(
                        search_url,
                        params=params,
                        headers=headers,
                    )

                    if response.status_code == 200:
                        data = response.json()
                        if data.get("data"):
                            item = data["data"][0]
                            pdf_url = resolve_pdf_url(item)
                            if not pdf_url and item.get("externalIds", {}).get("DOI"):
                                pdf_url = f"https://doi.org/{item['externalIds']['DOI']}"

                            enriched_papers.append(
                                {
                                    "title": item.get("title"),
                                    "abstract": item.get("abstract") or "",
                                    "url": pdf_url,
                                    "date": str(item.get("year")),
                                    "venue": item.get("venue"),
                                }
                            )
                    elif response.status_code == 429:
                        print("  Rate Limit. Skipping...")
                        await asyncio.sleep(2)
                except Exception as exc:
                    print(f"  Search Exception: {exc}")

    return enriched_papers


async def filter_by_keywords(
    papers: Sequence[Dict[str, Any]],
    expanded_mandatory: Dict[str, List[str]],
    expanded_bonus: Sequence[str],
) -> List[Dict[str, Any]]:
    """
    Local metadata filter with tiered logic.
    """
    _ = expanded_bonus
    filtered: List[Dict[str, Any]] = []
    exclusion_terms = [
        "survey",
        "review",
        "benchmark",
        "comprehensive evaluation",
        "roadmap",
    ]

    for paper in papers:
        title = str(paper.get("title", ""))
        abstract = str(paper.get("abstract", ""))
        title_lower = title.lower()
        if any(term in title_lower for term in exclusion_terms):
            continue

        text = f"{title} {abstract}".lower()
        all_mandatory_met = True

        for _root_keyword, variations in expanded_mandatory.items():
            group_met = any(v.lower() in text for v in variations)
            if not group_met:
                all_mandatory_met = False
                break

        if all_mandatory_met:
            filtered.append(dict(paper))

    return filtered


async def download_pdf(url: str, save_dir: str = INPUT_DIR) -> Optional[str]:
    """
    Download PDF to local disk, with landing-page fallback resolution.
    """

    def _is_pdf_file(file_path: str) -> bool:
        try:
            with open(file_path, "rb") as file:
                return file.read(4) == b"%PDF"
        except Exception:
            return False

    def _extract_pdf_url_from_html(html_text: str, base_url: str) -> Optional[str]:
        meta_match = re.search(
            r'<meta[^>]*name=["\']citation_pdf_url["\'][^>]*content=["\']([^"\']+)["\']',
            html_text,
            re.IGNORECASE,
        )
        if meta_match:
            return urljoin(base_url, unescape(meta_match.group(1)))

        download_match = re.search(
            r'href=["\']([^"\']*/article/download/[^"\']+)["\']',
            html_text,
            re.IGNORECASE,
        )
        if download_match:
            return urljoin(base_url, unescape(download_match.group(1)))

        pdf_match = re.search(
            r'href=["\']([^"\']*\.pdf(?:\?[^"\']*)?)["\']',
            html_text,
            re.IGNORECASE,
        )
        if pdf_match:
            return urljoin(base_url, unescape(pdf_match.group(1)))

        return None

    os.makedirs(save_dir, exist_ok=True)

    url_hash = hashlib.md5(url.encode("utf-8")).hexdigest()[:10]
    filename = url.rstrip("/").split("/")[-1] or "paper"
    if "?" in filename:
        filename = filename.split("?")[0]
    if not filename.lower().endswith(".pdf"):
        filename = f"{filename}_{url_hash}.pdf"

    filename = "".join(
        [c for c in filename if c.isalpha() or c.isdigit() or c in (" ", ".", "_", "-")]
    ).strip()
    if not filename:
        filename = f"paper_{url_hash}.pdf"

    save_path = os.path.join(save_dir, filename)

    if os.path.exists(save_path):
        if os.path.getsize(save_path) > 1024 and _is_pdf_file(save_path):
            print(f"File already exists: {save_path}")
            return save_path
        print(f"Cached file invalid or too small, re-downloading: {save_path}")
        try:
            os.remove(save_path)
        except Exception:
            pass

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/91.0.4472.124 Safari/537.36"
        ),
        "Accept": "application/pdf,text/html;q=0.9,*/*;q=0.8",
    }

    try:
        print(f"Downloading {url}...")
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            current_url = url
            for _attempt in range(2):
                async with client.stream("GET", current_url, headers=headers) as response:
                    if response.status_code != 200:
                        print(f"Failed to download {current_url}: {response.status_code}")
                        return None

                    content_type = (response.headers.get("Content-Type") or "").lower()
                    iterator = response.aiter_bytes(chunk_size=8192)
                    try:
                        first_chunk = await anext(iterator)
                    except StopAsyncIteration:
                        first_chunk = b""

                    is_pdf = first_chunk.startswith(b"%PDF") or "application/pdf" in content_type
                    if is_pdf:
                        with open(save_path, "wb") as file:
                            if first_chunk:
                                file.write(first_chunk)
                            async for chunk in iterator:
                                if chunk:
                                    file.write(chunk)

                        if os.path.getsize(save_path) < 1000 or not _is_pdf_file(save_path):
                            print(f"Warning: downloaded file is not a valid PDF: {save_path}")
                            try:
                                os.remove(save_path)
                            except Exception:
                                pass
                            return None
                        return save_path

                    body = bytearray(first_chunk)
                    async for chunk in iterator:
                        if chunk:
                            body.extend(chunk)

                    html_text = body.decode("utf-8", errors="ignore")
                    resolved_pdf_url = _extract_pdf_url_from_html(
                        html_text,
                        str(response.url),
                    )

                if resolved_pdf_url and resolved_pdf_url != current_url:
                    print(f"Resolved PDF URL from landing page: {resolved_pdf_url}")
                    current_url = resolved_pdf_url
                    continue

                print(f"Landing page does not contain resolvable PDF link: {current_url}")
                return None

        return None
    except Exception as exc:
        print(f"Error downloading {url}: {exc}")
        return None
