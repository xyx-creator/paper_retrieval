import os
import sys
import glob
import argparse
from datetime import datetime
from config import INPUT_DIR, OUTPUT_DIR, RELEVANCE_THRESHOLD, KEYWORDS
from modules.pdf_processor import extract_text_and_metadata, extract_all_captions, crop_specific_figure
from modules.glm_agent import GLMAgent
from modules.report_generator import generate_consolidated_report
from modules.paper_source import search_arxiv, search_semantic_scholar, filter_by_keywords, download_pdf, search_dblp, batch_fetch_s2

def main():
    parser = argparse.ArgumentParser(description="GLM Dual-Brain Paper Retrieval Agent")
    parser.add_argument("--source", choices=["local", "arxiv", "dblp"], default="local", help="Paper source")
    parser.add_argument("--days", type=int, default=1, help="Number of days to search (arXiv)")
    parser.add_argument("--venue", type=str, help="Venue for Semantic Scholar/DBLP (e.g., CVPR)")
    parser.add_argument("--year", type=int, help="Year for Semantic Scholar/DBLP")
    
    # New Arguments for Tiered Keywords
    parser.add_argument("--mandatory", type=str, help="Comma-separated MANDATORY keywords (Must match)")
    parser.add_argument("--bonus", type=str, help="Comma-separated BONUS keywords (Boost score)")
    
    # Deprecated but kept for backward compatibility (maps to mandatory)
    parser.add_argument("--keywords", type=str, help="[Deprecated] Comma-separated keywords")
    
    args = parser.parse_args()
    
    print("Initializing GLM Dual-Brain Agent...")
    try:
        agent = GLMAgent()
    except Exception as e:
        print(f"Failed to initialize GLMAgent: {e}")
        return

    # Ensure directories exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    if not os.path.exists(INPUT_DIR):
        os.makedirs(INPUT_DIR)
    IMAGES_DIR = os.path.join(OUTPUT_DIR, "images")
    if not os.path.exists(IMAGES_DIR):
        os.makedirs(IMAGES_DIR)

    # 1. Keyword Processing
    # Determine Mandatory and Bonus keywords
    mandatory_raw = []
    bonus_raw = []
    
    if args.mandatory:
        mandatory_raw = [k.strip() for k in args.mandatory.split(",")]
    elif args.keywords:
        mandatory_raw = [k.strip() for k in args.keywords.split(",")]
    else:
        mandatory_raw = KEYWORDS["mandatory"]
        
    if args.bonus:
        bonus_raw = [k.strip() for k in args.bonus.split(",")]
    else:
        # Only use default bonus if using default mandatory (i.e. user didn't override)
        if not args.mandatory and not args.keywords:
            bonus_raw = KEYWORDS["bonus"]
            
    print(f"\n[Step 0] Keywords Configuration:")
    print(f"  Mandatory: {mandatory_raw}")
    print(f"  Bonus: {bonus_raw}")
    
    # Batch Expand Mandatory (Strict Mode)
    print(f"  Expanding {len(mandatory_raw)} mandatory keywords (Strict)...")
    expanded_mandatory = agent.expand_keywords_batch(mandatory_raw, mode="strict")
    
    # Bonus keywords are not expanded to save time, using raw keywords instead.
    print(f"  Skipping expansion for {len(bonus_raw)} bonus keywords.")
    expanded_bonus = bonus_raw
            
    print(f"  Expanded Mandatory: {expanded_mandatory}")
    print(f"  Bonus (Raw): {expanded_bonus}")
    
    # Flatten mandatory for arXiv search query (OR logic between all mandatory terms? No, usually AND between groups)
    # For arXiv query, we just take the first term of each mandatory group + first term of bonus?
    # Or just use the first mandatory term.
    # search_keywords for APIs usually needs a list.
    search_keywords_flat = []
    for k in mandatory_raw:
        search_keywords_flat.append(k)
    # Add bonus for search context? Maybe better to keep search broad (mandatory only) and filter later.
    
    # 2. Search & Filter Phase 1 (Metadata)
    papers_to_process = []
    
    if args.source == "local":
        print("\n[Source] Local PDF Mode")
        pdf_files = glob.glob(os.path.join(INPUT_DIR, "*.pdf"))
        # Sort by modification time to process newest first
        pdf_files.sort(key=os.path.getmtime, reverse=True)
        
        for pdf_path in pdf_files:
            if len(papers_to_process) >= 15:
                print("Limit reached (15 papers) for local processing.")
                break
                
            # For local files, we extract metadata on the fly
            papers_to_process.append({
                "local_path": pdf_path,
                "title": os.path.basename(pdf_path), # Temp title
                "abstract": "", # Will extract later
                "venue": "Local"
            })
            
    else:
        print(f"\n[Source] Fetching metadata from {args.source.upper()}...")
        raw_papers = []
        if args.source == "arxiv":
            # Pass expanded mandatory for search query? 
            # search_arxiv takes a list. Let's pass the raw mandatory list.
            raw_papers = search_arxiv(mandatory_raw, days=args.days)
        elif args.source == "s2":
            if not args.venue or not args.year:
                print("Error: --venue and --year are required for Semantic Scholar mode.")
                return
            raw_papers = search_semantic_scholar(args.venue, args.year)
        elif args.source == "dblp":
            if not args.venue or not args.year:
                print("Error: --venue and --year are required for DBLP mode.")
                return
            
            # Node 1: DBLP + S2 Batch
            print(f"Step 1 (Street Sweep): Fetching from DBLP ({args.venue} {args.year})...")
            dblp_papers = search_dblp(args.venue, args.year)
            
            print(f"Step 2 (Precision Snipe): Batch fetching details from Semantic Scholar...")
            raw_papers = batch_fetch_s2(dblp_papers)
            
        print(f"Fetched {len(raw_papers)} candidates.")
        
        # Filter Phase 1 (Updated with Tiered Logic)
        print("\n[Filter Phase 1] Metadata Keyword Matching (Tiered)...")
        filtered_phase1 = filter_by_keywords(raw_papers, expanded_mandatory, expanded_bonus)
        print(f"Survivors: {len(filtered_phase1)}")
        
        # Filter Phase 2 (GLM Scoring)
        print("\n[Filter Phase 2] GLM-4 Relevance Scoring (Metadata) - PARALLEL MODE...")
        
        # Parallelize Phase 2 Scoring
        filtered_phase2 = []
        
        # We need to preserve order or just collect high scores?
        # Let's collect all high scores.
        
        import concurrent.futures
        import threading
        score_lock = threading.Lock()
        
        def score_paper(p):
            import time
            import random
            
            # Simple retry with backoff logic for API stability
            for attempt in range(3):
                try:
                    # Use updated score_relevance signature
                    score = agent.score_relevance(p["abstract"], ", ".join(mandatory_raw), ", ".join(bonus_raw))
                    with score_lock:
                        print(f"  - {p['title'][:50]}... -> Score: {score}")
                    if score >= RELEVANCE_THRESHOLD:
                        p["score"] = score
                        return p
                    return None # Score too low, don't retry
                except Exception as e:
                    if attempt < 2:
                        wait_time = random.uniform(2, 5) * (attempt + 1)
                        # Optional: print(f"Retry scoring {p['title'][:20]} in {wait_time:.1f}s due to error: {e}")
                        time.sleep(wait_time)
                    else:
                        with score_lock:
                            print(f"Error scoring {p['title'][:30]}: {e}")
            return None

        # Max workers set to 5 for faster processing
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            # Submit all tasks
            future_to_paper = {executor.submit(score_paper, p): p for p in filtered_phase1}
            
            for future in concurrent.futures.as_completed(future_to_paper):
                result = future.result()
                if result:
                    filtered_phase2.append(result)
                    
        # Sort by score descending
        filtered_phase2.sort(key=lambda x: x["score"], reverse=True)
        
        # Take top 10
        papers_to_process = filtered_phase2[:10]
        print(f"Top {len(papers_to_process)} papers selected after parallel scoring.")
                
    print(f"\n[Pipeline] {len(papers_to_process)} papers selected for Deep Analysis.")
    
    # 3. Download & Process (Parallel)
    results_list = []
    
    # Import concurrent.futures
    import concurrent.futures
    import threading
    
    # Lock for thread-safe list appending and printing
    results_lock = threading.Lock()
    
    def process_single_paper(paper_info, idx, total):
        try:
            with results_lock:
                print(f"\n[Thread Start] Processing Paper {idx+1}/{total}: {paper_info['title'][:50]}...")
            
            # Download if needed
            pdf_path = paper_info.get("local_path")
            if not pdf_path and "url" in paper_info:
                # Assuming download_pdf is thread-safe or file system handles it
                # We should probably use a unique temp name if possible, but download_pdf uses hash/name
                pdf_path = download_pdf(paper_info['url'], INPUT_DIR)
                
            if not pdf_path or not os.path.exists(pdf_path):
                with results_lock:
                    print(f"Skipping {idx+1}: PDF not found or download failed.")
                return None
                
            filename = os.path.basename(pdf_path)
            
            # Node 1: Metadata Extraction
            raw_metadata = extract_text_and_metadata(pdf_path)
            # In local mode, prioritize the title from PDF content over the filename.
            title = raw_metadata.get("title") if raw_metadata.get("title") and "Error" not in raw_metadata.get("title") else os.path.basename(pdf_path)
            abstract = raw_metadata.get("abstract", "")

            # If abstract is still empty in local mode, use the first page's text.
            if not abstract or len(abstract) < 100:
                import fitz
                try:
                    doc = fitz.open(pdf_path)
                    first_page_text = doc[0].get_text("text")
                    doc.close()
                    abstract = first_page_text[:2000] # Use first 2000 chars as abstract
                except Exception as e:
                    print(f"Could not extract fallback text from {pdf_path}: {e}")
            
            # Phase 2b: Strict Re-Scoring (Deep Analysis)
            # Metadata scoring (Phase 1) was fast/rough. Now we have full abstract from PDF (sometimes better).
            # But mostly we want to re-verify before spending Vision credits.
            
            # Use score from Phase 2 if available, else calculate (Local mode)
            score = paper_info.get("score")
            
            # If we are in local mode, we haven't scored yet. 
            # If we are in online mode, we have a preliminary score.
            # Let's re-score strictly if it's local, or trust the preliminary high score?
            # User asked for "Stricter Judge". The agent.score_relevance has been updated to be strict.
            # If we already scored in Phase 2 loop, that used the NEW strict prompt.
            # So we can trust `paper_info["score"]`.
            
            if score is None:
                 score = agent.score_relevance(abstract, ", ".join(mandatory_raw), ", ".join(bonus_raw))
            
            # Double check threshold (in case Local mode or logic change)
            if score < RELEVANCE_THRESHOLD:
                 with results_lock:
                     print(f"Skipping {idx+1} (Local/Re-eval): Low relevance {score}.")
                 return None

            with results_lock:
                print(f"[{idx+1}] Title: {title}")
                print(f"[{idx+1}] Relevance: {score}/10")

            # Node 3: Smart Figure Selection
            captions = extract_all_captions(pdf_path)
            if not captions:
                with results_lock:
                    print(f"[{idx+1}] No captions found.")
                return None
                
            captions_text_list = [f"{c['figure_id']}: {c['caption_text'][:200]}..." for c in captions]
            best_figure_id = agent.select_best_figure(captions_text_list)
            
            if not best_figure_id:
                with results_lock:
                    print(f"[{idx+1}] No relevant figure found.")
                return None
                
            # Image Org
            paper_slug = os.path.splitext(filename)[0].replace(" ", "_")[:50]
            import re
            paper_slug = re.sub(r'[^a-zA-Z0-9_]', '', paper_slug)
            
            paper_image_dir = os.path.join(IMAGES_DIR, paper_slug)
            
            # Ensure dir creation is thread safe (os.makedirs is usually fine)
            if not os.path.exists(paper_image_dir):
                try:
                    os.makedirs(paper_image_dir, exist_ok=True)
                except:
                    pass

            image_path = crop_specific_figure(pdf_path, best_figure_id, captions, output_dir=paper_image_dir)
            
            if not image_path:
                return None
                
            # Node 4: Dual-Brain
            import fitz
            doc = fitz.open(pdf_path)
            first_page_text = doc[0].get_text("text")
            doc.close()
            
            text_analysis = agent.analyze_text_brain(abstract, introduction=first_page_text)
            vision_analysis = agent.analyze_vision_brain(image_path)
            # Update synthesis to use mandatory keywords for context
            synthesis = agent.synthesize_report(text_analysis, vision_analysis, score, ", ".join(mandatory_raw))
            
            result = {
                "title": title,
                "filename": filename,
                "score": score,
                "image_path": image_path,
                "synthesis": synthesis
            }
            
            with results_lock:
                results_list.append(result)
                print(f"[{idx+1}] Analysis Complete.")
                
            return result
            
        except Exception as e:
            with results_lock:
                print(f"Error processing paper {idx+1}: {e}")
            return None

    # Run Parallel Execution
    # Max workers set to 5 for Deep Analysis (PDF download + Dual Brain)
    print(f"\n[Parallel Execution] Starting processing with max_workers=5...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        for i, paper_info in enumerate(papers_to_process):
            futures.append(executor.submit(process_single_paper, paper_info, i, len(papers_to_process)))
            
        # Wait for all to complete
        concurrent.futures.wait(futures)
        
    # Node 5: Report
    if results_list:
        print("\nGenerating Consolidated Report...")
        date_str = datetime.now().strftime('%Y-%m-%d')
        source_tag = args.source.upper()
        if args.source == "dblp" or args.source == "s2":
            source_tag = f"{args.venue}{args.year}"
            
        # Use first mandatory keyword for filename tag
        keywords_tag = mandatory_raw[0].replace(" ", "-") if mandatory_raw else "Papers"
        report_filename = f"{source_tag}_{keywords_tag}_{date_str}.md"
        report_path = os.path.join(OUTPUT_DIR, report_filename)
        
        generate_consolidated_report(results_list, report_path, ", ".join(mandatory_raw))
    else:
        print("\nNo papers processed.")

    print("\nProcessing Complete!")

if __name__ == "__main__":
    main()
