import fitz  # PyMuPDF
import re
import os
from config import FIGURE_KEYWORDS, OUTPUT_DIR
from PIL import Image

def extract_all_captions(pdf_path):
    """
    Extracts all figure captions from the PDF.
    Returns a list of dicts: {'page_num': int, 'figure_id': str, 'caption_text': str, 'rect': fitz.Rect}
    """
    captions = []
    try:
        doc = fitz.open(pdf_path)
        # Limit to first 15 pages to speed up
        max_pages = 15
        
        for page_num, page in enumerate(doc):
            if page_num >= max_pages:
                break
                
            text_blocks = page.get_text("blocks")
            # blocks: (x0, y0, x1, y1, "lines\n...", block_no, block_type)
            
            for block in text_blocks:
                block_text = block[4].strip()
                # Strict Regex: Starts with "Figure X" or "Fig. X"
                match = re.match(r'^(?:Figure|Fig\.?)\s+(\d+)', block_text, re.IGNORECASE)
                if match:
                    fig_num = match.group(1)
                    fig_id = f"Figure {fig_num}"
                    captions.append({
                        "page_num": page_num,
                        "figure_id": fig_id,
                        "caption_text": block_text.replace('\n', ' '), # Clean newlines
                        "rect": fitz.Rect(block[:4])
                    })
        return captions
    except Exception as e:
        print(f"Error extracting captions from {pdf_path}: {e}")
        return []

def crop_specific_figure(pdf_path, target_figure_id, captions, output_dir=OUTPUT_DIR):
    """
    Crops the specific figure identified by target_figure_id (e.g., "Figure 1").
    Uses drawing/image detection first, then text gap fallback.
    
    Args:
        pdf_path (str): Path to PDF file.
        target_figure_id (str): ID of the figure to crop (e.g. "Figure 1").
        captions (list): List of caption dicts.
        output_dir (str): Directory to save the cropped image. Defaults to global OUTPUT_DIR.
    """
    # Find the target caption in the list
    target_caption = None
    for cap in captions:
        # Match "Figure 1" vs "Figure 1" (case insensitive)
        # Or check if target_figure_id is IN the caption text (e.g. "Figure 1: ...")
        if target_figure_id.lower() in cap["figure_id"].lower():
            target_caption = cap
            break
            
    if not target_caption:
        print(f"Target figure {target_figure_id} not found in captions list.")
        return None

    page_num = target_caption["page_num"]
    caption_rect = target_caption["rect"]
    
    doc = fitz.open(pdf_path)
    page = doc[page_num]
    
    # Crop Strategy 1: Look for Drawings/Images immediately above the caption
    # We define a search area above the caption.
    # Top of search area: Top of page? Or reasonable limit (e.g. 500 points)?
    # Let's search the whole area above the caption.
    
    search_rect = fitz.Rect(0, 0, page.rect.width, caption_rect.y0)
    
    # Get all drawings and images in this area
    drawings = page.get_drawings()
    images = page.get_images() # Returns list of image info, not rects directly?
    # get_image_info(xrefs=True) gives rects? No, get_image_rects(xref) does.
    
    candidate_rects = []
    
    # 1. Check Drawings (Vector Graphics)
    for draw in drawings:
        r = draw["rect"]
        # Ignore full-page background rectangles (common in PDFs)
        if r.width > page.rect.width * 0.95 and r.height > page.rect.height * 0.95:
            continue
            
        if r.intersects(search_rect):
            # Check if it is *strictly* above the caption (with small overlap allowed)
            if r.y1 <= caption_rect.y0 + 10: 
                candidate_rects.append(r)
    
    # 2. Check Images (Raster)
    image_list = page.get_images()
    for img in image_list:
        xref = img[0]
        # An image can appear multiple times, get all rects
        img_rects = page.get_image_rects(xref)
        for r in img_rects:
             if r.intersects(search_rect):
                if r.y1 <= caption_rect.y0 + 10:
                    candidate_rects.append(r)

    final_crop_rect = None
    
    if candidate_rects:
        # Combine all candidate rects that are "close" to the caption
        # Heuristic: The figure is likely the union of all objects just above the caption.
        # But there might be header text or other figures above.
        # We only want objects that are *closest* to the caption.
        
        # Sort by y1 (bottom), descending (closest to caption first)
        candidate_rects.sort(key=lambda x: x.y1, reverse=True)
        
        # Take the bottom-most object
        base_rect = candidate_rects[0]
        
        # Expand to include other objects that are vertically close to this one (e.g. parts of same diagram)
        # Threshold: 20 points gap
        union_rect = base_rect
        
        for r in candidate_rects[1:]:
            # If r is close to union_rect
            if r.y1 >= union_rect.y0 - 50: # If bottom of r is near top of union
                 union_rect = union_rect | r # Union
            else:
                # Stop if we hit a large gap (likely another figure or header)
                pass
                
        # Add some padding
        padding = 10
        final_crop_rect = union_rect + (-padding, -padding, padding, padding) # (x0, y0, x1, y1) expansion
        
        # Ensure it doesn't overlap caption too much
        if final_crop_rect.y1 > caption_rect.y0:
            final_crop_rect.y1 = caption_rect.y0
            
        # Ensure it doesn't go off-page
        final_crop_rect = final_crop_rect & page.rect

    else:
        # Fallback: Text Gap Heuristic, constrained by caption column
        print("No vector/image objects found. Using Text Gap fallback.")
        text_blocks = page.get_text("blocks")
        bottom = caption_rect.y0
        
        # Find the highest point of any text block strictly above the caption
        candidates_y1 = [b[3] for b in text_blocks if fitz.Rect(b[:4]).y1 < bottom - 5]
        top = max(candidates_y1) if candidates_y1 else 0
        
        # HORIZONTAL FIX: Define the crop area based on the caption's horizontal
        # position, not the full page width. Add some padding.
        h_padding = 20
        left = max(0, caption_rect.x0 - h_padding)
        right = min(page.rect.width, caption_rect.x1 + h_padding)
        
        # Create a crop box constrained to the inferred column.
        final_crop_rect = fitz.Rect(left, top, right, bottom)
        
        # Ensure the box is within the page boundaries.
        final_crop_rect = final_crop_rect & page.rect

    # Sanity Check: Height
    if final_crop_rect.height < 20:
        print("Crop height too small.")
        return None

    # Render
    zoom = 2.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, clip=final_crop_rect)
    
    # White-Space Check (Is it blank?)
    # Convert to PIL to check entropy or extrema
    # But we want to avoid extra deps if possible. PyMuPDF pixmap has samples.
    # Simple check: if all bytes are 255 (white).
    
    # Check if pixmap is valid
    if pix.width < 10 or pix.height < 10:
         return None

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    output_filename = f"{os.path.basename(pdf_path)}_{target_figure_id.replace(' ', '_')}.png"
    output_path = os.path.join(output_dir, output_filename)
    pix.save(output_path)
    
    # Double check with PIL if it's all white (optional but good)
    # If the file size is very small, it might be blank.
    if os.path.getsize(output_path) < 1000: # < 1KB is suspicious for a figure
        print("Warning: Cropped image seems empty/too small.")
        
    print(f"Saved crop to {output_path}")
    return output_path

# Keep the old function for compatibility if needed, but we probably won't use it.
def extract_text_and_metadata(pdf_path):
    """
    Extracts the title and abstract from the first page of the PDF using a robust method.
    """
    try:
        doc = fitz.open(pdf_path)
        page = doc[0]
        
        # --- Robust Title Extraction ---
        dict_blocks = page.get_text("dict")["blocks"]
        max_size = 0
        title_block_no = -1

        # Find the block number with the largest font size, assuming it's the title
        for b in dict_blocks:
            if b['type'] == 0 and "lines" in b:  # 0 is a text block
                for line in b["lines"]:
                    for span in line["spans"]:
                        text = span["text"].strip()
                        # Simple heuristic to avoid picking up headers/footers
                        if "arxiv" in text.lower() or "http" in text.lower() or "cs.cv" in text.lower():
                            continue
                        if span["size"] > max_size:
                            max_size = span["size"]
                            title_block_no = b["number"]

        title = ""
        if title_block_no != -1:
            # Extract the full, properly formatted text from the identified title block
            text_blocks = page.get_text("blocks")
            for b in text_blocks:
                if b[5] == title_block_no:
                    # Replace newlines with spaces for a clean, single-line title
                    title = b[4].replace('\n', ' ').strip()
                    break
        
        # Fallback if the above logic fails
        if not title:
            # Fallback to the first non-empty line of the page
            full_text = page.get_text("text")
            lines = full_text.split('\n')
            for line in lines:
                if line.strip():
                    title = line.strip()
                    break
            if not title:
                 title = "Title not found" # Ultimate fallback

        # --- Abstract Extraction (with improvements) ---
        text = page.get_text("text")
        abstract = ""
        # Regex to find abstract, stopping at "introduction" or "1." (case-insensitive)
        abstract_match = re.search(r'(?i)abstract[:\s]*(.*?)(?:\n\s*(?:introduction|1\.|i\.\s*ntroduction))', text, re.DOTALL)
        if abstract_match:
            abstract = abstract_match.group(1).strip().replace('\n', ' ')
        else:
            # Fallback if regex fails
            start_idx = text.lower().find("abstract")
            if start_idx != -1:
                end_idx = text.lower().find("introduction", start_idx)
                if end_idx == -1:
                    end_idx = text.lower().find("1.", start_idx)
                if end_idx != -1:
                    abstract = text[start_idx+8:end_idx].strip().replace('\n', ' ')
                else:
                    # Grab a chunk of text if no end is found
                    abstract = text[start_idx+8:start_idx+1500].strip().replace('\n', ' ')

        return {
            "title": title,
            "abstract": abstract if abstract else "Abstract not found."
        }
    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")
        return {"title": "Error extracting title", "abstract": ""}
