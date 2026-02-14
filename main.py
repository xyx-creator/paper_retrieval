import os
import sys
import glob
from config import INPUT_DIR, OUTPUT_DIR, RELEVANCE_THRESHOLD
from modules.pdf_processor import extract_text_and_metadata, extract_all_captions, crop_specific_figure
from modules.glm_agent import GLMAgent
from modules.report_generator import generate_markdown_report

def main():
    print("Initializing GLM Dual-Brain Agent...")
    
    try:
        agent = GLMAgent()
    except Exception as e:
        print(f"Failed to initialize GLMAgent: {e}")
        return

    # Ensure output directory exists
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Get list of PDF files
    pdf_files = glob.glob(os.path.join(INPUT_DIR, "*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {INPUT_DIR}")
        return

    print(f"Found {len(pdf_files)} papers to process.")

    for pdf_path in pdf_files:
        filename = os.path.basename(pdf_path)
        print(f"\nProcessing: {filename}")
        
        # Node 1: Extract Metadata
        metadata = extract_text_and_metadata(pdf_path)
        title = metadata["title"]
        abstract = metadata["abstract"]
        
        if not abstract or len(abstract) < 50:
            print("Skipping: Abstract too short or not found.")
            continue

        print(f"Title: {title}")
        
        # Node 2: Relevance Scoring
        keywords = "Vision-Language Models, Multimodal Learning, Architecture"
        
        score = agent.score_relevance(abstract, keywords)
        print(f"Relevance Score: {score}/5")
        
        if score < RELEVANCE_THRESHOLD:
            print("Skipping: Low relevance.")
            continue
            
        # Node 3: Smart Figure Selection & Cropping
        print("Extracting all figure captions...")
        captions = extract_all_captions(pdf_path)
        
        if not captions:
            print("No figure captions found in paper.")
            continue
            
        # Extract just the text for the agent
        captions_text_list = [f"{c['figure_id']}: {c['caption_text'][:200]}..." for c in captions]
        
        print("Agent selecting the best architecture figure...")
        best_figure_id = agent.select_best_figure(captions_text_list)
        
        if not best_figure_id:
            print("Agent could not identify a relevant architecture figure.")
            continue
            
        print(f"Agent selected: {best_figure_id}")
        
        image_path = crop_specific_figure(pdf_path, best_figure_id, captions)
        
        if not image_path:
            print(f"Failed to crop {best_figure_id}.")
            continue
            
        # Node 4: Dual-Brain Analysis
        print("Starting Dual-Brain Analysis...")
        
        # Text Brain
        print("- Text Brain (GLM-4.7) analyzing claims...")
        import fitz
        doc = fitz.open(pdf_path)
        first_page_text = doc[0].get_text("text")
        
        text_analysis = agent.analyze_text_brain(abstract, introduction=first_page_text)
        
        # Vision Brain
        print("- Vision Brain (GLM-4.6V) analyzing image...")
        vision_analysis = agent.analyze_vision_brain(image_path)
        
        # Fusion
        print("- Fusion Brain synthesizing report...")
        synthesis = agent.synthesize_report(text_analysis, vision_analysis, score, keywords)
        
        # Node 5: Report Generation
        report_filename = f"{os.path.splitext(filename)[0]}_report.md"
        report_path = os.path.join(OUTPUT_DIR, report_filename)
        
        generate_markdown_report(title, filename, score, image_path, synthesis, report_path)
        
    print("\nProcessing Complete!")

if __name__ == "__main__":
    main()
