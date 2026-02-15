import base64
from zhipuai import ZhipuAI
from config import ZHIPUAI_API_KEY, MODEL_TEXT, MODEL_VISION, MODEL_KEYWORD_EXPANSION

class GLMAgent:
    def __init__(self):
        if not ZHIPUAI_API_KEY:
            raise ValueError("ZHIPUAI_API_KEY not found in environment variables.")
        self.client = ZhipuAI(api_key=ZHIPUAI_API_KEY)

    def _encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def expand_keywords_batch(self, keywords, mode="strict"):
        """
        Step 0: Use GLM-4-Air to expand MULTIPLE keywords in a single batch call.
        Returns a dictionary: {original_keyword: [expansions]}
        """
        if not keywords:
            return {}
            
        keywords_str = ", ".join(keywords)
        
        if mode == "strict":
            prompt = f"""
            You are a linguistic assistant.
            Keywords List: {keywords_str}
            
            Task: For EACH keyword in the list, generate grammatical variations ONLY (plural, singular, tense, hyphenation, abbreviation).
            Do NOT generate synonyms or related concepts.
            
            Constraint: Return MAX 3 variations per keyword.
            
            Output Format:
            Keyword1: Var1, Var2
            Keyword2: Var1, Var2
            ...
            """
        else:
            prompt = f"""
            You are a research assistant.
            Keywords List: {keywords_str}
            
            Task: For EACH keyword in the list, expand into synonyms, acronyms, and related academic terms.
            
            Constraint: Return MAX 5 most relevant terms per keyword.
            
            Output Format:
            Keyword1: Syn1, Syn2, Syn3
            Keyword2: Syn1, Syn2, Syn3
            ...
            """
        
        try:
            response = self.client.chat.completions.create(
                model=MODEL_KEYWORD_EXPANSION,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3 if mode == "strict" else 0.7
            )
            content = response.choices[0].message.content.strip()
            
            # Parse output
            result = {}
            lines = content.split('\n')
            for line in lines:
                if ':' in line:
                    parts = line.split(':', 1)
                    key = parts[0].strip()
                    # Check if key matches one of our inputs (fuzzy match or exact?)
                    # Let's try to map back to original input list
                    matched_original = None
                    for k in keywords:
                        if k.lower() == key.lower() or key.lower() in k.lower(): # Simple matching
                            matched_original = k
                            break
                    
                    if not matched_original:
                        # Fallback: just use the key from LLM if we can't match strictly
                        # But caller expects keys to match input list for logic.
                        # Let's trust the LLM output keys mostly match input.
                        matched_original = key

                    vars_str = parts[1].strip()
                    variations = [v.strip() for v in vars_str.split(',') if v.strip()]
                    
                    # Ensure original is in list
                    if matched_original not in variations:
                        variations.insert(0, matched_original)
                        
                    result[matched_original] = variations
            
            # Ensure all input keywords have entries
            for k in keywords:
                if k not in result:
                    result[k] = [k]
                    
            return result
            
        except Exception as e:
            print(f"Error in expand_keywords_batch: {e}")
            # Fallback: return original as single expansion
            return {k: [k] for k in keywords}

    def expand_keywords(self, keywords, mode="strict"):
        # Deprecated wrapper for single list usage, redirects to batch but returns flat list?
        # Actually, the old signature took a list and returned a flat list.
        # But our new main.py calls this in a loop.
        # We are replacing the loop in main.py, so we can deprecate this or keep for backward compat.
        # Let's keep it but implement via batch logic for simplicity?
        # No, let's just stick to the requested plan: Update main.py to use batch.
        pass

    def score_relevance(self, abstract, mandatory_keywords, bonus_keywords):
        """
        Node 2 / Phase 2: Score relevance based on tiered keywords.
        Returns int (1-10).
        """
        prompt = f"""
        You are a CRITICAL Academic Reviewer.
        MANDATORY Keywords (Must Match): {mandatory_keywords}
        BONUS Keywords (Boost Score): {bonus_keywords}
        Paper Abstract: {abstract}
        
        Task: Rate relevance (1-10).
        
        SCORING LOGIC:
        1. **Mandatory Check**: 
           - Does the paper address ALL mandatory keywords? 
           - If NO -> Score < 5 (Irrelevant).
           
        2. **Base Score (If Mandatory Met)**:
           - **Score 6-7**: Matches mandatory keywords but is a standard application.
           - **Score 8**: Matches mandatory keywords + novel contribution.
           
        3. **Bonus Boost**:
           - If paper ALSO addresses BONUS keywords (e.g., "{bonus_keywords}"), ADD +1 to +2 points.
           - Max Score: 10.
           
        4. **Penalties**:
           - Niche application without core improvement: -1 point.
           
        Output ONLY the integer score.
        """
        
        try:
            response = self.client.chat.completions.create(
                model=MODEL_TEXT,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            content = response.choices[0].message.content.strip()
            import re
            match = re.search(r'\d+', content)
            if match:
                return int(match.group(0))
            return 1
        except Exception as e:
            print(f"Error in score_relevance: {e}")
            return 1

    def select_best_figure(self, captions_list):
        """
        Selects the best figure ID from a list of captions.
        Input: list of strings (e.g. ["Figure 1: ...", "Figure 2: ..."])
        Output: str (e.g. "Figure 1") or None
        """
        if not captions_list:
            return None
            
        captions_text = "\n".join([f"- {cap}" for cap in captions_list])
        
        prompt = f"""
        You are an expert researcher. Below is a list of figure captions from a paper.
        
        {captions_text}
        
        Task: Identify the ONE figure that best illustrates the overall MODEL ARCHITECTURE, FRAMEWORK, or PIPELINE of the proposed method.
        
        Priority:
        1. Look for keywords like "Overview", "Architecture", "Framework", "Pipeline", "Model".
        2. Prefer early figures (Figure 1 or 2) if they match the description.
        3. Avoid "Ablation", "Results", "Comparison", "Dataset" figures.
        
        Output ONLY the Figure ID (e.g., "Figure 1"). Do not output any explanation.
        If no figure seems relevant to the architecture, output "None".
        """
        
        try:
            response = self.client.chat.completions.create(
                model=MODEL_TEXT,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            content = response.choices[0].message.content.strip()
            # Clean up content to just get "Figure X"
            import re
            match = re.search(r'(Figure\s+\d+|Fig\.?\s*\d+)', content, re.IGNORECASE)
            if match:
                # Normalize to "Figure X"
                num = re.search(r'\d+', match.group(0)).group(0)
                return f"Figure {num}"
            return None
        except Exception as e:
            print(f"Error in select_best_figure: {e}")
            return None


    def analyze_text_brain(self, abstract, introduction=""):
        """
        Node 4 (Text Brain): Extract claimed architecture and key metadata from text.
        """
        prompt = f"""
        Based on the following Abstract and Introduction, extract the following information:
        
        Abstract: {abstract}
        Introduction (Snippet): {introduction[:3000]}
        
        Tasks:
        1. Identify the Year and Venue (e.g., CVPR 2024, arXiv 2023) if mentioned. If not found, infer from context or mark "Unknown".
        2. Identify the ArXiv URL if present in text (unlikely in raw text but check). Mark "Unknown" if not found.
        3. Summarize the Core Motivation (Pain Points) in one sentence.
        4. Identify the Validation Tasks/Datasets used (e.g., POPE, MMBench).
        5. Summarize the Core Conclusion/Performance gains.
        6. Identify the Core Architecture Modules and Data Flow claimed.
        
        Output format (JSON-like structure preferred for parsing, but text is fine):
        - Year_Venue: ...
        - ArXiv_URL: ...
        - Motivation: ...
        - Validation_Tasks: ...
        - Core_Conclusion: ...
        - Core_Modules: ...
        - Data_Flow: ...
        """
        try:
            response = self.client.chat.completions.create(
                model=MODEL_TEXT,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error in text analysis: {e}"

    def analyze_vision_brain(self, image_path):
        """
        Node 4 (Vision Brain): Describe the image objectively.
        """
        base64_image = self._encode_image(image_path)
        
        prompt = """
        Please describe this architecture diagram objectively like a scanner.
        List the visible module names and the arrows/connections between them.
        Do NOT hallucinate or infer connections that are not visually present.
        """
        
        try:
            response = self.client.chat.completions.create(
                model=MODEL_VISION,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": base64_image
                                }
                            }
                        ]
                    }
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error in vision analysis: {e}"

    def synthesize_report(self, text_analysis, vision_analysis, relevance_score, keywords):
        """
        Node 4 (Fusion): Cross-validate and generate final explanation in specific Markdown format.
        """
        prompt = f"""
        You are a Dual-Brain Analyst. 
        
        Source 1 (Text Metadata & Claims):
        {text_analysis}
        
        Source 2 (Visual Facts):
        {vision_analysis}
        
        Context:
        - Relevance Score: {relevance_score}/10
        - Keywords: {keywords}
        
        Task:
        Generate a structured report strictly following the format below. 
        Cross-validate the "Method Description" by combining Text Claims with Visual Facts to ensure accuracy.
        
        IMPORTANT: Do NOT include markdown code block indicators (like ```markdown or ```) in your output. Just output the raw markdown text directly.
        
        Required Output Format (Markdown):
        
        ## 1. Basic Information
        * **Year/Venue**: [Extract from Source 1]
        * **Relevance Score**: {relevance_score} / 10 (Keywords: {keywords})
        * **Paper Link**: [Extract from Source 1 or "Unknown"]
        
        ## 2. Background
        * **Core Motivation**: [Extract from Source 1: Motivation]
        
        ## 3. Core Architecture and Method
        
        * **Method Description**: [Synthesize Source 1 (Modules/Flow) and Source 2 (Visuals). Explain how data flows through the core modules shown in the figure. Be concise and factual. Do NOT list discrepancies explicitly here, just output the corrected/verified explanation.]
        
        ## 4. Experimental Performance
        * **Validation Tasks**: [Extract from Source 1: Validation Tasks]
        * **Core Conclusion**: [Extract from Source 1: Core Conclusion]
        """
        try:
            response = self.client.chat.completions.create(
                model=MODEL_TEXT,
                messages=[{"role": "user", "content": prompt}]
            )
            content = response.choices[0].message.content
            # Post-processing to remove markdown code blocks if the model still includes them
            content = content.replace("```markdown", "").replace("```", "").strip()
            return content
        except Exception as e:
            return f"Error in synthesis: {e}"
