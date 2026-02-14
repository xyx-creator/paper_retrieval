import base64
from zhipuai import ZhipuAI
from config import ZHIPUAI_API_KEY, MODEL_TEXT, MODEL_VISION

class GLMAgent:
    def __init__(self):
        if not ZHIPUAI_API_KEY:
            raise ValueError("ZHIPUAI_API_KEY not found in environment variables.")
        self.client = ZhipuAI(api_key=ZHIPUAI_API_KEY)

    def _encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def score_relevance(self, abstract, keywords):
        """
        Node 2: Score relevance of the paper based on abstract and keywords.
        Returns int (1-5).
        """
        prompt = f"""
        You are a strict academic relevance scorer.
        Keywords: {keywords}
        Abstract: {abstract}
        
        Task: Rate the relevance of the abstract to the keywords on a scale of 1-5.
        
        Scoring Criteria:
        5: Highly Relevant. Core contribution revolves around the keywords.
        4: Relevant. Uses the technology as a key component.
        3: Mentioned. Referenced in background or baselines.
        1-2: Irrelevant.
        
        Output ONLY the integer score (e.g., 5). Do not output any explanation.
        """
        
        try:
            response = self.client.chat.completions.create(
                model=MODEL_TEXT,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            content = response.choices[0].message.content.strip()
            # Extract number
            import re
            match = re.search(r'\d', content)
            if match:
                return int(match.group(0))
            return 1 # Default to low relevance if parsing fails
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
        - Relevance Score: {relevance_score}/5
        - Keywords: {keywords}
        
        Task:
        Generate a structured report strictly following the format below. 
        Cross-validate the "Method Description" by combining Text Claims with Visual Facts to ensure accuracy.
        
        Required Output Format (Markdown):
        
        ## 1. 基本信息
        * **年份/等级**: [Extract from Source 1]
        * **相关度打分**: {relevance_score} / 5 分 (关键词: {keywords})
        * **原文链接**: [Extract from Source 1 or "Unknown"]
        
        ## 2. 研究背景
        * **核心动机**: [Extract from Source 1: Motivation]
        
        ## 3. 核心架构与方法解析
        
        * **方法简述**: [Synthesize Source 1 (Modules/Flow) and Source 2 (Visuals). Explain how data flows through the core modules shown in the figure. Be concise and factual. Do NOT list discrepancies explicitly here, just output the corrected/verified explanation.]
        
        ## 4. 实验表现
        * **验证任务**: [Extract from Source 1: Validation Tasks]
        * **核心结论**: [Extract from Source 1: Core Conclusion]
        """
        try:
            response = self.client.chat.completions.create(
                model=MODEL_TEXT,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error in synthesis: {e}"
