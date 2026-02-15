import os

# API Configuration
ZHIPUAI_API_KEY = os.environ.get("ZHIPUAI_API_KEY")
S2_API_KEY = os.environ.get("S2_API_KEY")

# Model Configuration
MODEL_KEYWORD_EXPANSION = "glm-4.5-air"
MODEL_RELEVANCE_SCORING = "glm-4.7"
MODEL_TEXT = "glm-4.7"
MODEL_VISION = "glm-4.6v"

# Path Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "paper")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# Constants
RELEVANCE_THRESHOLD = 8
FIGURE_KEYWORDS = ["Figure 1", "Overview", "Architecture", "Framework", "Pipeline"]
KEYWORDS = {
    "mandatory": ["Vision-Language Models", "Hallucination"],
    "bonus": ["Training-free"]
}
ARXIV_CATEGORY = "cat:cs.CV"
