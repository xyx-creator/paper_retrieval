import os

# API Configuration
ZHIPUAI_API_KEY = os.environ.get("ZHIPUAI_API_KEY")

# Model Configuration
MODEL_TEXT = "glm-4-plus"  # Using glm-4-plus as GLM-4.7 equivalent/best available
MODEL_VISION = "glm-4v-plus" # Using glm-4v-plus as GLM-4.6V equivalent/best available

# Path Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "paper")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# Constants
RELEVANCE_THRESHOLD = 4
FIGURE_KEYWORDS = ["Figure 1", "Overview", "Architecture", "Framework", "Pipeline"]
