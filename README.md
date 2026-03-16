# paper_retrieval

## Environment Setup (Conda)

This project uses the conda environment `pr`.

```powershell
conda activate pr
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

If you want to avoid interpreter ambiguity, you can always use the full path:

```powershell
C:/Users/lenovo/anaconda3/envs/pr/python.exe main.py --help
```

## Run Program

Main entry is `main.py`.

```powershell
C:/Users/lenovo/anaconda3/envs/pr/python.exe main.py [ARGS]
```

### ReAct Dynamic Mode

Use a free-text query and let the agent decide which tools to call.

```powershell
C:/Users/lenovo/anaconda3/envs/pr/python.exe main.py --query "Help me find two papers about CVPR 2024 and download the first one"
```

If you also want deep analysis and markdown report generation, ask explicitly in query:

```powershell
C:/Users/lenovo/anaconda3/envs/pr/python.exe main.py --query "Find two CVPR 2024 papers, download one, analyze the PDF, and generate a markdown report"
```

### More Examples

Search arXiv and then filter with natural language:

```powershell
C:/Users/lenovo/anaconda3/envs/pr/python.exe main.py --query "Search arXiv for recent vision-language model papers from the last 3 days"
```

Search DBLP by venue/year intent and download PDF:

```powershell
C:/Users/lenovo/anaconda3/envs/pr/python.exe main.py --query "Find CVPR 2025 papers on training-free hallucination mitigation and download one PDF"
```

Notes:
- Core input argument is now `--query`.
- Agent behavior is dynamic and no longer constrained by a fixed step-by-step DAG route.
- When query includes analysis/report intent, agent can use `analyze_pdf` + `generate_markdown_report` tools.
- Reports are written to `output/`.
- Generated figure crops are written to `output/images/`.

## Tests

Run unit tests (no external API calls required):

```bash
C:/Users/lenovo/anaconda3/envs/pr/python.exe -m unittest discover -s tests -v
```