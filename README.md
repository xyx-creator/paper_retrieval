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

### 1) Local Mode

Analyze local PDFs under `paper/`.

```powershell
C:/Users/lenovo/anaconda3/envs/pr/python.exe main.py --source local --mandatory "Vision Language Models,training-free" --top-k 10
```

### 2) arXiv Mode

Retrieve from arXiv (default `--days` is 3).

```powershell
C:/Users/lenovo/anaconda3/envs/pr/python.exe main.py --source arxiv --mandatory "Vision Language Models,training-free" --days 3 --top-k 10
```

### 3) DBLP Mode (e.g. CVPR 2025)

Retrieve from DBLP and enrich via Semantic Scholar.

```powershell
C:/Users/lenovo/anaconda3/envs/pr/python.exe main.py --source dblp --venue CVPR --year 2025 --mandatory "Vision Language Models,training-free" --top-k 10
```

Notes:
- `--source dblp` requires both `--venue` and `--year`.
- Reports are written to `output/`.
- Generated figure crops are written to `output/images/`.

## Tests

Run unit tests (no external API calls required):

```bash
C:/Users/lenovo/anaconda3/envs/pr/python.exe -m unittest discover -s tests -v
```