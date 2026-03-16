# Paper Retrieval

This repository provides a paper retrieval and analysis system with three completed phases:

1. Plan 1: MCP infrastructure decoupling.
2. Plan 2: ReAct agent with dynamic tool routing.
3. Plan 3: Multi-agent architecture (Researcher -> Vision Expert -> Writer).

## Current Status

All 3 plans are completed and integrated.

## Environment Setup

This project uses conda environment `pr`.

```powershell
conda activate pr
python -m pip install -r requirements.txt
```

If needed, use a fixed interpreter path:

```powershell
C:/Users/lenovo/anaconda3/envs/pr/python.exe main.py --help
```

## Architecture Overview

### Plan 1: MCP Server

- MCP server entry: `mcp_server.py`
- Server name: `PaperBrain`
- Exposed MCP tools include:
	- `search_arxiv_tool`
	- `search_dblp_tool`
	- `download_and_extract_captions_tool`
	- `crop_figure_tool`

Run MCP server:

```powershell
C:/Users/lenovo/anaconda3/envs/pr/python.exe mcp_server.py
```

### Plan 2: ReAct Dynamic Routing

- Runtime core: `agent_runner.py`
- Main entry: `main.py`
- ReAct builds on `create_react_agent` with `ALL_TOOLS`
- Streaming logs include Thought / Action / Observation / Final Answer

### Plan 3: Multi-Agent Workflow

- State: `workflow/multi_agent_state.py`
- Nodes: `workflow/multi_agent_nodes.py`
- Graph: `workflow/multi_agent_graph.py`
- Runtime handoff logs:
	- `[Researcher Agent working...]`
	- `[Vision Expert Agent working...]`
	- `[Writer Agent working...]`

## Run Modes

Main program entry is `main.py`.

```powershell
C:/Users/lenovo/anaconda3/envs/pr/python.exe main.py [ARGS]
```

### Mode 1: ReAct (default)

```powershell
C:/Users/lenovo/anaconda3/envs/pr/python.exe main.py --query "Search arXiv for recent vision-language model papers from the last 3 days"
```

### Mode 2: Multi-Agent

```powershell
C:/Users/lenovo/anaconda3/envs/pr/python.exe main.py --mode multi-agent --query "Search arXiv for Vision-Language Models and Hallucination papers from last 3 days, then create a concise report"
```

## Local vs Remote Retrieval Policy

The current behavior is:

1. Local intent is local-only.
	 - For local queries, remote retrieval tools are disabled in ReAct mode.
	 - In multi-agent mode, local intent processes local PDFs directly.
2. Remote retrieval is used when query intent is arXiv/DBLP/online retrieval.
3. Both modes still support report generation after analysis.

Local query example:

```powershell
C:/Users/lenovo/anaconda3/envs/pr/python.exe main.py --query "Analyze local PDFs under paper folder about Vision-Language Models and training-free, then generate one markdown report named react_local_training_free.md"
```

Remote query examples:

```powershell
C:/Users/lenovo/anaconda3/envs/pr/python.exe main.py --query "Search arXiv for recent Vision-Language Model papers from last 3 days, download the first one, analyze it, and generate one markdown report named arxiv_mode_report.md"

C:/Users/lenovo/anaconda3/envs/pr/python.exe main.py --query "Find CVPR 2025 papers from DBLP, download the first available PDF, analyze it with keywords Vision-Language Models and Hallucination, and generate one markdown report named dblp_mode_report.md"
```

## Outputs

- Markdown reports are written to `output/`
- Extracted/cropped figures are written to `output/images/`

## Validation

Run unit tests:

```powershell
C:/Users/lenovo/anaconda3/envs/pr/python.exe -m unittest discover -s tests -v
```

Recent validation confirms tests pass in conda environment `pr`.