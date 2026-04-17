# LLM Transaction Extraction Pipeline

Evaluating how well Large Language Models convert unstructured banking transaction strings into structured JSON representations.

**Author:** Beethoven Marhone  
**Course:** CAP 6640 – Natural Language Processing, University of Central Florida  

## Overview

This project tests whether LLMs can reliably extract structured data (merchant names, dates, amounts, categories, cities) from messy, real-world bank statement text. The pipeline sends raw transaction strings to an LLM with a schema-constrained prompt, validates the JSON output, and compares each field against manually curated ground truth labels.

## Dataset

- **74 real banking transactions** from everyday debit/credit card activity
- Includes: restaurants, gas stations, fintech apps, remittances, subscriptions, transfers, deposits
- Ground truth labels manually defined for all six extraction fields

## Pipeline

```
Raw Transaction Text → Prompt Construction → LLM (Gemini) → JSON Output → Validation → Field Comparison → Accuracy Metrics
```

## Setup

1. Get a free Google Gemini API key at [aistudio.google.com](https://aistudio.google.com)
2. Set your key:
   ```bash
   export GEMINI_API_KEY=your-key-here
   ```

## Usage

### Step 1: Run extraction
```bash
python pipeline.py
```
This sends all 74 transactions to the LLM and saves outputs to `llm_outputs.json`.

### Step 2: Create ground truth
Copy `ground_truth_template.json` to `ground_truth.json` and correct any wrong fields.

### Step 3: Evaluate
```bash
python pipeline.py --evaluate
```
This computes field-level accuracy metrics and saves results to `metrics.json`, `results.csv`, and `error_analysis.json`.

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| JSON Validity | % of responses that are valid JSON |
| Amount Accuracy | Correct numeric extraction |
| Date Accuracy | Correct transaction date selection |
| Merchant Normalization | Cleaned merchant name matches ground truth |
| Category Classification | Correct category assignment |
| Transaction Type | Correct type identification |
| City Extraction | Correct city identification |

## Project Structure

```
├── README.md
├── pipeline.py              # Main extraction & evaluation pipeline
├── llm_outputs.json         # Raw LLM responses (generated)
├── ground_truth.json        # Manually verified correct labels
├── metrics.json             # Accuracy metrics (generated)
├── results.csv              # Per-transaction results (generated)
├── error_analysis.json      # Detailed error patterns (generated)
└── figures/
    ├── pipeline_figure.png  # Pipeline diagram for paper
    └── bar_chart.png        # Accuracy bar chart for paper
```