# LLM Transaction Extraction

Testing how well LLMs handle messy bank statement text — can they turn noisy transaction descriptions into clean JSON?

**Beethoven Marhone**  
CAP 6640 – NLP, University of Central Florida  

## What this does

Bank statements are messy. Merchants show up as "Rmtly* Fffe7" instead of "Remitly." Store numbers, phone numbers, and state codes get mixed in. This project takes 74 real transactions like that, feeds them to Claude (Haiku, Sonnet, and Opus), and checks how well each model extracts merchant names, dates, amounts, categories, and cities.

## How to run it

Get an API key from [console.anthropic.com](https://console.anthropic.com), then:

```bash
export ANTHROPIC_API_KEY=sk-ant-your-key-here
python3 pipeline.py
```

After it finishes, copy `ground_truth_template.json` to `ground_truth.json`, fix any wrong labels, then:

```bash
python3 pipeline.py --evaluate
```

That prints accuracy numbers and saves results to `metrics.json`, `results.csv`, and `error_analysis.json`.

## What's in the repo

- `pipeline.py` — runs extraction and evaluation
- `ground_truth.json` — manually labeled correct answers
- `metrics.json` — accuracy results per model
- `results.csv` — per-transaction breakdown
- `error_analysis.json` — what went wrong and where
- `figures/` — pipeline diagram and bar chart for the paper

## Results (short version)

All three models got 100% on JSON validity, amounts, dates, and cities. Semantic fields were harder — merchant normalization peaked at 51.4% with Opus, category hit 78.4%, and transaction type hit 74.3% with Haiku. Bigger model does not always mean better.