"""
LLM Transaction Extraction & Evaluation Pipeline
==================================================
Evaluates how well LLMs convert messy bank transaction strings
into structured JSON representations.

Usage (ChatGPT / OpenAI):
  1. Get an OpenAI API key from platform.openai.com
  2. export OPENAI_API_KEY=your-key-here
  3. python pipeline.py

Usage (Claude / Anthropic):
  1. Get an Anthropic API key from console.anthropic.com
  2. export ANTHROPIC_API_KEY=your-key-here
  3. python pipeline.py --provider anthropic

Author: Beethoven Marhone
Course: CAP 6640 - NLP, University of Central Florida
"""

import json
import csv
import os
import time
import urllib.request
import urllib.error
from collections import Counter
from typing import Optional

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# ============================================================
# CONFIGURATION
# ============================================================
PROVIDER = "openai"   # "openai" or "anthropic"

# OpenAI / ChatGPT
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL   = "gpt-4o"
OPENAI_URL     = "https://api.openai.com/v1/chat/completions"

# Anthropic / Claude
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL   = "claude-sonnet-4-20250514"
ANTHROPIC_URL     = "https://api.anthropic.com/v1/messages"

# ============================================================
# SCHEMA PROMPT
# ============================================================
PROMPT_TEMPLATE = """Extract structured transaction data from the text below using the following strict schema and rules.

Required JSON fields:
- merchant_name (Cleaned and normalized. Remove phone numbers, card numbers, store numbers, addresses, state abbreviations, and transaction-type words like "POS Debit", "Debit Card", "Card Purchase", "With Pin", "POS Credit Adjustment", "Transaction", "Paid To", etc.)
- transaction_date (If multiple dates appear, select the merchant transaction date. Return format MM/DD.)
- total_amount (Numeric only.)
- transaction_type (Must be one of: "purchase", "atm_withdrawal", "recurring", "transfer", "deposit", "other")
- category (Must be one of: "grocery", "restaurant", "gas", "retail", "services", "utilities", "entertainment", "fintech", "transfer", "other")
- city (Return only the city name if explicitly present in the text. Do not infer. If not clearly stated, return null.)

Rules:
- Do NOT include state abbreviations in merchant_name.
- Do NOT include city inside merchant_name.
- Do NOT include card numbers, check numbers, or reference codes in merchant_name.
- Do NOT guess missing information.
- If uncertain, return null.
- Output must be strictly valid JSON (single object, not an array).
- Do not include explanations, markdown, or code fences.

Transaction text:
{transaction_text}

Amount: ${amount}
"""

# ============================================================
# DATASET — 74 real banking transactions
# ============================================================
DATASET = [
    {"id": 1, "raw": "POS Credit Adjustment 0898 Transaction 02-27-26 Earnin Gc Mountain View CA", "amount": 50.00},
    {"id": 2, "raw": "Deposit - ACH Paid From Outcomes Operati Payroll 01Afd1", "amount": 963.48},
    {"id": 3, "raw": "POS Debit- Debit Card 0898 02-26-26 Floatme Subscripti San Antonio TX", "amount": 4.99},
    {"id": 4, "raw": "POS Debit- Debit Card 0898 02-26-26 Rmtly* Fffe7 Remitly.Com WA", "amount": 15.08},
    {"id": 5, "raw": "POS Debit- Debit Card 0898 02-26-26 Cleo* Advance Repa Meetcleo.Com DE", "amount": 45.00},
    {"id": 6, "raw": "POS Debit- Debit Card 0898 02-27-26 Zip* App Pay Later 183-37823729 Ny", "amount": 46.50},
    {"id": 7, "raw": "POS Debit- Debit Card 0898 02-27-26 Zip* App Pay Later 183-37823729 Ny", "amount": 55.75},
    {"id": 8, "raw": "POS Debit- Debit Card 0898 02-27-26 *creditgenie Cash Conshohocken PA", "amount": 114.99},
    {"id": 9, "raw": "POS Debit- Debit Card 0898 02-27-26 Gptzero* Trial Ove Www.Gptzero.M Ny", "amount": 155.88},
    {"id": 10, "raw": "POS Debit- Debit Card 0898 02-26-26 Rmtly* B610A Remitly.Com WA", "amount": 499.93},
    {"id": 11, "raw": "Paid To - Brigit-com Membership Chk 8410676", "amount": 8.99},
    {"id": 12, "raw": "Paid To - Brigit-com Protection Chk 8410676", "amount": 52.99},
    {"id": 13, "raw": "Dividend", "amount": 0.01},
    {"id": 14, "raw": "POS Credit Adjustment 0898 Transaction 02-28-26 Dave Inc Los Angeles CA", "amount": 25.00},
    {"id": 15, "raw": "POS Credit Adjustment 0898 Transaction 02-28-26 Dave Inc Los Angeles CA", "amount": 25.00},
    {"id": 16, "raw": "Dispute - Temp Credit", "amount": 155.88},
    {"id": 17, "raw": "Deposit - RTP Paid From Ml Plus LLC", "amount": 100.00},
    {"id": 18, "raw": "Deposit - RTP Paid From Ml Plus LLC", "amount": 100.00},
    {"id": 19, "raw": "Zelle DB Anne Vante", "amount": 80.00},
    {"id": 20, "raw": "Zelle DB Kircy Millien", "amount": 100.00},
    {"id": 21, "raw": "POS Debit- Debit Card 0898 02-27-26 Dave Inc Dave.Com CA", "amount": 5.00},
    {"id": 22, "raw": "POS Debit- Debit Card 0898 02-27-26 Affirm * Pay B2Oao 855-423-3729 CA", "amount": 24.92},
    {"id": 23, "raw": "POS Debit- Debit Card 0898 02-28-26 Pollo Tropical 100 Kissimmee FL", "amount": 25.78},
    {"id": 24, "raw": "POS Debit- Debit Card 0898 02-27-26 Affirm * Pay T4Tsu 855-423-3729 CA", "amount": 33.72},
    {"id": 25, "raw": "POS Debit- Debit Card 0898 02-27-26 Dave Inc Dave.Com CA", "amount": 55.00},
    {"id": 26, "raw": "POS Debit- Debit Card 0384 05-23-25 Klarna* Duke Energ Klarna.Com OH", "amount": 112.50},
    {"id": 27, "raw": "POS Debit- Debit Card 0384 05-27-25 Klarna* Apple Klarna.Com OH", "amount": 165.84},
    {"id": 28, "raw": "POS Debit- Debit Card 0384 05-23-25 Enableloans 855-2115599 SD", "amount": 361.12},
    {"id": 29, "raw": "Paid To - Amazon Marketpla Internet Chk 4330513", "amount": 18.18},
    {"id": 30, "raw": "Paid To - Best Buy Auto Pymt Chk 12240215", "amount": 101.94},
    {"id": 31, "raw": "POS Credit Adjustment 0384 Transaction 05-28-25 Earnin Cdhbd_b Palo Alto CA", "amount": 150.00},
    {"id": 32, "raw": "POS Credit Adjustment 0384 Transaction 05-28-25 Hard Rock Bet Ap 5 Davie FL", "amount": 899.00},
    {"id": 33, "raw": "POS Credit Adjustment 0384 Transaction 05-28-25 Hard Rock Bet Ap 5 Davie FL", "amount": 1102.39},
    {"id": 34, "raw": "POS Debit- Debit Card 0384 05-27-25 Pp*apple.Com/Bill 402-935-7733 CA", "amount": 2.99},
    {"id": 35, "raw": "POS Debit- Debit Card 0384 05-27-25 365 Market J 888 4 Troy MI", "amount": 3.61},
    {"id": 36, "raw": "POS Debit- Debit Card 0384 05-28-25 Afterpay 185-52896014 CA", "amount": 59.76},
    {"id": 37, "raw": "POS Debit- Debit Card 0384 05-28-25 Zelle*stephane Ly Visa Direct AZ", "amount": 370.00},
    {"id": 38, "raw": "POS Debit- Debit Card 0384 05-27-25 Zelle*stephane Ly Visa Direct AZ", "amount": 425.00},
    {"id": 39, "raw": "POS Debit- Debit Card 0384 05-28-25 Apple.Com/Bill 866-712-7753 CA", "amount": 19.15},
    {"id": 40, "raw": "POS Debit- Debit Card 0384 05-28-25 Cash App*ter Ro Oakland CA", "amount": 20.00},
    {"id": 41, "raw": "POS Debit- Debit Card 0384 05-28-25 Wawa 5105 Orlando FL", "amount": 20.31},
    {"id": 42, "raw": "POS Debit- Debit Card 0384 05-27-25 Hard Rock Bet Appl Davie FL", "amount": 21.60},
    {"id": 43, "raw": "POS Debit - Debit Card 0384 Transaction 05-28-25 Denny's #7954 Orlando", "amount": 23.00},
    {"id": 44, "raw": "POS Debit- Debit Card 0384 05-27-25 Denny's #7954 Orlando FL", "amount": 27.04},
    {"id": 45, "raw": "POS Debit- Debit Card 0384 05-28-25 Zelle*stephane Ly Visa Direct AZ", "amount": 842.00},
    {"id": 46, "raw": "POS Credit Adjustment 0384 Transaction 05-30-25 Hard Rock Bet Ap 5 Davie FL", "amount": 915.00},
    {"id": 47, "raw": "POS Debit- Debit Card 0384 05-30-25 Playstation Networ 650-2956540 CA", "amount": 9.99},
    {"id": 48, "raw": "POS Debit- Debit Card 0384 05-28-25 Hard Rock Bet Appl Davie FL", "amount": 20.00},
    {"id": 49, "raw": "POS Debit- Debit Card 0384 05-28-25 Hard Rock Bet Appl Davie FL", "amount": 100.00},
    {"id": 50, "raw": "POS Debit- Debit Card 0384 05-30-25 Cash App*ter Ro*ad Oakland CA", "amount": 915.00},
    {"id": 51, "raw": "Paid To - Dave Davesubfee Chk 8410676", "amount": 2.00},
    {"id": 52, "raw": "Dividend", "amount": 0.02},
    {"id": 53, "raw": "POS Credit Adjustment 0384 Transaction 06-02-25 Zelle*anne R Vant Visa Direct AZ", "amount": 30.00},
    {"id": 54, "raw": "POS Credit Adjustment 0384 Transaction 05-31-25 Earnin Cecdb_b Palo Alto CA", "amount": 150.00},
    {"id": 55, "raw": "Deposit - ACH Paid From Cleo Ai Inc 12855771 ! 060225", "amount": 0.01},
    {"id": 56, "raw": "POS Debit- Debit Card 0384 06-01-25 Idt Boss Intl Call Www.Idt.Net NJ", "amount": 1.75},
    {"id": 57, "raw": "POS Debit- Debit Card 0384 05-30-25 Wawa 5105 Orlando FL", "amount": 2.65},
    {"id": 58, "raw": "POS Debit- Debit Card 0384 05-31-25 Wawa 5132 Orlando FL", "amount": 3.00},
    {"id": 59, "raw": "POS Debit - Debit Card 0384 Transaction 06-01-25 7-Eleven Kissimmee", "amount": 4.29},
    {"id": 60, "raw": "POS Debit- Debit Card 0384 06-01-25 Cleo Ai Meetcleo.Com DE", "amount": 6.60},
    {"id": 61, "raw": "POS Debit- Debit Card 0384 05-31-25 Hard Rock Bet Appl Davie FL", "amount": 11.00},
    {"id": 62, "raw": "POS Debit- Debit Card 0384 06-01-25 Zelle*beethoven M Visa Direct AZ", "amount": 20.00},
    {"id": 63, "raw": "POS Debit- Debit Card 0384 06-01-25 Wawa 5105 Orlando FL", "amount": 23.71},
    {"id": 64, "raw": "POS Debit- Debit Card 0384 05-30-25 Re *reversephone.C 213-894-9165 Ny", "amount": 29.99},
    {"id": 65, "raw": "POS Debit- Debit Card 0384 06-02-25 Cash App*ter Ro Oakland CA", "amount": 30.00},
    {"id": 66, "raw": "POS Debit- Debit Card 0384 06-01-25 Hard Rock Bet Appl Davie FL", "amount": 40.00},
    {"id": 67, "raw": "POS Debit- Debit Card 0384 05-31-25 Hard Rock Bet Appl Davie FL", "amount": 50.00},
    {"id": 68, "raw": "POS Debit- Debit Card 0384 05-29-25 Enableloans 855-2115599 SD", "amount": 538.10},
    {"id": 69, "raw": "POS Credit Adjustment 0384 Transaction 06-03-25 Zelle*vladimir VI Visa Direct CA", "amount": 20.00},
    {"id": 70, "raw": "POS Debit- Debit Card 0384 06-02-25 Wawa 5105 Orlando FL", "amount": 0.93},
    {"id": 71, "raw": "Deposit - RTP Paid From Enableloans", "amount": 1250.00},
    {"id": 72, "raw": "POS Debit- Debit Card 0384 06-04-25 Paypal *marhone Be Visa Direct CA", "amount": 3.00},
    {"id": 73, "raw": "POS Debit- Debit Card 0384 06-04-25 Paypal *marhone Be Visa Direct CA", "amount": 19.00},
    {"id": 74, "raw": "POS Debit- Debit Card 0384 06-04-25 Afterpay 044-4123456 CA", "amount": 26.61},
]


# ============================================================
# LLM CALLS
# ============================================================

def call_openai(raw_text: str, amount: float, retries: int = 3) -> str:
    """Send transaction to OpenAI ChatGPT API and return raw response."""
    prompt = PROMPT_TEMPLATE.format(transaction_text=raw_text, amount=amount)

    payload = json.dumps({
        "model": OPENAI_MODEL,
        "temperature": 0.1,
        "max_tokens": 500,
        "messages": [{"role": "user", "content": prompt}]
    }).encode("utf-8")

    for attempt in range(retries):
        req = urllib.request.Request(
            OPENAI_URL,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {OPENAI_API_KEY}",
            },
            method="POST"
        )
        try:
            with urllib.request.urlopen(req) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = 10 * (2 ** attempt)
                print(f"  Rate limit hit, retrying in {wait}s... (attempt {attempt + 1}/{retries})")
                time.sleep(wait)
                continue
            print(f"  API Error {e.code}: {e.read().decode()[:200]}")
            return ""
        except Exception as e:
            print(f"  Error: {e}")
            return ""
    return ""


def call_claude(raw_text: str, amount: float, retries: int = 3) -> str:
    """Send transaction to Claude API and return raw response."""
    prompt = PROMPT_TEMPLATE.format(transaction_text=raw_text, amount=amount)

    payload = json.dumps({
        "model": ANTHROPIC_MODEL,
        "max_tokens": 500,
        "temperature": 0.1,
        "messages": [{"role": "user", "content": prompt}]
    }).encode("utf-8")

    for attempt in range(retries):
        req = urllib.request.Request(
            ANTHROPIC_URL,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
            },
            method="POST"
        )
        try:
            with urllib.request.urlopen(req) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                content = data.get("content", [])
                if not content:
                    return ""
                return content[0]["text"]
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = 10 * (2 ** attempt)
                print(f"  Rate limit hit, retrying in {wait}s... (attempt {attempt + 1}/{retries})")
                time.sleep(wait)
                continue
            print(f"  API Error {e.code}: {e.read().decode()[:200]}")
            return ""
        except Exception as e:
            print(f"  Error: {e}")
            return ""
    return ""


# ============================================================
# JSON PARSER
# ============================================================

def parse_response(text: str) -> Optional[dict]:
    """Parse LLM response into dict. Handles markdown fences."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()
    if text.startswith("json"):
        text = text[4:].strip()

    try:
        parsed = json.loads(text)
        if isinstance(parsed, list) and len(parsed) == 1:
            parsed = parsed[0]
        return parsed
    except json.JSONDecodeError:
        return None


# ============================================================
# FIELD COMPARISON
# ============================================================

def norm(s):
    """Normalize string for comparison."""
    if s is None:
        return None
    return str(s).strip().lower()


def compare(predicted: dict, ground_truth: dict) -> dict:
    """Compare each field. Returns dict of field_name -> bool."""
    results = {}

    # Amount — exact numeric match
    results["amount"] = (predicted.get("total_amount") == ground_truth.get("total_amount"))

    # Date — string match
    results["date"] = (norm(predicted.get("transaction_date")) == norm(ground_truth.get("transaction_date")))

    # Merchant — normalized string match
    results["merchant"] = (norm(predicted.get("merchant_name")) == norm(ground_truth.get("merchant_name")))

    # Category — exact match
    results["category"] = (norm(predicted.get("category")) == norm(ground_truth.get("category")))

    # Transaction type — exact match
    results["type"] = (norm(predicted.get("transaction_type")) == norm(ground_truth.get("transaction_type")))

    # City — both None or matching
    pred_city = norm(predicted.get("city"))
    true_city = norm(ground_truth.get("city"))
    results["city"] = (pred_city == true_city)

    return results


# ============================================================
# MAIN PIPELINE
# ============================================================

def call_llm(raw_text: str, amount: float) -> str:
    """Dispatch to the configured LLM provider."""
    if PROVIDER == "openai":
        return call_openai(raw_text, amount)
    return call_claude(raw_text, amount)


def active_model() -> str:
    return OPENAI_MODEL if PROVIDER == "openai" else ANTHROPIC_MODEL


def run_pipeline():
    """Run extraction on all transactions and evaluate."""

    if PROVIDER == "openai" and not OPENAI_API_KEY:
        print("ERROR: Set your OpenAI API key first:")
        print("  export OPENAI_API_KEY=your-key-here")
        print("\nGet a key at: https://platform.openai.com/api-keys")
        return
    if PROVIDER == "anthropic" and not ANTHROPIC_API_KEY:
        print("ERROR: Set your Anthropic API key first:")
        print("  export ANTHROPIC_API_KEY=your-key-here")
        print("\nGet a key at: https://console.anthropic.com")
        return

    # Resume: load any previously completed results
    completed = {}
    if os.path.exists("llm_outputs.json"):
        with open("llm_outputs.json") as f:
            for r in json.load(f):
                if r.get("json_valid"):
                    completed[r["id"]] = r
        if completed:
            print(f"Resuming — {len(completed)} transactions already done, skipping.")

    remaining = [e for e in DATASET if e["id"] not in completed]
    print(f"Running pipeline on {len(remaining)}/{len(DATASET)} transactions...")
    print(f"Provider: {PROVIDER.upper()}  |  Model: {active_model()}\n")

    all_results = list(completed.values())
    json_valid = len(completed)

    for entry in remaining:
        i = entry["id"]
        raw = entry["raw"]
        amt = entry["amount"]
        print(f"[{i:3d}/{len(DATASET)}] {raw[:60]}...")

        # Call LLM
        response = call_llm(raw, amt)
        time.sleep(5)  # 5s gap between calls

        # Parse
        predicted = parse_response(response)

        if predicted is None:
            print(f"         -> INVALID JSON")
            all_results.append({
                "id": i, "raw": raw, "amount": amt,
                "json_valid": False, "predicted": response[:300],
                "fields": None
            })
            continue

        json_valid += 1
        all_results.append({
            "id": i, "raw": raw, "amount": amt,
            "json_valid": True, "predicted": predicted,
            "fields": None  # filled after ground truth comparison
        })

    # ---- Save raw LLM outputs ----
    with open("llm_outputs.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nLLM outputs saved to llm_outputs.json")
    print(f"JSON validity: {json_valid}/{len(DATASET)} ({json_valid/len(DATASET)*100:.1f}%)")
    print("\n" + "="*55)
    print("NEXT STEP: Create ground_truth.json")
    print("="*55)
    print("""
Now you need to create ground_truth.json with the CORRECT
answers for each transaction. Run:

  python pipeline.py --evaluate

after creating ground_truth.json to compute accuracy metrics.

See ground_truth_template.json for the format.
""")

    # Generate ground truth template
    template = []
    for r in all_results:
        if r["json_valid"]:
            template.append({
                "id": r["id"],
                "raw": r["raw"],
                "llm_output": r["predicted"],
                "ground_truth": r["predicted"]  # pre-fill with LLM output, user corrects
            })
        else:
            template.append({
                "id": r["id"],
                "raw": r["raw"],
                "llm_output": "INVALID_JSON",
                "ground_truth": {
                    "merchant_name": "",
                    "transaction_date": "",
                    "total_amount": r["amount"],
                    "transaction_type": "",
                    "category": "",
                    "city": None
                }
            })

    with open("ground_truth_template.json", "w") as f:
        json.dump(template, f, indent=2)
    print("Template saved to ground_truth_template.json")
    print("Edit the 'ground_truth' field for each transaction, then run --evaluate")


def run_evaluation():
    """Compare LLM outputs against ground truth and compute metrics."""

    if not os.path.exists("llm_outputs.json"):
        print("ERROR: Run 'python pipeline.py' first to generate LLM outputs.")
        return
    if not os.path.exists("ground_truth.json"):
        print("ERROR: Create ground_truth.json first.")
        print("Copy ground_truth_template.json -> ground_truth.json")
        print("Then edit the 'ground_truth' fields with correct values.")
        return

    with open("llm_outputs.json") as f:
        outputs = json.load(f)
    with open("ground_truth.json") as f:
        truths = json.load(f)

    # Index ground truth by id
    gt_map = {item["id"]: item["ground_truth"] for item in truths}

    total = len(outputs)
    valid = [r for r in outputs if r["json_valid"]]
    json_rate = len(valid) / total * 100

    # Maps compare() short key -> actual JSON field name
    FIELD_KEY = {
        "amount":   "total_amount",
        "date":     "transaction_date",
        "merchant": "merchant_name",
        "category": "category",
        "type":     "transaction_type",
        "city":     "city",
    }

    fields_by_id = {}   # id -> field_results dict
    errors = []

    for r in valid:
        gt = gt_map.get(r["id"])
        if not gt:
            continue

        field_results = compare(r["predicted"], gt)
        fields_by_id[r["id"]] = field_results

        for field, correct in field_results.items():
            if not correct:
                key = FIELD_KEY[field]
                errors.append({
                    "id": r["id"],
                    "field": field,
                    "predicted": r["predicted"].get(key, ""),
                    "expected":  gt.get(key, ""),
                    "raw": r["raw"][:80]
                })

    # Compute per-field accuracy
    scored = list(fields_by_id.values())
    n = len(scored)
    metrics = {
        "total_transactions": total,
        "json_valid": len(valid),
        "json_validity_rate": round(json_rate, 1),
    }

    for field in ["amount", "date", "merchant", "category", "type", "city"]:
        correct = sum(1 for f in scored if f.get(field, False))
        metrics[f"{field}_accuracy"] = round(correct / n * 100, 1) if n else 0

    # ---- Print Report ----
    print("\n" + "=" * 55)
    print("       EXTRACTION ACCURACY REPORT")
    print("=" * 55)
    print(f"  Model:                    {active_model()}  ({PROVIDER})")
    print(f"  Total Transactions:       {metrics['total_transactions']}")
    print(f"  JSON Validity:            {metrics['json_validity_rate']}%")
    print(f"  Amount Accuracy:          {metrics['amount_accuracy']}%")
    print(f"  Date Accuracy:            {metrics['date_accuracy']}%")
    print(f"  Merchant Normalization:   {metrics['merchant_accuracy']}%")
    print(f"  Category Classification:  {metrics['category_accuracy']}%")
    print(f"  Transaction Type:         {metrics['type_accuracy']}%")
    print(f"  City Extraction:          {metrics['city_accuracy']}%")
    print("=" * 55)

    # Error breakdown
    if errors:
        print(f"\n  Total field-level errors: {len(errors)}")
        print("\n  Error Breakdown by Field:")
        field_counts = Counter(e["field"] for e in errors)
        for field, count in field_counts.most_common():
            print(f"    {field}: {count} errors")

        print("\n  Sample Errors (first 10):")
        for e in errors[:10]:
            print(f"    [ID {e['id']}] {e['field']}: expected '{e['expected']}' got '{e['predicted']}'")
            print(f"      Raw: {e['raw']}...")

    # ---- Save outputs ----
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    with open("results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "raw_text", "json_valid",
                         "amount_ok", "date_ok", "merchant_ok",
                         "category_ok", "type_ok", "city_ok"])
        for r in valid:
            fld = fields_by_id.get(r["id"])
            if fld:
                writer.writerow([
                    r["id"], r["raw"][:80], True,
                    fld["amount"], fld["date"], fld["merchant"],
                    fld["category"], fld["type"], fld["city"]
                ])

    with open("error_analysis.json", "w") as f:
        json.dump(errors, f, indent=2)

    print(f"\nFiles saved:")
    print(f"  metrics.json         - accuracy numbers for your paper")
    print(f"  results.csv          - per-transaction breakdown")
    print(f"  error_analysis.json  - detailed error patterns")

    if HAS_MATPLOTLIB:
        generate_figures(metrics)
    else:
        print("\n  (Install matplotlib to auto-generate figures: pip install matplotlib)")


# ============================================================
# FIGURE GENERATION
# ============================================================

def generate_figures(metrics: dict):
    """Generate bar_chart.png and pipeline_figure.png into figures/."""
    os.makedirs("figures", exist_ok=True)

    # ── Bar chart ──────────────────────────────────────────────
    fields = ["Amount", "Date", "Merchant", "Category", "Type", "City"]
    keys   = ["amount_accuracy", "date_accuracy", "merchant_accuracy",
              "category_accuracy", "type_accuracy", "city_accuracy"]
    values = [metrics.get(k, 0) for k in keys]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#2196F3" if v >= 80 else "#FF9800" if v >= 60 else "#F44336"
              for v in values]
    bars = ax.bar(fields, values, color=colors, edgecolor="white", linewidth=0.8)

    ax.set_ylim(0, 110)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title(
        f"Gemini {metrics.get('json_validity_rate', 0):.0f}% JSON Validity — "
        f"Field-Level Extraction Accuracy\n"
        f"Model: {ANTHROPIC_MODEL}  |  n = {metrics.get('total_transactions', 0)} transactions",
        fontsize=11
    )
    ax.axhline(100, color="gray", linestyle="--", linewidth=0.7, alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    legend_handles = [
        mpatches.Patch(color="#2196F3", label="≥ 80%"),
        mpatches.Patch(color="#FF9800", label="60–79%"),
        mpatches.Patch(color="#F44336", label="< 60%"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=9)

    fig.tight_layout()
    fig.savefig("figures/bar_chart.png", dpi=150)
    plt.close(fig)
    print("  figures/bar_chart.png")

    # ── Pipeline diagram ───────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 3.5)
    ax.axis("off")
    ax.set_facecolor("#FAFAFA")
    fig.patch.set_facecolor("#FAFAFA")

    steps = [
        (0.6,  "Raw Bank\nTransaction", "#E3F2FD", "#1565C0"),
        (2.8,  "Prompt\nTemplate",      "#F3E5F5", "#6A1B9A"),
        (5.0,  "Gemini\nAPI Call",      "#E8F5E9", "#2E7D32"),
        (7.2,  "JSON\nParser",          "#FFF3E0", "#E65100"),
        (9.4,  "Field\nComparison",     "#FCE4EC", "#880E4F"),
        (11.4, "Accuracy\nMetrics",     "#E0F2F1", "#004D40"),
    ]

    box_w, box_h = 1.7, 1.1
    y_center = 1.75

    for x, label, bg, fg in steps:
        fancy = mpatches.FancyBboxPatch(
            (x - box_w / 2, y_center - box_h / 2), box_w, box_h,
            boxstyle="round,pad=0.08", linewidth=1.2,
            edgecolor=fg, facecolor=bg
        )
        ax.add_patch(fancy)
        ax.text(x, y_center, label, ha="center", va="center",
                fontsize=9, fontweight="bold", color=fg)

    # Arrows between boxes
    arrow_props = dict(arrowstyle="-|>", color="#555555",
                       lw=1.4, mutation_scale=14)
    for i in range(len(steps) - 1):
        x_start = steps[i][0]   + box_w / 2 + 0.04
        x_end   = steps[i+1][0] - box_w / 2 - 0.04
        ax.annotate("", xy=(x_end, y_center), xytext=(x_start, y_center),
                    arrowprops=arrow_props)

    ax.set_title(
        "LLM Transaction Extraction Pipeline  ·  CAP 6640 NLP  ·  UCF",
        fontsize=12, fontweight="bold", pad=10, color="#333333"
    )

    fig.tight_layout()
    fig.savefig("figures/pipeline_figure.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  figures/pipeline_figure.png")


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    import sys

    if "--provider" in sys.argv:
        idx = sys.argv.index("--provider")
        if idx + 1 < len(sys.argv):
            PROVIDER = sys.argv[idx + 1].lower()

    if "--evaluate" in sys.argv:
        run_evaluation()
    elif "--figures" in sys.argv:
        if not os.path.exists("metrics.json"):
            print("ERROR: Run --evaluate first to generate metrics.json")
        elif not HAS_MATPLOTLIB:
            print("ERROR: matplotlib not installed. Run: pip install matplotlib")
        else:
            with open("metrics.json") as f:
                generate_figures(json.load(f))
    else:
        run_pipeline()