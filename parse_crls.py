import os
import re
import csv
import json
import time
import traceback
from typing import List, Dict, Any, Optional, Tuple

# PDF parsing
import fitz  # PyMuPDF

# OpenRouter via OpenAI client
import openai


def call_openrouter_api_simple(
    system_prompt: str,
    user_prompt: str,
    model: str = "google/gemini-2.5-flash-lite-preview-06-17",
    temperature: float = 0.0,
    top_p: float = 1,
    max_tokens: int = 8000,
    json_mode: bool = False,
    json_schema: Optional[Dict[str, Any]] = None,
    json_schema_name: str = 'Schema Json',
    ignore_safety_stop: bool = False,
):
    """
    Makes a call to the OpenRouter API with the specified parameters.

    Returns:
        tuple: (response_content, usage) - The model's response text and token usage
    """
    response_content = ""
    usage = {'input_tokens': 0, 'output_tokens': 0}
    try:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY not set in environment")

        client = openai.OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )

        if json_mode:
            completion_args = {
                "extra_headers": {
                    "HTTP-Referer": "https://github.com/OpenRouterTeam/openrouter-python",
                    "X-Title": "OpenRouter API",
                },
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if json_schema:
                completion_args["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": json_schema_name,
                        "strict": True,
                        "schema": json_schema,
                    },
                }
            else:
                completion_args["response_format"] = {"type": "json_object"}

            chat_completion = client.chat.completions.create(**completion_args)
        else:
            chat_completion = client.chat.completions.create(
                extra_headers={
                    "HTTP-Referer": "https://github.com/OpenRouterTeam/openrouter-python",
                    "X-Title": "OpenRouter API",
                },
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )

        finish_reason = chat_completion.choices[0].finish_reason
        if finish_reason == 'error':
            native_reason = getattr(chat_completion.choices[0], 'native_finish_reason', None)
            print(f"API response stopped early: {native_reason}")

            if native_reason == 'SAFETY' and not ignore_safety_stop:
                print("SAFETY STOP DETECTED: Halting processing")
                usage_info = {
                    'input_tokens': chat_completion.usage.prompt_tokens if chat_completion.usage else 0,
                    'output_tokens': chat_completion.usage.completion_tokens if chat_completion.usage else 0,
                }
                return None, usage_info

            response_content = chat_completion.choices[0].message.content
        else:
            response_content = chat_completion.choices[0].message.content

        usage = {
            'input_tokens': chat_completion.usage.prompt_tokens if chat_completion.usage else 0,
            'output_tokens': chat_completion.usage.completion_tokens if chat_completion.usage else 0,
        }
    except Exception as e:
        print(f"Error calling OpenRouter API: {str(e)}")
        print(traceback.format_exc())
        return None, usage

    return response_content, usage


def extract_text_signals(doc: fitz.Document, max_chars: int = 20000) -> str:
    """Return a condensed, high-signal text snippet from the PDF.
    - Full text of first page
    - Lines from other pages matching key patterns
    - Truncated to max_chars
    """
    key_patterns = re.compile(
        r"(complete response|CRL|FDA|application|submitted|NDA|BLA|ANDA|efficacy|safety|risk|benefit|trial|endpoint|manufactur|CMC|deficien|request|respon|sponsor|applicant|drug|biolog)",
        re.IGNORECASE,
    )

    parts: List[str] = []

    if len(doc) > 0:
        try:
            first = doc.load_page(0)
            parts.append(first.get_text("text"))
        except Exception:
            pass

    for i in range(1, len(doc)):
        try:
            page = doc.load_page(i)
            text = page.get_text("text")
            lines = []
            for line in text.splitlines():
                if key_patterns.search(line):
                    lines.append(line)
            if lines:
                parts.append("\n".join(lines[:50]))
        except Exception:
            continue

    joined = "\n\n".join([p for p in parts if p])
    if len(joined) > max_chars:
        return joined[:max_chars]
    return joined


def build_json_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "drug_name": {"type": "string"},
            "company": {"type": "string"},
            "date_of_application": {"type": "string", "description": "Prefer YYYY-MM-DD if present; otherwise best-effort"},
            "reason_for_crl": {"type": "string", "description": "Concise summary of the primary stated reasons"},
            "category": {"type": "string", "enum": ["efficacy", "safety", "both", "other"]},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "evidence": {
                "type": "object",
                "properties": {
                    "date_source_text": {"type": "string"},
                    "reason_source_text": {"type": "string"},
                    "pages_hint": {"type": "array", "items": {"type": "integer"}},
                },
                "required": [],
                "additionalProperties": False,
            },
            "notes": {"type": "string"},
        },
        "required": ["drug_name", "company", "reason_for_crl", "category"],
        "additionalProperties": False,
    }


SYSTEM_PROMPT = (
    "You are a precise regulatory analyst extracting key fields from FDA Complete Response Letters (CRLs). "
    "Return only JSON following the provided schema. If unknown, use empty string and set confidence low. "
    "Categorize reasons as: efficacy, safety, both, or other (e.g., CMC/manufacturing)."
)


USER_PROMPT_TEMPLATE = (
    "Extract the following from the provided CRL text.\n"
    "- drug_name\n"
    "- company (applicant/sponsor)\n"
    "- date_of_application\n"
    "- reason_for_crl (short summary)\n"
    "- category (efficacy|safety|both|other)\n\n"
    "CRL EXCERPTS:\n\n{doc_text}"
)


def parse_pdf_with_llm(pdf_path: str) -> Optional[Dict[str, Any]]:
    try:
        with fitz.open(pdf_path) as doc:
            text = extract_text_signals(doc)
    except Exception as e:
        print(f"Failed to read PDF {pdf_path}: {e}")
        return None

    if not text or len(text.strip()) < 20:
        print(f"Low/no extractable text for {os.path.basename(pdf_path)}; may be scanned/OCR needed.")
        return {
            "drug_name": "",
            "company": "",
            "date_of_application": "",
            "reason_for_crl": "",
            "category": "other",
            "confidence": 0.0,
            "evidence": {"date_source_text": "", "reason_source_text": "", "pages_hint": []},
            "notes": "No extractable text; likely scanned PDF without OCR.",
            "_file": os.path.basename(pdf_path),
        }

    user_prompt = USER_PROMPT_TEMPLATE.format(doc_text=text)
    schema = build_json_schema()

    response, usage = call_openrouter_api_simple(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        json_mode=True,
        json_schema=schema,
        json_schema_name="CRLExtraction",
        temperature=0.0,
        max_tokens=1500,
    )

    if response is None:
        print(f"LLM call returned None (possibly safety stop) for {pdf_path}")
        return None

    try:
        data = json.loads(response)
    except Exception:
        # Some models may wrap in code fences; try to strip
        cleaned = response.strip()
        cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
        try:
            data = json.loads(cleaned)
        except Exception as e:
            print(f"Failed to parse JSON for {pdf_path}: {e}\nRaw: {response[:500]}...")
            return None

    # Attach filename for traceability
    if isinstance(data, dict):
        data["_file"] = os.path.basename(pdf_path)
    return data


def write_outputs(rows: List[Dict[str, Any]], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    details_dir = os.path.join(out_dir, "details")
    os.makedirs(details_dir, exist_ok=True)

    jsonl_path = os.path.join(out_dir, "crl_results.jsonl")
    csv_path = os.path.join(out_dir, "crl_results.csv")

    # JSONL
    with open(jsonl_path, "w", encoding="utf-8") as jf:
        for r in rows:
            jf.write(json.dumps(r, ensure_ascii=False) + "\n")

    # CSV
    fieldnames = [
        "_file",
        "drug_name",
        "company",
        "date_of_application",
        "reason_for_crl",
        "category",
        "confidence",
        "notes",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as cf:
        writer = csv.DictWriter(cf, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})

    # Per-file JSONs for inspection
    for r in rows:
        fname = r.get("_file", "result")
        with open(os.path.join(details_dir, f"{fname}.json"), "w", encoding="utf-8") as f:
            json.dump(r, f, ensure_ascii=False, indent=2)

    print(f"Saved: {jsonl_path} and {csv_path} (and per-file JSONs in details/)")


def find_pdfs(input_dir: str) -> List[str]:
    if not os.path.isdir(input_dir):
        return []
    paths: List[str] = []
    for root, _, files in os.walk(input_dir):
        for name in files:
            if name.lower().endswith(".pdf"):
                paths.append(os.path.join(root, name))
    return sorted(paths)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Parse CRL PDFs and extract structured fields with OpenRouter LLM")
    parser.add_argument("--input", default="CRL Unapprovved", help="Directory containing CRL PDFs")
    parser.add_argument("--out", default="outputs", help="Directory to write outputs")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of PDFs to process (0 = all)")
    args = parser.parse_args()

    os.makedirs(args.input, exist_ok=True)
    pdfs = find_pdfs(args.input)
    if not pdfs:
        print(f"No PDFs found in '{args.input}'. Place files there and re-run.")
        return

    if args.limit > 0:
        pdfs = pdfs[: args.limit]

    results: List[Dict[str, Any]] = []
    for i, pdf in enumerate(pdfs, 1):
        print(f"[{i}/{len(pdfs)}] Processing {pdf}...")
        data = parse_pdf_with_llm(pdf)
        if data:
            results.append(data)
        time.sleep(0.5)  # small pacing

    if results:
        write_outputs(results, args.out)
    else:
        print("No results produced.")


if __name__ == "__main__":
    main()

