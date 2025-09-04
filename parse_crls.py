import os
import re
import csv
import json
import time
import traceback
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# PDF parsing
import pymupdf4llm

# Thread lock for PyMuPDF4LLM operations
_pdf_extraction_lock = threading.Lock()

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




def extract_markdown_text(pdf_path: str) -> str:
    """Extract text as markdown using PyMuPDF4LLM with thread safety."""
    try:
        # Ensure the file path is absolute and exists
        if not os.path.isabs(pdf_path):
            pdf_path = os.path.abspath(pdf_path)
        
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        # Use thread lock to ensure thread-safe operation
        with _pdf_extraction_lock:
            # Extract markdown text
            markdown_text = pymupdf4llm.to_markdown(pdf_path)
        
        if not markdown_text or len(markdown_text.strip()) < 20:
            raise ValueError(f"Extracted text too short or empty from {os.path.basename(pdf_path)}")
            
        return markdown_text
        
    except Exception as e:
        print(f"Error extracting markdown from {os.path.basename(pdf_path)}: {e}")
        print(f"Error type: {type(e).__name__}")
        # Re-raise the exception so it can be handled by the calling function
        raise


def build_json_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "drug_name": {"type": "string"},
            "company": {"type": "string"},
            "date": {"type": "string", "description": "Prefer YYYY-MM-DD if present; otherwise best-effort"},
            "summary": {"type": "string", "description": "Concise summary of the reason for CRL"},
            "classification": {"type": "string", "enum": ["efficacy", "safety", "others"]},
            "path_to_future_approval": {"type": "string", "description": "What the sponsor must do to gain future approval"},
        },
        "required": [
            "drug_name",
            "company",
            "date",
            "summary",
            "classification",
            "path_to_future_approval",
        ],
        "additionalProperties": False,
    }


SYSTEM_PROMPT = (
    "You are a precise regulatory analyst extracting key fields from FDA Complete Response Letters (CRLs). "
    "Return only JSON following the provided schema with exact keys. If a value is unknown, use an empty string. "
    "Classify the reason using ONLY: efficacy, safety, or others (e.g., CMC/manufacturing/logistics)."
)


USER_PROMPT_TEMPLATE = (
    "Extract these fields from the CRL text and adhere to the schema: \n"
    "- drug_name\n"
    "- company (applicant/sponsor)\n"
    "- date (application or key date mentioned)\n"
    "- summary (concise reason for CRL)\n"
    "- classification (use exactly one: efficacy | safety | others)\n"
    "- path_to_future_approval (what is required to gain approval)\n\n"
    "CRL EXCERPTS:\n\n{doc_text}"
)


def parse_pdf_with_llm(pdf_path: str) -> Optional[Dict[str, Any]]:
    try:
        markdown_text = extract_markdown_text(pdf_path)
    except Exception as e:
        print(f"Failed to extract text from {os.path.basename(pdf_path)}: {e}")
        return None
    
    if not markdown_text or len(markdown_text.strip()) < 20:
        print(f"Low/no extractable text for {os.path.basename(pdf_path)}; may be scanned/OCR needed.")
        return None

    user_prompt = USER_PROMPT_TEMPLATE.format(doc_text=markdown_text)
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

    # Attach filename for traceability (internal key)
    if isinstance(data, dict):
        data["_file"] = os.path.basename(pdf_path)
    return data


def write_outputs(rows: List[Dict[str, Any]], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "crl_results.csv")

    # CSV only
    fieldnames = [
        "drug_name",
        "company",
        "date",
        "summary",
        "classification",
        "path_to_future_approval",
        "link",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as cf:
        writer = csv.DictWriter(cf, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            file_name = r.get("_file", "")
            link = f"https://github.com/plainyogurt21/CRLs/blob/main/unapproved_CRLs/{file_name}" if file_name else ""
            row = {k: r.get(k, "") for k in fieldnames}
            row["link"] = link
            writer.writerow(row)

    print(f"Saved: {csv_path}")


def find_pdfs(input_dir: str) -> List[str]:
    if not os.path.isdir(input_dir):
        return []
    paths: List[str] = []
    for root, _, files in os.walk(input_dir):
        for name in files:
            if name.lower().endswith(".pdf"):
                paths.append(os.path.join(root, name))
    return sorted(paths)


def process_batch_parallel(pdf_batch: List[str], batch_num: int, total_batches: int) -> List[Dict[str, Any]]:
    """Process a batch of PDFs in parallel using ThreadPoolExecutor."""
    batch_results = []
    
    print(f"Processing batch {batch_num}/{total_batches} ({len(pdf_batch)} PDFs)...")
    
    with ThreadPoolExecutor(max_workers=len(pdf_batch)) as executor:
        # Submit all PDFs in the batch
        future_to_pdf = {executor.submit(parse_pdf_with_llm, pdf): pdf for pdf in pdf_batch}
        
        # Collect results as they complete
        for future in as_completed(future_to_pdf):
            pdf = future_to_pdf[future]
            try:
                result = future.result()
                if result:
                    batch_results.append(result)
                print(f"  ✓ Completed: {os.path.basename(pdf)}")
            except Exception as e:
                print(f"  ✗ Error processing {os.path.basename(pdf)}: {e}")
    
    return batch_results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Parse CRL PDFs and extract structured fields with OpenRouter LLM")
    parser.add_argument("--input", default="unapproved_CRLs", help="Directory containing CRL PDFs")
    parser.add_argument("--out", default="outputs", help="Directory to write outputs")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of PDFs to process (0 = all)")
    parser.add_argument("--batch-size", type=int, default=25, help="Number of PDFs to process in parallel per batch")
    args = parser.parse_args()

    os.makedirs(args.input, exist_ok=True)
    pdfs = find_pdfs(args.input)
    if not pdfs:
        print(f"No PDFs found in '{args.input}'. Place files there and re-run.")
        return

    if args.limit > 0:
        pdfs = pdfs[: args.limit]

    print(f"Found {len(pdfs)} PDFs to process in batches of {args.batch_size}")
    
    # Process in batches
    results: List[Dict[str, Any]] = []
    batch_size = args.batch_size
    total_batches = (len(pdfs) + batch_size - 1) // batch_size
    
    for i in range(0, len(pdfs), batch_size):
        batch_pdfs = pdfs[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        
        batch_results = process_batch_parallel(batch_pdfs, batch_num, total_batches)
        results.extend(batch_results)
        
        # Small delay between batches to avoid overwhelming the API
        if batch_num < total_batches:
            print("Waiting 2 seconds before next batch...")
            time.sleep(2)

    if results:
        write_outputs(results, args.out)
        print(f"Successfully processed {len(results)}/{len(pdfs)} PDFs")
    else:
        print("No results produced.")


if __name__ == "__main__":
    main()
