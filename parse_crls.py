import os
import csv
import time
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils import (
    call_openrouter_api_simple,
    call_openrouter_api_with_pdf,
    extract_markdown_text,
    build_json_schema,
    find_pdfs,
    parse_json_response
)


SYSTEM_PROMPT = (
    "You are a precise regulatory analyst extracting key fields from FDA Complete Response Letters (CRLs). "
    "Return only JSON following the provided schema with exact keys. If a value is unknown, use an empty string. "
    "Classify the reason using fields:  efficacy, safety, manufacturing, multiple reasons, other."
)


USER_PROMPT_TEMPLATE = (
    "Extract these fields from the CRL text and adhere to the schema: \n"
    "- drug_name\n"
    "- company (applicant/sponsor) found on page 1\n"
    "- summary (reason for CRL)\n"
    "- classification \n"
    "- path_to_future_approval (what is required to gain approval)\n\n"
    "CRL EXCERPTS:\n\n{doc_text}"
)


def parse_pdf_with_text_extraction(pdf_path: str) -> Optional[Dict[str, Any]]:
    """Parse PDF by first extracting text, then sending to LLM."""
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

    return parse_json_response(response, pdf_path)


def parse_pdf_direct_multimodal(pdf_path: str) -> Optional[Dict[str, Any]]:
    """Parse PDF by sending it directly to the LLM using multimodal capabilities."""
    user_prompt = (
        "Extract these fields from this CRL PDF and adhere to the schema: \n"
        "- drug_name\n"
        "- company (applicant/sponsor) found on page 1\n"
        "- summary (reason for CRL)\n"
        "- classification\n"
        "- path_to_future_approval (what is required to gain approval)\n\n"
        "Please analyze the PDF document attached to this message."
    )
    
    schema = build_json_schema()

    response, usage = call_openrouter_api_with_pdf(
        pdf_path=pdf_path,
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

    return parse_json_response(response, pdf_path)


def parse_pdf_with_llm(pdf_path: str, use_multimodal: bool = False) -> Optional[Dict[str, Any]]:
    """
    Parse PDF using either text extraction or direct multimodal approach.
    
    Args:
        pdf_path: Path to the PDF file
        use_multimodal: If True, send PDF directly to LLM. If False, extract text first.
    """
    if use_multimodal:
        return parse_pdf_direct_multimodal(pdf_path)
    else:
        return parse_pdf_with_text_extraction(pdf_path)


def write_outputs(rows: List[Dict[str, Any]], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "crl_results.csv")

    # CSV only
    fieldnames = [
        "drug_name",
        "company",
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


def process_batch_parallel(pdf_batch: List[str], batch_num: int, total_batches: int, use_multimodal: bool = False) -> List[Dict[str, Any]]:
    """Process a batch of PDFs in parallel using ThreadPoolExecutor."""
    batch_results = []
    
    print(f"Processing batch {batch_num}/{total_batches} ({len(pdf_batch)} PDFs)...")
    
    with ThreadPoolExecutor(max_workers=len(pdf_batch)) as executor:
        # Submit all PDFs in the batch
        future_to_pdf = {executor.submit(parse_pdf_with_llm, pdf, use_multimodal): pdf for pdf in pdf_batch}
        
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
    parser.add_argument("--no-multimodal", action="store_true", help="Use text extraction instead of direct PDF processing")
    args = parser.parse_args()
    
    # Default to multimodal unless --no-multimodal is specified
    use_multimodal = not args.no_multimodal

    os.makedirs(args.input, exist_ok=True)
    pdfs = find_pdfs(args.input)
    if not pdfs:
        print(f"No PDFs found in '{args.input}'. Place files there and re-run.")
        return

    if args.limit > 0:
        pdfs = pdfs[: args.limit]

    processing_method = "multimodal PDF" if use_multimodal else "text extraction"
    print(f"Found {len(pdfs)} PDFs to process in batches of {args.batch_size} using {processing_method}")
    
    # Process in batches
    results: List[Dict[str, Any]] = []
    batch_size = args.batch_size
    total_batches = (len(pdfs) + batch_size - 1) // batch_size
    
    for i in range(0, len(pdfs), batch_size):
        batch_pdfs = pdfs[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        
        batch_results = process_batch_parallel(batch_pdfs, batch_num, total_batches, use_multimodal)
        results.extend(batch_results)
        
        # Small delay between batches to avoid overwhelming the API
        if batch_num < total_batches:
            print("Waiting .1 seconds before next batch...")
            time.sleep(.1)

    if results:
        write_outputs(results, args.out)
        print(f"Successfully processed {len(results)}/{len(pdfs)} PDFs")
    else:
        print("No results produced.")


if __name__ == "__main__":
    main()
