import os
import re
import json
import traceback
import threading
import requests
import base64
from typing import List, Dict, Any, Optional
import pymupdf4llm
import openai


# Thread lock for PyMuPDF4LLM operations
_pdf_extraction_lock = threading.Lock()


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


def encode_pdf_to_base64(pdf_path: str) -> str:
    """Encode PDF file to base64 data URL format."""
    with open(pdf_path, "rb") as pdf_file:
        base64_pdf = base64.b64encode(pdf_file.read()).decode('utf-8')
    
    return f"data:application/pdf;base64,{base64_pdf}"


def call_openrouter_api_with_pdf(
    pdf_path: str,
    system_prompt: str,
    user_prompt: str,
    model: str = "google/gemini-2.5-flash-Lite",
    temperature: float = 0.0,
    max_tokens: int = 8000,
    json_mode: bool = False,
    json_schema: Optional[Dict[str, Any]] = None,
    json_schema_name: str = 'Schema Json',
):
    """
    Makes a call to the OpenRouter API with PDF directly attached using requests.
    
    Args:
        pdf_path: Path to the PDF file
        system_prompt: System prompt for the model
        user_prompt: User prompt for the model
        Other parameters same as call_openrouter_api_simple
    
    Returns:
        tuple: (response_content, usage) - The model's response text and token usage
    """
    response_content = ""
    usage = {'input_tokens': 0, 'output_tokens': 0}
    
    try:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY not set in environment")

        # Ensure the file path is absolute and exists
        if not os.path.isabs(pdf_path):
            pdf_path = os.path.abspath(pdf_path)
        
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        # Encode PDF to base64
        base64_pdf = encode_pdf_to_base64(pdf_path)
        filename = os.path.basename(pdf_path)

        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/OpenRouterTeam/openrouter-python",
            "X-Title": "OpenRouter API",
        }

        # Create messages with PDF attachment according to OpenRouter docs
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user", 
                "content": [
                    {"type": "text", "text": user_prompt},
                    {
                        "type": "file",
                        "file": {
                            "filename": filename,
                            "file_data": base64_pdf
                        }
                    }
                ]
            }
        ]

        # Configure PDF processing engine
        plugins = [
            {
                "id": "file-parser",
                "pdf": {
                    "engine": "native"  # Free text extraction engine
                }
            }
        ]

        payload = {
            "model": model,
            "messages": messages,
            "plugins": plugins,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if json_mode:
            if json_schema:
                payload["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": json_schema_name,
                        "strict": True,
                        "schema": json_schema,
                    },
                }
            else:
                payload["response_format"] = {"type": "json_object"}

        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        response_data = response.json()

        if not response_data.get("choices"):
            print(f"No choices in response: {response_data}")
            return None, usage

        choice = response_data["choices"][0]
        finish_reason = choice.get("finish_reason")
        
        if finish_reason == 'error':
            print(f"API response stopped early: {finish_reason}")
            return None, usage

        response_content = choice["message"]["content"]

        # Extract usage information
        if "usage" in response_data:
            usage = {
                'input_tokens': response_data["usage"].get("prompt_tokens", 0),
                'output_tokens': response_data["usage"].get("completion_tokens", 0),
            }

    except requests.exceptions.RequestException as e:
        print(f"HTTP error calling OpenRouter API with PDF: {str(e)}")
        if hasattr(e, 'response') and e.response is not None:
            try:
                error_data = e.response.json()
                print(f"Error response: {error_data}")
            except json.JSONDecodeError:
                print(f"Error response text: {e.response.text}")
        return None, usage
    except Exception as e:
        print(f"Error calling OpenRouter API with PDF: {str(e)}")
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
            "summary": {"type": "string", "description": "Summary of the reason for CRL"},
            "classification": {"type": "string", "enum": ["efficacy", "safety", "multiple", "manufacturing", "others"]},
            "path_to_future_approval": {"type": "string", "description": "What the sponsor must do to gain future approval"},
        },
        "required": [
            "drug_name",
            "company",
            "summary",
            "classification",
            "path_to_future_approval",
        ],
        "additionalProperties": False,
    }


def find_pdfs(input_dir: str) -> List[str]:
    if not os.path.isdir(input_dir):
        return []
    paths: List[str] = []
    for root, _, files in os.walk(input_dir):
        for name in files:
            if name.lower().endswith(".pdf"):
                paths.append(os.path.join(root, name))
    return sorted(paths)


def parse_json_response(response: str, pdf_path: str) -> Optional[Dict[str, Any]]:
    """Parse JSON response from LLM, handling potential code fences."""
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