import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from groq import Groq
try:
    from groq import APIConnectionError, APIStatusError, RateLimitError
except ImportError:
    APIConnectionError = Exception
    APIStatusError = Exception
    RateLimitError = Exception

MODEL = "openai/gpt-oss-120b"
INPUT_FILE = "next.txt"
OUTPUT_FILE = "summary.txt"
GROUNDING_REPORT_FILE = "grounding_report.txt"
MIN_BULLETS = 5
MAX_BULLETS = 7
MAX_WORDS_PER_BULLET = 25
MAX_WORDS_PER_CHUNK = 1200
MAX_RETRIES = 4
INITIAL_BACKOFF_SECONDS = 2
REASONING_EFFORT = "low"
ENABLE_GROUNDING_CHECK = True
EVALUATION_FILE = None

SYSTEM_PROMPT = (
    "You are an analytical research assistant who specializes in "
    "condensing long documents into clear, accurate summaries for busy readers." )

SUMMARY_FEWSHOT = """
Example:
Document:
The university announced that applications for the exchange program will close on June 30. 
Students must submit their documents online. Successful applicants will be contacted within two weeks.
Ideal output:
Exchange Program Applications
- Applications close on June 30.
- Students must submit their documents online.
- Successful applicants will be contacted within two weeks.
"""

def build_chunk_prompt(document_text: str, chunk_index: int, total_chunks: int) -> str:
    return f"""
Context:
The text below is section {chunk_index} of {total_chunks} from one long document.

Task:
Summarize this section into its most important factual points so the partial summary can later be combined into a final document summary.

Constraints:
- Use only information explicitly stated in the source.
- Do not infer, guess, or add outside knowledge.
- Preserve important names, dates, figures, statistics, and claims exactly.
- Remove repetition and minor details.
- Keep important information in the same general order.
- Output only the partial summary.
- Use concise bullet points.

{SUMMARY_FEWSHOT}

Output format:
- concise factual bullet
- concise factual bullet
- concise factual bullet

Section {chunk_index}:
{document_text}
"""

def build_final_prompt(partial_summaries: str) -> str:
    return f"""
Context:
The following are summaries of consecutive sections from the same long document.

Task:
Create one final summary that represents the entire document.

Constraints:
- Output exactly {MIN_BULLETS}-{MAX_BULLETS} bullet points.
- Each bullet must be a single sentence.
- Each bullet must contain no more than {MAX_WORDS_PER_BULLET} words.
- Include only information supported by the source summaries.
- Do not add outside knowledge or personal opinions.
- Preserve figures, dates, names, and statistics exactly when they appear.
- Remove duplicated information.
- Prioritize the most important events, findings, arguments, or conclusions.
- Keep the main information in the same general order as the document.
- Output a short title followed by the bullet list.
- Nothing before the title and nothing after the final bullet.

{SUMMARY_FEWSHOT}

Partial summaries:
{partial_summaries}

Final output:
"""

def build_grounding_prompt(source_text: str, summary: str) -> str:
    return f"""
Role:
You are a factuality and source-grounding reviewer.

Context:
A summary was generated from the source document below.

Task:
Check whether every important claim in the summary is supported by the source.

Rules:
- Do not use outside knowledge.
- Identify unsupported, altered, or invented claims.
- Pay special attention to names, dates, numbers, percentages, statistics, and causal claims.
- If all major claims are supported, report PASS.
- If any claim is unsupported or altered, report FAIL and explain exactly which claims need correction.

Source document:
{source_text}

Generated summary:
{summary}

Output format:
Status: PASS or FAIL
Issues:
- issue 1, or None
Recommended correction:
- concise correction, or None
"""

def create_client() -> Groq:
    api_key = "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"
    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY is not set. Set it as an environment variable; "
            "never place the key directly in source code.")
    return Groq(api_key=api_key)

def validate_input_file(file_path: str) -> Path:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Input file was not found: {path.resolve()}. "
            "Make sure next.txt exists or update INPUT_FILE.")
    if not path.is_file():
        raise ValueError(f"Input path is not a file: {path.resolve()}")
    if path.suffix.lower() != ".txt":
        raise ValueError(f"Expected a .txt file, but received: {path.name}")
    if path.stat().st_size == 0:
        raise ValueError(f"The input file is empty: {path.resolve()}")
    return path

def read_document(file_path: str) -> str:
    path = validate_input_file(file_path)
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"Could not decode {path.name} as UTF-8.") from exc
    text = text.strip()
    if not text:
        raise ValueError(f"The input file contains no usable text: {path.resolve()}")
    return text

def split_into_paragraphs(text: str) -> List[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

def split_into_sentences(text: str) -> List[str]:
    sentences = re.split(r"(?<=[.!?\u061f])\s+", text.strip())
    return [s.strip() for s in sentences if s.strip()]

def split_into_chunks(text: str, max_words: int = MAX_WORDS_PER_CHUNK) -> List[str]:
    if max_words <= 0:
        raise ValueError("Invalid chunking parameters.")
    paragraphs = split_into_paragraphs(text)
    if not paragraphs:
        return [text] if text.strip() else []
    chunks: List[str] = []
    current: List[str] = []
    current_words = 0
    def flush() -> None:
        nonlocal current, current_words
        if current:
            chunks.append("\n\n".join(current).strip())
            current, current_words = [], 0

    for para in paragraphs:
        para_words = len(para.split())
        if para_words > max_words:
            flush()
            sent_chunk: List[str] = []
            sent_words = 0
            for sent in split_into_sentences(para):
                w = len(sent.split())
                if sent_words + w > max_words and sent_chunk:
                    chunks.append(" ".join(sent_chunk).strip())
                    sent_chunk, sent_words = [], 0
                sent_chunk.append(sent)
                sent_words += w
            if sent_chunk:
                chunks.append(" ".join(sent_chunk).strip())
            continue
        if current_words + para_words > max_words and current:
            flush()

        current.append(para)
        current_words += para_words
    flush()
    return chunks if chunks else [text]

def is_retryable_error(exc: Exception) -> bool:
    if isinstance(exc, (RateLimitError, APIConnectionError)):
        return True
    if isinstance(exc, APIStatusError):
        status = getattr(exc, "status_code", None)
        return status is None or status in {408, 409, 429, 500, 502, 503, 504}
    message = str(exc).lower()
    return any(k in message for k in ("rate limit", "timeout", "connection", "429", "500", "502", "503", "504", "empty response"))

def call_groq_with_retry(client: Groq, messages: List[Dict[str, str]], max_tokens: int) -> str:
    last_error: Optional[Exception] = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=0.0,
                max_tokens=max_tokens,
                reasoning_effort=REASONING_EFFORT,
                include_reasoning=False, )
            content = response.choices[0].message.content
            if not content or not content.strip():
                raise RuntimeError("The model returned an empty response.")
            return content.strip()
        except Exception as exc:
            last_error = exc
            if not is_retryable_error(exc) or attempt == MAX_RETRIES:
                break
            delay = INITIAL_BACKOFF_SECONDS * (2 ** (attempt - 1))
            print(f"Request failed: {type(exc).__name__}. Retrying in {delay}s...")
            time.sleep(delay)
    raise RuntimeError(f"Groq request failed after {attempt} attempt(s): {last_error}") from last_error

def summarize_chunk(client: Groq, chunk: str, index: int, total: int) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_chunk_prompt(chunk, index, total)}, ]
    return call_groq_with_retry(client, messages, max_tokens=1000)

def summarize_document(client: Groq, document_text: str) -> Tuple[str, List[str]]:
    chunks = split_into_chunks(document_text)
    print(f"Document split into {len(chunks)} chunks.")
    partials = []
    for i, chunk in enumerate(chunks, 1):
        print(f"Summarizing chunk {i}/{len(chunks)}...")
        partials.append(summarize_chunk(client, chunk, i, len(chunks)))

    combined = "\n\n".join(f"SECTION {i}: {s}" for i, s in enumerate(partials, 1))
    final_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_final_prompt(combined)}, ]
    print("Creating final summary...")
    final = call_groq_with_retry(client, final_messages, max_tokens=1300)
    return final, partials

def check_grounding(client: Groq, source_text: str, summary: str) -> str:
    messages = [
        {"role": "system", "content": "You are a strict factuality reviewer. Judge only against the supplied source."},
        {"role": "user", "content": build_grounding_prompt(source_text, summary)}, ]
    return call_groq_with_retry(client, messages, max_tokens=1000)

def extract_numbers(text: str) -> List[str]:
    return re.findall(r"\b\d+(?:[.,]\d+)?%?\b", text)

def number_consistency_check(source: str, summary: str) -> Dict[str, object]:
    source_numbers = set(extract_numbers(source))
    summary_numbers = extract_numbers(summary)
    unsupported = [n for n in summary_numbers if n not in source_numbers]
    return {
        "summary_numbers": summary_numbers,
        "unsupported_summary_numbers": unsupported,
        "passed": not unsupported, }

def validate_summary_format(summary: str) -> Dict[str, object]:
    lines = [x.strip() for x in summary.splitlines() if x.strip()]
    title = lines[0] if lines else ""
    bullets = [x for x in lines[1:] if x.startswith("-")]
    counts = [len(re.sub(r"^-\s*", "", b).split()) for b in bullets]
    return { "title_present": bool(title),
        "title_word_count": len(title.split()),
        "bullet_count": len(bullets),
        "bullet_count_valid": MIN_BULLETS <= len(bullets) <= MAX_BULLETS,
        "all_bullets_within_limit": all(c <= MAX_WORDS_PER_BULLET for c in counts), }

def evaluate_summary(generated: str, reference: str) -> Dict[str, object]:
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        return {"error": "Install rouge-score with: pip install rouge-score"}
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference, generated)
    return {
        "ROUGE-1 F1": round(scores["rouge1"].fmeasure, 4),
        "ROUGE-2 F1": round(scores["rouge2"].fmeasure, 4),
        "ROUGE-L F1": round(scores["rougeL"].fmeasure, 4), }

def evaluate_test_set(client: Groq, evaluation_file: str) -> None:
    path = Path(evaluation_file)
    if not path.exists():
        raise FileNotFoundError(f"Evaluation file not found: {evaluation_file}")
    data = json.loads(path.read_text(encoding="utf-8"))
    for i, item in enumerate(data, 1):
        document = read_document(item["document"])
        generated, _ = summarize_document(client, document)
        print(f"Test {i}: {evaluate_summary(generated, item['reference_summary'])}")

def main() -> int:
    print("=== Improved Long-Document Summarization ===")
    try:
        client = create_client()
        source = read_document(INPUT_FILE)
        print(f"Loaded {INPUT_FILE}: {len(source):,} characters")
        summary, _ = summarize_document(client, source)
        Path(OUTPUT_FILE).write_text(summary, encoding="utf-8")
        print("\n=== FINAL SUMMARY ===\n")
        print(summary)
        print(f"\nSaved to: {OUTPUT_FILE}")
        print("\n=== FORMAT CHECK ===")
        print(json.dumps(validate_summary_format(summary), indent=2))
        print("\n=== NUMBER CHECK ===")
        print(json.dumps(number_consistency_check(source, summary), indent=2))

        if ENABLE_GROUNDING_CHECK:
            print("\n=== SOURCE-GROUNDING CHECK ===")
            report = check_grounding(client, source, summary)
            Path(GROUNDING_REPORT_FILE).write_text(report, encoding="utf-8")
            print(report)
            print(f"Grounding report saved to: {GROUNDING_REPORT_FILE}")

        if EVALUATION_FILE:
            evaluate_test_set(client, EVALUATION_FILE)

        return 0
    except (FileNotFoundError, ValueError, EnvironmentError) as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"\nUNEXPECTED ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    raise SystemExit(main())