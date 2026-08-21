import os
import re
from groq import Groq

client = Groq(api_key="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")

SYSTEM_PROMPT = (
    "You are a certified professional translator specializing in formal "
    "business, academic, and official correspondence between Arabic and "
    "English. You have native-level fluency in both languages and deep "
    "familiarity with Modern Standard Arabic (MSA) conventions, formal "
    "register, and cross-cultural business etiquette. Your translations "
    "are precise, idiomatic, and faithfully preserve the tone, intent, "
    "and level of formality of the source text. You never add "
    "commentary, explanations, disclaimers, or personal opinions — you "
    "return only the requested translation."
)

_ARABIC_RANGE = re.compile(r"[\u0600-\u06FF\u0750-\u077F]")


def detect_language(text: str) -> str:
    """Return 'Arabic' if the text contains Arabic script, else 'English'."""
    return "Arabic" if _ARABIC_RANGE.search(text) else "English"


def translate(
    text: str,
    source_lang: str | None = None,
    target_lang: str | None = None,
    model: str = "qwen/qwen3.6-27b",
) -> str:
    """
    Translate text between Arabic and English in either direction.

    If source_lang / target_lang are omitted, the direction is
    auto-detected: Arabic input -> English, any other input -> Arabic.
    """
    if source_lang is None:
        source_lang = detect_language(text)
    if target_lang is None:
        target_lang = "English" if source_lang == "Arabic" else "Arabic"

    user_prompt = f"""Context: The text below must be translated for a professional reader whose working language is {target_lang}. The source text is in {source_lang}.

Task: Produce a formal, publication-quality translation of the following text from {source_lang} into {target_lang}, faithfully preserving meaning, tone, register, and intent.

Constraints:
- Output the translation only — no explanations, notes, disclaimers, or commentary of any kind.
- Preserve names, dates, numbers, and figures exactly as given in the source.
- Maintain the formal, professional register of the source text; do not casualize, embellish, or simplify.
- Do not translate proper nouns unless an official or standard {target_lang} equivalent exists.
- Resolve any ambiguity using the most contextually appropriate professional interpretation, without noting the ambiguity in the output.
- Ensure grammatical correctness and natural, idiomatic phrasing in {target_lang}; avoid literal or word-for-word translation where it would sound unnatural.

Output format: Plain text containing only the translated sentence(s) — no quotation marks, no labels, no extra formatting.

Text to translate:
\"{text}\""""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        reasoning_effort="none",
    )
    return response.choices[0].message.content.strip()


if __name__ == "__main__":
    sample_en = "Thank you for your application. We will contact you within two weeks."
    print(translate(sample_en))

    print("---")

    sample_ar = "شكرًا لتقديمكم طلب التوظيف. سنتواصل معكم خلال أسبوعين."
    print(translate(sample_ar))