import re
import requests
import pyarabic.araby as araby
from camel_tools.utils.dediac import dediac_ar
from camel_tools.utils.normalize import (
    normalize_unicode,
    normalize_alef_ar,
    normalize_alef_maksura_ar,
    normalize_teh_marbuta_ar,
)
from camel_tools.tokenizers.word import simple_word_tokenize
from ar_corrector.corrector import Corrector

class ArabicTextProcessor:
    
    _NON_ARABIC = re.compile(r'[^\u0621-\u064A\u0660-\u0669\s،؛؟.!]')
    _ELONGATION = re.compile(r'(.)\1{2,}')
    _MULTI_SPACE = re.compile(r'\s+')

    def __init__(self, enable_spell_correction: bool = True):
        self.corrector = Corrector() if enable_spell_correction else None

    def normalize_alef_hamza(self, text: str) -> str:
        return normalize_alef_ar(text)

    def normalize_for_matching(self, text: str) -> str:
        text = normalize_teh_marbuta_ar(text)       
        text = normalize_alef_maksura_ar(text)      
        text = text.replace('ؤ', 'و').replace('ئ', 'ي')
        return text

    def remove_diacritics(self, text: str) -> str:
        return dediac_ar(text)

    def remove_elongation(self, text: str) -> str:
        return self._ELONGATION.sub(r'\1', text)

    def remove_non_arabic(self, text: str) -> str:
        return self._NON_ARABIC.sub('', text)

    def normalize_whitespace(self, text: str) -> str:
        return self._MULTI_SPACE.sub(' ', text).strip()

    def clean(
        self,
        text: str,
        remove_diacritics: bool = True,
        remove_elongation: bool = True,
        remove_non_arabic: bool = True,
        for_matching: bool = False,
    ) -> str:
        text = araby.strip_tatweel(text)
        text = normalize_unicode(text)         
        if remove_diacritics:
            text = self.remove_diacritics(text)
        if remove_elongation:
            text = self.remove_elongation(text)
        text = self.normalize_alef_hamza(text)
        if for_matching:
            text = self.normalize_for_matching(text)
        if remove_non_arabic:
            text = self.remove_non_arabic(text)
        return self.normalize_whitespace(text)

    def tokenize(self, text: str) -> list:
        return simple_word_tokenize(text)

    def correct_spelling(self, text: str) -> str:
        if not self.corrector:
            raise RuntimeError(
                "Spell correction is disabled - init with enable_spell_correction=True"
            )
        return self.corrector.contextual_correct(text)

    def process(self, text: str, fix_spelling: bool = True) -> str:
        cleaned = self.clean(text)
        if fix_spelling:
            cleaned = self.correct_spelling(cleaned)
        return cleaned

def correct_grammar_llm(text: str, api_key: str, model: str = "llama-3.3-70b-versatile") -> str:
    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content" : """
أنت مدقق لغوي متخصص في اللغة العربية الفصحى.
المطلوب:
1- صحح الأخطاء الإملائية.
2- صحح الأخطاء النحوية.
3- صحح علامات الترقيم إذا كانت خاطئة.
4- لا تعيد صياغة الجملة.
5- لا تستخدم مرادفات.
6- لا تختصر النص.
7- لا تشرح ما قمت به.
8- أعد النص المصحح فقط.

أمثلة:

الإدخال:
ذهبت الي المدرسه
الإخراج:
ذهبت إلى المدرسة

الإدخال:
هاذا كتاب جميل
الإخراج:
هذا كتاب جميل

الإدخال:
إن المعلمون هم بناة الأجيال
الإخراج:
إن المعلمين هم بناة الأجيال
"""
                },
                {"role": "user", "content": text},
            ],
            "temperature": 0,
            "top_p": 0.1,
        },
        timeout=30,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()

if __name__ == "__main__":
    sample = "إإن المعلمون هم بناة الأجيال، ولذالك يجب على الطلاب أن يستمعوا إلي نصائحهم بعناية. ذهب أحمد وعلي إلي المدرسة المبكرة، ولاكنهم وجدوا الأبواب مغلقةً لأنهم وصلوا قبل الموعد بساعةٍ كاملة. لم يتأخر أحداً عن الطابور الصباحي اليوم، فالجميع حرصوا على الحضور مبكرين."
    processor = ArabicTextProcessor()

    print("Original :", sample)
    print("Cleaned  :", processor.clean(sample))
    print(correct_grammar_llm(processor.process(sample), api_key="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"))
