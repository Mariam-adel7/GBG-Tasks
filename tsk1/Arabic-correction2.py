import re
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

    def __init__(self, enable_spell_correction: bool = True, enable_grammar_check: bool = True):
        self.corrector = Corrector() if enable_spell_correction else None
        self.ged_pipeline = None
        self.gec_tokenizer = None
        self.gec_model = None
        if enable_grammar_check:
            from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
            self.ged_pipeline = pipeline(
                'token-classification',
                model='CAMeL-Lab/camelbert-msa-qalb14-ged-13',
                aggregation_strategy='first',
            )
            gec_model_name = 'CAMeL-Lab/arabart-qalb15-gec-ged-13'
            self.gec_tokenizer = AutoTokenizer.from_pretrained(gec_model_name)
            self.gec_model = AutoModelForSeq2SeqLM.from_pretrained(gec_model_name)

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

    def correct_spelling_report(self, text: str):
        if not self.corrector:
            raise RuntimeError(
                "Spell correction is disabled - init with enable_spell_correction=True"
            )
        tokens = self.corrector.processor.tokenize(text)
        fixed_tokens = []
        errors = []
        for token in tokens:
            if not self.corrector.is_known(token):
                possible_fixes = self.corrector.get_possible_fixes(token, 5)
                fix = (
                    self.corrector.rank_possibilities(fixed_tokens, possible_fixes)
                    if possible_fixes != token
                    else token
                )
            else:
                fix = token
            if fix != token:
                errors.append((token, fix))
            fixed_tokens.append(fix)
        return ' '.join(fixed_tokens).strip(), errors

    def process(self, text: str, fix_spelling: bool = True) -> str:
        cleaned = self.clean(text)
        if fix_spelling:
            cleaned = self.correct_spelling(cleaned)
        return cleaned

    def detect_grammar_errors(self, text: str):
        if not self.ged_pipeline:
            raise RuntimeError(
                "Grammar check is disabled - init with enable_grammar_check=True"
            )
        predictions = self.ged_pipeline(text)
        return [(p['word'], p['entity_group']) for p in predictions if p['entity_group'] != 'UC']

    def correct_grammar(self, text: str, max_new_tokens: int = 128) -> str:
        if not self.gec_model:
            raise RuntimeError(
                "Grammar check is disabled - init with enable_grammar_check=True"
            )
        inputs = self.gec_tokenizer(text, return_tensors='pt', truncation=True, max_length=256)
        outputs = self.gec_model.generate(**inputs, max_new_tokens=max_new_tokens, num_beams=4)
        return self.gec_tokenizer.decode(outputs[0], skip_special_tokens=True)

    def correct_full(self, text: str):
        cleaned = self.clean(text)
        spelling_fixed, spelling_errors = self.correct_spelling_report(cleaned)
        grammar_fixed = self.correct_grammar(spelling_fixed)
        return grammar_fixed, spelling_errors


if __name__ == "__main__":
    sample = "ااننا لممم اذهنب للمدرسهه الليومم."
    processor = ArabicTextProcessor()

    print("Original :", sample)
    print("Cleaned  :", processor.clean(sample))

    corrected, errors = processor.correct_spelling_report(processor.clean(sample))
    print("Corrected:", corrected)
    print("Spelling errors found:")
    for wrong, fixed in errors:
        print(f"  {wrong} -> {fixed}")

    print("Grammar issues flagged (detection only):")
    for word, tag in processor.detect_grammar_errors(corrected):
        print(f"  {word} -> {tag}")

    final_text, _ = processor.correct_full(sample)
    print("Final (spelling + grammar corrected):", final_text)