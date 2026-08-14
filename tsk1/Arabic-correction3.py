import numpy as np
import torch
from datasets import load_dataset, disable_progress_bars
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)

disable_progress_bars()
MAX_LENGTH = 160
PREFIX = "صحح الأخطاء الإملائية والنحوية: "
SAMPLE_SIZE = 30000
VALID_SIZE = 2000


def preprocess(examples, tokenizer):
    sources = [PREFIX + s for s in examples["source"]]
    model_inputs = tokenizer(sources, max_length=MAX_LENGTH, truncation=True)
    labels = tokenizer(text_target=examples["target"], max_length=MAX_LENGTH, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def compute_metrics(eval_preds, tokenizer):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    exact_match = sum(
        p.strip() == l.strip() for p, l in zip(decoded_preds, decoded_labels)
    ) / len(decoded_preds)
    return {"exact_match": exact_match}


def correct(text, tokenizer, model, device):
    prompt = f"{PREFIX}{text}"

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
    ).to(device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_LENGTH,
        num_beams=4,
        early_stopping=True,
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


if __name__ == "__main__":
    model_name = "UBC-NLP/AraT5v2-base-1024"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    raw = load_dataset("s3h/arabic-grammar-corrections")["train"]
    raw = raw.rename_columns({"src": "source", "trg": "target"})

    if SAMPLE_SIZE:
        raw = raw.shuffle(seed=42).select(range(SAMPLE_SIZE))

    split = raw.train_test_split(test_size=VALID_SIZE, seed=42)
    dataset = {"train": split["train"], "validation": split["test"]}

    tokenized_dataset = {
        name: ds.map(
            preprocess,
            batched=True,
            num_proc=1,
            fn_kwargs={"tokenizer": tokenizer},
            remove_columns=ds.column_names,
        )
        for name, ds in dataset.items()
    }

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir="./mt5-arabic",
        learning_rate=3e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4, 
        num_train_epochs=3,
        weight_decay=0.01,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        predict_with_generate=True,
        generation_max_length=MAX_LENGTH,  
        load_best_model_at_end=True,
        metric_for_best_model="exact_match",
        greater_is_better=True,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda eval_preds: compute_metrics(eval_preds, tokenizer),
    )

    trainer.train()
    trainer.save_model("./arabic_corrector")
    tokenizer.save_pretrained("./arabic_corrector")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(
        "./arabic_corrector",
        use_fast=False
    )

    model = AutoModelForSeq2SeqLM.from_pretrained( "./arabic_corrector" ).to(device)
    text = "ذهبتت الي المدرسه"
    prediction = correct(text, tokenizer, model, device)

    print("Original :", text)
    print("Corrected:", prediction)