## Task 1: Arabic Text Correction
The first task focuses on detecting and correcting errors in Arabic text, including:

- Spelling mistakes
- Grammar mistakes
- Punctuation errors
- Repeated or elongated characters
- Arabic character normalization

Three approaches were implemented and compared:

1. **Hybrid + LLM**
   - Arabic preprocessing
   - Contextual spelling correction
   - Llama 3.3 70B for grammar and punctuation correction

2. **Specialized Arabic Correction Pipeline**
   - Arabic preprocessing
   - Spelling correction
   - CAMeL-BERT for Grammar Error Detection (GED)
   - AraBART for Grammar Error Correction (GEC)

3. **Fine-Tuned AraT5**
   - Uses `UBC-NLP/AraT5v2-base-1024`
   - Fine-tuned on the `s3h/arabic-grammar-corrections` dataset
   - Learns to transform incorrect Arabic text into corrected text

## Project Structure

```text
tsk1/
│
├── Arabic-correction.py
├── Arabic-correction2.py
├── Arabic-correction3.py
└── Arabic_Text_Correction.docx
```

## Task 2: RNN, GRU, and LSTM Comparison

This task compares three recurrent neural network architectures:
- **RNN (Recurrent Neural Network)**
- **GRU (Gated Recurrent Unit)**
- **LSTM (Long Short-Term Memory)**

The models are trained and evaluated on NLP tasks to compare their ability to
process sequential text using them in Next-Word Prediction, Text classification and Named Entity Recognition (NER)

### Main objectives
- Compare model performance
- Compare training/inference speed
- Compare the number of parameters
- Study the effect of sequence length
- Understand long-term dependencies
- Compare the trade-off between GRU and LSTM

### Models
| Model | Main Characteristic |
|---|---|
| RNN | Simple recurrent architecture |
| GRU | Uses update and reset gates |
| LSTM | Uses cell state and multiple gates |

The comparison focuses on metrics such as **Precision, Recall, F1-score,
Entity-level F1**, and computational performance where available.

```text
tsk2/
│
├── NER.csv
├── NER.ipynb
├── Next_WordPerd.ipynb
├── nextword.csv
├── Text_classification.ipynb
├── text_class.csv
└── Results and Comparison.docx
```
