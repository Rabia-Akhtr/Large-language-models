# Large-language-models
# AG News Text Classification — Baseline vs Full Fine-tuned BERT (LLM)

**Task:** 4-class news topic classification on **AG News** (World, Sports, Business, Sci/Tech).  
**Main goal:** Compare a strong classical baseline (**TF-IDF + Logistic Regression**) against a **BERT-style** model (**bert-base-uncased**) fine-tuned end-to-end, using **validation-only** model selection and **one-time** test evaluation (no leakage).

---

## Project Highlights (Assignment Requirements)
- ✅ Public dataset: **AG News** (official train/test splits)
- ✅ Baseline model: **TF-IDF + Logistic Regression**
- ✅ LLM model: **Fine-tuned BERT** (`bert-base-uncased`)
- ✅ Validation split created from train only (**no test leakage**)
- ✅ Hyperparameter tuning via small grid search (**selection on validation Macro-F1 only**)
- ✅ Evaluation metrics: **Accuracy, Macro-Precision, Macro-Recall, Macro-F1**
- ✅ Visuals: learning curves, confusion matrix, model comparison bar chart
- ✅ Reproducibility artifacts saved (config, grid results, history, predictions, error analysis)

---


> Note: Large checkpoint files (e.g., `best_model.pt`) are not required for marking and are often excluded from GitHub to keep the repo lightweight.

---

## Dataset
- **AG News** is a standard benchmark for topic classification with **4 balanced classes**.
- Official splits used:
  - **Train:** 120,000 samples
  - **Test:** 7,600 samples

---

## Methods

### Baseline (Classical ML)
- **TF-IDF** features (unigrams + bigrams)
- **Multinomial Logistic Regression**
- Trained on the **official train split**, evaluated on the **official test split**

### Fine-tuned BERT (LLM)
- Model: **`bert-base-uncased`**
- Tokenisation: WordPiece, truncation to **max_length=128**, dynamic padding per batch
- Optimisation: **AdamW** + linear warmup/decay schedule
- Model selection: **best validation Macro-F1**
- Test set: used **once** after selecting best config

---

## Hyperparameter Optimisation (Validation-only)
A small grid search explored:
- learning rate: `2e-5`, `3e-5`
- batch size: `16`, `32`
- epochs: `3`, `4`
- weight decay: `0.0`, `0.01`

**Best config (by validation Macro-F1):**
- `lr = 3e-5`, `batch_size = 32`, `epochs = 3`, `weight_decay = 0.01`

See: `artifacts_agnews/grid_results.csv`

---

## Results (Official Test Set)

### Overall metrics
| Model | Accuracy | Macro-Precision | Macro-Recall | Macro-F1 |
|------|---------:|----------------:|-------------:|---------:|
| TF-IDF + Logistic Regression | 0.9213 | 0.9211 | 0.9213 | 0.9211 |
| Fine-tuned BERT (selected by VAL Macro-F1) | 0.9492 | 0.9493 | 0.9492 | 0.9492 |

**Gain from fine-tuning:** ~**+2.8 percentage points** in both Accuracy and Macro-F1.

### Per-class behaviour (BERT)
- **Sports** and **World** achieve the strongest performance (very distinctive vocabulary).
- Most remaining errors occur between **Business** and **Sci/Tech**, which often share overlapping terms (companies, products, markets, innovation).

**Visuals**
- Model comparison bar chart: `figures/model_comparison_bar_percent.png`
- Learning curves (Log loss + Val Macro-F1): `figures/bert_training_curves_one_figure.png`
- Confusion matrix (BERT): `figures/confusion_matrix_bert.png`

---

## Interpretation (What the results mean)
The baseline already performs strongly because AG News contains clear topic-specific keywords, which TF-IDF captures well. However, **fine-tuning BERT improves performance** by leveraging **contextual representations** learned during pretraining. This is consistent with common findings in the Transformer literature that pretrained encoder models provide strong general-purpose language features that boost downstream classification, especially when headlines contain ambiguity or shared vocabulary across classes.

---

## Limitations 
Performance is near a ceiling on AG News, so further tuning yields diminishing returns. Some headlines are inherently ambiguous—especially **Business vs Sci/Tech**—so perfect separation is unlikely without more context. The hyperparameter search was intentionally small for compute constraints, and results were validated only on AG News, so robustness under domain shift (other news sources/time periods) remains untested.

---

## How to Run

### Option 1: Run in Google Colab
1. Open `notebooks/Large_language_models_(1).ipynb`
2. Run all cells from top to bottom

### Option 2: Run locally
```bash
pip install -r requirements.txt
jupyter notebook

