# Extractive Summarization of Bodo News Articles 

This repository contains a comparative project on extractive text summarization for **Bodo (brx)**, a low-resource, Devanagari-script language spoken primarily in Assam, India. 

Due to the lack of off-the-shelf NLP tooling (like sentence tokenizers) and annotated summarization datasets for Bodo, this project builds a summarization pipeline from first principles. It implements and compares two distinct unsupervised approaches:
1. **TF-IDF (Lexical approach):** A classical statistical method based on word rarity and frequency.
2. **IndicBERTv2 (Semantic approach):** A modern transformer-based method utilizing contextual sentence embeddings and document-centroid similarity.

Both systems guarantee a minimum retention of 30% of the original article's sentences, reassembling them in their original chronological order.

---

## 📊 Dataset

The corpus was scraped from the Bodo-language edition of *The Sentinel* (`bodo.sentinelassam.com`).
* **Total Articles:** 2,056 (after cleaning missing rows)
* **Columns:** `title`, `article`, `source`, `url`
* **Data Split:** 80% Train (1,644 articles) / 20% Held-out Test (412 articles). 
  * *Note: The train split is only used to fit the TF-IDF vectorizer. IndicBERT requires no corpus-specific fitting.*

---

## ⚙️ Preprocessing & Tokenization

Since standard NLP tokenizers (like NLTK's `punkt`) do not support Bodo punctuation, a custom pipeline was developed:
* **Cleaning:** A strict character whitelist preserves only the Devanagari Unicode block (U+0900–U+097F), ASCII digits, the danda (।), apostrophes (used in Bodo orthography, e.g., बर'), and whitespace.
* **Sentence Tokenization:** Text is split using the danda (।) rather than the Latin full stop, correctly parsing Bodo sentence boundaries.

---

## 🧠 Methodologies

### Model 1: TF-IDF (Lexical Rarity)
This classical approach scores sentences based on the aggregate weight of their constituent words.
* **Stop-words:** Uses a manually curated list of 34 Bodo function words (pronouns, conjunctions, particles) to prevent them from dominating scores.
* **Vectorization:** A `TfidfVectorizer` is fitted exclusively on the 1,644-article training corpus to learn IDF statistics. 
* **Scoring:** Each sentence is scored by summing the TF-IDF weights of its words. Top-scoring sentences are extracted.
* **Pros & Cons:** Extremely fast, CPU-friendly, and highly interpretable. However, it cannot capture semantic meaning or recognize paraphrased content.

### Model 2: IndicBERT (Semantic Centrality)
This approach leverages `ai4bharat/IndicBERTv2-MLM-only`, a pretrained multilingual transformer covering 26 Indian languages, to understand semantic context.
* **Embeddings:** Because the base model lacks a sentence-pooling head, it uses mean-pooling over the token-level hidden states (768-dimensional) to create a single dense vector per sentence.
* **Scoring:** Calculates a "document centroid" (the mean of all sentence embeddings in the article). Sentences are ranked by their **cosine similarity** to this centroid.
* **Pros & Cons:** Captures deep semantic meaning and requires no corpus fitting (works natively on unseen text). However, it is computationally expensive (requires a GPU for batch processing) and acts as a "black box" compared to TF-IDF.

---

## 📈 Results & Evaluation

Both systems were evaluated on the exact same 412-article held-out test set. Because no human-authored gold-standard summaries exist for Bodo, ROUGE scores were computed using the *source article* as the reference (a standard proxy metric for unsupervised extractive summarization).

| Metric | TF-IDF Pipeline (F1) | IndicBERT Pipeline (F1) |
| :--- | :--- | :--- |
| **ROUGE-1** | 0.4014 | 0.4819 |
| **ROUGE-2** | 0.2224 | 0.2970 |
| **ROUGE-L** | 0.4014 | 0.4819 |

*Note: In extractive summarization where sentences are retained verbatim and in original order, ROUGE-1 and ROUGE-L naturally yield identical results. While IndicBERT shows higher overlap with the source text, qualitative human evaluation by Bodo speakers is required to definitively judge the coherence and informativeness of the summaries.*

---

## 💻 Implementation & Usage

Both pipelines are built for **Google Colab** and feature an interactive UI built with `ipywidgets`.

### Requirements
* `pandas`, `numpy`, `re`, `scikit-learn`
* `torch`, `transformers` (for IndicBERT)
* `rouge-score`
* `ipywidgets`

### Interactive UI
Both notebooks include a UI panel (Textarea + FloatSlider + Button) that allows you to:
1. Paste any Bodo article (even unseen ones).
2. Adjust the summarization ratio (default is 0.3 / 30%).
3. Generate the summary on-the-fly and view the compression ratio.

### Note on Hardware
* **TF-IDF:** Runs seamlessly on standard CPU runtimes.
* **IndicBERT:** It is highly recommended to use a **T4 GPU** runtime in Colab for the transformer model, especially if processing batches of articles.

---

## 🚀 Limitations & Future Work

* **Stop-word Expansion:** The manual 34-word Bodo stop-word list used in the TF-IDF model can be expanded using corpus frequency statistics.
* **Hybrid Scoring:** Future iterations could implement a hybrid scoring mechanism combining both methods (e.g., `0.7 * bert_score + 0.3 * tfidf_score`) to balance semantic understanding with strict lexical relevance.
* **Evaluation:** Developing a small, human-annotated gold-standard dataset of Bodo summaries to replace the source-text proxy evaluation.
* **Advanced NLP Tooling:** Developing a more linguistically informed normalizer for Bodo (handling schwa deletion, named-entity preservation) to replace the aggressive regex whitelist.