# Data

No data is committed to this repository. Put the files below in this `data/` folder before running the notebooks. MD5 checksums of the exact files I used are listed so you can confirm you have the same versions.

## Project Gutenberg texts (Part A)

Shakespeare (*Julius Caesar*, *Hamlet*, *Macbeth*) and Jane Austen (*Emma*, *Persuasion*, *Sense and Sensibility*) from NLTK's Gutenberg sample. The notebooks download them automatically with `nltk.download("gutenberg")`. Public domain.

## Fake or real news (Part A)

- **File:** `fake_or_real_news.csv` (6,335 articles; columns: unnamed index, `title`, `text`, `label` = FAKE/REAL)
- **Source:** compiled by George McIntire. Fake articles come from a 2016 Kaggle fake-news dataset, and real articles from mainstream outlets collected via AllSides. The original GitHub location no longer appears to be online, but the file is widely mirrored (search Kaggle or GitHub for `fake_or_real_news.csv`).
- **Licence:** none stated by the original author. Treat it as research/educational use only, and don't redistribute it.
- **MD5:** `073d0c904a036702556ee551bff9b0b4`

## MTSamples medical transcriptions (Part B)

- **File:** `mtsamples.csv` (4,999 rows; columns include `medical_specialty` and `transcription`)
- **Source:** Kaggle, "Medical Transcriptions" by Tara Boyle: <https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions>. The copy I used came from a CC0 mirror of the same file: <https://raw.githubusercontent.com/aneuraz/mtsamplesFR/master/data/mtsamples.csv>
- **Licence:** CC0 1.0 (public domain) on Kaggle. The text was scraped from <https://mtsamples.com>, which publishes de-identified sample reports under its own terms of use, so check them before any commercial use.
- **MD5:** `d8547837775283c95b8a70119da806f8`

## Medical Abstracts text classification corpus (Part B)

- **Files:** `medical_tc_train.csv` (11,550 abstracts), `medical_tc_test.csv` (2,888), `medical_tc_labels.csv` (5 condition names)
- **Source:** <https://github.com/sebischair/Medical-Abstracts-TC-Corpus>, also on Hugging Face as `TimSchopf/medical_abstracts`
- **Licence:** CC BY-SA 3.0
- **Citation:** Schopf T, Braun D, Matthes F. *Evaluating Unsupervised Text Classification: Zero-Shot and Similarity-Based Approaches.* NLPIR 2022, pp. 6–15. <https://doi.org/10.1145/3582768.3582795>
- **MD5:** train `792daf8b6ca8557f88ba68f7cec22b03`, test `98978fd21c156efe960d901451161817`, labels `77e5f74d409bcc8b08495c2b743fc7fc`

```bash
cd data
curl -LO https://raw.githubusercontent.com/sebischair/Medical-Abstracts-TC-Corpus/main/medical_tc_train.csv
curl -LO https://raw.githubusercontent.com/sebischair/Medical-Abstracts-TC-Corpus/main/medical_tc_test.csv
curl -LO https://raw.githubusercontent.com/sebischair/Medical-Abstracts-TC-Corpus/main/medical_tc_labels.csv
```
