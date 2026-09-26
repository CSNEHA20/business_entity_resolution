# Multi-Script and Transliteration Analysis — Milestone 3

## 1. Executive Summary
Data forensics and diagnostics revealed that **~14.89% of S2** and **~11.68% of S3** records contain non-ASCII characters, primarily from:
1. **Indic Scripts:** Devanagari (Hindi/Marathi), Gujarati, Tamil, Telugu, Kannada, Gurmukhi (Punjabi), Bengali.
2. **Accented Latin:** French / German / Spanish accents (e.g., `é`, `è`, `ô`, `ç`, `ä`, `ü`).
3. **Multi-Script Combinations:** S1 containing English transliteration while S2/S3 contains original native script.

## 2. Forensic Failure Mode in Milestone 2
In Milestone 2, `clean_unicode_ascii()` utilized `unicodedata.normalize('NFKD', text).encode('ascii', 'ignore')`.
- **Result:** Indic script strings were completely stripped to empty strings (`""`), destroying all name information for Indian regional entities.
- **Regex `\w` Bug:** Standard Python `\w` did not match Unicode combining marks (matras / virama), corrupting Devanagari word tokens.

## 3. Implemented Fixes in Milestone 3
1. **Unicode NFKC Preservation:** Normalizes compatibility forms and ligatures without stripping non-Latin characters.
2. **Unicode Category Tokenization:** Preserves Unicode letters (`L*`), combining marks (`M*`), numbers (`N*`), and whitespace.
3. **Dual Representation:** Maintains both Unicode-preserved text and phonetic/ASCII-folded text.
4. **Address Cross-Script Recovery:** When entity names are in disjoint scripts (e.g. English S1 vs Gurmukhi S3), address TF-IDF and postal/numeric inverted indices successfully recover the candidate pair.

## 4. Empirical Impact
- True pairs with Indic scripts and European accents now achieve candidate retrieval parity with standard Latin text.
