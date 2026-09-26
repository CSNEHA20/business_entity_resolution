# Milestone 6: Test-Distribution Compatibility & Open-Set Country Audit

## 1. Distribution & Schema Inspection

| Observable Feature | Training Distribution | Test Distribution (Sampled) | Compatibility Status |
| :--- | :--- | :--- | :--- |
| **TSV Header Schema** | `['entity_id', 'business_name', 'business_address', 'country']` | `['entity_id', 'business_name', 'business_address', 'country']` | **EXACT MATCH** |
| **S1 Entities** | 100,000 | ~1,730,000 | Handled via Chunking |
| **S2 Entities** | 5,034,616 | ~4,890,000 | Indexed in Memory |
| **S3 Entities** | 5,069,963 | ~5,080,000 | Indexed in Memory |
| **S1 Country Distribution** | {'US': 1323633, 'INDIA': 883188} | {'US': 19215, 'FRANCE': 7485, 'INDIA': 23300} | **Open-Set Country Support Verified** |
| **S2 Country Distribution** | - | {'INDIA': 23534, 'FRANCE': 7228, 'US': 19238} | **Open-Set Country Support Verified** |
| **S3 Country Distribution** | - | {'INDIA': 23690, 'FRANCE': 7061, 'US': 19249} | **Open-Set Country Support Verified** |

## 2. France / Multilingual Support Verification

1. **Country Normalization:** `normalize_country()` maps `"FR"`, `"FRANCE"`, `"FRA"`, `"RÉPUBLIQUE FRANÇAISE"` cleanly to standard representations without hardcoding US or India.
2. **Accented Unicode Normalization:** `normalize_business_name_suffixes()` and RapidFuzz pairwise metrics operate on full UTF-8 Unicode strings with token sort and set matching, correctly handling accents (e.g. `['Maison de Santé Generation', 'École primaire Sainte Pierre', 'Établissements Demployeurs SARL']`).
3. **French Business Suffixes:** Suffixes such as `SARL`, `SAS`, `SA`, `EURL`, `SCI`, `SNC` are standardly recognized in token normalization.
4. **Postal Codes:** French 5-digit postal codes (e.g. `75001`, `69002`, `13001`) match `RE_PIN = re.compile(r"\b\d{5,6}\b")` without modification.
5. **No Hard-Coded Exclusions:** Zero country-specific filtering exists in blocking, features, or decision logic.
