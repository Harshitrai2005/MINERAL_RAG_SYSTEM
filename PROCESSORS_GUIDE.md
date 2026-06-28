# PROCESSORS DOCUMENTATION
## Understanding Data Processors in Mineral Exploration RAG System

Two processors handle all supported file types:

1. PDF PROCESSOR — geological reports and text files
2. MINERAL DATASET PROCESSOR — geochemical CSV and JSON data

---

## 1. PDF PROCESSOR (Geological Report Extractor)
**File: `backend/ingestion/pdf_processor.py`**

### What It Does
Extracts text from geological survey reports and creates searchable chunks that preserve
page references and detect important sections.

### Input Types
- `.pdf` — geological reports, survey documents, company reports
- `.txt` — plain text geological reports and field notes

### Processing Strategy
1. **Page-by-page extraction** — preserves page numbers for citations
2. **OCR artifact cleaning** — removes scanner noise and formatting garbage
3. **Section detection** — identifies geological report sections automatically
4. **Smart chunking** — splits into semantic chunks respecting section boundaries

### Detected Report Sections
- Executive Summary
- Geological Setting / Regional Geology
- Mineralization / Mineral Zones
- Geochemistry / Geochemical Data
- Lithology / Rock Types
- Structural Geology / Tectonics / Faults
- Drilling Results / Boreholes
- Resource Estimates / Mineral Resources
- Recommendations / Exploration Targets

### Output Metadata Per Chunk
- `source` — filename
- `doc_type` — "geological_report"
- `page` — page number in original PDF
- `total_pages` — total pages in report
- `section` — detected section name
- `chunk_index` — position within page

### Key Features
- Handles multi-page PDFs efficiently
- Preserves geological terminology (coordinates, assays, elements)
- Skips empty/image-only pages automatically
- Deduplication via MD5 hash of file content

### Example
```
Input: whitehorse_report.pdf (35 pages)

Output:
- 47 chunks total
- Metadata preserves pages 3-7 as "Geological Setting"
- Coordinates and element values intact
- Each chunk ~500 words for the embedding model
```

---

## 2. MINERAL DATASET PROCESSOR (Geochemical Data Analyzer)
**File: `backend/ingestion/mineral_dataset_processor.py`**

### What It Does
Processes structured data (CSV/JSON) with geochemical assays, drill results, and soil surveys.
Creates multiple levels of analysis — from raw numbers to deposit model interpretations.

### Input Types
- `.csv` — assay tables, geochemical surveys, drill results
- `.json` — structured assay records, nested databases (supports GeoJSON, JSONL, 100MB+)

### Expected Data Columns

**Coordinate columns:**
X, Y, Z, Easting, Northing, Elevation, Depth

**Element columns (any of these):**
Au, Cu, Zn, Pb, Ag, Mo, As, Sb, Te, W, Bi, Fe, Mn

**Categorical columns (optional):**
sample_id, zone, lithology, alteration, rock_type, grade_category, priority

### What Each Output Chunk Represents

**Chunk 1 — Dataset Overview**
Sample count, column inventory, element statistics (min/mean/median/max).
Answers: "What data do we have? How many samples?"

**Chunk 2 — Geochemical Anomaly Analysis**
Anomalous samples identified per element, count above thresholds, peak concentrations.
Answers: "Where are the anomalies? How strong are they?"

**Chunk 3 — Deposit Model Interpretation**
Matches element suite against known deposit pathfinders, scores by deposit type, assigns confidence.
Answers: "What deposit type is this?"

**Chunk 4 — High-Grade Samples**
Top 20 samples ranked by primary element (Au or Cu) with complete assay values.
Answers: "Where are the best samples? What are the grades?"

**Chunk 5+ — Row-Level Chunks**
One chunk per sample record with all columns and values.
Answers: "What was sample X like? Specific sample lookup?"

**Chunk N — Zone Summaries**
Grouped statistics by zone with aggregated stats and top 3 samples per zone.
Answers: "How do zones compare? What's the best zone?"

### Deposit Type Pathfinder Suites

| Deposit Type           | Pathfinder Elements        | Key Signal                  |
|------------------------|----------------------------|-----------------------------|
| Epithermal Au          | Au, Ag, As, Sb, Te, Tl, Hg | Au + As + Sb + Te together  |
| Porphyry Cu            | Cu, Mo, Au, Re, W          | Cu + Mo + Au + W together   |
| VMS                    | Cu, Zn, Pb, Ag, Ba, S      | Zn + Pb + Cu co-occurrence  |
| SEDEX                  | Zn, Pb, Ag, Ba, Tl         | Zn + Pb + Ag without Cu     |
| IOCG                   | Cu, Au, Fe, U, Bi, Co      | Cu + Au + Fe + U combination|
| Pegmatite              | Li, Be, Cs, Ta, Nb, Sn     | Rare element suite          |

### Example Processing
```
Input: drilling_assays_porphyry_cu_au.csv
- 24 samples, 8 zones
- Elements: Au_ppm, Cu_ppm, Mo_ppm, As_ppm, Ag_ppm, Zn_ppm

Output Chunks:
1. Overview: 24 samples from 4 drill holes; Cu avg=1,456 ppm; Au avg=0.45 ppm
2. Anomalies: 18 Cu-anomalous (>100 ppm); 12 Au-anomalous (>0.05 ppm)
3. Deposit Model: 92% match to Porphyry Cu (Cu+Mo+Au+W present)
4. High Grade: Top 3: Cu=3,456ppm, Cu=3,245ppm, Cu=2,456ppm
5-9. Zone Summaries: Primary Zone avg Cu=2,145ppm; Phyllic Zone avg=1,235ppm
10+. Row chunks: one per sample
```

---

## PRACTICAL USAGE GUIDE

### When to Use Each Processor

**Use PDF PROCESSOR when:**
- You have geological reports or technical papers
- You need to extract specific sections (drilling results, resource estimates)
- You want to preserve page citations in answers
- Example query: "What was the resource estimate on page 12?"

**Use MINERAL DATASET PROCESSOR when:**
- You have assay/drill data, soil surveys, or geochemical databases
- You need to identify anomalies automatically
- You want deposit model interpretation from element suites
- Example query: "Which samples are high-grade? What deposit type is this?"

### Query Examples

**PDF queries:**
- "What was the resource estimate?"
- "What drilling results were reported?"
- "What is the geological setting?"
- "What are the exploration recommendations?"

**Dataset queries:**
- "What is the highest gold grade sample?"
- "How many copper anomalies do we have?"
- "Is this a porphyry or epithermal system?"
- "Compare Cu grades between zones"

---

## SAMPLE DATA PROVIDED

### CSV Samples (`sample_data/csv/`)
1. **drilling_assays_porphyry_cu_au.csv** — 24 drill samples from 4 holes, Porphyry Cu-Au system, oxide/transition/primary zoning
2. **geochemistry_expanded.csv** — expanded multi-element geochemical survey
3. **soil_geochem_survey.csv** — 14 soil/rock samples across multiple zones

### JSON Samples (`sample_data/`)
1. **mineral_zones_sample.json** — mineral zone records with spatial and grade data
2. **mineral_zones_expanded.json** — expanded version with nested zone attributes

### Report Samples (`sample_data/`)
1. **whitehorse_porphyry_exploration_report.txt** — simulated full geological report with all standard sections
2. **sample_geological_report.txt** — shorter geological report for quick testing

---

## TESTING RECOMMENDATIONS

### Quick Validation
1. Upload `drilling_assays_porphyry_cu_au.csv` as "dataset" → check anomaly detection
2. Upload `whitehorse_porphyry_exploration_report.txt` as "report" → check section detection
3. Upload `mineral_zones_sample.json` as "dataset" → check JSON flattening

### Query Testing
```
Query: "What is the deposit type?"
Expected: 92% match to Porphyry Cu (Cu+Mo+Au+W suite)

Query: "Show me high-grade samples"
Expected: Top samples by Au/Cu grade with zone breakdown

Query: "Compare zones by copper grade"
Expected: Zone summary table with avg Cu per zone

Query: "What are the exploration recommendations?"
Expected: Answer citing report sections with page references
```

---

## STRENGTHS & RELIABILITY

### PDF Processor
- Page-aware — all answers include page citations
- Section-aware — understands standard geological report structure
- OCR-robust — handles scanned PDFs
- Deduplication — tracks file changes via MD5

### Dataset Processor
- Multi-level analysis — from dataset overview down to individual sample lookup
- Deposit model matching — identifies system type from element suite
- Zone clustering — enables zone-by-zone comparison queries
- Flexible schema — handles various column naming conventions
- Large file support — streaming batch processing for 100MB+ JSON files

### General
- All processors fail gracefully — never crash on bad data
- Metadata preserved on every chunk — enables traceable citations
- Chunking strategy optimized for the embedding model window size

---

## GEOCHEMICAL THRESHOLD REFERENCE

```
Element  | Anomalous | High    | Unit
---------|-----------|---------|-----
Au       | 0.05      | 0.5     | ppm
Cu       | 100       | 1,000   | ppm
Zn       | 200       | 1,000   | ppm
Pb       | 50        | 500     | ppm
Ag       | 1         | 10      | ppm
Mo       | 10        | 100     | ppm
As       | 20        | 200     | ppm (pathfinder)
Sb       | 5         | 50      | ppm (pathfinder)
Te       | 0.5       | 5       | ppm (pathfinder)
W        | 5         | 50      | ppm
Bi       | 1         | 20      | ppm (pathfinder)
Fe       | 5         | 15      | %
Mn       | 500       | 2,000   | ppm
```

Note: Always calibrate thresholds against your regional background values.
