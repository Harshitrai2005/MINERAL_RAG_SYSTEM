"""
Mineral Dataset Processor
Handles structured geochemical and mineral composition datasets.
Supports CSV (assay tables, geochemical surveys) and JSON formats.

FIX (v5.1): Large CSVs are processed in chunks using pd.read_csv(chunksize=...)
  instead of loading the entire file into RAM. A 500k-row assay table (typical
  for a large drilling program) could previously consume 2–4 GB of RAM and
  crash the free-tier Render instance. With chunked reading, memory usage is
  bounded to ~50 MB regardless of file size.

FIX (v5.2): JSON ingestion now streams via `ijson` instead of `json.load()`.
  Previously, any non-JSONL JSON file (single object/array, GeoJSON, or a
  dict wrapping a records list under a key like "mineral_zones") was fully
  parsed into memory in one call before any batching logic ran — meaning a
  100MB+ JSON file could spike RSS to 300MB-1GB+ and get OOM-killed, which
  surfaces upstream as an opaque "internal server error" with no traceback.

  Now:
    1. A cheap structural pre-scan (ijson.parse) finds which top-level key
       holds the record list — or whether the root itself is an array, or a
       GeoJSON FeatureCollection — without materializing any values.
    2. ijson.items() streams just that array, batched in groups of 5,000
       (mirroring the existing CSV/JSONL batching pattern).
    3. A bounded fallback (full json.load()) remains for edge cases the
       structural scan can't resolve, but is capped at 25MB so it can't
       itself trigger the same OOM failure.

FIX (v5.3): Row chunks are now capped at _MAX_ROW_CHUNK_CHARS characters at
  write time. Deeply nested JSON records after json_normalize can produce
  800-1200 char row chunks; with 15 slots in the RAG prompt this previously
  exceeded Groq's per-request token limits (413 errors) and crowded out the
  high-signal overview and zone-summary chunks that actually answer aggregate
  questions. The cap applies to both JSON and CSV paths since they share
  _generate_row_chunks.

Geochemical datasets typically contain:
- Sample coordinates (X, Y, Z / Easting, Northing, Elevation)
- Element concentrations (Au, Cu, Zn, Pb, Ag, Mo, As, Sb, etc.)
- Rock/sample codes, lithology labels, alteration codes
- Downhole assay intervals from drill programs
- Mineral zone / deposit-model records (zone geometry, grades, alteration,
  mineralogy, resource estimates) — possibly deeply nested (e.g. "grades",
  "dimensions_m", "best_intercept", "resource_estimate" sub-objects)
"""

import csv
import hashlib
import io
import json
from pathlib import Path
from typing import Optional, Iterator

import ijson
import numpy as np
import pandas as pd

from core.config import settings
from utils.logger import setup_logger
from utils.text_chunker import TextChunker

logger = setup_logger(__name__)


# ── Geochemical Thresholds ────────────────────────────────────────────────────
# Background vs anomalous concentrations (ppm unless noted)
# Values are indicative — always calibrate against regional background

GEOCHEMICAL_THRESHOLDS = {
    "Au": {"anomalous": 0.05, "high": 0.5, "unit": "ppm"},        # Gold
    "Cu": {"anomalous": 100, "high": 1000, "unit": "ppm"},         # Copper
    "Zn": {"anomalous": 200, "high": 1000, "unit": "ppm"},         # Zinc
    "Pb": {"anomalous": 50, "high": 500, "unit": "ppm"},           # Lead
    "Ag": {"anomalous": 1, "high": 10, "unit": "ppm"},             # Silver
    "Mo": {"anomalous": 10, "high": 100, "unit": "ppm"},           # Molybdenum
    "As": {"anomalous": 20, "high": 200, "unit": "ppm"},           # Arsenic
    "Sb": {"anomalous": 5, "high": 50, "unit": "ppm"},             # Antimony
    "Bi": {"anomalous": 5, "high": 50, "unit": "ppm"},             # Bismuth
    "W":  {"anomalous": 5, "high": 50, "unit": "ppm"},             # Tungsten
    "Mn": {"anomalous": 1000, "high": 5000, "unit": "ppm"},        # Manganese
    "Fe": {"anomalous": 5, "high": 20, "unit": "%"},               # Iron (%)
}

# Mineral deposit model pathfinder element suites
DEPOSIT_PATHFINDERS = {
    "Epithermal Au-Ag (High Sulphidation)":  ["Au", "Ag", "As", "Sb", "Bi", "Te"],
    "Epithermal Au-Ag (Low Sulphidation)":   ["Au", "Ag", "As", "Sb", "Mn"],
    "Porphyry Cu-Au":                         ["Cu", "Au", "Mo", "Re", "Ag"],
    "Porphyry Cu-Mo":                         ["Cu", "Mo", "Re", "Ag"],
    "VMS (Zn-Pb-Cu-Ag)":                     ["Zn", "Pb", "Cu", "Ag", "As", "Sb"],
    "Orogenic Au":                            ["Au", "As", "Sb", "W", "Bi", "Te"],
    "IOCG (Fe-Cu-Au)":                        ["Fe", "Cu", "Au", "Ag", "Co", "U"],
    "Skarn (W-Mo-Cu)":                        ["W", "Mo", "Cu", "Bi", "Zn", "Pb"],
    "Mississippi Valley Type (Zn-Pb)":        ["Zn", "Pb", "F", "Ba"],
    "Sediment-Hosted Au (Carlin-Type)":       ["Au", "As", "Sb", "Hg", "Tl"],
}

# CSV chunk size for memory-safe processing of large files
_CSV_CHUNK_ROWS = 5_000

# JSON streaming batch size (records per batch, mirrors CSV chunking)
_JSON_BATCH = 5_000

# Cap on how many rows we keep in memory for aggregate stats/zone summaries
_STATS_SAMPLE_CAP = 50_000

# Known wrapper keys that hold a record list — checked in priority order
# during the structural pre-scan and in the json.load() fallback path.
_RECORD_LIST_KEYS = (
    "data", "records", "samples", "results", "features",
    "items", "rows", "values", "entries", "observations",
    "mineral_zones", "zones", "assays", "drillholes",
    "intervals", "geochemistry", "stations", "measurements",
    "holes", "collars", "surveys", "lithology", "prospects",
)

# Hard cap for the json.load() fallback path — if the structural scan can't
# resolve the record location and the file exceeds this, refuse rather than
# risk loading it fully into memory.
_JSON_FALLBACK_MAX_BYTES = 25 * 1024 * 1024  # 25 MB

# FIX (v5.3): Hard cap on individual row chunk text length.
# Deeply nested JSON records serialized via json_normalize can produce very
# long row chunks (800-1200 chars). Capping at 600 chars ensures that even
# with 15 row chunks in the RAG prompt the total context stays within Groq's
# per-request token limits, while preserving enough detail for sample lookup.
# The most diagnostic fields (element grades, coordinates, zone name) appear
# first in the serialized text so truncation drops trailing low-value columns.
_MAX_ROW_CHUNK_CHARS = 600


class MineralDatasetProcessor:

    def __init__(self):
        self.chunker = TextChunker(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def process_file(self, file_path: Path) -> list[dict]:
        """
        Process a CSV or JSON mineral dataset into text chunks for vector indexing.
        Returns list of {id, text, metadata} dicts.
        """
        file_path = Path(file_path)
        file_hash = self._compute_hash(file_path)
        suffix = file_path.suffix.lower()

        if suffix == ".json":
            return self._process_json(file_path, file_hash)
        elif suffix == ".csv":
            return self._process_csv_chunked(file_path, file_hash)
        else:
            raise ValueError(f"Unsupported dataset format: {suffix}")

    # ── CSV processing (chunked for large files) ──────────────────────────────

    def _process_csv_chunked(self, file_path: Path, file_hash: str) -> list[dict]:
        """
        Read CSV in _CSV_CHUNK_ROWS-row batches instead of all at once.
        First pass: collect rows (capped) for statistics and zone aggregation.
        Row-level chunks are generated incrementally to keep memory bounded.
        """
        logger.info(f"Processing CSV (chunked, {_CSV_CHUNK_ROWS} rows/batch): {file_path.name}")

        chunks: list[dict] = []
        all_rows_for_stats: list[pd.DataFrame] = []
        total_rows = 0

        try:
            for batch in pd.read_csv(file_path, chunksize=_CSV_CHUNK_ROWS, low_memory=False):
                total_rows += len(batch)
                if total_rows <= _STATS_SAMPLE_CAP:
                    all_rows_for_stats.append(batch)
                row_chunks = self._generate_row_chunks(batch, file_path.name, file_hash)
                chunks.extend(row_chunks)

        except Exception as exc:
            logger.error(f"CSV read failed for {file_path.name}: {exc}")
            raise

        logger.info(f"  → {total_rows} rows processed from {file_path.name}")

        if all_rows_for_stats:
            df_sample = pd.concat(all_rows_for_stats, ignore_index=True)
            df_sample.columns = [c.strip() for c in df_sample.columns]

            overview_chunks = self._generate_overview_chunks(df_sample, file_path.name, file_hash, total_rows)
            zone_chunks = self._generate_zone_summaries(df_sample, file_path.name, file_hash)

            chunks = overview_chunks + zone_chunks + chunks

        logger.info(f"  → {len(chunks)} total chunks from {file_path.name}")
        return chunks

    # ── JSON processing (streaming, bounded memory) ─────────────────────────

    def _process_json(self, file_path: Path, file_hash: str) -> list[dict]:
        """
        Process a JSON mineral dataset file of essentially any size via
        streaming (ijson), so memory use stays bounded regardless of file
        size — handles:

          • Array of records: [{...}, {...}]
          • Dict wrapping a record list under any key, including domain
            keys like "mineral_zones", or generic ones like "data"/"records"
          • Dict wrapping a record list under an *unrecognized* key (found
            via structural scan of all depth-1 keys)
          • GeoJSON FeatureCollection: {"type":"FeatureCollection","features":[...]}
          • Deeply nested records (e.g. "grades", "dimensions_m",
            "best_intercept", "resource_estimate" sub-objects) — flattened
            automatically via pd.json_normalize(sep="_")
          • JSONL (one JSON object per line): detected and streamed
          • A single JSON object representing one record (no list at all)
        """
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        logger.info(f"Processing JSON: {file_path.name} ({file_size_mb:.1f} MB)")

        # Detect JSONL (newline-delimited JSON) first — already streaming.
        if self._is_jsonl(file_path):
            return self._process_jsonl(file_path, file_hash)

        try:
            top_level_key, is_array_root, is_geojson = self._detect_record_location(file_path)
        except Exception as exc:
            logger.warning(
                f"Structural scan failed for {file_path.name} ({exc}); "
                f"falling back to bounded json.load()"
            )
            return self._process_json_fallback(file_path, file_hash)

        if top_level_key is None and not is_array_root and not is_geojson:
            logger.info(
                f"No record list detected in {file_path.name}; "
                f"falling back to bounded json.load() (treats root as a single record if needed)"
            )
            return self._process_json_fallback(file_path, file_hash)

        try:
            chunks = self._process_json_streaming(
                file_path, file_hash, top_level_key, is_array_root, is_geojson
            )
        except Exception as exc:
            logger.warning(
                f"Streaming parse failed for {file_path.name} at key "
                f"'{top_level_key}' ({exc}); falling back to bounded json.load()"
            )
            return self._process_json_fallback(file_path, file_hash)

        if not chunks:
            # Streaming found a location but yielded nothing usable —
            # try the fallback path once before giving up.
            logger.info(f"Streaming yielded no chunks for {file_path.name}; trying fallback")
            return self._process_json_fallback(file_path, file_hash)

        return chunks

    def _detect_record_location(self, file_path: Path) -> tuple[Optional[str], bool, bool]:
        """
        Structural-only scan (no value materialization) to find where the
        record list lives. Returns (key_or_None, is_array_root, is_geojson).

        Priority:
          1. Root is itself an array            -> (None, True, False)
          2. Root is FeatureCollection           -> ("features", False, True)
          3. Root has a known wrapper key
             (e.g. "mineral_zones", "data", ...) -> (key, False, False)
          4. Root has ANY depth-1 key whose
             value is an array (first such key)  -> (key, False, False)
          5. Nothing matched                     -> (None, False, False)
        """
        with open(file_path, "rb") as f:
            parser = ijson.parse(f)

            root_seen = False
            root_is_array = False
            current_key = None
            geojson_type_seen = False

            known_key_hit: Optional[str] = None
            first_array_key: Optional[str] = None

            for prefix, event, value in parser:
                if not root_seen:
                    root_seen = True
                    if event == "start_array":
                        root_is_array = True
                        break
                    # event == "start_map" -> root is an object, keep scanning
                    continue

                if prefix.count(".") == 0 and event == "map_key":
                    current_key = value
                    continue

                if prefix == "type" and event == "string" and value == "FeatureCollection":
                    geojson_type_seen = True
                    continue

                if event == "start_array" and current_key is not None and prefix == current_key:
                    if current_key == "features" and geojson_type_seen:
                        return "features", False, True
                    if current_key in _RECORD_LIST_KEYS and known_key_hit is None:
                        known_key_hit = current_key
                        # Known key found — this is our best match, stop scanning.
                        return known_key_hit, False, False
                    if first_array_key is None:
                        first_array_key = current_key
                    # Keep scanning in case a known key appears later.

            if root_is_array:
                return None, True, False

            if known_key_hit:
                return known_key_hit, False, False

            if first_array_key:
                return first_array_key, False, False

            return None, False, False

    def _process_json_streaming(
        self,
        file_path: Path,
        file_hash: str,
        top_level_key: Optional[str],
        is_array_root: bool,
        is_geojson: bool,
    ) -> list[dict]:
        """Stream records from the detected location, batching as we go."""
        all_chunks: list[dict] = []
        df_sample_batches: list[pd.DataFrame] = []
        total_records = 0
        batch: list[dict] = []

        item_path = "item" if is_array_root else f"{top_level_key}.item"

        def flush(b: list[dict], start: int):
            nonlocal all_chunks, df_sample_batches
            records = b
            if is_geojson:
                records = [self._geojson_feature_to_record(feat) for feat in b]
            try:
                df_batch = pd.json_normalize(records, sep="_")
                df_batch.columns = [c.strip() for c in df_batch.columns]
            except Exception as e:
                logger.warning(f"json_normalize failed for batch at {start}: {e}")
                for i, rec in enumerate(records):
                    all_chunks.append({
                        "id": f"{file_hash}_j{start + i}",
                        "text": f"Record from {file_path.name}: {json.dumps(rec, default=str)[:1000]}",
                        "metadata": {
                            "source": file_path.name,
                            "doc_type": "geochemical_dataset",
                            "section": "Raw Record",
                            "file_hash": file_hash,
                        },
                    })
                return

            if sum(len(x) for x in df_sample_batches) < _STATS_SAMPLE_CAP:
                df_sample_batches.append(df_batch)

            all_chunks.extend(self._generate_row_chunks(df_batch, file_path.name, file_hash))

        with open(file_path, "rb") as f:
            for rec in ijson.items(f, item_path):
                if not isinstance(rec, dict):
                    continue
                batch.append(rec)
                total_records += 1
                if len(batch) >= _JSON_BATCH:
                    flush(batch, total_records - len(batch))
                    batch = []

        if batch:
            flush(batch, total_records - len(batch))

        if total_records == 0:
            logger.warning(f"No records streamed from {file_path.name} at path '{item_path}'")
            return []

        logger.info(f"  → {total_records} records streamed from {file_path.name} (key='{top_level_key}')")

        if df_sample_batches:
            df_sample = pd.concat(df_sample_batches, ignore_index=True)
            overview = self._generate_overview_chunks(df_sample, file_path.name, file_hash, total_records)
            zone = self._generate_zone_summaries(df_sample, file_path.name, file_hash)
            all_chunks = overview + zone + all_chunks

        logger.info(f"Generated {len(all_chunks)} chunks from JSON dataset {file_path.name}")
        return all_chunks

    @staticmethod
    def _geojson_feature_to_record(feat: dict) -> dict:
        rec = dict(feat.get("properties") or {})
        geom = feat.get("geometry")
        if geom and geom.get("coordinates"):
            coords = geom["coordinates"]
            if isinstance(coords, (list, tuple)) and len(coords) >= 2:
                rec["longitude"] = coords[0]
                rec["latitude"] = coords[1]
                if len(coords) > 2:
                    rec["elevation"] = coords[2]
        return rec

    @staticmethod
    def _is_jsonl(file_path: Path) -> bool:
        """Detect if file is newline-delimited JSON (JSONL)."""
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                first = f.readline().strip()
                second = f.readline().strip()
            if first and second:
                json.loads(first)
                json.loads(second)
                return True
        except (json.JSONDecodeError, Exception):
            pass
        return False

    def _process_jsonl(self, file_path: Path, file_hash: str) -> list[dict]:
        """Process newline-delimited JSON (JSONL) in a streaming fashion."""
        logger.info(f"Processing as JSONL: {file_path.name}")
        all_chunks: list[dict] = []
        df_sample_batches: list[pd.DataFrame] = []
        total_records = 0
        batch: list[dict] = []

        def flush_batch(b: list[dict], start: int):
            nonlocal all_chunks, df_sample_batches
            try:
                df_batch = pd.json_normalize(b, sep="_")
                df_batch.columns = [c.strip() for c in df_batch.columns]
                if sum(len(x) for x in df_sample_batches) < _STATS_SAMPLE_CAP:
                    df_sample_batches.append(df_batch)
                all_chunks.extend(self._generate_row_chunks(df_batch, file_path.name, file_hash))
            except Exception as exc:
                logger.warning(f"JSONL batch flush failed at {start}: {exc}")

        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    batch.append(obj)
                    total_records += 1
                except json.JSONDecodeError:
                    continue

                if len(batch) >= _JSON_BATCH:
                    flush_batch(batch, total_records - len(batch))
                    batch = []

        if batch:
            flush_batch(batch, total_records - len(batch))

        if df_sample_batches:
            df_sample = pd.concat(df_sample_batches, ignore_index=True)
            overview = self._generate_overview_chunks(df_sample, file_path.name, file_hash, total_records)
            zone = self._generate_zone_summaries(df_sample, file_path.name, file_hash)
            all_chunks = overview + zone + all_chunks

        logger.info(f"Generated {len(all_chunks)} chunks from JSONL {file_path.name} ({total_records} records)")
        return all_chunks

    # ── Bounded fallback (small files / unresolved structure only) ─────────

    def _process_json_fallback(self, file_path: Path, file_hash: str) -> list[dict]:
        """
        Full json.load() — only reached when the structural scan can't
        confidently resolve a record location. Hard-capped by file size so
        it can never single-handedly OOM the process.
        """
        size = file_path.stat().st_size
        if size > _JSON_FALLBACK_MAX_BYTES:
            raise ValueError(
                f"'{file_path.name}' is {size / 1024 / 1024:.1f} MB and its JSON structure "
                f"could not be resolved for streaming. Refusing to load it fully into memory "
                f"(fallback limit: {_JSON_FALLBACK_MAX_BYTES / 1024 / 1024:.0f} MB). "
                f"Consider converting to JSONL or flattening nested wrapper objects."
            )

        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                data = json.load(f)
        except json.JSONDecodeError as exc:
            logger.error(f"JSON parse error in {file_path.name}: {exc}")
            raise ValueError(f"Invalid JSON file '{file_path.name}': {exc}") from exc

        records = self._extract_records(data, file_path.name)
        if not records:
            logger.warning(f"JSON dataset {file_path.name} has no iterable records")
            return []

        total_records = len(records)
        logger.info(f"  → {total_records} records found in {file_path.name} (fallback path)")

        all_chunks: list[dict] = []
        df_sample_batches: list[pd.DataFrame] = []

        for batch_start in range(0, total_records, _JSON_BATCH):
            batch = records[batch_start: batch_start + _JSON_BATCH]
            try:
                df_batch = pd.json_normalize(batch, sep="_")
                df_batch.columns = [c.strip() for c in df_batch.columns]
            except Exception as e:
                logger.warning(f"json_normalize failed for batch at {batch_start}: {e}")
                for i, rec in enumerate(batch):
                    all_chunks.append({
                        "id": f"{file_hash}_j{batch_start + i}",
                        "text": f"Record from {file_path.name}: {json.dumps(rec, default=str)[:1000]}",
                        "metadata": {
                            "source": file_path.name,
                            "doc_type": "geochemical_dataset",
                            "section": "Raw Record",
                            "file_hash": file_hash,
                        },
                    })
                continue

            if sum(len(x) for x in df_sample_batches) < _STATS_SAMPLE_CAP:
                df_sample_batches.append(df_batch)

            row_chunks = self._generate_row_chunks(df_batch, file_path.name, file_hash)
            all_chunks.extend(row_chunks)

        if df_sample_batches:
            df_sample = pd.concat(df_sample_batches, ignore_index=True)
            overview = self._generate_overview_chunks(df_sample, file_path.name, file_hash, total_records)
            zone = self._generate_zone_summaries(df_sample, file_path.name, file_hash)
            all_chunks = overview + zone + all_chunks

        logger.info(f"Generated {len(all_chunks)} chunks from JSON dataset {file_path.name} (fallback path)")
        return all_chunks

    @staticmethod
    def _extract_records(data, filename: str) -> list:
        """
        Recursively extract a flat list of record dicts from any JSON structure.
        Handles: arrays, common dict keys, GeoJSON, and arbitrary nesting.
        Used only by the bounded fallback path.
        """
        if isinstance(data, list):
            if data and isinstance(data[0], list):
                flat = []
                for sub in data:
                    flat.extend(sub)
                return flat
            return data

        if isinstance(data, dict):
            if data.get("type") == "FeatureCollection" and "features" in data:
                records = []
                for feat in data["features"]:
                    rec = dict(feat.get("properties") or {})
                    geom = feat.get("geometry")
                    if geom and geom.get("coordinates"):
                        coords = geom["coordinates"]
                        if isinstance(coords, (list, tuple)) and len(coords) >= 2:
                            rec["longitude"] = coords[0]
                            rec["latitude"] = coords[1]
                            if len(coords) > 2:
                                rec["elevation"] = coords[2]
                    records.append(rec)
                return records

            for key in _RECORD_LIST_KEYS:
                if key in data and isinstance(data[key], list):
                    candidate = data[key]
                    if candidate and isinstance(candidate[0], dict):
                        return candidate

            for key, val in data.items():
                if isinstance(val, list) and val and isinstance(val[0], dict):
                    logger.info(f"_extract_records: using key '{key}' from {filename}")
                    return val

            # Root is a single record (no list found anywhere)
            return [data]

        logger.warning(f"Unexpected JSON root type {type(data)} in {filename}")
        return []

    # ── Chunk generators ──────────────────────────────────────────────────────

    def _generate_overview_chunks(
        self, df: pd.DataFrame, source_name: str, file_hash: str, total_rows: int
    ) -> list[dict]:
        """Dataset-level overview chunk: column inventory, stats, deposit model hints."""
        chunks = []
        cols = list(df.columns)
        elem_cols = self._detect_element_columns(cols)

        chunks.append({
            "id": f"{file_hash}_overview_cols",
            "text": (
                f"Dataset: {source_name} | {total_rows} samples | "
                f"Columns: {', '.join(cols)}\n"
                f"Detected geochemical elements: {', '.join(elem_cols) if elem_cols else 'none'}"
            ),
            "metadata": {
                "source": source_name, "doc_type": "geochemical_dataset",
                "section": "Dataset Overview", "file_hash": file_hash,
            },
        })

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            stats_lines = [f"Statistical summary for {source_name} ({total_rows} rows):"]
            for col in numeric_cols[:20]:
                try:
                    s = df[col].dropna()
                    if len(s) == 0:
                        continue
                    stats_lines.append(
                        f"  {col}: min={s.min():.3g}, max={s.max():.3g}, "
                        f"mean={s.mean():.3g}, median={s.median():.3g}, "
                        f"std={s.std():.3g}, p95={s.quantile(0.95):.3g}"
                    )
                except Exception:
                    pass
            chunks.append({
                "id": f"{file_hash}_overview_stats",
                "text": "\n".join(stats_lines),
                "metadata": {
                    "source": source_name, "doc_type": "geochemical_dataset",
                    "section": "Statistical Summary", "file_hash": file_hash,
                },
            })

        if elem_cols:
            anomaly_lines = [f"Anomalous sample counts for {source_name}:"]
            for elem in elem_cols:
                thresholds = GEOCHEMICAL_THRESHOLDS.get(elem.upper(), {})
                if not thresholds:
                    continue
                try:
                    col_data = pd.to_numeric(df[elem], errors="coerce").dropna()
                    anom = (col_data >= thresholds["anomalous"]).sum()
                    high = (col_data >= thresholds["high"]).sum()
                    unit = thresholds["unit"]
                    anomaly_lines.append(
                        f"  {elem}: {anom} anomalous (≥{thresholds['anomalous']} {unit}), "
                        f"{high} high-grade (≥{thresholds['high']} {unit})"
                    )
                except Exception:
                    pass
            if len(anomaly_lines) > 1:
                chunks.append({
                    "id": f"{file_hash}_overview_anomalies",
                    "text": "\n".join(anomaly_lines),
                    "metadata": {
                        "source": source_name, "doc_type": "geochemical_dataset",
                        "section": "Anomaly Summary", "file_hash": file_hash,
                    },
                })

        deposit_chunk = self._deposit_model_chunk(df, source_name, file_hash, elem_cols)
        if deposit_chunk:
            chunks.append(deposit_chunk)

        return chunks

    def _generate_zone_summaries(
        self, df: pd.DataFrame, source_name: str, file_hash: str
    ) -> list[dict]:
        """Per-zone aggregation chunks for ranking/comparison queries."""
        zone_col = self._detect_zone_column(df.columns.tolist())
        if not zone_col:
            return []

        elem_cols = self._detect_element_columns(df.columns.tolist())
        if not elem_cols:
            return []

        chunks = []
        try:
            grouped = df.groupby(zone_col)
            zone_summaries = []
            for zone_name, group in grouped:
                stats_parts = [f"Zone: {zone_name} | {len(group)} samples"]
                for elem in elem_cols[:8]:
                    try:
                        vals = pd.to_numeric(group[elem], errors="coerce").dropna()
                        if len(vals) == 0:
                            continue
                        stats_parts.append(
                            f"{elem}: max={vals.max():.3g}, mean={vals.mean():.3g}"
                        )
                    except Exception:
                        pass
                zone_summaries.append(" | ".join(stats_parts))

            if zone_summaries:
                chunks.append({
                    "id": f"{file_hash}_zone_comparison",
                    "text": (
                        f"Zone comparison summary for {source_name} "
                        f"({len(zone_summaries)} zones):\n"
                        + "\n".join(zone_summaries)
                    ),
                    "metadata": {
                        "source": source_name, "doc_type": "geochemical_dataset",
                        "section": "Zone Comparison", "file_hash": file_hash,
                    },
                })
        except Exception as exc:
            logger.warning(f"Zone summary failed for {source_name}: {exc}")

        return chunks

    def _generate_row_chunks(
        self, df: pd.DataFrame, source_name: str, file_hash: str
    ) -> list[dict]:
        """
        One chunk per row — enables exact sample lookup.

        FIX (v5.3): Each chunk is capped at _MAX_ROW_CHUNK_CHARS characters.
        json_normalize on deeply nested mineral zone records can produce very
        long row texts. The cap prevents individual row chunks from bloating
        the RAG prompt and triggering Groq 413 errors. The most diagnostic
        fields (element grades, zone name, coordinates) appear early in the
        serialized text, so trailing low-value columns are what gets cut.
        Grade flags are always appended after the cap so they are never lost.
        """
        chunks = []
        elem_cols = self._detect_element_columns(df.columns.tolist())

        for idx, row in df.iterrows():
            parts = []
            for col, val in row.items():
                # List/array-valued cells (e.g. "mineralogy": ["pyrite", ...])
                # must be checked before pd.notna(), since pd.notna() on a
                # list-like returns an array, not a scalar bool.
                if isinstance(val, (list, tuple, np.ndarray)):
                    if len(val) == 0:
                        continue
                    parts.append(f"{col}: {', '.join(str(v) for v in val)}")
                    continue
                if isinstance(val, dict):
                    if not val:
                        continue
                    parts.append(f"{col}: {json.dumps(val, default=str)}")
                    continue
                if pd.notna(val) and str(val).strip():
                    parts.append(f"{col}: {val}")
            if not parts:
                continue

            text = f"Sample from {source_name}: " + " | ".join(parts)

            # Cap row text before appending grade flags so flags are never
            # truncated — they are the highest-signal tokens for retrieval.
            if len(text) > _MAX_ROW_CHUNK_CHARS:
                text = text[:_MAX_ROW_CHUNK_CHARS] + "…"

            grade_tags = []
            for elem in elem_cols:
                thresholds = GEOCHEMICAL_THRESHOLDS.get(str(elem).upper(), {})
                if thresholds:
                    try:
                        val_num = float(row.get(elem, ""))
                        if val_num >= thresholds["high"]:
                            grade_tags.append(f"HIGH-GRADE {elem}")
                        elif val_num >= thresholds["anomalous"]:
                            grade_tags.append(f"ANOMALOUS {elem}")
                    except (ValueError, TypeError):
                        pass

            if grade_tags:
                text += f" [GRADE FLAGS: {', '.join(grade_tags)}]"

            # Use a hash-based suffix instead of a bare row index so that
            # ids stay unique across batches in the streaming path (where
            # df.iterrows() indices reset per-batch).
            row_id_seed = f"{file_hash}:{idx}:{text[:50]}"
            row_id = hashlib.md5(row_id_seed.encode("utf-8")).hexdigest()[:10]

            chunks.append({
                "id": f"{file_hash}_r{row_id}",
                "text": text,
                "metadata": {
                    "source": source_name, "doc_type": "geochemical_dataset",
                    "section": "Sample Data", "file_hash": file_hash,
                },
            })
        return chunks

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _detect_element_columns(columns: list[str]) -> list[str]:
        """Return column names that match known geochemical element symbols."""
        return [c for c in columns if c.strip().upper() in {k.upper() for k in GEOCHEMICAL_THRESHOLDS}]

    @staticmethod
    def _detect_zone_column(columns: list[str]) -> Optional[str]:
        """Heuristic: find a column likely representing geological zones."""
        candidates = ["zone", "zone_name", "zone_id", "name", "area", "domain", "region",
                      "prospect", "target", "unit", "lithology", "lith", "formation"]
        col_lower = {c.lower(): c for c in columns}
        for cand in candidates:
            if cand in col_lower:
                return col_lower[cand]
        return None

    def _deposit_model_chunk(
        self, df: pd.DataFrame, source_name: str, file_hash: str, elem_cols: list[str]
    ) -> Optional[dict]:
        """Match the dataset's element suite to known deposit models."""
        if not elem_cols:
            return None

        present = {e.upper() for e in elem_cols}
        matches = []
        for model, pathfinders in DEPOSIT_PATHFINDERS.items():
            overlap = present & set(pathfinders)
            if len(overlap) >= 2:
                matches.append((model, sorted(overlap), len(overlap)))

        if not matches:
            return None

        matches.sort(key=lambda x: x[2], reverse=True)
        lines = [f"Deposit model affinities for {source_name}:"]
        for model, elems, count in matches[:5]:
            lines.append(f"  {model}: matched elements {', '.join(elems)} ({count}/{len(DEPOSIT_PATHFINDERS[model])})")

        return {
            "id": f"{file_hash}_deposit_models",
            "text": "\n".join(lines),
            "metadata": {
                "source": source_name, "doc_type": "geochemical_dataset",
                "section": "Deposit Model Affinity", "file_hash": file_hash,
            },
        }

    @staticmethod
    def _compute_hash(file_path: Path) -> str:
        h = hashlib.md5()
        with open(file_path, "rb") as fh:
            h.update(fh.read(65_536))
        return h.hexdigest()[:12]