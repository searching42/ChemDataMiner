import argparse
import glob
import json
import os
import random
import re
from collections import Counter, defaultdict

import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm

from src.config import FILES, GARBAGE_ROW_STARTS, RAW_DIR
from src.utils import clean_header_text

def extract_tables_from_md(file_path):
    """Extract HTML <table> elements from a Mineru-produced markdown file.

    Returns: list[pd.DataFrame]
    """
    extracted: list[pd.DataFrame] = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f.read(), 'html.parser')
            tables = soup.find_all('table')
            for idx, table in enumerate(tables):
                # read_html can return multiple tables if nested; usually 1.
                df_list = pd.read_html(str(table), flavor="lxml")
                if df_list:
                    extracted.append(df_list[0])
    except Exception:
        pass
    return extracted

TARGET_PROPERTIES = [
    "HOMO",
    "LUMO",
    "GAP",
    "IC50",
    "Dipole_Moment",
    "Isotropic_Polarizability",
    "LogP",
    "Solubility",
    "Internal_Energy",
    "Enthalpy",
    "Gibbs_Free_Energy",
    "Toxicity",
    "Hydration_Free_Energy",
    "BBB_Permeability",
    "CYP450",
    "Total_Energy",
    # The following are often structure/atom-level; we detect but may drop as "atom-level" by default.
    "Atomic_Forces",
    "Virial_Stress_Tensor",
    "Dielectric_Constant",
    "Electron_Density",
    "Charge_Density",
]


PROPERTY_PATTERNS: list[tuple[str, list[str]]] = [
    ("HOMO", [r"\bhomo\b", r"e[_\s-]*homo", r"\bip\b"]),
    ("LUMO", [r"\blumo\b", r"e[_\s-]*lumo", r"\bea\b"]),
    ("GAP", [r"\bgap\b", r"\beg\b", r"e[_\s-]*g", r"band\s*gap", r"h[\s-]*l\s*gap", r"delta\s*e"]),
    ("IC50", [r"\bic\s*50\b", r"\bic50\b"]),
    ("Dipole_Moment", [r"dipole", r"\bmu\b", r"μ"]),
    ("Isotropic_Polarizability", [r"polariz", r"\balpha\b", r"α"]),
    ("LogP", [r"\blogp\b", r"\bxlogp\b", r"c\s*logp"]),
    ("Solubility", [r"solubility", r"\blog\s*s\b", r"\blogs\b"]),
    ("Internal_Energy", [r"internal\s*energy"]),
    ("Enthalpy", [r"enthalpy"]),
    ("Gibbs_Free_Energy", [r"gibbs", r"free\s*energy"]),
    ("Toxicity", [r"toxicity", r"\btoxic\b"]),
    ("Hydration_Free_Energy", [r"hydration.*free\s*energy", r"hydration\s*free"]),
    ("BBB_Permeability", [r"\bbbb\b", r"blood[\s-]*brain", r"permeab"]),
    ("CYP450", [r"cyp\s*450", r"cyp\d"]),
    ("Total_Energy", [r"total\s*energy", r"e[_\s-]*total", r"\betotal\b"]),
    ("Atomic_Forces", [r"atomic\s*forces?", r"\bforces?\b", r"\bf[_\s-]*[xyz]\b"]),
    ("Virial_Stress_Tensor", [r"\bvirial\b", r"stress\s*tensor", r"\bstress\b"]),
    ("Dielectric_Constant", [r"dielectric", r"epsilon", r"ε"]),
    ("Electron_Density", [r"electron\s*density", r"\brho\b", r"ρ"]),
    ("Charge_Density", [r"charge\s*density"]),
]


COMPOUND_COL_PATTERNS = [
    r"\bcompound\b",
    r"\bentry\b",
    r"\bmolecule\b",
    r"\bstructure\b",
    r"\bname\b",
    r"\bligand\b",
    r"\bid\b",
    r"\bno\.?\b",
]


NEGATIVE_TABLE_HINTS = [
    # Instrument/experimental conditions (often not trainable molecular properties)
    "instrument",
    "detector",
    "column",
    "wavelength",
    "lambda",
    "flow",
    "rpm",
    "pressure",
    "temperature",
    "temp",
    "solvent",
    "time",
    "catalyst",
    "reagent",
    "equiv",
    "conditions",
    "yield",
    # Common non-target table types
    "coordinates",
    "angstrom",
    "space group",
    "bond length",
    "bond angle",
]


ELEMENT_SYMBOLS = {
    "H", "He",
    "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
    "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr",
    "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "In", "Sn", "Sb", "Te", "I", "Xe",
}


_NUM_RE = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [" ".join([str(x) for x in tup if str(x) != "nan"]).strip() for tup in df.columns]
    return df


def _drop_empty(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")
    # Drop columns that are literally all empty strings after str conversion
    empty_cols = []
    for c in df.columns:
        s = df[c].astype(str).str.strip()
        if (s == "").all():
            empty_cols.append(c)
    if empty_cols:
        df = df.drop(columns=empty_cols)
    return df


def _maybe_fix_header(df: pd.DataFrame) -> pd.DataFrame:
    """Fix tables where pandas inferred numeric/Unnamed headers."""
    df = df.copy()
    try:
        cols = [str(c) for c in df.columns]
        if len(cols) > 0:
            bad = sum(1 for c in cols if c.isdigit() or "unnamed" in c.lower())
            if bad / len(cols) > 0.8 and len(df) >= 1:
                df.columns = df.iloc[0]
                df = df[1:].reset_index(drop=True)
    except Exception:
        return df
    return df


def _normalize_header_token(text: str) -> str:
    # Strip common LaTeX-ish wrappers and normalize whitespace.
    s = str(text)
    s = s.replace("$", "").replace("\\", "")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _split_unit(header: str) -> tuple[str, str | None]:
    """Split unit from header like 'HOMO (eV)' -> ('HOMO', 'eV')."""
    h = _normalize_header_token(header)
    # Prefer explicit () / [].
    m = re.search(r"^(.*?)[\(\[]\s*([^\)\]]+)\s*[\)\]]\s*$", h)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return h.strip(), None


def _normalize_property(header: str) -> tuple[str | None, str | None]:
    """Return (canonical_property, unit) if header matches our targets."""
    base, unit = _split_unit(header)
    base_clean = clean_header_text(base)
    base_low = base_clean.lower()

    # Guardrails + mapping for single-letter energy tokens (U/H/G/E/V).
    if base_clean in {"U", "H", "G", "E", "V"}:
        unit_low = (unit or "").lower()
        if not any(x in unit_low for x in ["ev", "ha", "hartree", "au", "a.u", "kj", "kcal"]):
            return None, None
        if base_clean == "U":
            return "Internal_Energy", unit
        if base_clean == "H":
            return "Enthalpy", unit
        if base_clean == "G":
            return "Gibbs_Free_Energy", unit
        if base_clean in {"E", "V"}:
            return "Total_Energy", unit

    for canon, pats in PROPERTY_PATTERNS:
        for pat in pats:
            if re.search(pat, base_low, flags=re.IGNORECASE):
                return canon, unit

    # Fallback: exact match against canonical tokens after cleaning.
    canon_map = {p.lower(): p for p in TARGET_PROPERTIES}
    if base_low in canon_map:
        return canon_map[base_low], unit

    return None, None


def _numeric_ratio(s: pd.Series) -> float:
    """Fraction of non-null values that can be parsed into numbers."""
    ss = s.dropna().astype(str).str.strip()
    if len(ss) == 0:
        return 0.0
    hits = ss.apply(lambda x: _NUM_RE.search(x) is not None).sum()
    return float(hits) / float(len(ss))


def _looks_like_atom_table(df: pd.DataFrame) -> bool:
    cols = [clean_header_text(c).lower() for c in df.columns]
    # Strong signal: x/y/z columns or explicit atom/element columns.
    xyz_like = {"x", "y", "z"}
    if len(xyz_like.intersection(set(cols))) >= 2:
        return True
    if any("atom" in c or "element" in c for c in cols):
        if any(k in " ".join(cols) for k in ["x", "y", "z"]):
            return True

    # Values signal: first column mostly element symbols.
    try:
        first = df.iloc[:, 0].dropna().astype(str).str.strip()
        if len(first) >= 10:
            sym = first.apply(lambda x: x in ELEMENT_SYMBOLS).mean()
            if sym >= 0.5:
                return True
    except Exception:
        pass
    return False


def _infer_compound_col(df: pd.DataFrame) -> str | None:
    """Heuristic: pick a column that represents compound identifiers/names."""
    best = (0.0, None)  # (score, col)
    for c in df.columns:
        cname = clean_header_text(c)
        low = cname.lower()
        score = 0.0

        if any(re.search(p, low) for p in COMPOUND_COL_PATTERNS):
            score += 3.0

        # Prefer a column with mostly non-numeric identifiers.
        nr = 1.0 - _numeric_ratio(df[c])
        score += 1.5 * nr

        # Penalize columns that look like conditions.
        if any(h in low for h in NEGATIVE_TABLE_HINTS):
            score -= 2.0

        if score > best[0]:
            best = (score, str(c))

    # If the best is still weak, default to first column if it looks non-numeric.
    if best[1] is None or best[0] < 1.5:
        try:
            if _numeric_ratio(df.iloc[:, 0]) < 0.6:
                return str(df.columns[0])
        except Exception:
            return None
        return None
    return best[1]


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame | None:
    """Base cleaning: normalize headers, drop empty rows/cols, and remove obvious garbage rows."""
    if df is None:
        return None

    df = _flatten_columns(df)
    df = _maybe_fix_header(df)
    df = _drop_empty(df)

    if df.empty or df.shape[1] < 2:
        return None

    # Preserve original column headers (units/symbols) for later parsing.
    try:
        df.attrs["original_columns"] = [str(c) for c in df.columns]
    except Exception:
        df.attrs["original_columns"] = None

    # Normalize column names (unit text will likely be stripped here; we keep originals in attrs).
    df = df.copy()
    df.columns = [clean_header_text(_normalize_header_token(c)) for c in df.columns]

    # Drop obvious separator rows (based on first cell prefix).
    try:
        first = df.iloc[:, 0].astype(str).str.strip()
        df = df[~first.str.startswith(tuple(GARBAGE_ROW_STARTS), na=False)]
    except Exception:
        pass

    df = _drop_empty(df)
    if df.empty or df.shape[1] < 2:
        return None

    return df


def classify_and_normalize_table(
    df: pd.DataFrame,
    keep_atom_tables: bool = False,
) -> tuple[pd.DataFrame | None, dict]:
    """Return (normalized_df or None, metadata)."""
    meta: dict = {
        "compound_col": None,
        "property_cols": [],
        "property_units": {},
        "drop_reason": None,
    }

    if df is None or df.empty:
        meta["drop_reason"] = "empty"
        return None, meta

    if _looks_like_atom_table(df) and not keep_atom_tables:
        meta["drop_reason"] = "atom_level_table"
        return None, meta

    # Identify property columns.
    prop_cols: list[str] = []
    prop_units: dict[str, str | None] = {}
    canon_by_col: dict[str, str] = {}
    original_cols = df.attrs.get("original_columns")
    for idx, c in enumerate(df.columns):
        header_for_unit = c
        if isinstance(original_cols, list) and idx < len(original_cols):
            header_for_unit = original_cols[idx]

        canon, unit = _normalize_property(str(header_for_unit))
        if canon is None:
            continue
        # Require at least some numeric-ish content to reduce false positives.
        if _numeric_ratio(df[c]) < 0.2:
            continue
        prop_cols.append(str(c))
        canon_by_col[str(c)] = canon
        if unit:
            prop_units[canon] = unit

    if not prop_cols:
        meta["drop_reason"] = "no_target_properties"
        return None, meta

    compound_col = _infer_compound_col(df)
    if compound_col is None:
        meta["drop_reason"] = "no_compound_col"
        return None, meta

    # Negative-hints override: if this looks like conditions/instrument table, drop unless strong property signal.
    joined = " ".join([str(c).lower() for c in df.columns])
    neg_hits = sum(1 for h in NEGATIVE_TABLE_HINTS if h in joined)
    if neg_hits >= 6 and len(prop_cols) <= 1:
        meta["drop_reason"] = "likely_conditions_or_instrument"
        return None, meta

    # Build normalized DF: Compound + canonical property columns.
    out = pd.DataFrame()
    out["Compound"] = df[compound_col].astype(str).str.strip()

    # Deduplicate canonical property names; keep first occurrence.
    seen = set()
    kept_props: list[str] = []
    for c in prop_cols:
        canon = canon_by_col[c]
        if canon in seen:
            continue
        seen.add(canon)
        kept_props.append(c)
        out[canon] = df[c]

    # Drop empty/black rows.
    out = _drop_empty(out)
    if out.empty or out.shape[0] == 0:
        meta["drop_reason"] = "normalized_empty"
        return None, meta

    meta["compound_col"] = compound_col
    meta["property_cols"] = [canon_by_col[c] for c in kept_props]
    meta["property_units"] = prop_units
    return out, meta


def _to_long_records(
    normalized_df: pd.DataFrame,
    source_file: str,
    table_id: int,
    property_units: dict[str, str | None],
) -> list[dict]:
    records: list[dict] = []
    if normalized_df is None or normalized_df.empty:
        return records
    props = [c for c in normalized_df.columns if c != "Compound"]
    for _, row in normalized_df.iterrows():
        compound = str(row.get("Compound", "")).strip()
        if not compound:
            continue
        for p in props:
            raw = row.get(p)
            if raw is None or (isinstance(raw, float) and pd.isna(raw)):
                continue
            s = str(raw).strip()
            m = _NUM_RE.search(s)
            if not m:
                continue
            try:
                val = float(m.group())
            except Exception:
                continue
            records.append(
                {
                    "compound": compound,
                    "property": p,
                    "value": val,
                    "unit": property_units.get(p),
                    "source_file": source_file,
                    "table_id": table_id,
                }
            )
    return records

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Step 1: Extract, filter, and normalize tables from Mineru markdown.")
    p.add_argument("--sample-files", type=int, default=0, help="Sample N markdown files (0 = use all).")
    p.add_argument("--seed", type=int, default=0, help="Random seed for sampling.")
    p.add_argument("--round", dest="round_id", type=int, default=0, help="Round id to get a deterministic different sample.")
    p.add_argument("--max-files", type=int, default=0, help="Hard cap on number of files processed (0 = no cap).")
    p.add_argument("--keep-atom-tables", action="store_true", help="Keep atom-level coordinate/force tables (default: drop).")
    p.add_argument("--write-normalized", action="store_true", help="Write normalized/long-form outputs in addition to tables_final.txt.")
    p.add_argument("--max-drop-examples", type=int, default=3, help="How many drop examples to keep per drop reason in report.")
    return p.parse_args(argv)


def _tagged_path(path: os.PathLike | str, tag: str | None) -> str:
    """Insert '.{tag}' before the last extension (used to keep multiple sample rounds)."""
    s = str(path)
    if not tag:
        return s
    base, ext = os.path.splitext(s)
    if ext:
        return f"{base}.{tag}{ext}"
    return f"{s}.{tag}"


def run(argv: list[str] | None = None):
    args = _parse_args(argv)

    print(">>> STEP 1: Extracting, Filtering, and Normalizing Tables...")
    md_files = glob.glob(str(RAW_DIR / "**/*.md"), recursive=True)
    md_files.sort()
    print(f"Found {len(md_files)} markdown files under {RAW_DIR}")

    # Deterministic sampling across rounds.
    if args.sample_files and args.sample_files > 0:
        rng = random.Random(int(args.seed) + int(args.round_id))
        n = min(args.sample_files, len(md_files))
        md_files = rng.sample(md_files, n)
        md_files.sort()
        print(f"Sampling enabled: processing {len(md_files)} markdown files (seed={args.seed}, round={args.round_id})")

    if args.max_files and args.max_files > 0:
        md_files = md_files[: args.max_files]
        print(f"Max-files cap enabled: processing first {len(md_files)} markdown files")

    tag = None
    if (args.sample_files and args.sample_files > 0) or (args.max_files and args.max_files > 0):
        parts: list[str] = []
        if args.sample_files and args.sample_files > 0:
            parts.append(f"s{int(args.sample_files)}")
            parts.append(f"seed{int(args.seed)}")
            parts.append(f"r{int(args.round_id)}")
        if args.max_files and args.max_files > 0:
            parts.append(f"max{int(args.max_files)}")
        tag = ".".join(parts) if parts else None

    counters = Counter()
    drop_examples: dict[str, list[dict]] = defaultdict(list)

    # Stream writes to avoid holding huge lists in memory.
    tables_final_path = _tagged_path(FILES["tables_final"], tag)
    tables_norm_path = _tagged_path(FILES["tables_normalized"], tag)
    tables_long_path = _tagged_path(FILES["tables_long"], tag)
    report_path = _tagged_path(FILES["step1_report"], tag)

    out_tables = open(tables_final_path, "w", encoding="utf-8")
    out_norm = open(tables_norm_path, "w", encoding="utf-8") if args.write_normalized else None
    out_long = open(tables_long_path, "w", encoding="utf-8") if args.write_normalized else None

    try:
        for fpath in tqdm(md_files, desc="Markdown"):
            counters["md_files_total"] += 1
            tables = extract_tables_from_md(fpath)
            counters["tables_total"] += len(tables)

            for i, raw_df in enumerate(tables, start=1):
                counters["tables_seen"] += 1
                clean_df = clean_dataframe(raw_df)
                if clean_df is None:
                    counters["drop_clean_failed"] += 1
                    continue

                normalized_df, meta = classify_and_normalize_table(
                    clean_df,
                    keep_atom_tables=bool(args.keep_atom_tables),
                )
                if normalized_df is None:
                    reason = meta.get("drop_reason") or "dropped"
                    counters[f"drop_{reason}"] += 1
                    if len(drop_examples[reason]) < int(args.max_drop_examples):
                        drop_examples[reason].append(
                            {
                                "source_file": fpath,
                                "table_id": i,
                                "columns": list(clean_df.columns)[:50],
                                "shape": list(clean_df.shape),
                            }
                        )
                    continue

                counters["tables_kept"] += 1

                # Write filtered table for downstream LLM steps.
                table_str = normalized_df.to_markdown(index=False, tablefmt="grid")
                out_tables.write(f"FILE: {fpath}\nTABLE_ID: {i}\n{table_str}\n======\n")

                if out_norm is not None and out_long is not None:
                    rec = {
                        "source_file": fpath,
                        "table_id": i,
                        "shape": list(clean_df.shape),
                        "normalized_shape": list(normalized_df.shape),
                        "compound_col": meta.get("compound_col"),
                        "property_cols": meta.get("property_cols", []),
                        "property_units": meta.get("property_units", {}),
                        "table_rows": normalized_df.to_dict(orient="records"),
                    }
                    out_norm.write(json.dumps(rec, ensure_ascii=False) + "\n")

                    long_records = _to_long_records(
                        normalized_df,
                        source_file=fpath,
                        table_id=i,
                        property_units=meta.get("property_units", {}),
                    )
                    for lr in long_records:
                        out_long.write(json.dumps(lr, ensure_ascii=False) + "\n")

        print(f"Saved {counters['tables_kept']} kept tables to {tables_final_path}")
        if args.write_normalized:
            print(f"Saved normalized tables to {tables_norm_path}")
            print(f"Saved long-form records to {tables_long_path}")
    finally:
        out_tables.close()
        if out_norm is not None:
            out_norm.close()
        if out_long is not None:
            out_long.close()

    report = {
        "raw_dir": str(RAW_DIR),
        "tables_final": str(tables_final_path),
        "tables_normalized": str(tables_norm_path),
        "tables_long": str(tables_long_path),
        "args": {
            "sample_files": args.sample_files,
            "seed": args.seed,
            "round": args.round_id,
            "max_files": args.max_files,
            "keep_atom_tables": bool(args.keep_atom_tables),
            "write_normalized": bool(args.write_normalized),
        },
        "counters": dict(counters),
        "drop_examples": drop_examples,
        "target_properties": TARGET_PROPERTIES,
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Wrote Step 1 report to {report_path}")

if __name__ == "__main__":
    run()
