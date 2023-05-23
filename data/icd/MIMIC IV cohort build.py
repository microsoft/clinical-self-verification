# A python script to create a dataset for ICD 9 and ICD 10 extraction from MIMIC IV.
# Adapted from https://github.com/JoakimEdin/medical-coding-reproducibility


import collections
import json
import os
import random
import pandas as pd

from collections import Counter
from functools import partial
from pathlib import Path

import numpy as np


ID_COLUMN = "_id"
TEXT_COLUMN = "text"
TARGET_COLUMN = "target"
SUBJECT_ID_COLUMN = "subject_id"


MIN_TARGET_COUNT = 10  # Minimum number of times a code must appear to be included
mimic4_root = #file path


def filter_codes(df: pd.DataFrame, columns: list[str], min_count: int) -> pd.DataFrame:
    """Filter the codes dataframe to only include codes that appear at least min_count times
    """
    for col in columns:
        code_counts = Counter([code for codes in df[col] for code in codes])
        codes_to_keep = set(
            code for code, count in code_counts.items() if count >= min_count
        )
        df[col] = df[col].apply(lambda x: [code for code in x if code in codes_to_keep])
        print(f"Number of unique codes in {col} before filtering: {len(code_counts)}")
        print(f"Number of unique codes in {col} after filtering: {len(codes_to_keep)}")

    return df


def parse_codes_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Parse the codes dataframe"""
    df = df.rename(columns={"hadm_id": ID_COLUMN, "subject_id": SUBJECT_ID_COLUMN})
    df = df.dropna(subset=["icd_code"])
    df = df.drop_duplicates(subset=[ID_COLUMN, "icd_code"])
    df = (
        df.groupby([SUBJECT_ID_COLUMN, ID_COLUMN, "icd_version"])
        .apply(partial(reformat_code_dataframe, col="icd_code"))
        .reset_index()
    )
    return df


def parse_notes_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Parse the notes dataframe"""
    df = df.rename(
        columns={
            "hadm_id": ID_COLUMN,
            "subject_id": SUBJECT_ID_COLUMN,
            "text": TEXT_COLUMN,
        }
    )
    df = df.dropna(subset=[TEXT_COLUMN])
    df = df.drop_duplicates(subset=[ID_COLUMN, TEXT_COLUMN])
    return df

def reformat_icd(code: str, version: int, is_diag: bool) -> str:
    """format icd code depending on version"""
    if version == 9:
        return reformat_icd9(code, is_diag)
    elif version == 10:
        return reformat_icd10(code, is_diag)
    else:
        raise ValueError("version must be 9 or 10")


def reformat_icd10(code: str, is_diag: bool) -> str:
    """
    Put a period in the right place because the MIMIC-3 data files exclude them.
    Generally, procedure codes have dots after the first two digits,
    while diagnosis codes have dots after the first three digits.
    """
    code = "".join(code.split("."))
    if not is_diag:
        return code
    return code[:3] + "." + code[3:]


def reformat_icd9(code: str, is_diag: bool) -> str:
    """
    Put a period in the right place because the MIMIC-3 data files exclude them.
    Generally, procedure codes have dots after the first two digits,
    while diagnosis codes have dots after the first three digits.
    """
    code = "".join(code.split("."))
    if is_diag:
        if code.startswith("E"):
            if len(code) > 4:
                return code[:4] + "." + code[4:]
        else:
            if len(code) > 3:
                return code[:3] + "." + code[3:]
    else:
        if len(code) > 2:
            return code[:2] + "." + code[2:]
    return code

def reformat_code_dataframe(row: pd.DataFrame, col: str) -> pd.Series:
    """Takes a dataframe and a column name and returns a series with the column name and a list of codes.
    """
    return pd.Series({col: row[col].sort_values().tolist()})    

def preprocess_text(df):
  df[TEXT_COLUMN] = df[TEXT_COLUMN].apply(lambda x: x.replace("[^A-Za-z0-9]+", " ").replace("(\s\d+)+\s", " ").replace("\s+", " ").lower())
  return df     





# Diagnosis & procedure
mimic_diag = pd.read_csv(os.path.join(mimic4_root, 'hosp', 'diagnoses_icd.csv'),
                      dtype={"icd_code": "string", "subject_id": "string", "hadm_id": "string"})
mimic_proc = pd.read_csv(os.path.join(mimic4_root, 'hosp', 'procedures_icd.csv.gz'),
                      dtype={"icd_code": "string", "subject_id": "string", "hadm_id": "string"})

# Discharge summary
mimic_notes = pd.read_csv(os.path.join(mimic4_root, 'discharge.csv.gz'),
                       dtype={'note_id': 'string', 'subject_id': 'string', 'hadm_id': 'string', 'text': 'string'})


# Format the codes by adding decimal points
mimic_proc["icd_code"] = mimic_proc.apply(
    lambda row: reformat_icd(
        code=row["icd_code"], version=row["icd_version"], is_diag=False
    ),
    axis=1,
)
mimic_diag["icd_code"] = mimic_diag.apply(
    lambda row: reformat_icd(
        code=row["icd_code"], version=row["icd_version"], is_diag=True
    ),
    axis=1,
)

# Process codes and notes
mimic_proc = parse_codes_dataframe(mimic_proc)
mimic_diag = parse_codes_dataframe(mimic_diag)
mimic_notes = parse_notes_dataframe(mimic_notes)

# Merge the codes and notes into a icd9 and icd10 dataframe
mimic_proc_9 = mimic_proc[mimic_proc["icd_version"] == 9]
mimic_proc_9 = mimic_proc_9.rename(columns={"icd_code": "icd9_proc"})
mimic_proc_10 = mimic_proc[mimic_proc["icd_version"] == 10]
mimic_proc_10 = mimic_proc_10.rename(columns={"icd_code": "icd10_proc"})

mimic_diag_9 = mimic_diag[mimic_diag["icd_version"] == 9]
mimic_diag_9 = mimic_diag_9.rename(columns={"icd_code": "icd9_diag"})
mimic_diag_10 = mimic_diag[mimic_diag["icd_version"] == 10]
mimic_diag_10 = mimic_diag_10.rename(columns={"icd_code": "icd10_diag"})

mimiciv_9 = mimic_notes.merge(
    mimic_proc_9[[ID_COLUMN, "icd9_proc"]], on=ID_COLUMN, how="left"
)
mimiciv_9 = mimiciv_9.merge(
    mimic_diag_9[[ID_COLUMN, "icd9_diag"]], on=ID_COLUMN, how="left"
)

mimiciv_10 = mimic_notes.merge(
    mimic_proc_10[[ID_COLUMN, "icd10_proc"]], on=ID_COLUMN, how="left"
)
mimiciv_10 = mimiciv_10.merge(
    mimic_diag_10[[ID_COLUMN, "icd10_diag"]], on=ID_COLUMN, how="left"
)                       


# remove notes with no codes
mimiciv_9 = mimiciv_9.dropna(subset=["icd9_proc", "icd9_diag"], how="all")
mimiciv_10 = mimiciv_10.dropna(subset=["icd10_proc", "icd10_diag"], how="all")

# convert NaNs to empty lists
mimiciv_9["icd9_proc"] = mimiciv_9["icd9_proc"].apply(
    lambda x: [] if x is np.nan else x
)
mimiciv_9["icd9_diag"] = mimiciv_9["icd9_diag"].apply(
    lambda x: [] if x is np.nan else x
)
mimiciv_10["icd10_proc"] = mimiciv_10["icd10_proc"].apply(
    lambda x: [] if x is np.nan else x
)
mimiciv_10["icd10_diag"] = mimiciv_10["icd10_diag"].apply(
    lambda x: [] if x is np.nan else x
)

mimiciv_9 = filter_codes(mimiciv_9, ["icd9_proc", "icd9_diag"], MIN_TARGET_COUNT)
mimiciv_10 = filter_codes(mimiciv_10, ["icd10_proc", "icd10_diag"], MIN_TARGET_COUNT)

# define target
mimiciv_9[TARGET_COLUMN] = mimiciv_9["icd9_proc"] + mimiciv_9["icd9_diag"]
mimiciv_10[TARGET_COLUMN] = mimiciv_10["icd10_proc"] + mimiciv_10["icd10_diag"]

# remove empty target
mimiciv_9 = mimiciv_9[mimiciv_9[TARGET_COLUMN].apply(lambda x: len(x) > 0)]
mimiciv_10 = mimiciv_10[mimiciv_10[TARGET_COLUMN].apply(lambda x: len(x) > 0)]

# format text 
mimiciv_9 = preprocess_text(mimiciv_9)
mimiciv_10 = preprocess_text(mimiciv_10)

# reset index
mimiciv_9 = mimiciv_9.reset_index(drop=True)
mimiciv_10 = mimiciv_10.reset_index(drop=True)


# Save the dataset as a pickle file 
output_filename_9 = # output directory/file name for icd9 codes
output_filename_10 = # output directory/file name for icd10 codes

os.makedirs(os.path.dirname(output_filename_9), exist_ok = True)
mimiciv_9.to_pickle(output_filename)
os.makedirs(os.path.dirname(output_filename_10), exist_ok = True)
mimiciv_10.to_pickle(output_filename)