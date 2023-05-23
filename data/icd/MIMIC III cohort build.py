
# A python script to create a dataset for ICD 9 extraction from MIMIC III.
# Adapted from https://github.com/JoakimEdin/medical-coding-reproducibility

import os
import random
import shutil
from collections import Counter
from functools import partial
from pathlib import Path
import pandas as pd


ID_COLUMN = "_id"
TEXT_COLUMN = "text"
TARGET_COLUMN = "target"
SUBJECT_ID_COLUMN = "subject_id"

CODE_SYSTEMS = [
    ("ICD9-DIAG", "DIAGNOSES_ICD.csv.gz", "ICD9_CODE", "icd9_diag"),
    ("ICD9-PROC", "PROCEDURES_ICD.csv.gz", "ICD9_CODE", "icd9_proc"),
]

MIN_TARGET_COUNT = 10  # Minimum number of times a code must appear to be included
download_dir = # root directory to mimic iii
note_dir = # directory to NOTEEVENTS.csv file


def format_code_dataframe(df: pd.DataFrame, col_in: str, col_out: str) -> pd.DataFrame:
    """Formats the code dataframe.
    Args:
        df (pd.DataFrame): The dataframe containing the codes.
        col_in (str): The name of the column containing the codes.
        col_out (str): The new name of the column containing the codes
    Returns:
        pd.DataFrame: The formatted dataframe.
    """
    df = df.rename(
        columns={
            "HADM_ID": ID_COLUMN,
            "SUBJECT_ID": SUBJECT_ID_COLUMN,
            "TEXT": TEXT_COLUMN,
        }
    )
    df = df.sort_values([SUBJECT_ID_COLUMN, ID_COLUMN])
    df[col_in] = df[col_in].astype(str).str.strip()
    df = df[[SUBJECT_ID_COLUMN, ID_COLUMN, col_in]].rename({col_in: col_out}, axis=1)
    # remove codes that are nan
    df = df[df[col_out] != "nan"]
    return (
        df.groupby([SUBJECT_ID_COLUMN, ID_COLUMN])
        .apply(partial(reformat_code_dataframe, col=col_out))
        .reset_index()
    )

def merge_code_dataframes(code_dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """Merges all code dataframes into a single dataframe.
    Args:
        code_dfs (list[pd.DataFrame]): List of code dataframes.
    Returns:
        pd.DataFrame: Merged code dataframe.
    """
    merged_codes = code_dfs[0]
    for code_df in code_dfs[1:]:
        merged_codes = merged_codes.merge(
            code_df, how="outer", on=[SUBJECT_ID_COLUMN, ID_COLUMN]
        )
    return merged_codes
def merge_report_addendum_helper_function(row: pd.DataFrame) -> pd.Series:
    """Merges the report and addendum text."""
    dout = dict()
    if len(row) == 1:
        dout["DESCRIPTION"] = row.iloc[0].DESCRIPTION
        dout[TEXT_COLUMN] = row.iloc[0][TEXT_COLUMN]
    else:
        # row = row.sort_values(["DESCRIPTION", "CHARTDATE"], ascending=[False, True])
        dout["DESCRIPTION"] = "+".join(row.DESCRIPTION)
        dout[TEXT_COLUMN] = " ".join(row[TEXT_COLUMN])
    return pd.Series(dout)

def merge_reports_addendum(mimic_notes: pd.DataFrame) -> pd.DataFrame:
    """Merges the reports and addendum into one dataframe.
    Args:
        mimic_notes (pd.DataFrame): The dataframe containing the notes from the mimiciii dataset.
    Returns:
        pd.DataFrame: The dataframe containing the discharge summaries consisting of reports and addendum.
    """
    discharge_summaries = mimic_notes[mimic_notes["CATEGORY"] == "Discharge summary"]
    #discharge_summaries = mimic_notes
    discharge_summaries = discharge_summaries.dropna(subset=[ID_COLUMN])
    discharge_summaries[ID_COLUMN] = discharge_summaries[ID_COLUMN].astype(int)
    return (
        discharge_summaries.groupby([SUBJECT_ID_COLUMN, ID_COLUMN])
        .apply(merge_report_addendum_helper_function)
        .reset_index()
    )  

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
    return 

def replace_nans_with_empty_lists(
    df: pd.DataFrame, columns: list[str] = ["icd9_diag", "icd9_proc"]
) -> pd.DataFrame:
    """Replaces nans in the columns with empty lists."""
    for column in columns:
        df.loc[df[column].isnull(), column] = df.loc[df[column].isnull(), column].apply(
            lambda x: []
        )
    return df
    
def remove_duplicated_codes(df: pd.DataFrame, column_names: list[str]) -> pd.DataFrame:
    """Remove duplicated codes in the dataframe"""
    df = df.copy()
    for col in column_names:
        df[col] = df[col].apply(lambda codes: list(set(codes)))
    return df

def get_duplicated_icd9_proc_codes() -> set:
    """Get the duplicated ICD9-PROC codes. The codes are duplicated because they are saved as integers,
    removing any zeros at the beginning of the codes. These codes will not be included in the dataset.
    Returns:
        set: The duplicated ICD9-PROC codes
    """
    icd9_proc_codes = pd.read_csv(
        download_dir+"/D_ICD_PROCEDURES.csv.gz",
        compression="gzip",
        dtype={"ICD9_CODE": str},
    )
    return set(
        icd9_proc_codes[icd9_proc_codes["ICD9_CODE"].astype(str).duplicated()][
            "ICD9_CODE"
        ]
    )


def prepare_discharge_summaries(mimic_notes: pd.DataFrame) -> pd.DataFrame:
    """Format the notes dataframe into the discharge summaries dataframe
    Args:
        mimic_notes (pd.DataFrame): The notes dataframe
    Returns:
        pd.DataFrame: Formatted discharge summaries dataframe
    """
    mimic_notes = mimic_notes.rename(
        columns={
            "HADM_ID": ID_COLUMN,
            "SUBJECT_ID": SUBJECT_ID_COLUMN,
            "TEXT": TEXT_COLUMN,
        }
    )
    print(f"{mimic_notes[ID_COLUMN].nunique()} number of admissions")
    discharge_summaries = merge_reports_addendum(mimic_notes)
    discharge_summaries = discharge_summaries.sort_values(
        [SUBJECT_ID_COLUMN, ID_COLUMN]
    )

    discharge_summaries = discharge_summaries.reset_index(drop=True)
    print(
        f"{discharge_summaries[SUBJECT_ID_COLUMN].nunique()} subjects, {discharge_summaries[ID_COLUMN].nunique()} admissions"
    )
    return discharge_summaries


def filter_codes(df: pd.DataFrame, columns: list[str], min_count: int) -> pd.DataFrame:
    """Filter the codes dataframe to only include codes that appear at least min_count times
    Args:
        df (pd.DataFrame): The codes dataframe
        col (str): The column name of the codes
        min_count (int): The minimum number of times a code must appear
    Returns:
        pd.DataFrame: The filtered codes dataframe
    """
    for col in columns:
        code_counts = Counter([code for codes in df[col] for code in codes])
        codes_to_keep = set(
            code for code, count in code_counts.items() if count >= min_count
        )
        df[col] = df[col].apply(lambda x: [code for code in x if code in codes_to_keep])
    return df


def download_and_preprocess_code_systems(code_systems: list[tuple]) -> pd.DataFrame:
    """Download and preprocess the code systems dataframe
    Args:
        code_systems (List[tuple]): The code systems to download and preprocess
    Returns:
        pd.DataFrame: The preprocessed code systems dataframe"""
    code_dfs = []
    for name, fname, col_in, col_out in code_systems:
        print(f"Loading {name} codes...")
        df = pd.read_csv(download_dir+'/'+fname, compression="gzip", dtype={"ICD9_CODE": str}
        )
        df = format_code_dataframe(df, col_in, col_out)
        df = remove_duplicated_codes(df, [col_out])
        code_dfs.append(df)

    merged_codes = merge_code_dataframes(code_dfs)
    merged_codes = replace_nans_with_empty_lists(merged_codes)
    merged_codes["icd9_diag"] = merged_codes["icd9_diag"].apply(
        lambda codes: list(map(partial(reformat_icd9, is_diag=True), codes))
    )
    merged_codes["icd9_proc"] = merged_codes["icd9_proc"].apply(
        lambda codes: list(map(partial(reformat_icd9, is_diag=False), codes))
    )
    merged_codes[TARGET_COLUMN] = merged_codes["icd9_proc"] + merged_codes["icd9_diag"]
    return merged_codes

def reformat_code_dataframe(row: pd.DataFrame, col: str) -> pd.Series:
    """Takes a dataframe and a column name and returns a series with the column name and a list of codes.
    """
    return pd.Series({col: row[col].sort_values().tolist()})


get_duplicated_icd9_proc_codes()
# MIMIC-III full

mimic_notes = pd.read_csv(note_dir)
discharge_summaries = prepare_discharge_summaries(mimic_notes)
merged_codes = download_and_preprocess_code_systems(CODE_SYSTEMS)

full_dataset = discharge_summaries.merge(
    merged_codes, on=[SUBJECT_ID_COLUMN, ID_COLUMN], how="inner"
)
full_dataset = replace_nans_with_empty_lists(full_dataset)
# Remove codes that appear less than 10 times
full_dataset = filter_codes(
    full_dataset, [TARGET_COLUMN, "icd9_proc", "icd9_diag"], min_count=MIN_TARGET_COUNT
)
# Remove admissions with no codes
full_dataset = full_dataset[full_dataset[TARGET_COLUMN].apply(len) > 0]

#full_dataset = preprocess_documents(df=full_dataset, preprocessor=preprocessor)

print(f"{full_dataset[ID_COLUMN].nunique()} number of admissions")
full_dataset = full_dataset.reset_index(drop=True)

# Save the dataset as a pickle file 
output_filename = # output directory/file name 
os.makedirs(os.path.dirname(output_filename), exist_ok = True)
full_dataset.to_pickle(output_filename)