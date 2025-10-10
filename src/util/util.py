import pandas as pd

def merge_keep_first(left, right):
    merged = pd.merge(left, right, on="fiscalDateEnding", how="outer", suffixes=('', '_dup'))

    for col in right.columns:
        if col != "fiscalDateEnding" and col + "_dup" in merged.columns:
            merged.drop(columns=[col + "_dup"], inplace=True)

    return merged