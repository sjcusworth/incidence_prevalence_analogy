import datetime
import sys
import csv
import multiprocessing as mp
from itertools import repeat
import polars as pl
import pyarrow.dataset as ds
import yaml
import os
from pandas import read_csv
import numpy as np
from main.ANALOGY_SCIENTIFIC.IncPrevMethods import StrdIncPrev

def standardise_incprev(
        dir_data: str,
        dir_out: str,
        config_strd: dict,
        ) -> None:

    col_condition: str = "Condition"
    col_category: str = "Group"
    col_group: str = "Subgroup"

    incprev = StrdIncPrev(
            standard_breakdowns=config_strd["standard_breakdowns"],
            col_condition = col_condition,
            col_category = col_category,
            col_group = col_group,
            )

    column_dtypes = {
            col_condition: "string",
            col_category: "string",
            col_group: "string",
            "Date": "string",
            "Numerator": "int32",
            "Denominator": "float64",
            "Prevalence": "float64",
            "Lower_CI": "float64",
            "Upper_CI": "float64",
            }

    incprev.raw_data_inc = read_csv(f"{dir_out}inc_crude.csv", dtype=column_dtypes,)
    incprev.raw_data_inc["Incidence"] = incprev.raw_data_inc["Incidence"].fillna(0)
    incprev.raw_data_inc["Upper_CI"] = incprev.raw_data_inc["Upper_CI"].apply(lambda x: x if x!=np.inf else 0)#np.NaN)

    incprev.raw_data_prev = read_csv(f"{dir_out}prev_crude.csv", dtype=column_dtypes,)
    incprev.raw_data_prev["Prevalence"] = incprev.raw_data_prev["Prevalence"].fillna(0)
    incprev.raw_data_prev["Upper_CI"] = incprev.raw_data_prev["Upper_CI"].apply(lambda x: x if x!=np.inf else 0)#np.NaN)

    def fmt_data(df):
        df["Condition"] = df["Condition"].map(lambda x: ''.join(letter for letter in x if letter.isalnum()).replace("BDMEDI",""))

        overall_map = (df["Group"] == "Overall")
        df["Subgroup"][overall_map] = ""

        df["Subgroup"] = df["Subgroup"].fillna("NotSpecified")

        df["std_group"] = df["Subgroup"].map(lambda x: ", ".join(
            "".join([letter for letter in x if letter != " "]).split(",")[0:2]
            ))
        df["Subgroup"] = df["Subgroup"].map(lambda x: ", ".join(
            "".join([letter for letter in x if letter != " "]).split(",")[2:]
            ))

        rm_map = df["std_group"].apply(lambda x: ''.join(letter for letter in x if letter != " ").split(","))
        #removing intersex
        rm_map = rm_map.apply(lambda x: False if "I" in x else True).to_numpy()
        df = df[(rm_map)]

        return df

    incprev.raw_data_prev = fmt_data(incprev.raw_data_prev)
    incprev.raw_data_inc = fmt_data(incprev.raw_data_inc)

    incprev.getReference(f"{dir_data}{config_strd['reference_population']}",
                         bins=config_strd["age_bins"],
                         groups=config_strd["age_group_labels"],)

    prev = incprev.standardise_all_conditions()
    inc = incprev.standardise_all_conditions(measure="Incidence")

    inc.to_csv(f"{dir_out}inc_DSR.csv")
    prev.to_csv(f"{dir_out}prev_DSR.csv")

