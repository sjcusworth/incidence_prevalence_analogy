import os
import shutil
import polars as pl
from main.IncPrev import run_incprev
from main.smallNumCens import getCrudeMap
from tests.make_test_dat import get_example_dat

def test_ageSex_map_sums():
    """
    Compare the grouped age-sex sum data above to a single variable
    e.g. sum(`age, sex, ethnicity`) vs `ethnicity`
    """
    path_dat = "dat_temp.csv"
    dir_incprev = "temp/"
    os.mkdir(dir_incprev)

    get_example_dat(5_000).write_csv(path_dat)

    #calc incprev
    run_incprev(
            {
                "filename": path_dat,
                "col_end_date": "END_DATE",
                "col_index_date": "INDEX_DATE",
                "start_date": {
                    "inc": {"year": 2001, "month": 1, "day": 1,},
                    "prev": {"year": 2001, "month": 7, "day": 1,}
                    },
                "end_date": {
                    "inc": {"year": 2003, "month": 12, "day": 31,},
                    "prev": {"year": 2003, "month": 12, "day": 31,},
                    },
                "n_processes": 1,
                "BD_LIST": ["condition_a", "condition_b"],
                "batch_size": 1,
                "INCREMENT_BY_MONTH": 12,
                "calc_grouped": True,
                "DEMOGRAPHY": [
                    'pat_catg_a',
                    'pat_catg_b',
                    ["pat_catg_a", "pat_catg_b"],
                    ["AGE_CATEGORY", "SEX"],
                    ["AGE_CATEGORY", "SEX", "pat_catg_a"],
                    ["AGE_CATEGORY", "SEX", "pat_catg_b"],
                    ["AGE_CATEGORY", "SEX", "pat_catg_a", "pat_catg_b"],
                    ],
                },
            "./",
            dir_incprev,
            "%d-%m-%Y",
            )

    path_dat_incprev = f"{dir_incprev}prev_crude.csv"
    crude_map = getCrudeMap(path_dat_incprev)

    for pat_catg_ in ("pat_catg_a", "pat_catg_b", "pat_catg_a, pat_catg_b"):
        dat_check = (
                pl.read_csv(path_dat_incprev, infer_schema_length=0,)
                .with_columns(
                    pl.col("Numerator").cast(pl.Int64)
                    )
                .select(pl.col(["Subgroup", "Date", "Condition", "Numerator", "Group",]))
                .filter(pl.col("Group")==pat_catg_)
                .with_columns(
                    pl.col("Subgroup").str.replace_all("'", "")
                    )
                .select(pl.all().exclude("Group"))
                )
        dat_check = (
                dat_check
                .join(
                    crude_map.filter(pl.col("Subgroup")!="Overall"),
                    on=["Subgroup", "Date", "Condition"],
                    how="left",
                    )
                )

        # ensure no rows that don't match numerator_right and numerator
        assert dat_check.filter(pl.col("Numerator_right")!=pl.col("Numerator")).shape[0] == 0

    # clean-up
    os.remove(path_dat)
    shutil.rmtree(dir_incprev)


if __name__=="__main__":
    test_ageSex_map_sums()
