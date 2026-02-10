from os import remove
from random import randint, seed
from datetime import datetime
import polars as pl

def get_example_dat(n_rows = 500) -> pl.DataFrame:
    dates_index = [f"{randint(1,28)}-{randint(1,12)}-{randint(1990,2005)}" for x in range(0,n_rows)]
    dates_end = [f"{x[:-4]}{int(x[-4:])+randint(2,20)}" for x in dates_index]

    def get_dates_condition(dates_index, dates_end):
        dates_cond = [randint(0,1) for x in dates_index]
        dates_cond = [True if x else False for x in dates_cond]

        dates_period = [f"{x[:-4]}{randint(int(x[-4:]), int(y[-4:]))}" for x,y in zip(dates_index,dates_end)]

        return [y if x else None for x,y in zip(dates_cond, dates_period)]

    def get_rnd_catg(opts, n,):
        return [opts[randint(0,len(opts)-1)] for x in range(0, n)]

    dat = pl.DataFrame({
        "INDEX_DATE": dates_index,
        "END_DATE": dates_end,
        "condition_a": get_dates_condition(dates_index, dates_end,),
        "condition_b": get_dates_condition(dates_index, dates_end,),
        "pat_catg_a": get_rnd_catg(
            ["a"]*100 + ["b"]*136 + ["c"],
            len(dates_index)),
        "pat_catg_b": get_rnd_catg(
            ["1"]*5 + ["2"] + ["3"]*3 + ["4"]*2,
            len(dates_index)),
        "SEX": get_rnd_catg(
            ["M"]*10 + ["F"]*9,
            len(dates_index)),
        "AGE_CATEGORY": get_rnd_catg(
            ["0-16"]*3 + ["17-30"]*4 + ["31-40"]*5 + ["41-50"]*3 + ["51-60"]*2 + ["61-70"]*4 + ["71-80"]*2 + ["81+"],
            len(dates_index)),
        })

    return dat

if __name__=="__main__":
    seed(117)
    get_example_dat(50_000).write_csv("data/data_test.csv")
