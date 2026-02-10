import os
import datetime
import csv
from itertools import repeat
import multiprocessing as mp
from re import match, compile
import pyarrow.dataset as ds
import polars as pl
from main.ANALOGY_SCIENTIFIC.IncPrevMethods_polars import IncPrev

def processBatch(batch,
                 STUDY_START_DATE,
                 STUDY_END_DATE,
                 FILENAME,
                 DEMOGRAPHY,
                 col_end_date,
                 col_index_date,
                 date_fmt,
                 dir_out,
                 batchId,) -> None:
    #Get unique categories
    CATGS = list(set([sublist if isinstance(sublist, str) else item \
            for sublist in DEMOGRAPHY for item in sublist]))

    if isinstance(batch, str):
        cols = ['INDEX_DATE', 'END_DATE',] + [batch]
    else:
        batch = list(batch)
        cols = ['INDEX_DATE', 'END_DATE',] + list(batch)

    if len(CATGS) > 0:
        cols = cols + CATGS

    #Incidence
    dat_incprev = IncPrev(STUDY_END_DATE[0],
                            STUDY_START_DATE[0],
                            FILENAME,
                            batch,
                            DEMOGRAPHY,
                            cols,
                            col_end_date=col_end_date,
                            col_index_date=col_index_date,
                            date_fmt=date_fmt,
                            verbose=False,)

    results_inc = dat_incprev.runAnalysis(inc=True, prev=False)[0]

    #Prevalence
    dat_incprev = IncPrev(STUDY_END_DATE[1],
                            STUDY_START_DATE[1],
                            FILENAME,
                            batch,
                            DEMOGRAPHY,
                            cols,
                            col_end_date=col_end_date,
                            col_index_date=col_index_date,
                            date_fmt=date_fmt,
                            verbose=False,)

    results_prev = dat_incprev.runAnalysis(inc=False, prev=True)[1]

    results = tuple([results_inc, results_prev])

    for result_ in results:
        if "Prevalence" in result_.columns:
            metric = "prev"
        else:
            metric = "inc"
        result_.write_csv(f"{dir_out}out_{metric}_{batchId}.csv")

    # grouped incprev options


def run_incprev(conf_incprev: dict,
                dir_data: str,
                dir_out: str,
                date_fmt: str) -> None:
    FILENAME = f"{dir_data}{conf_incprev['filename']}"

    STUDY_START_DATE_INC = datetime.datetime(year=conf_incprev["start_date"]["inc"]["year"],
                                         month=conf_incprev["start_date"]["inc"]["month"],
                                         day=conf_incprev["start_date"]["inc"]["day"])
    STUDY_END_DATE_INC = datetime.datetime(year=conf_incprev["end_date"]["inc"]["year"],
                                         month=conf_incprev["end_date"]["inc"]["month"],
                                         day=conf_incprev["end_date"]["inc"]["day"])

    STUDY_START_DATE_PREV = datetime.datetime(year=conf_incprev["start_date"]["prev"]["year"],
                                         month=conf_incprev["start_date"]["prev"]["month"],
                                         day=conf_incprev["start_date"]["prev"]["day"])
    STUDY_END_DATE_PREV = datetime.datetime(year=conf_incprev["end_date"]["prev"]["year"],
                                         month=conf_incprev["end_date"]["prev"]["month"],
                                         day=conf_incprev["end_date"]["prev"]["day"])

    STUDY_START_DATE = [STUDY_START_DATE_INC, STUDY_START_DATE_PREV]
    STUDY_END_DATE = [STUDY_END_DATE_INC, STUDY_END_DATE_PREV]

    #get condition date columns
    if conf_incprev["BD_LIST"] is None:
        if FILENAME[-7:] == "parquet":
            dataset = ds.dataset(FILENAME, format="parquet")
            col_head = dataset.head(1).to_pylist()[0].keys()
            del dataset
            BASELINE_DATE_LIST = [col for col in col_head if col.startswith('BD_')]
            del col_head
        elif FILENAME[-3:] == "csv":
            with open(FILENAME,
                      "r",
                      encoding="utf8") as f:
                reader=csv.reader(f)
                col_head = next(reader)
            BASELINE_DATE_LIST = [col for col in col_head if col.startswith('BD_')]
            del col_head
        else:
            raise Exception("Cannot determine file type")
    else:
        BASELINE_DATE_LIST = conf_incprev["BD_LIST"]

    if len(BASELINE_DATE_LIST) == 1:
           processBatch(
                   BASELINE_DATE_LIST,
                   STUDY_START_DATE,
                   STUDY_END_DATE,
                   FILENAME,
                   conf_incprev["DEMOGRAPHY"],
                   conf_incprev["col_end_date"],
                   conf_incprev["col_index_date"],
                   date_fmt,
                   dir_out,
                   0,
            )
    else:
        BASELINE_DATE_LIST = \
                [tuple(BASELINE_DATE_LIST[i:i + conf_incprev["batch_size"]]) \
                for i in range(0, len(BASELINE_DATE_LIST), conf_incprev["batch_size"])]
        batches = list(zip(
            BASELINE_DATE_LIST,
            repeat(STUDY_START_DATE),
            repeat(STUDY_END_DATE),
            repeat(FILENAME),
            repeat(conf_incprev["DEMOGRAPHY"]),
            repeat(conf_incprev["col_end_date"]),
            repeat(conf_incprev["col_index_date"]),
            repeat(date_fmt),
            repeat(dir_out),
            list(range(0, len(BASELINE_DATE_LIST))),#batchId
        ))

        N_PROCESSES = conf_incprev["n_processes"]

        if N_PROCESSES is None or N_PROCESSES == 1:
            for batch_ in batches:
                processBatch(*batch_)
        else:
            pool = mp.get_context("spawn").Pool(processes = N_PROCESSES)
            pool.starmap(processBatch, batches)
            pool.close()
            pool.join()

    files_out = os.listdir(dir_out)
    pattern_inc = compile(r'.*inc_[0-9].*')
    pattern_prev = compile(r'.*prev_[0-9].*')

    file_names_inc = [x for x in files_out if match(pattern_inc, x)]
    file_names_prev = [x for x in files_out if match(pattern_prev, x)]

    output_file_inc = "inc_crude.csv"
    output_file_prev = "prev_crude.csv"

    def write_out(file_names, output_file, dir_):
        with open(f"{dir_}{output_file}", 'w') as outfile:
            for i, file_name in enumerate(file_names):
                with open(f"{dir_}{file_name}", 'r') as infile:
                #skip header if not 1st out file
                    if i!=0:
                        next(infile)
                        outfile.write(infile.read())
                    else:
                        outfile.write(infile.read())

    write_out(file_names_inc, output_file_inc, dir_out)
    write_out(file_names_prev, output_file_prev, dir_out)

    for file_ in file_names_inc:
        os.remove(f"{dir_out}{file_}")
    for file_ in file_names_prev:
        os.remove(f"{dir_out}{file_}")
