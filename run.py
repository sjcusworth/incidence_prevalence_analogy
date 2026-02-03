import sys
import yaml
import os

with open("config.yml",
          "r",
          encoding="utf8") as file_config:
    config = yaml.safe_load(file_config)


if len(sys.argv) == 1:
    print("Insert argument at commandline: `process` for preprocessing; `incprev` for incprev calculations; `strd` for direct standardisation; `censor` for small number censoring; `report` for producing graphs and tables")
    opt = None
else:
    opt = sys.argv[1]


## Preprocessing
if opt == "process":
    from main.preprocessing import preprocessing
    preprocessing(
            config["dir_data"],
            config["processing"],
            )


## IncPrev
if opt == "incprev":
    import datetime
    import csv
    from itertools import repeat
    import multiprocessing as mp
    from re import match, compile
    import pyarrow.dataset as ds
    import polars as pl
    from main.ANALOGY_SCIENTIFIC.IncPrevMethods_polars import IncPrev
    from main.IncPrev import processBatch

    conf_incprev = config["incprev"]
    FILENAME = f"{config['dir_data']}{conf_incprev['filename']}"

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
                   conf_incprev["date_fmt"],
                   config["dir_out"],
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
            repeat(conf_incprev["date_fmt"]),
            repeat(config["dir_out"]),
            list(range(0, len(BASELINE_DATE_LIST))),#batchId
        ))

        N_PROCESSES = conf_incprev["n_processes"]

        if N_PROCESSES is None:
            pool = mp.Pool(processes = mp.cpu_count() - 2)
        else:
            pool = mp.Pool(processes = N_PROCESSES)
        pool.starmap(processBatch, batches)

    files_out = os.listdir(config['dir_out'])
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

    write_out(file_names_inc, output_file_inc, config["dir_out"])
    write_out(file_names_prev, output_file_prev, config["dir_out"])

    for file_ in file_names_inc:
        os.remove(f"{config['dir_out']}{file_}")
    for file_ in file_names_prev:
        os.remove(f"{config['dir_out']}{file_}")


## Standardising
# only compatible with age-sex direct standardisation
if opt == "strd":
    from main.strd import standardise_incprev
    standardise_incprev(
            config["dir_data"],
            config["dir_out"],
            config["strd"],
            )


## Small number censoring
if opt == "censor":
    from main.smallNumCens import small_num_censor
    small_num_censor(
            n = config["censor"]["n"],
            crude = config["censor"]["crude"],
            strd = config["censor"]["strd"],
            dir_out = config["dir_out"],
            )


## reportResults
if opt == "report":
    from main.reportResults import report_results
    report_results(
            config,
            table1=config["report"]["table1"],
            crude=config["report"]["crude"],
            strd=config["report"]["strd"],
            )
