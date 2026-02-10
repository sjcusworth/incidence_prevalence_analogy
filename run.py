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
            date_fmt=config["date_fmt"],
            )


## IncPrev
if opt == "incprev":
    from main.IncPrev import run_incprev
    if __name__ == "__main__":
        run_incprev(
                config["incprev"],
                config["dir_data"],
                config["dir_out"],
                config["date_fmt"],
                )


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
