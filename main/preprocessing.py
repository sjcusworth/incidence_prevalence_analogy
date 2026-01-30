import sys
import os
import datetime
import csv
import multiprocessing as mp
from itertools import repeat
import logging
import yaml

import polars as pl
import pyarrow.dataset as ds
from main.preprocessing_functions import process_imd, rmDup, mergeCols, combineLevels

def preprocessing(
        dir_data: str,
        config_preproc: dict,
        path_log: str = "log_sBatch_1Python.txt"
        ) -> None:
    ## Log
    logging.basicConfig(filename=path_log,
                        filemode='a',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.DEBUG)
    logger = logging.getLogger()
    ##

    flag_temp_file: bool = False

    ## Format Null ################################################################
    logger.info("Formatting null values")

    if config_preproc["filename"] is None:
        filesToFormat = [f"{dir_data}{config_preproc['filename_gold']}",
                         f"{dir_data}{config_preproc['processing']['filename_aurum']}"]
        config_preproc['filename_gold'] = f"{config_preproc['filename_gold'][:-4]}_formNulls.parquet"
        config_preproc['filename_aurum'] = f"{config_preproc['filename_aurum'][:-4]}_formNulls.parquet"
    else:
        filesToFormat = [f"{dir_data}{config_preproc['filename']}"]
        config_preproc["filename"] = filesToFormat[0]

    for file_ in filesToFormat:
        dat = (
                pl.scan_csv(file_, infer_schema_length=0)
                .with_columns(
                    pl.when(pl.all().str.len_chars() == 0)
                        .then(None)
                        .otherwise(pl.all())
                        .name.keep()
                    )
                )
        dat.sink_parquet(f"{file_[:-4]}_formNulls.parquet")
    logger.info("    Formatting null values finished")
    del dat

    ###LinkingAurumGold############################################################
    if config_preproc["filename"] is None:
        print("Linking")
        logger.info("Linking Gold and Aurum")

        dat_a = config_preproc['filename_gold']
        dat_b = config_preproc['filename_aurum']

        outFile="dat_linked.parquet"
        rmDup(
            f"{dir_data}{dat_a}",
            f"{dir_data}{dat_b}",
            A_ind=0,
            B_ind=2,
            map_file=f"{dir_data}{config_preproc['map_file_AtoB']}",
            map_delim=config_preproc['map_delim_AtoB'],
            low_memory=False,
            wdir=dir_data,
            logger=logger,
            outFile=outFile,
            )
        logger.info("    Linking finished")
    else:
        outFile = config_preproc["filename"]

    ###MergeCols#####################################################################
    if outFile.find(".csv") != -1:
        file_type = "csv"
    else:
        file_type = "parquet"

    if config_preproc["mergeCols_AtoB"] is not None:
        print("Merging Cols")
        logger.info("Merging columns")

        mergeCols(
                dir_data,
                outFile,
                config_preproc["mergeCols_AtoB"],
                file_type = file_type,
                low_memory = False,
                logger=logger,
                outFile="condMerged.parquet",
                )
        if flag_temp_file:
            os.remove(outFile)
        else:
            flag_temp_file = True
        outFile = "condMerged.parquet"
        logger.info("    Merging Cols finished")

    ###CombineLevels#####################################################################
    if outFile.find(".csv") != -1:
        file_type = "csv"
    else:
        file_type = "parquet"

    if config_preproc["combineLevels"] is not None:
        print("Processing Column Levels")
        logger.info("Combining levels")

        combineLevels(dir_data,
                      outFile,
                      config_preproc["combineLevels"],
                      file_type=file_type,
                      outFile="dat_updatedLevels.parquet",
                      )
        if flag_temp_file:
            os.remove(outFile)
        else:
            flag_temp_file = True
        outFile = "dat_updatedLevels.parquet"
        logger.info("    Processing Column Levels finished")

    ###LinkImd#####################################################################
    if outFile.find(".csv") != -1:
        is_parquet = False
    else:
        is_parquet = True

    if config_preproc['link_imd']:
        print("Linking IMD")
        logger.info("Linking IMD")

        process_imd(
                outFile,
                dir_data,
                file_map = config_preproc["imd_map_file"],
                low_memory=False,
                is_parquet=is_parquet,
                logger=logger,
                outFile="dat_processed.parquet",
                )
        if flag_temp_file:
            os.remove(outFile)
        else:
            flag_temp_file = True
        outFile="dat_processed.parquet"
        logger.info("    Linking IMD finished")

    os.rename(outFile, "dat_processed.parquet")
