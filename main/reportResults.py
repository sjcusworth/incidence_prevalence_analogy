import polars as pl
from shutil import move
import yaml
from main.dataScienceWorkflows.graphing import Visualisation
from main.dataScienceWorkflows.table1 import table1_polars

def report_results(
        config: dict,
        table1: bool,
        crude: bool,
        strd: bool,
        ) -> None:

    file_dat = f"{config['dir_data']}{config['incprev']['filename']}"

    ###############################################################################
    if table1:
        ## Table1
        catgs = [
                'AGE_CATEGORY',
                'SEX',
                'ETHNICITY',
                'REGION',
                ]

        null_values = [
                "MISSING",
                "",
                "null",
                "none",
                ]

        tb1 = table1_polars(
                file_dat,
                catgs,
                [],
                null_values,
                )
        tb1.write_csv(f"{config['dir_out']}table1.csv")

    ###############################################################################
    if crude:
        ## Graphs crude
        graphObj = Visualisation()

        catgs = [
                'Overall',
                'AGE_CATEGORY',
                'SEX',
                'ETHNICITY',
                'REGION',
                ]

        for dir_ in (config['dir_out'], f"{config['dir_out']}Publish/"):
            # check files exist (e.g. where not run censoring)
            try:
                data_prev = pl.read_csv(f"{dir_}prev_crude.csv", infer_schema_length=0,)
                data_inc = pl.read_csv(f"{dir_}inc_crude.csv", infer_schema_length=0,)
            except:
                continue

            for dat_, incprev in [(data_prev, "prev"), (data_inc, "inc")]:
                for cond_ in data_prev.get_column("Condition").unique().to_list():
                    for group_ in catgs:
                        if incprev == "prev":
                            y_var_ = "Prevalence"
                            y_var_rename_ = "Prevalence (per 100,000 persons)"
                        else:
                            y_var_ = "Incidence"
                            y_var_rename_ = "Incidence (per 100,000 py)"

                        label_ = graphObj.plot_scatter(
                                data = dat_.filter(
                                    pl.col("Condition")==cond_,
                                    pl.col("Group")==group_,
                                    pl.col("Subgroup")!="I",
                                    ).with_columns(
                                        pl.col("Date").str.strptime(pl.Date, "%Y-%m-%d"),
                                        pl.col(y_var_).cast(pl.Float64),
                                        pl.col("Lower_CI").cast(pl.Float64),
                                        pl.col("Upper_CI").cast(pl.Float64),
                                        ).filter(
                                            pl.col("Date").dt.year() >= 2006,
                                            ).rename({y_var_: y_var_rename_}).to_pandas(),
                                y_var = y_var_rename_,
                                x_var = "Date",
                                c_name = "Subgroup",
                                interactive = True,
                                meta_vars = ["Subgroup", y_var_rename_, "Lower_CI", "Upper_CI",],
                                out_type = "interactive",
                                toDisk = True,
                                withLine = True,
                                is_errorY = True,
                                cols_errorY = ["Lower_CI", "Upper_CI"],
                                )
                        move(f"{label_}.html", f"{dir_}{label_}_crude_{cond_}_{group_}.html")

    ###############################################################################
    ## Graphs strd
    if strd:
        graphObj = Visualisation()

        catgs = [
                "Overall",
                "Ethnicity",
                ]

        for dir_ in (config['dir_out'], f"{config['dir_out']}Publish/"):
            # check files exist (e.g. where not run censoring)
            try:
                data_prev = pl.read_csv(f"{dir_}prev_DSR.csv", infer_schema_length=0,)
                data_inc = pl.read_csv(f"{dir_}inc_DSR.csv", infer_schema_length=0,)
            except:
                continue

            for dat_, incprev in [(data_prev, "prev"), (data_inc, "inc")]:
                for cond_ in data_prev.get_column("Condition").unique().to_list():
                    for group_ in catgs:
                        if incprev == "prev":
                            y_var_ = "Prevalence"
                            y_var_rename_ = "Prevalence (per 100,000 persons)"
                        else:
                            y_var_ = "Incidence"
                            y_var_rename_ = "Incidence (per 100,000 py)"

                        dat_cond_plot_ = dat_.filter(
                                    pl.col("Condition")==cond_,
                                    pl.col("Group")==group_,
                                    pl.col("Subgroup")!="I",
                                    ).with_columns(
                                        pl.col("Date").str.strptime(pl.Date, "%Y-%m-%d"),
                                        pl.col(y_var_).cast(pl.Float64),
                                        pl.col("Lower_CI").cast(pl.Float64),
                                        pl.col("Upper_CI").cast(pl.Float64),
                                        ).filter(
                                            pl.col("Date").dt.year() >= 2006,
                                            ).rename({y_var_: y_var_rename_})
                        if dat_cond_plot_.filter(pl.col(y_var_rename_).is_not_null()).shape[0] == 0:
                            continue
                        label_ = graphObj.plot_scatter(
                                data = dat_cond_plot_.to_pandas(),
                                y_var = y_var_rename_,
                                x_var = "Date",
                                c_name = "Subgroup",
                                interactive = True,
                                meta_vars = ["Subgroup", y_var_rename_, "Lower_CI", "Upper_CI",],
                                out_type = "interactive",
                                toDisk = True,
                                withLine = True,
                                is_errorY = True,
                                cols_errorY = ["Lower_CI", "Upper_CI"],
                                )
                        move(f"{label_}.html", f"{dir_}{label_}_strd_{cond_}_{group_}.html")
