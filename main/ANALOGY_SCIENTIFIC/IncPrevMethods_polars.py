from datetime import date, datetime
from typing import List, Dict, Callable
import numpy as np
from tqdm import tqdm
from scipy.stats import chi2
from scipy.special import ndtri
from dateutil.relativedelta import relativedelta
import polars as pl

class IncPrev():
    """
    |Manages the automated incidence rate and point prevalence calculations on an output from Dexter.
    |Normal use would initialise object, then use any of:
    |            calculate_incidence
    |            calculate_prevalence
    |            calculate_grouped_incidence
    |            calculate_grouped_prevalence

    |Required Args:
    |    STUDY_END_DATE (datetime): End date of the study. Should be same as the value entered in dexter for data extract.
    |    STUDY_START_DATE (datetime): Start date of the study. Should be same as the value entered in dexter for data extract.
    |    FILENAME (string): Name of the data file.
    |    DATABASE_NAME (string): OPTIONS: AURUM, GOLD, THIN, IMRD.

    |Optional Args:
    |    BASELINE_DATE_LIST (list of strings): Default: []. Defines the columns to base event dates on. Defaults to all column names beginning with "BD_".
    |    DEMOGRAPHY (list of strings): Default: []. Defines columns to use as grouping variables. Defaults to any of AGE_CATG, SEX, ETHNICITY, COUNTRY, HEALTH_AUTH, TOWNSEND found in the column names.
    |    cols (list of strings): Default: None. List of column names to pass into usecols of pd.read_csv in IncPrev.read(). Must include PRACTICE_PATIENT_ID, PRACTICE_ID, INDEX_DATE, START_DATE, END_DATE, COLLECTION_DATE, TRANSFER_DATE, DEATH_DATE, REGISTRATION_STATUS, additionally with all BASELINE_DATE_LIST and DEMOGRAPHY. Used for efficiency.
    |    skiprows_i (list of ints): Default: None. List of indexes to pass into skiprows of pd.read_csv in IncPrev.read(). Indexes must be inclusive of header (header index = 0).
    |    read_data (bool): Default True. If True, FILENAME arge must be defined. If false, DEMOGRAPHY, BASELINE_DATE_LIST args must be defined. Defines whether raw_data will be assigned during (True) or post (False) initialisation.
    |    SMALL_FP_VAL (float): Default: 1e-8. Small constant used during calculations to avoid values of 0? [check].
    |    DAYS_IN_YEAR (float): Default: 365.25. Constant defining number of days in a year.
    |    PER_PY (string): Default: 100_000. Number of person years represented by incidence and prevalence values.
    |    ALPHA (float): Default: 0.05. Significance level for calulating error using Byar's method.
    |    INCREMENT_BY_MONTH (int): Default: 12. Number of months in each incidence/prevalence calculation.

    |Attributes:
    |    PER_PY (string): Defined above.
    |    SMALL_FP_VAL (string): Defined above.
    |    DAYS_IN_YEAR (string): Defined above.
    |    ALPHA (float): Defined above.
    |    STUDY_END_DATE (string): Defined above.
    |    STUDY_START_DATE (string): Defined above.
    |    INCREMENT_BY_MONTH (int): Defined above.
    |    FILENAME (string): Defined above.
    |    DATABASE_NAME (string): Defined above.
    |    BASELINE_DATE_LIST (list): Defined above.
    |    DEMOGRAPHY (list): Defined above.
    |    raw_data (pl.DataFrame): The data used to calculate incidence rate and point prevalence.

    |Methods:
    |    read:
    |    byars_lower:
    |    byars_higher:
    |    save_dataframe_inc:
    |    save_dataframe_prev:
    |    point_incidence:
    |    point_prevalence:
    |    calculate_incidence:
    |    calculate_prevalence:
    |    calculate_grouped_incidence:
    |    calculate_grouped_prevalence:

    """

    __slots__ = 'PER_PY', 'ALPHA', \
        'STUDY_END_DATE', 'STUDY_START_DATE', \
        'BASELINE_DATE_LIST', 'DEMOGRAPHY', 'FILENAME', 'raw_data', \
        'DATABASE_NAME', "dat_fmt", "verbose", "DataKeys", "StudyDesignKeys", \
        "increment_years", "increment_months", "increment_days", \
        "BASELINE_DATE_LIST"

    def __init__(self,
                 STUDY_END_DATE,
                 STUDY_START_DATE,
                 FILENAME,
                 DATABASE_NAME,
                 BASELINE_DATE_LIST=[],
                 DEMOGRAPHY=[],
                 cols = None,
                 read_data = True,
                 increment_years=1,
                 increment_months=0,
                 increment_days=0,
                 PER_PY=100_000,
                 ALPHA=0.05,
                 fileType = None,
                 verbose = False):

        self.PER_PY, self.ALPHA,\
        self.STUDY_END_DATE, self.STUDY_START_DATE, \
        self.DEMOGRAPHY, self.FILENAME, \
        self.DATABASE_NAME, self.increment_years, \
        self.increment_months, self.increment_days, \
        self.BASELINE_DATE_LIST = \
        PER_PY, ALPHA, \
        STUDY_END_DATE, STUDY_START_DATE, \
        DEMOGRAPHY, FILENAME,\
        DATABASE_NAME, increment_years, increment_months, increment_days, \
        BASELINE_DATE_LIST

        self.dat_fmt = "%Y-%m-%d"
        self.verbose = verbose

        self.STUDY_END_DATE += relativedelta(years=0, months=0, days=1)
        if read_data == True:
            self.read(cols, fileType)
        else:
            self.raw_data = None

        self.DataKeys = {
                "INDEX_DATE_COL": "INDEX_DATE",
                "END_DATE_COL": "END_DATE",
                "EVENT_DATE_COL": "EVENT_DATE",
                }
        self.StudyDesignKeys = {
                "SMALL_FP_VAL": 1e-8,
                }

    def read(self, cols, fileType="csv"):
        if cols is None:
            cols = "*"
        if fileType=="csv":
            self.raw_data = (
                    pl.scan_csv(self.FILENAME).lazy()
                    .select(pl.col(cols))
            )
        elif fileType == "parquet":
            self.raw_data = (
                    pl.scan_parquet(self.FILENAME).lazy()
                    .select(pl.col(cols))
            )

        self.raw_data = (
            self.raw_data
            .with_columns(
                pl.col(["INDEX_DATE",
                    "START_DATE",
                    "END_DATE",
                    "COLLECTION_DATE",
                    "TRANSFER_DATE",
                    "DEATH_DATE"]).str.strptime(pl.Date, format="%Y-%m-%d")
            )
        )
         # This logic is IMRD database specific. REGISTRATION_STATUS with value 99 in IMRD database means patient is dead.
         # To ensure records are properly updated the DEATH_DATE for those patients whose REGISTRATION_STATUS == 99 are
         # set to the TRANSFER_DATE.
        if self.DATABASE_NAME == "IMRD":
            self.raw_data = (
                self.raw_data
                .with_columns(
                    pl.when(
                        (pl.col("*").is_null()) & (pl.col("REGISTRATION_STATUS")==99)
                            .then(pl.col("TRANSFER_DATE"))
                            .otherwise(pl.col("DEATH_DATE"))
                    ).alias("DEATH_DATE")
                )
            )

        if len(self.BASELINE_DATE_LIST) == 0:
             self.BASELINE_DATE_LIST = [col for col in self.raw_data.columns if col.startswith('BD_')]

        self.raw_data = self.raw_data.with_columns(
            pl.col(self.BASELINE_DATE_LIST).str.strptime(pl.Date,
                                                         format=self.dat_fmt)
        )

          # Only run if the database is IMRD.
        if self.DATABASE_NAME == "IMRD":
            self.raw_data.with_columns(
                pl.min(["END_DATE", "DEATH_DATE"]).alias("END_DATE")
            )

        #Missing stratification vars set to "null"
        catgs = list(set([sublist if isinstance(sublist, str) else item \
                for sublist in self.DEMOGRAPHY for item in sublist]))

        self.raw_data = self.raw_data.with_columns(
                pl.col(catgs).fill_null("null")
                )

    def byars_lower(self, count, denominator):
        if count < 10:
            b = chi2.ppf((self.ALPHA / 2), (count * 2)) / 2
            lower_ci = b / denominator
            return lower_ci
        else:
            z = ndtri(1 - self.ALPHA / 2)
            c = 1 / (9 * count)
            b = 3 * np.sqrt(count)
            lower_o = count * ((1 - c - (z / b)) ** 3)
            lower_ci = lower_o / denominator
            return lower_ci

    def byars_higher(self, count, denominator):
        if count < 10:
            b = chi2.ppf(1 - (self.ALPHA / 2), 2 * count + 2) / 2
            upper_ci = b / denominator
            return upper_ci
        else:
            z = ndtri(1 - self.ALPHA / 2)
            c = 1 / (9 * (count + 1))
            b = 3 * (np.sqrt(count + 1))
            upper_o = (count + 1) * ((1 - c + (z / b)) ** 3)
            upper_ci = upper_o / denominator
            return upper_ci

    def runAnalysis(self, inc=True, prev=True,):
        if inc:
            results_inc = self.calculate_overall_inc_prev(is_incidence=True)
        else:
            results_inc = None
        if prev:
            results_prev = self.calculate_overall_inc_prev(is_incidence=False)
        else:
            results_prev = None
        if len(self.DEMOGRAPHY) > 0:
            if inc:
                results_inc = pl.concat(tuple([
                    results_inc,
                    self.calculate_grouped_inc_prev(is_incidence=True)
                    ]),
                                        how="vertical",
                                        )
            if prev:
                results_prev = pl.concat(tuple([
                    results_prev,
                    self.calculate_grouped_inc_prev(is_incidence=False)
                    ]),
                                        how="vertical",
                                        )

        return tuple([results_inc, results_prev])

    #######################################################################

    def date_range(
        self,
        start_date: datetime,
        end_date: datetime,
        increment_years: int = 0,
        increment_months: int = 12,
        increment_days: int = 0,
    ) -> datetime:
        """
        A generator function to get list of datetimes between start_date and end_date, with increments.

        Args:
          start_date (datetime): study start date as provided in the study design.
          end_date (datetime): study end date as provided in the study design.
          increment_years (int): yearly increments by between start date and end date.
          increment_months (int): monthly increments by between start date and end date.
          increment_days (int): daily increments by between start date and end date.

        Returns:
            current_period (datetime)
        """
        current_period = start_date
        delta = relativedelta(years=increment_years, months=increment_months, days=increment_days)
        while current_period <= end_date:
            yield current_period
            current_period += delta


    """
    Incidence and Prevalence Rule definitions.
    """


    def prevalence_numerator_rule(self, d_range: List[datetime]) -> List[pl.Expr]:
        """
        Function definition for rules to calculate prevalence numerator.

        Args:
            d_range(List[datetime]): a list of datetime values to calculate point prevalence at.

        Returns:
            query(List[pl.Expr])

        """
        query = []
        for d in d_range:
            query.append(
                pl.when(
                    (pl.col(self.DataKeys["INDEX_DATE_COL"]) <= d)
                    & (pl.col(self.DataKeys["END_DATE_COL"]) >= d)
                    & (pl.col(self.DataKeys["EVENT_DATE_COL"]) <= d)
                )
                .then(pl.lit(1))
                .otherwise(pl.lit(0))
                .alias(str(d.date()))
            )
        return query


    def prevalence_denominator_rule(self, d_range: List[datetime]) -> List[pl.Expr]:
        """
        Function definition for rules to calculate prevalence denominator.

        Args:
            d_range(List[datetime]): a list of datetime values to calculate point prevalence at.

        Returns:
            query(List[pl.Expr])
        """
        query = []
        for d in d_range:
            query.append(
                pl.when((pl.col(self.DataKeys["INDEX_DATE_COL"]) <= d) & \
                        (pl.col(self.DataKeys["END_DATE_COL"]) >= d))
                .then(pl.lit(1))
                .otherwise(pl.lit(0))
                .alias(str(d.date()))
            )
        return query


    def incidence_numerator_rule(self, d_range: List[datetime]) -> List[pl.Expr]:
        """
        Function definition for rules to calculate incidence numerator.

        Args:
            d_range(List[datetime]): a list of datetime values to calculate period incidence at.

        Returns:
            query(List[pl.Expr])
        """
        query = []
        for i in range(1, len(d_range)):
            query.append(
                pl.when(
                    (
                        pl.col(self.DataKeys["EVENT_DATE_COL"]).is_between(
                            d_range[i - 1], d_range[i], closed="left"
                        )
                    )
                    & (pl.col(self.DataKeys["EVENT_DATE_COL"]) > \
                            pl.col(self.DataKeys["INDEX_DATE_COL"]))
                    & (pl.col(self.DataKeys["END_DATE_COL"]) >= d_range[i - 1])
                    & (pl.col(self.DataKeys["INDEX_DATE_COL"]) < d_range[i])
                )
                .then(pl.lit(1))
                .otherwise(pl.lit(0))
                .alias(str(d_range[i - 1].date()))
            )
        return query


    def incidence_denominator_rule(self, d_range: List[datetime]) -> List[pl.Expr]:
        """
        Function definition for rules to calculate incidence denominator.

        Args:
            d_range(List[datetime]): a list of datetime values to calculate period incidence at.

        Returns:
            query(List[pl.Expr])
        """
        query = []
        for i in range(1, len(d_range)):
            delta = d_range[i] - d_range[i - 1]
            query.append(
                pl.when(
                    (
                        (
                            (pl.col(self.DataKeys["EVENT_DATE_COL"]) >= d_range[i - 1])
                            & (pl.col(self.DataKeys["EVENT_DATE_COL"]) >\
                               pl.col(self.DataKeys["INDEX_DATE_COL"]))
                        )
                        | (pl.col(self.DataKeys["EVENT_DATE_COL"]).is_null())
                    )
                    & (pl.col(self.DataKeys["END_DATE_COL"]) >= d_range[i - 1])
                    & (pl.col(self.DataKeys["INDEX_DATE_COL"]) < d_range[i])
                )
                .then(
                    pl.min_horizontal(
                        pl.col(self.DataKeys["END_DATE_COL"]),
                        pl.col(self.DataKeys["EVENT_DATE_COL"]), d_range[i]
                    )
                    .sub(pl.max_horizontal(pl.col(self.DataKeys["INDEX_DATE_COL"]), d_range[i - 1]))
                    .dt.days()
                    .cast(pl.Float64)
                    / delta.days
                )
                .otherwise(pl.lit(self.StudyDesignKeys["SMALL_FP_VAL"]))
                .alias(str(d_range[i - 1].date()))
            )
        return query


    def calculate_metrics(
        self,
        melted_df: pl.LazyFrame,
        rule_fn: Callable[[List[datetime]], List[pl.Expr]],
        d_range: List[datetime],
        col_list: List[str],
        rename: Dict[str, str],
    ):
        query = rule_fn(d_range)

        melted_df = (
            melted_df.with_columns(query)
            .group_by(["Condition", "Group", "Subgroup"])
            .agg(pl.col(col_list).sum())
        )

        return melted_df.melt(id_vars=["Condition", "Group", "Subgroup"], value_vars=col_list).rename(
            rename
        )


    def filter_data_for_combination(self, data: pl.LazyFrame, condition: List[str], demography: List[str]):
        if self.verbose:
            print(demography)
            print(condition)
        # Extract relevant columns
        filtered_data = data.select(
            [self.DataKeys["INDEX_DATE_COL"], self.DataKeys["END_DATE_COL"]] +\
                    condition + demography
        )

        # Rename columns to match the desired structure
        if len(demography) > 0:
            return filtered_data.rename(
                {condition[0]: self.DataKeys["EVENT_DATE_COL"], demography[0]: "Subgroup"}
            ).with_columns(
                pl.lit(condition[0]).alias("Condition"), pl.lit(demography[0]).alias("Group")
            )
        else:
            return filtered_data.rename({condition[0]:\
            self.DataKeys["EVENT_DATE_COL"]}).with_columns(
                pl.lit(condition[0]).alias("Condition"),
                pl.lit("Overall").alias("Group"),
                pl.lit("").alias("Subgroup"),
            )


    def calculate_overall_inc_prev(
        self,
        is_incidence: bool = False,
    ) -> pl.DataFrame:
        """
        Function to calculate overall incidence or prevalence for each condition.

        Args:
            data (LazyFrame): the processed study polars LazyFrame.
            study_start_date (date): study start date as defined during study extract.
            study_end_date (date): study end date as defined during study extract.
            condition_list (List): List of conditions.
            is_incidence (bool): flag for Incidence or Prevalence study
            increment_days (int): increment period by n days.
            increment_months (int): increment period by n months.
            increment_years (int): increment period by n years.

        Returns:
            polars DataFrame
        """
        #confidence_method = ByarsConfidenceInterval()
        drange = list(
            self.date_range(
                self.STUDY_START_DATE,
                self.STUDY_END_DATE,
                self.increment_years,
                self.increment_months,
                self.increment_days
            )
        )
        if self.verbose:
            print(drange)
        col_list = [str(d.date()) for d in drange]
        if self.verbose:
            print(col_list)

        all_num_results = []
        all_den_results = []
        rename_num = {"variable": "Date", "value": "Numerator"}
        rename_den = {"variable": "Date", "value": "Denominator"}

        for datecol_name in self.BASELINE_DATE_LIST:
            # Filter the data
            filtered_data = self.filter_data_for_combination(self.raw_data,
                                                             [datecol_name],
                                                             [])

            # Calculate numerator and denominator
            if is_incidence:
                df_num = self.calculate_metrics(
                    filtered_data,
                    self.incidence_numerator_rule,
                    drange,
                    col_list[:-1],
                    rename_num,
                )
                df_den = self.calculate_metrics(
                    filtered_data,
                    self.incidence_denominator_rule,
                    drange,
                    col_list[:-1],
                    rename_den,
                )
            else:
                df_num = self.calculate_metrics(
                    filtered_data,
                    self.prevalence_numerator_rule,
                    drange,
                    col_list,
                    rename_num,
                )
                df_den = self.calculate_metrics(
                    filtered_data,
                    self.prevalence_denominator_rule,
                    drange,
                    col_list,
                    rename_den,
                )

            # Store results
            all_num_results.append(df_num)
            all_den_results.append(df_den)
        # Combine all results
        final_num_df = pl.concat(all_num_results, how="vertical").collect()
        final_den_df = pl.concat(all_den_results, how="vertical").collect()

        df_overall = final_num_df.join(final_den_df, on=["Condition", "Group", "Subgroup", "Date"])
        col_name = "Incidence" if is_incidence else "Prevalence"
        df_overall = df_overall.with_columns(
            ((pl.col("Numerator") / pl.col("Denominator"))*self.PER_PY).alias(col_name),
            pl.struct(["Numerator", "Denominator"])
            .map_elements(lambda x: self.byars_lower(x["Numerator"], x["Denominator"])*self.PER_PY)
            .alias("Lower_CI"),
            pl.struct(["Numerator", "Denominator"])
            .map_elements(lambda x: self.byars_higher(x["Numerator"], x["Denominator"])*self.PER_PY)
            .alias("Upper_CI"),
        )

        return df_overall


    def calculate_grouped_inc_prev(
        self,
        is_incidence: bool = False,
    ) -> pl.DataFrame:
        """
        Function to calculate subgroup incidence or prevalence for each condition.

        Args:
            data (LazyFrame): the processed study polars LazyFrame.
            study_start_date (date): study start date as defined during study extract.
            study_end_date (date): study end date as defined during study extract.
            condition_list (List): List of conditions.
            demography_list (List): List of demography
            is_incidence (bool): flag for Incidence or Prevalence study
            increment_days (int): increment period by n days.
            increment_months (int): increment period by n months.
            increment_years (int): increment period by n years.

        Returns:
            polars DataFrame
        """
        drange = list(
            self.date_range(
                self.STUDY_START_DATE,
                self.STUDY_END_DATE,
                self.increment_years,
                self.increment_months,
                self.increment_days
            )
        )
        if self.verbose:
            print(drange)
        col_list = [str(d.date()) for d in drange]
        if self.verbose:
            print(col_list)

        all_num_results = []
        all_den_results = []
        rename_num = {"variable": "Date", "value": "Numerator"}
        rename_den = {"variable": "Date", "value": "Denominator"}

        for datecol_name in self.BASELINE_DATE_LIST:
            for demo in self.DEMOGRAPHY:
                if isinstance(demo, list):
                    demo_ = ", ".join(demo)
                    data_ = self.raw_data.with_columns(
                            pl.concat_str(pl.col(demo),
                                          separator=", ").alias(demo_)
                            )
                    filtered_data = self.filter_data_for_combination(data_,
                                                                     [datecol_name],
                                                                     [demo_])
                else:# Filter the data
                    filtered_data = self.filter_data_for_combination(self.raw_data,
                                                                     [datecol_name],
                                                                     [demo])
                # Calculate numerator and denominator
                if is_incidence:
                    df_num = self.calculate_metrics(
                        filtered_data,
                        self.incidence_numerator_rule,
                        drange,
                        col_list[:-1],
                        rename_num,
                    )
                    df_den = self.calculate_metrics(
                        filtered_data,
                        self.incidence_denominator_rule,
                        drange,
                        col_list[:-1],
                        rename_den,
                    )
                else:
                    df_num = self.calculate_metrics(
                        filtered_data,
                        self.prevalence_numerator_rule,
                        drange,
                        col_list,
                        rename_num,
                    )
                    df_den = self.calculate_metrics(
                        filtered_data,
                        self.prevalence_denominator_rule,
                        drange,
                        col_list,
                        rename_den,
                    )
                # Store results
                all_num_results.append(df_num)
                all_den_results.append(df_den)
        # Combine all results
        final_num_df = pl.concat(all_num_results, how="vertical").collect()
        final_den_df = pl.concat(all_den_results, how="vertical").collect()

        df_overall = final_num_df.join(final_den_df, on=["Condition", "Group", "Subgroup", "Date"])
        col_name = "Incidence" if is_incidence else "Prevalence"
        df_overall = df_overall.with_columns(
            ((pl.col("Numerator") / pl.col("Denominator"))*self.PER_PY).alias(col_name),
            pl.struct(["Numerator", "Denominator"])
            .map_elements(lambda x: self.byars_lower(x["Numerator"], x["Denominator"])*self.PER_PY)
            .alias("Lower_CI"),
            pl.struct(["Numerator", "Denominator"])
            .map_elements(lambda x: self.byars_higher(x["Numerator"], x["Denominator"])*self.PER_PY)
            .alias("Upper_CI"),
        )

        return df_overall



