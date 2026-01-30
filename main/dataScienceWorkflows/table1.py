import yaml
import polars as pl
from scipy.stats import chi2_contingency, ttest_ind

class table1_polars:
    """
    Table1 from a Parquet file using Polars.

    Attributes:
    ----------
    delim : str
        The delimiter used for CSV output.
    catgs : list
        List of categorical columns to analyze.
    nums : list
        List of continuous columns to analyze.
    rnd : int
        Number of decimal places to round percentages.
    datLazy_raw : pl.LazyFrame
        LazyFrame containing the raw data with specified columns.
    n_total : int
        Total number of rows in the dataset.
    tb1 : dict
        Dictionary to store analysis results for each category.
    group_col : str, optional
        Column to denote exposed and unexposed groups (must be 2 groups).
    calc_sig : bool
        Evaluate significance of differences between groups if group_col is not None.
    """
    def __init__(
            self,
            fileDat,
            catgs,
            nums,
            null_values=["", "null",],
            delim=",",
            rnd=2,
            group_col=None,
            calc_sig=False,
            ):
        """
        Initializes the table1_polars class with the given parameters.

        Parameters:
        ----------
        fileDat : str
            Path to the Parquet file.
        catgs : list
            List of categorical columns to analyze.
        nums : list
            List of continuous columns to analyze.
        null_values : list, optional
            List of values to consider as null (default is ["", "null"]).
        delim : str, optional
            Delimiter for CSV output (default is ",").
        rnd : int, optional
            Number of decimal places to round percentages (default is 2).
        group_col : str, optional
            Column to denote exposed and unexposed groups (must be 2 groups).
        calc_sig : bool
            Evaluate significance of differences between groups if group_col is not None.
        """
        self.delim = delim
        self.catgs = catgs
        self.nums = nums
        self.rnd = rnd

        self.calc_sig = calc_sig
        if self.calc_sig:
            self.group_col = group_col
            self.groups_uniq = (
                    pl.scan_parquet(fileDat)
                    .select(pl.col(self.group_col))
                    .unique().collect().get_column(group_col).to_list()
                    )
            cols_select = self.catgs + self.nums + [self.group_col]
        else:
            self.group_col = "group_"
            self.groups_uniq = ["group"]
            cols_select = self.catgs + self.nums

        self.datLazy_raw = (
                pl.scan_parquet(fileDat)
                .select(
                    pl.col(cols_select)
                    )
                .with_columns(
                    pl.when(pl.col(pl.Utf8).is_in(null_values))
                    .then(None)
                    .otherwise(pl.col(pl.Utf8))
                    .name.keep()
                    )
                )

        self.n_total = self.datLazy_raw.select(pl.len()).collect().item()

        self.tb1 = {x_:{} for x_ in self.catgs+self.nums}
        for c_ in self.catgs:
            self.tb1[c_] = self.calcCatg(c_)
        for n_ in self.nums:
            self.tb1[n_] = self.calcNum(n_)


    def calcCatg(self, catg,):
        """
        Calculates the count, percentage, and missing values for each subcategory in a given category.

        Parameters:
        ----------
        catg : str
            The category column to analyze.

        Returns:
        -------
        pl.DataFrame
            A DataFrame of table1 stats for catg col.
        """
        dat_catg = self.datLazy_raw
        if not self.calc_sig:
            dat_catg = (
                    dat_catg
                    .with_columns(
                        pl.lit("group").alias(self.group_col),
                        )
                    )

        dat_missing = (
                dat_catg
                .select(pl.col(catg, self.group_col))
                .group_by(pl.col(self.group_col))
                .agg(
                    n_missing=-pl.col(catg).is_not_null().sum().cast(pl.Int64) + pl.col(catg).len(),
                    group_total=pl.col(catg).len()
                    )
                )

        dat_groupcounts = (
                dat_catg
                .filter(
                    # % calculated without missing
                    pl.col(catg).is_not_null()
                    )
                .group_by(pl.col(self.group_col))
                .len(name="subgrp_n_total")
                )

        dat_catg = (
                dat_catg
                .join(
                    dat_groupcounts,
                    how="left",
                    on=self.group_col,
                    )
                .join(
                    dat_missing,
                    how="left",
                    on=self.group_col,
                    )
                .group_by(pl.col([self.group_col, catg]))
                #.first() used where all values in a col are the same
                .agg(
                    n=pl.count(catg),
                    percent=pl.count(catg)/pl.col("subgrp_n_total").first()*100,
                    n_missing=pl.col("n_missing").first(),
                    group_total=pl.col("group_total").first(),
                    )
                .filter(
                    pl.col(catg).is_not_null()
                    )
                .collect()
                )


        if self.calc_sig:
            dat_chi = (
                    dat_catg
                    .filter(pl.col(self.group_col)==self.groups_uniq[0])
                    .select(pl.col([catg, "n"]))
                    .rename({"n":"left"})
                    .join(
                        (
                            dat_catg
                            .filter(pl.col(self.group_col)==self.groups_uniq[1])
                            .select(pl.col([catg, "n"]))
                            .rename({"n":"right"})
                            ),
                        how="full",
                        on=catg,
                        )
                    .select(pl.col(["left", "right"]))
                    .with_columns(pl.all().fill_null(strategy="zero"))
                    )
            p = chi2_contingency(dat_chi.to_numpy()).pvalue
            dat_catg = dat_catg.with_columns(
                    pvalue=pl.lit(p),
                    )

        return dat_catg


    def calcNum(self, num,):
        """
        Calculates ... for a numeric column.

        Parameters:
        ----------
        num : str
            The numeric column to analyze.

        Returns:
        -------
        pl.DataFrame
            A DataFrame of table1 stats for num col.
        """
        dat_num = self.datLazy_raw
        if not self.calc_sig:
            dat_num = (
                    dat_num
                    .with_columns(
                        pl.lit("group").alias(self.group_col),
                        )
                    )

        dat_num = (
                dat_num
                .select(pl.col(num), pl.col(self.group_col))
                )
        dat_groupcounts = (
                dat_num
                # used for missing data calculation, unlike in catgs
                # therefore, need all, including missing
                .group_by(pl.col(self.group_col))
                .len(name="subgrp_n_total")
                )
        dat_num = (
                dat_num
                .join(
                    dat_groupcounts,
                    how="left",
                    on=self.group_col,
                    )
                .group_by(pl.col(self.group_col))
                .agg(
                    mean=pl.col(num).mean(),
                    std=pl.col(num).std(),
                    median=pl.col(num).median(),
                    min=pl.col(num).min(),
                    max=pl.col(num).max(),
                    n_missing=-pl.col(num).is_not_null().sum().cast(pl.Int64) + pl.col("subgrp_n_total").first(),
                    group_total=pl.col(num).len(),
                    )
                .collect()
                )

        if self.calc_sig:
            dat_ttest = [
                    (
                        self.datLazy_raw
                        .select(pl.col(num), pl.col(self.group_col))
                        .filter(pl.col(self.group_col)==self.groups_uniq[0])
                        .select(pl.col(num))
                        .filter(pl.col(num).is_not_null())
                        .collect()
                        .get_column(num)
                        .to_numpy()
                        ),
                    (
                        self.datLazy_raw
                        .select(pl.col(num), pl.col(self.group_col))
                        .filter(pl.col(self.group_col)==self.groups_uniq[1])
                        .select(pl.col(num))
                        .filter(pl.col(num).is_not_null())
                        .collect()
                        .get_column(num)
                        .to_numpy()
                        ),
                    ]
            p = ttest_ind(*dat_ttest).pvalue

            dat_num = dat_num.with_columns(
                    pvalue=pl.lit(p),
                    )

        return dat_num


    def formatCsv_catg(self, group,):
        """
        Formats the analysis results of a category into CSV format.

        Parameters:
        ----------
        group : str
            The category to format.

        Returns:
        -------
        list
            A list of formatted CSV lines for the given category.
        """
        out = [[], []]
        subCatgs = [x for x in self.tb1[group][group].unique().to_list() if x is not None]

        for i, expstatus_ in enumerate(self.groups_uniq):
            add_missing=True

            for sc_ in subCatgs:
                sc_dat = self.tb1[group].filter(pl.col(group).eq(sc_) & pl.col(self.group_col).eq(expstatus_))

                #if category doesn't exist in either exposure or control
                if sc_dat.shape[0] == 0:
                    out_ = [
                            "", #group
                            str(sc_), #subgroup
                            "", #missing
                            f"0 (0.0%)", #count
                            ]

                else:
                    out_ = [
                            "", #group
                            str(sc_), #subgroup
                            "", #missing
                            f"{sc_dat['n'][0]} ({round(sc_dat['percent'][0], self.rnd)}%)", #count
                            ]

                # needs amending
                    # currently, if category non-existent in a group,
                    # will skip and add missing to the 1st existent group.
                    # Works, but will lead to missing not being in 1st row
                    # NOTE: Fixed below; doesn't work if one is missing from 1 group but not the other
                if add_missing:
                    n_missing_ = (
                            self.tb1[group]
                            .filter(
                                pl.col(self.group_col).eq(expstatus_)
                                )
                            .get_column("n_missing")
                            .to_list()[0]
                            )
                    group_total_ = (
                            self.tb1[group]
                            .filter(
                                pl.col(self.group_col).eq(expstatus_)
                                )
                            .get_column("group_total")
                            .to_list()[0]
                            )
                    if self.calc_sig:
                        pvalue_ = (
                                self.tb1[group]
                                .filter(
                                    pl.col(self.group_col).eq(expstatus_)
                                    )
                                .get_column("pvalue")
                                .to_list()[0]
                                )

                    out_[0] = str(group)
                    missing_percent_ = (n_missing_/group_total_)*100
                    out_[2] = f"{n_missing_} ({round(missing_percent_, self.rnd)}%)"
                    if self.calc_sig:
                        out_.append(str(round(pvalue_, self.rnd)))

                    add_missing=False

                elif self.calc_sig:
                    out_.append("")

                #only need group info for one of the exposure groups
                if i==1:
                    out_ = out_[2:]
                else:
                    if self.calc_sig:
                        out_ = out_[:-1] #only need pvalue for one exposure group

                out[i].append(self.delim.join(out_))

        if len(out[1]) == 0:
            out = out[0]
        elif len(out[1]) != 0:
            out = [",".join([out[0][i],out[1][i]]) for i in range(0,len(out[0]),1)]

        return out


    def formatCsv_num(self, group,):
        """
        Formats the analysis results of a continuous variable into CSV format.

        Parameters:
        ----------
        group : str
            The category to format.

        Returns:
        -------
        list
            A list of formatted CSV lines for the given category.
        """
        out = [[], []]

        for i, expstatus_ in enumerate(self.groups_uniq):
            add_missing=True

            for sc_ in [["mean", "std"], ["median", "min", "max"]]:
                #sc_dat = self.tb1[group].select(sc_ + [self.group_col, "n_missing", "pvalue"]).filter(pl.col(self.group_col).eq(expstatus_))
                sc_dat = self.tb1[group].filter(pl.col(self.group_col).eq(expstatus_))
                out_ = [
                        "", #group
                        " ".join(sc_), #subgroup
                        "", #missing
                        " ".join([str(round(sc_dat[x][0], self.rnd)) for x in sc_]), #count
                        ]

                if add_missing:
                    group_total_ = (
                            self.tb1[group]
                            .filter(
                                pl.col(self.group_col).eq(expstatus_)
                                )
                            .get_column("group_total")
                            .to_list()[0]
                            )

                    out_[0] = str(group)
                    out_[2] = f"{sc_dat['n_missing'][0]} ({round((sc_dat['n_missing'][0]/group_total_)*100, self.rnd)}%)"
                    if self.calc_sig:
                        out_.append(str(round(sc_dat['pvalue'][0], self.rnd)))

                    add_missing=False

                elif self.calc_sig:
                    out_.append("")

                #only need group info for one of the exposure groups
                if i==1:
                    out_ = out_[2:]
                else:
                    if self.calc_sig:
                        out_ = out_[:-1] #only need pvalue for one exposure group

                out[i].append(self.delim.join(out_))

        if len(out[1]) == 0:
            out = out[0]
        elif len(out[1]) != 0:
            out = [",".join([out[0][i],out[1][i]]) for i in range(0,2,1)]

        return out


    def write_csv(self, outFile,):
        """
        Writes the analysis results to a CSV file.

        Parameters:
        ----------
        outFile : str
            Path to the output CSV file.
        """
        file = open(outFile, "w",)

        col_labels = [
                'Group',
                'Subgroup',
                'Missing',
                'Count',
                ]

        if self.calc_sig:
            file.write(f",,Group_{self.groups_uniq[0]},,Group_{self.groups_uniq[1]},,\n")
            col_labels = col_labels + ["Missing", "Count", "pvalue",]

        file.write(f"{self.delim.join(col_labels)}\n")

        for c_ in self.catgs:
            for line in self.formatCsv_catg(c_):
                file.write(f"{line}\n")

        for n_ in self.nums:
            for line in self.formatCsv_num(n_):
                file.write(f"{line}\n")

        file.close()


if __name__=="__main__":
    ## Load config
    with open("wdir.yml",
              "r",
              encoding="utf8") as file_config:
        config = yaml.safe_load(file_config)

    PATH = config["PATH"]
    DIR_DATA = f"{PATH}{config['dir_data']}"
    file_dat = f"{DIR_DATA}{config['incprev']['filename']}"
    ##

    catgs = [
            "AGE_CATEGORY",
            "AGE_CATEGORY_broader",
            "SEX",
            "IMD_pracid",
            "IMD_pracid_threeBin",
            "REGION",
            "ETHNICITY",
            ]

    null_values = [
            "MISSING",
            "",
            "Scotland_imd",
            "Wales_imd",
            "Northern Ireland_imd",
            ]

    tb1 = table1_polars(
            file_dat,
            catgs,
            null_values,
            )
    tb1.write_csv("out/table1.csv")
