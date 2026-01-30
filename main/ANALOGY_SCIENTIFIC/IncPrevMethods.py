import re
import warnings
import pandas as pd
import numpy as np
import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2
from scipy.special import ndtri
from dateutil.relativedelta import relativedelta
from csv import DictReader
from math import sqrt

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
    |    raw_data (pd.DataFrame): The data used to calculate incidence rate and point prevalence.

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

    __slots__ = 'PER_PY', 'SMALL_FP_VAL', 'DAYS_IN_YEAR', 'ALPHA', \
        'STUDY_END_DATE', 'STUDY_START_DATE', 'INCREMENT_BY_MONTH', \
        'BASELINE_DATE_LIST', 'DEMOGRAPHY', 'FILENAME', 'raw_data', \
        'DATABASE_NAME'

    def __init__(self, \
                 STUDY_END_DATE, STUDY_START_DATE, \
                 FILENAME, DATABASE_NAME, \
                 BASELINE_DATE_LIST=[], DEMOGRAPHY=[], \
                 cols = None, skiprows_i = None, \
                 read_data = True, SMALL_FP_VAL=1e-8, DAYS_IN_YEAR=365.25, \
                 PER_PY=100_000, ALPHA=0.05, INCREMENT_BY_MONTH=12, \
                 fileType = None):

        self.PER_PY, self.SMALL_FP_VAL, self.DAYS_IN_YEAR, self.ALPHA,\
        self.STUDY_END_DATE, self.STUDY_START_DATE, self.INCREMENT_BY_MONTH, \
        self.BASELINE_DATE_LIST, self.DEMOGRAPHY, self.FILENAME, \
        self.DATABASE_NAME = \
        PER_PY, SMALL_FP_VAL, DAYS_IN_YEAR, ALPHA, \
        STUDY_END_DATE, STUDY_START_DATE, INCREMENT_BY_MONTH, \
        BASELINE_DATE_LIST, DEMOGRAPHY, FILENAME,\
        DATABASE_NAME

        self.STUDY_END_DATE += relativedelta(years=0, months=0, days=1)
        if read_data == True:
            self.read(cols, skiprows_i, fileType)
        else:
            self.raw_data = None

    def read(self, cols, skiprows_i = [], fileType="csv"):
        if fileType=="csv":
            self.raw_data = pd.read_csv(self.FILENAME, usecols = cols, skiprows=skiprows_i)
        elif fileType == "parquet":
            print("skiprows_i attribute not used")
            self.raw_data = pd.read_parquet(self.FILENAME, columns = cols)

         # This logic is IMRD database specific. REGISTRATION_STATUS with value 99 in IMRD database means patient is dead.
         # To ensure records are properly updated the DEATH_DATE for those patients whose REGISTRATION_STATUS == 99 are
         # set to the TRANSFER_DATE.

        if self.DATABASE_NAME == "IMRD":
             self.raw_data.loc[((pd.isnull(self.raw_data['DEATH_DATE'])) & (self.raw_data['REGISTRATION_STATUS'] == 99)),
                               'DEATH_DATE'] = self.raw_data.loc[((pd.isnull(self.raw_data['DEATH_DATE'])) & (self.raw_data['REGISTRATION_STATUS'] == 99)),
                                                            'TRANSFER_DATE']

         # When data is loaded from file pandas reads date column as strings. We convert the columns to datetime type.
        self.raw_data['INDEX_DATE'] = pd.to_datetime(self.raw_data['INDEX_DATE'])
        self.raw_data['START_DATE'] = pd.to_datetime(self.raw_data['START_DATE'])
        self.raw_data['END_DATE'] = pd.to_datetime(self.raw_data['END_DATE'])
        self.raw_data['COLLECTION_DATE'] = pd.to_datetime(self.raw_data['COLLECTION_DATE'])
        self.raw_data['DEATH_DATE'] = pd.to_datetime(self.raw_data['DEATH_DATE'])

        if len(self.BASELINE_DATE_LIST) == 0:
             self.BASELINE_DATE_LIST = [col for col in self.raw_data.columns if col.startswith('BD_')]

        if len(self.DEMOGRAPHY)  == 0:
             self.DEMOGRAPHY = list(set(self.raw_data.columns).intersection(set(['AGE_CATG', 'SEX', 'ETHNICITY', 'COUNTRY',
                                                                        'HEALTH_AUTH', 'TOWNSEND'])))

        for baseline_col in self.BASELINE_DATE_LIST:
             self.raw_data[baseline_col] = pd.to_datetime(self.raw_data[baseline_col])

          # Only run if the database is IMRD.
        if self.DATABASE_NAME == "IMRD":
             self.raw_data['END_DATE'] = self.raw_data[['DEATH_DATE', 'END_DATE']].min(axis=1)

    def byars_lower(self, count, denominator, return_limits=False): #return_limit change name return_freq
        if count < 10:
            b = chi2.ppf((self.ALPHA / 2), (count * 2)) / 2
            if return_limits:
                return b
            else:
                lower_ci = b / denominator
                return lower_ci
        else:
            z = ndtri(1 - self.ALPHA / 2)
            c = 1 / (9 * count)
            b = 3 * np.sqrt(count)
            lower_o = count * ((1 - c - (z / b)) ** 3)
            if return_limits:
                return lower_o
            else:
                lower_ci = lower_o / denominator
                return lower_ci

    def byars_higher(self, count, denominator, return_limits=False):
        if count < 10:
            b = chi2.ppf(1 - (self.ALPHA / 2), 2 * count + 2) / 2
            if return_limits:
                return b
            else:
                upper_ci = b / denominator
                return upper_ci
        else:
            z = ndtri(1 - self.ALPHA / 2)
            c = 1 / (9 * (count + 1))
            b = 3 * (np.sqrt(count + 1))
            upper_o = (count + 1) * ((1 - c + (z / b)) ** 3)
            if return_limits:
                return upper_o
            else:
                upper_ci = upper_o / denominator
                return upper_ci


    def save_dataframe_inc(self, data_list, filename, sub_group="OVERALL",
                           path_out="./"):
        if sub_group == "":
            df = pd.DataFrame(data_list, columns =['Year', 'Incidence', 'PersonYears', 'Numerator', 'Lower', 'Upper'])
        elif len(data_list[0]) < 7: #where sub_group is one constant level
            data_list = [[x[0], sub_group] + list(x[1:]) for x in data_list]
            df = pd.DataFrame(data_list, columns =['Year', sub_group, 'Incidence', 'PersonYears', 'Numerator', 'Lower', 'Upper'])
        else:
            df = pd.DataFrame(data_list, columns =['Year', sub_group, 'Incidence', 'PersonYears', 'Numerator', 'Lower', 'Upper'])
        df.to_csv(f"{path_out}{filename}", index=False)
        return df

    def save_dataframe_prev(self, data_list, filename, sub_group="OVERALL",
                            path_out="./"):
        if sub_group == "":
            df = pd.DataFrame(data_list, columns =['Year', 'Prevalence', 'Denominator', 'Numerator', 'Lower', 'Upper'])
        elif len(data_list[0]) < 7: #where sub_group is one constant level
            data_list = [[x[0], sub_group] + list(x[1:]) for x in data_list]
            df = pd.DataFrame(data_list, columns =['Year', sub_group, 'Prevalence', 'Denominator', 'Numerator', 'Lower', 'Upper'])
        else:
            df = pd.DataFrame(data_list, columns =['Year', sub_group, 'Prevalence', 'Denominator', 'Numerator', 'Lower', 'Upper'])
        df.to_csv(f"{path_out}{filename}", index=False)
        return df

################################################################################
    def get_numerator_filter_prev(self, dataframe, study_year, datecol_name):
        """
        Filter function for prevalence numerator.

        Args:
            dataframe: the full or grouped pandas dataframe.
            study_year: datetime value to calculate point prevalence on.
            datecol_name: baseline variable column name.

        Return:
            numpy array boolean masks for eligible patients.
        """
        return np.where((
            (dataframe['INDEX_DATE'] <= study_year) &  # Patient follow-up began before or on the start year for the analysis.
            (dataframe['END_DATE'] >= study_year)) & # The patient end date occurred on or after the start year for the analysis.
            (dataframe[datecol_name] <= study_year)) # The event date occurred before or on the start year for the analysis.

    def get_denominator_filter_prev(self, dataframe, study_year):
        """
        Filter function for prevalence denominator.

        Args:
            dataframe: the full or grouped pandas dataframe.
            study_year: datetime value to calculate point prevalence on.

        Return:
            numpy array boolean masks for eligible population.
        """
        return np.where(
            (dataframe['INDEX_DATE'] <= study_year) & # Patient follow-up began before or on the start year for the analysis.
            (dataframe['END_DATE'] >= study_year)) # The patient end date occurred on or after the start year for the analysis.

    def point_prevalence(self, dataframe, study_year, datecol_name, sub_group=""):
        """
        Function definition for point pervalence calculation.

        Args:
            dataframe: the full or grouped pandas dataframe.
            study_year: datetime value to calculate point prevalence on.
            datecol_name: baseline variable column name.

        Return:
            tuple (year, prevalence proportion, denominator, numerator, lower_ci, upper_ci, error_delta)

        """
        # event that occured before the year of interest which is a combination of outcome recording that is
        # before that year of interest
        numerator_mask = self.get_numerator_filter_prev(dataframe, study_year, datecol_name)
        numerator_count = len(dataframe.iloc[numerator_mask])

        # Patients who are in the practice at the start of the interested year, that is they enter the cohort
        # before the start of the interested year and have not exited before the start of the interested year
        denominator_mask = self.get_denominator_filter_prev(dataframe, study_year)
        denominator_count = len(dataframe.iloc[denominator_mask])

        denominator_count = denominator_count + self.SMALL_FP_VAL # adding a small constant to avoid division by zero.
        point_prev = (numerator_count/denominator_count)*self.PER_PY

        lower_ci = self.byars_lower(numerator_count, denominator_count)
        upper_ci = self.byars_higher(numerator_count, denominator_count)

        if sub_group=="":
            return (study_year.date(), point_prev, int(denominator_count), numerator_count, lower_ci*self.PER_PY, upper_ci*self.PER_PY)
        else:
            return (study_year.date(), sub_group, point_prev, int(denominator_count), numerator_count,
                    lower_ci*self.PER_PY, upper_ci*self.PER_PY)


    def get_numerator_filter_inc(self, dataframe, start_yr, end_yr, datecol_name):
        """
        Filter function for incidence numerator.

        Args:
            dataframe: the full or grouped pandas dataframe.
            start_yr: datetime value to start calculating incidence on.
            end_yr: datetime value to end calculating incidence on.
            datecol_name: baseline variable column name.

        Return:
            numpy array boolean masks for eligible patients.
        """
        return np.where((
            # The event date occurred on or after the start year but before the end year.
            dataframe[datecol_name].between(start_yr, end_yr, inclusive='left')) &
            ((dataframe['END_DATE'] >= start_yr) & # The patient end date occurred on or after the start date for the analysis.
             (dataframe['INDEX_DATE'] < end_yr)) & # Patient follow-up occurred before the end year.
            (dataframe[datecol_name] > dataframe['INDEX_DATE'])) # The event date occurred after patient follow up began.

    def get_denominator_filter_inc(self, dataframe, start_yr, end_yr, datecol_name):
        """
        Filter function for incidence denominator.

        Args:
            dataframe: the full or grouped pandas dataframe.
            start_yr: datetime value to start calculating incidence on.
            end_yr: datetime value to end calculating incidence on.
            datecol_name: baseline variable column name.

        Return:
            numpy array boolean masks for eligible population.
        """
        return np.where((
            (dataframe['END_DATE'] >= start_yr) & # The patient end date occurred on or after the start date for the analysis.
            (dataframe['INDEX_DATE'] < end_yr)) & # Patient follow-up occurred before the end year.
            ((dataframe[datecol_name] >= start_yr) & # The event date occurred on or after the start date for the analysis.

             # The event date occurred after patient follow up began or there was no event.
             (dataframe[datecol_name] > dataframe['INDEX_DATE']) |
             (dataframe[datecol_name].isna())))

    def get_start_period(self, dataframe, start_yr):
        """
        Function to get the start period when the patient starts contributing to the study.

        Args:
            dataframe: the eligible population pandas dataframe.
            start_yr: datetime value to start calculating incidence on.

        Return:
            pandas series with datetime object.

        """
        return dataframe['INDEX_DATE'].where(dataframe['INDEX_DATE'] > start_yr, start_yr)

    def get_end_period(self, dataframe, end_yr, datecol_name):
        """
        Function to get the end period when the patient stops contributing to the study.

        Args:
            dataframe: the eligible population pandas dataframe.
            end_yr: datetime value to end calculating incidence on.
            datecol_name: baseline variable column name.

        Return:
            pandas series with datetime object.

        """
        end_period = dataframe[['END_DATE', datecol_name]].min(axis=1)
        return end_period.where(end_period < end_yr, end_yr)



    def point_incidence(self, dataframe, start_yr, end_yr, datecol_name, sub_group=""):
        """
        Function definition for point incidence calculation.

        Args:
            dataframe: the full or grouped pandas dataframe.
            start_yr: datetime value to start calculating incidence on.
            end_yr: datetime value to end calculating incidence on.
            datecol_name: baseline variable column name.

        Return:
            tuple (year, incidence rate, denominator, numerator, lower_ci, upper_ci, error_delta, count)

        """
        numerator_mask = self.get_numerator_filter_inc(dataframe, start_yr, end_yr, datecol_name)
        numerator_count = len(dataframe.iloc[numerator_mask])


        denominator_mask = self.get_denominator_filter_inc(dataframe, start_yr, end_yr, datecol_name)
        denominator_df = dataframe.iloc[denominator_mask]


        start_period = self.get_start_period(denominator_df, start_yr)
        end_period = self.get_end_period(denominator_df, end_yr, datecol_name)

        person_years = (end_period - start_period).dt.days

        # if patient start date and end date are same add 1 day of contribution.
        person_years = person_years.replace(0, 1)

        delta = end_yr - start_yr

        person_years = person_years/delta.days

        denominator_time = person_years.sum() + self.SMALL_FP_VAL

        point_inc = (numerator_count/denominator_time)*self.PER_PY

        lower_ci = self.byars_lower(numerator_count, denominator_time)
        upper_ci = self.byars_higher(numerator_count, denominator_time)

        if sub_group=="":
            return (start_yr.date(), point_inc, denominator_time, numerator_count, lower_ci*self.PER_PY, upper_ci*self.PER_PY)
        else:
            return (start_yr.date(), sub_group, point_inc, denominator_time, numerator_count, lower_ci*self.PER_PY, upper_ci*self.PER_PY)


    def calculate_prevalence(self, sub_name="_Prev", path_out="./"):
        year_start = self.raw_data['START_DATE'].dt.date.min().year
        year_end = self.raw_data['END_DATE'].dt.date.max().year+1
        for datecol_name in tqdm(self.BASELINE_DATE_LIST):
            prev_list = []
            current_period = self.STUDY_START_DATE
            while current_period < self.STUDY_END_DATE:
                delta = relativedelta(years=0, months=self.INCREMENT_BY_MONTH, days=0)
                year_tuple = self.point_prevalence(self.raw_data, current_period, datecol_name)
                prev_list.append(year_tuple)
                current_period += delta

            filename = datecol_name.replace("BD_MEDI:", '').replace("_BIRM_CAM", '')
            filename = re.sub('[^A-Za-z0-9]+', '', filename)
            df = self.save_dataframe_prev(prev_list,
                                          filename + "_OVERALL" + sub_name + ".csv",
                                          path_out=path_out)



    def calculate_grouped_prevalence(self, sub_name = "_Prev", path_out="./"):
        year_start = self.raw_data['START_DATE'].dt.date.min().year
        year_end = self.raw_data['END_DATE'].dt.date.max().year+1
        for datecol_name in tqdm(self.BASELINE_DATE_LIST):
            for demo in self.DEMOGRAPHY:
                prev_list = []
                for name, group in self.raw_data.groupby(demo):
                    current_period = self.STUDY_START_DATE
                    while current_period < self.STUDY_END_DATE:
                        delta = relativedelta(years=0, months=self.INCREMENT_BY_MONTH, days=0)
                        year_tuple = self.point_prevalence(group, current_period, datecol_name, name)
                        prev_list.append(year_tuple)
                        current_period += delta
                filename = datecol_name.replace("BD_MEDI:", '').replace("_BIRM_CAM", '')
                filename = re.sub('[^A-Za-z0-9]+', '', filename)
                if type(demo) is list:
                    self.save_dataframe_prev(prev_list,
                                             filename + "_" + str(demo)[1:-1] + sub_name + ".csv",
                                             str(demo)[1:-1],
                                             path_out=path_out)
                else:
                    self.save_dataframe_prev(prev_list, filename + "_" + str(demo) + sub_name + ".csv",
                                             str(demo),
                                             path_out=path_out)


    def calculate_incidence(self, sub_name="_Inc", path_out="./"):
        year_start = self.raw_data['START_DATE'].dt.date.min().year
        year_end = self.raw_data['END_DATE'].dt.date.max().year+1
        for datecol_name in tqdm(self.BASELINE_DATE_LIST):
            prev_list = []
            current_period = self.STUDY_START_DATE
            while current_period < self.STUDY_END_DATE:
                delta = relativedelta(years=0, months=self.INCREMENT_BY_MONTH, days=0)
                end_period = min(self.STUDY_END_DATE, current_period + delta)
                year_tuple = self.point_incidence(self.raw_data, current_period, end_period, datecol_name)
                prev_list.append(year_tuple)
                current_period += delta

            filename = datecol_name.replace("BD_MEDI:", '').replace("_BIRM_CAM", '')
            filename = re.sub('[^A-Za-z0-9]+', '', filename)
            df = self.save_dataframe_inc(prev_list,
                                         filename + "_OVERALL" + sub_name +".csv",
                                         path_out=path_out)


    def calculate_grouped_incidence(self, sub_name = "_Inc", path_out="./"):
        year_start = self.raw_data['START_DATE'].dt.date.min().year
        year_end = self.raw_data['END_DATE'].dt.date.max().year+1
        for datecol_name in tqdm(self.BASELINE_DATE_LIST):
            for demo in self.DEMOGRAPHY:
                inc_list = []
                for name, group in self.raw_data.groupby(demo):
                    current_period = self.STUDY_START_DATE
                    while current_period < self.STUDY_END_DATE:
                        delta = relativedelta(years=0, months=self.INCREMENT_BY_MONTH, days=0)
                        end_period = min(self.STUDY_END_DATE, current_period + delta)
                        year_tuple = self.point_incidence(group, current_period, end_period, datecol_name, name)
                        inc_list.append(year_tuple)
                        current_period += delta
                filename = datecol_name.replace("BD_MEDI:", '').replace("_BIRM_CAM", '')
                filename = re.sub('[^A-Za-z0-9]+', '', filename)
                if type(demo) is list:
                    self.save_dataframe_inc(inc_list,
                                            filename + "_" + str(demo)[1:-1] + sub_name + ".csv",
                                            str(demo)[1:-1],
                                            path_out=path_out)
                else:
                    self.save_dataframe_inc(inc_list,
                                            filename + "_" + str(demo) + sub_name + ".csv",
                                            str(demo),
                                            path_out=path_out)



## Standardisation Feature ##############################

class StrdIncPrev(IncPrev):
    """
    Not currently developed for general use yet
    """
    #add attribute dict of units at each step
    def __init__(self,
            standard_breakdowns,
            col_condition: str = "Condition",
            col_category: str = "Group",
            col_group: str = "Subgroup",
            ):
        # only using IncPrev for CI methods, do don't need to initialise all
        # values
        super().__init__(None, None,
                None, None, None,
                read_data = False,)
        self.standard_breakdowns = standard_breakdowns

        self.condition_col = col_condition
        self.category_col = col_category
        self.group_col = col_group

        self.raw_data_prev = None
        self.raw_data_inc = None


    def getReference(self,
                     file="UK Age-Sex Pop Structure.csv",
                     bins = [0,16,30,40,50,60,70,80,115],
                     groups = ['0-16','17-30','31-40','41-50','51-60','61-70','71-80','81+'],
                    ):
        df_census = pd.read_csv(file)
        df_census = df_census[df_census.Age!='Total']
        df_census['Age Group'] = pd.cut(df_census.Age.replace('90+',90).astype(int),
                                         bins,
                                         right=False,
                                         labels=groups)


        df_census['Groups'] = (df_census['Age Group'].astype(str) + df_census.Sex.map({'Female':', F',
                           'Male':', M'}))

        self.ref_standardDenom = df_census[['Groups','Count']].groupby('Groups').sum()


    #%% Standardisation Functions
    def read_num_files(self,
                       condition,
                       standard_breakdown,
                       measure='Prevalence',
                       remove_intersex=True):
        """read file with numerators and denominators which can be used to create a standardised estimate"""
        def ignoreRow(x, labs):
            skip = False
            if x in labs:
                skip = True
            else:
                for lab in labs:
                    if str(x).find(f"'{lab}'") != -1:
                        skip=True
                        break
            return skip

        dtypes = {'Numerator': None,
                  measure: 'float64'}

        if measure == 'Prevalence':
            file_name_ending = "_Prev.csv"
            col_denom = "Denominator"

            dtypes[col_denom] = 'int64'
            dtypes['Numerator'] = 'int64'

        elif measure == 'Incidence':
            file_name_ending = "_Inc.csv"
            col_denom = "PersonYears"

            dtypes[col_denom] = 'float64'
            dtypes['Numerator'] = 'float64'

        else:
            raise 'Invalid measure of burden of disease'

        stand_subgroup_file = self.standard_breakdowns[standard_breakdown]

        if measure == "Prevalence":
            df = self.raw_data_prev
        elif measure == "Incidence":
            df = self.raw_data_inc

        df = df[
                (df["Group"]==stand_subgroup_file).to_numpy()\
                        & (df["Condition"]==condition).to_numpy()
                ]

        df = df[["Date", "Subgroup", measure, "Denominator",
                "Numerator", "Lower_CI", "Upper_CI", "std_group"]]

        df.columns = ['Date', self.group_col, measure, col_denom,
                      'Numerator', 'Lower', 'Upper', 'std_group']

        df = df.astype(dtypes)

        return df

    def standardise_subgroups_years(self, df:pd.DataFrame, measure):
        dict_years = dict()

        for i in df.Date.unique():
            #print(i)
            df_year = df[df.Date==i].copy()
#            df_year = self.split_subgroup_strings(df_year)
            df_year = df_year[df_year[self.group_col]!='Ireland'] #Ireland is not part of IMD
#            dict_years[i] = df_year.groupby([self.group_col]).apply(self.standardise_year, measure)
            dict_years[f"{i}_UpperCI"], dict_years[i] = self.dobsons_ci(df_year, True, measure, return_DSR=True)
            dict_years[f"{i}_LowerCI"] = self.dobsons_ci(df_year, False, measure)

        return dict_years

    def standardise_overall_years(self, df, measure):
        dict_years = dict()
        for i in df.Date.unique():
            df_year = df[df.Date==i].copy()
            #Not sure why std_group col has surrounding (), but below is a quick fix
            df_year["std_group"] = df_year["std_group"].apply(lambda x, rep1, rep2: x.replace(rep1, rep2), args=tuple(["(", ""]))
            df_year["std_group"] = df_year["std_group"].apply(lambda x, rep1, rep2: x.replace(rep1, rep2), args=tuple([")", ""]))
            #dict_years[i] = self.standardise_year(df_year, measure)
            dict_years[f"{i}_UpperCI"], dict_years[i] = self.dobsons_ci(df_year, True, measure, False, return_DSR=True)
            dict_years[f"{i}_LowerCI"] = self.dobsons_ci(df_year, False, measure, False)

        return dict_years

    def standardise_condition_results(self,
                                      condition,
                                      measure,
                                      overall = "Overall"):
        subgroup_dict = {}

        for i in [x for x in self.standard_breakdowns.keys() if x != "Overall"]:
            print(condition)
            df_subgroups = self.read_num_files(condition,
                                               i,
                                               measure=measure)
            subgroup_dict[i] = pd.DataFrame(self.standardise_subgroups_years(df_subgroups, measure))

        df_standardised = pd.concat(subgroup_dict,axis=0, names=[self.category_col, self.group_col])

        #overall
        df_subgroups = self.read_num_files(condition, "Overall", measure=measure)
        df_standardised.loc[(overall,overall),:] = pd.Series(self.standardise_overall_years(df_subgroups, measure))

        return df_standardised

    def standardise_all_conditions(self, measure='Prevalence'):
        disease_rates = {}

        if measure == "Prevalence":
            for cond_ in self.raw_data_prev[self.condition_col].unique():
                disease_rates[cond_] = self.standardise_condition_results(cond_, measure)
        else:
            for cond_ in self.raw_data_inc[self.condition_col].unique():
                disease_rates[cond_] = self.standardise_condition_results(cond_, measure)

        df_all_rates = pd.concat(disease_rates, names=[self.condition_col,
                self.category_col,
                self.group_col])

        def reformat(dat, measure):
            dat = dat.melt(ignore_index=False)
            cols = [re.sub(".*[:0-9:].(.*)","\\1",x) for x in list(dat.variable)]
            if measure == "Prevalence":
                cols = ["Prevalence" if x=="" else x for x in cols]
            elif measure == "Incidence":
                cols = ["Incidence" if x=="" else x for x in cols]

            dates = [re.sub("(.*[:0-9:]).*","\\1",x) for x in list(dat.variable)]

            dat.variable = cols
            dat["Year"] = dates

            dat = dat.pivot_table(columns="variable", values="value", index=[self.condition_col, "Year", self.category_col, self.group_col])

            return dat

        df_all_rates = reformat(df_all_rates, measure)

        df_all_rates = df_all_rates.reset_index(level=["Group", "Year"])

        return df_all_rates

    def calc_ci_group(self, df_group, denom_col, wi_map, upper, return_DSR=False,):
        """
        df_group: grouped dataframe; groups are specific to a group catg, with information across ref catgs

        out: upper or lower CIs across every ref-catg for a specific group
        """
        index = df_group["std_group"]
        df_group["std_group"] = df_group["std_group"].apply(lambda x, rep1, rep2: x.replace(rep1, rep2), args=tuple(["'", ""]))

        wi = np.vectorize(wi_map.get)(df_group["std_group"].values)
        if (wi==None).any(): #No standardisation group
            print("Cannot find Standardisation Group for...")
            print(df_group)
            if return_DSR:
                return np.NAN, np.NAN
            else:
                return np.NAN

        ni = df_group[denom_col].values

        Oi = df_group.Numerator.values
        #sum_wi = self.ref_standardDenom.Count.sum()
        sum_wi = sum(wi)

        if (ni==0).any() and (Oi==0).any():
            if return_DSR:
                return np.NAN, np.NAN
            else:
                return np.NAN

        DSR = (( 1/sum_wi ) * sum( (wi*Oi)/ni ) )
        DSR_var = (( 1/( sum_wi**2 )) * sum( ((wi**2)*Oi) / (ni**2) ))
        O = Oi.sum()
        O_var = O

        if upper:
            O_higher = self.byars_higher(O, sum(ni), return_limits=True)
            out = DSR + ( sqrt( (DSR_var / O_var) ) * (O_higher - O) )
        else:
            O_lower = self.byars_lower(O, sum(ni), return_limits=True)
            out = DSR + ( sqrt( (DSR_var / O_var) ) * (O_lower - O) )

        if return_DSR:
            #print(df_group)
            #print(pd.DataFrame({"grp": df_group["std_group"], "ni":ni, "Oi":Oi, "wi":wi, "sum_wi":[sum_wi]*len(wi)}))
            #breakpoint()
            return out, DSR
        else:
            return out

    def dobsons_ci(self, df_year, upper = True, measure="Prevalence", group=True, return_DSR=False):
        """
        df_year: entire dataframe for a single year
        """
        if measure == "Prevalence":
            denom_col = "Denominator"
        elif measure == "Incidence":
            denom_col = "PersonYears"
        else:
            raise("Invalid metric")

        wi_map = self.ref_standardDenom.Count.to_dict() #wi of each age-sex catg in df_year

        #remove null 'measure' values
       # df_year = df_year[(df_year[measure]).notna()]
        if group:
            out = df_year.groupby([self.group_col]).apply(self.calc_ci_group, denom_col, wi_map, upper, return_DSR,)

            if return_DSR:
                return out.apply(lambda x: x[0])*self.PER_PY, out.apply(lambda x: x[1])*self.PER_PY
            else:
                return out*self.PER_PY
        else:
            out = self.calc_ci_group(df_year, denom_col, wi_map, upper, return_DSR)
            if return_DSR:
                return out[0]*self.PER_PY, out[1]*self.PER_PY
            else:
                return out*self.PER_PY



