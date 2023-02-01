from datetime import datetime
import numpy as np
import pandas as pd
from math import floor
import io

class Data_Preprocess:
    """
    Raw data is aggregated over specified time window (or not if not defined,
    and each time window will be represented by mean/std-dev (if aggregation type was provided).
    """

    def __init__(self, aggregate_type='', aggregate_amount=1, threshold_hours=12, feature_num=None):
        """
        Initialize and define the aggregation type and amount.
        :param aggregate_type: 's' = seconds , 'min' = minutes , 'H' = hours , 'D' = days
        :param aggregate_amount: positive integer number
        """
        self.df = None
        self.dfs = []
        self.fname = None
        self.feature_n = feature_num
        self.th_hours = threshold_hours
        self.agg_type = aggregate_type
        self.agg_cnt = aggregate_amount
        self.agg_type_list = ['s', 'min', 'H', 'D']
        self.agg_type_dict_hours = {'s': 3600, 'min': 60, 'H': 1, 'D': 1.0/24.0}  # points per hour
        if type(aggregate_type) != str or type(aggregate_amount) != int or \
                aggregate_type not in self.agg_type_list:
            print("Aggregation type not recognized, set to no aggregation.")
            self.agg_str = ''
            self.agg_type = ''
        else:
            self.agg_str = str(aggregate_amount) + aggregate_type

    def load_data(self, filename, delimiter, load_all=True):
        """
        Load a text data file into memory and prepare column names, and the aggregation from raw if defined.
        :param load_all: to read all lines from file or specific series per self.feature_n.
        :param filename: including path if in other path
        :param delimiter: the delimiter character, usually a '\t' or ','
        :return:
        """
        self.fname = filename
        if load_all or self.feature_n is None:
            df = pd.read_csv(filename, delimiter=delimiter)
        else:
            ser = 'Feature' + str(self.feature_n)
            with open(filename) as f:
                headers = f.readline() + "\n"
                text = "\n".join([line for line in f if ser in line])
                text = headers + text
            df = pd.read_csv(io.StringIO(text),  delimiter=delimiter)
        fns = filename.rsplit("\\", 1)
        if len(fns) == 1:
            fname_ = fns
        else:
            fname_ = fns[1]
        print(f"File {fname_} loaded with {len(df)} rows.")
        all_cols = True
        needed_cols = ['RUN_START_DATE', 'Equip', 'Feature', 'PREP_VALUE']
        cols = list(df.columns)
        print(f"DF columns: {cols}")
        quit()
        miss_cols = []
        for col in needed_cols:
            if col not in cols:
                all_cols = False
                miss_cols.append(col)
        if len(miss_cols) > 0:
            print("Could not find the following columns: " + ",".join(miss_cols))
        if all_cols:
            df['RUN_START_DATE'] = pd.to_datetime(df['RUN_START_DATE'])
            df = df.sort_values('RUN_START_DATE')
            if self.agg_str == '':
                df.insert(0, 'time', df['RUN_START_DATE'])
            else:
                df.insert(0, 'time', df['RUN_START_DATE'].dt.floor(self.agg_str))
            df = df.drop(['RUN_START_DATE'], axis=1)
            # df['mean'] = df.groupby(['Equip', 'Feature'], as_index=False)['PREP_VALUE'].transform('mean')
            # df['std'] = df.groupby(['Equip', 'Feature'], as_index=False)['PREP_VALUE'].transform('std')
            # df['norm'] = (df['PREP_VALUE'] - df['mean']) / df['std']
            # df = df.drop(['PREP_VALUE', 'mean', 'std'], axis=1)
            if self.agg_str == '':
                df = df.groupby(['time', 'RUN_START_WW', 'Equip', 'Feature'], as_index=False)['PREP_VALUE']. \
                    agg(['mean']).reset_index().fillna(0)
                df = df.rename(columns={'mean': 'value'})
                # df = df.melt(id_vars=['time', 'RUN_START_WW', 'Equip', 'Feature'], value_vars=['mean'])
            else:
                df = df.groupby(['time', 'RUN_START_WW', 'Equip', 'Feature'], as_index=False)['PREP_VALUE'].\
                    agg(['mean', 'std']).reset_index().fillna(0)
                df = df.melt(id_vars=['time', 'RUN_START_WW', 'Equip', 'Feature'], value_vars=['mean', 'std'])
                # results in column name 'value'
            df['series'] = df['Feature'] + "_" + df['variable']
            df = df.drop(['Feature', 'variable'], axis=1)
            self.df = df

    def prepare_series(self):
        """
        analyze the data, split data at high change point if exists, and normalize (Z norm)
        :return:
        """
        s_time = datetime.now()
        self.df['key'] = self.df['Equip'] + "_" + self.df['series']
        uniq_keys = self.df['key'].unique()
        dfs = []
        fns = self.fname.rsplit("\\", 1)
        if len(fns) == 1:
            fname_ = fns
        else:
            fname_ = fns[1]
        for ukey in uniq_keys:
            print(f"Preparing key series {ukey} from file {fname_}...")
            ddf = self.df[self.df['key'] == ukey]
            ddf = ddf.sort_values(by=['series', 'Equip', 'time'], ascending=[True, True, True])
            ddf = ddf.reset_index(drop=True)
            ddf = self.znorm_df(ddf)
            ddf_list = []
            if self.agg_type != '':
                # number of points for threshold
                points = floor(self.agg_type_dict_hours[self.agg_type] * self.th_hours / self.agg_cnt)
            else:  # assuming raw data is per second
                points = floor(self.th_hours * 3600)  # amount of seconds in threshold window size
            self.recur_split_series_no_multi_clusters(ukey, ddf, ddf_list, ave_size=points, threshold=1.8)
            for ddfl in ddf_list:
                ddfl = ddfl.sort_values(by=['series', 'Equip', 'time'], ascending=[True, True, True])
                ddfl = ddfl.reset_index(drop=True)
                ddfl = self.znorm_df(ddfl)
                dfs.append(ddfl)
        self.dfs = dfs
        e_time = datetime.now()
        d_time = e_time - s_time
        print(f"Data preparation run time {d_time}")

    def recur_split_series_no_multi_clusters(self, ukey, df, ddf_list, ave_size=10, threshold=1.5):
        delta, indx = self.get_series_split_max_distance(df, ave_size=ave_size)
        if delta > threshold:
            print(f"Found two clusters or more and need split, in dataset {ukey}, indx {indx}, delta {delta}.")
            self.recur_split_series_no_multi_clusters(ukey, df[indx:], ddf_list, ave_size, threshold)
            self.recur_split_series_no_multi_clusters(ukey, df[:indx], ddf_list, ave_size, threshold)
        ddf_list.append(df)

    def get_series_split_max_distance(self, df, ave_size=10):
        delta = 0
        ki = 0
        for i in range(ave_size, (len(df) - ave_size)):
            k1 = np.mean(df[(i - ave_size + 1):i]['value'].values)
            k2 = np.mean(df[i:(i + ave_size - 1)]['value'].values)
            d = abs(k1 - k2)
            if d > delta:
                delta = d
                ki = i
        return delta, ki

    def znorm_df(self, df, val_name='value'):
        tmp_name = val_name + '2'
        df = df.rename(columns={val_name: tmp_name})
        m = np.mean(df[tmp_name])
        s = np.std(df[tmp_name])
        if s > 0:
            df[val_name] = (df[tmp_name] - m) / s
        else:
            df[val_name] = df[tmp_name] - m
        df = df.drop([tmp_name], axis=1)
        return df
