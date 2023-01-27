from datetime import datetime
import numpy as np
import pandas as pd

class Data_Preprocess:
    """
    Raw data is aggregated over specified time window (or not if not defined,
    and each time window will be represented by mean/std-dev (if aggregation type was provided).
    """

    def __init__(self, aggregate_type='', aggregate_amount=1):
        """
        Initialize and define the aggregation type and amount.
        :param aggregate_type: 's' = seconds , 'min' = minutes , 'H' = hours , 'D' = days
        :param aggregate_amount: positive integer number
        """
        self.df = None
        self.dfs = []
        if type(aggregate_type) != str or type(aggregate_amount) != int or \
                aggregate_type not in ['s', 'min', 'H', 'D']:
            print("Aggregation type not recognized, set to no aggregation.")
            self.agg_str = ''
        else:
            self.agg_str = str(aggregate_amount) + aggregate_type

    def load_data(self, filename, delimiter):
        """
        Load a text data file into memory and prepare column names, and the aggregation from raw if defined.
        :param filename: including path if in other path
        :param delimiter: the delimiter character, usually a '\t' or ','
        :return:
        """
        df = pd.read_csv(filename, delimiter=delimiter)
        print(f"File loaded with {len(df)} rows.")
        all_cols = True
        needed_cols = ['RUN_START_DATE', 'Equip', 'Feature', 'PREP_VALUE']
        cols = list(df.columns)
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
                print("Equal '' !!!")
                df.insert(0, 'time', df['RUN_START_DATE'])
            else:
                print("Aggregating !!!")
                df.insert(0, 'time', df['RUN_START_DATE'].dt.floor(self.agg_str))
            df['mean'] = df.groupby(['Equip', 'Feature'], as_index=False)['PREP_VALUE'].transform('mean')
            df['std'] = df.groupby(['Equip', 'Feature'], as_index=False)['PREP_VALUE'].transform('std')
            df['norm'] = (df['PREP_VALUE'] - df['mean']) / df['std']
            df = df.drop(['PREP_VALUE', 'mean', 'std'], axis=1)
            df = df.groupby(['time', 'RUN_START_WW', 'Equip', 'Feature'], as_index=False)['norm']. \
                agg(['mean', 'std']).reset_index().fillna(0)
            df = df.melt(id_vars=['time', 'RUN_START_WW', 'Equip', 'Feature'], value_vars=['mean', 'std'])
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
        for ukey in uniq_keys:
            print(f"Preparing key series {ukey} ...")
            ddf = self.df[self.df['key'] == ukey]
            ddf = ddf.sort_values(by=['series', 'Equip', 'time'], ascending=[True, True, True])
            ddf = ddf.reset_index(drop=True)
            ddf_list = []
            self.recur_split_series_no_multi_clusters(ukey, ddf, ddf_list, ave_size=20, threshold=1.8)
            for ddfl in ddf_list:
                ddfl = ddfl.sort_values(by=['series', 'Equip', 'time'], ascending=[True, True, True])
                ddfl = ddfl.reset_index(drop=True)
                m = np.mean(ddfl['value'])
                s = np.std(ddfl['value'])
                ddfl['norm'] = (ddfl['value'] - m) / s
                ddfl = ddfl.drop(['value'], axis=1)
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
        for i in range(ave_size, len(df) - ave_size):
            k1 = np.mean(df[(i - ave_size + 1):i]['value'].values)
            k2 = np.mean(df[i:(i + ave_size - 1)]['value'].values)
            d = abs(k1 - k2)
            if d > delta:
                delta = d
                ki = i
        return delta, ki