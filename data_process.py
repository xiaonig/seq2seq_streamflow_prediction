import os.path
from os.path import relpath

import numpy as np
import pandas as pd
from pandas import DatetimeIndex, DataFrame
from sklearn.preprocessing import StandardScaler

step_cut_total_df = pd.read_csv(relpath('datas/step_modes_total_2020.csv'), engine='c')


# 根据水文站点时间步长，生成归一化后水位数据，每条数据时间间隔相等
def df_resample_cut(cut_df: DataFrame, stcd: str):
    scaler = StandardScaler()
    time_str = (step_cut_total_df['STEP'][step_cut_total_df['STCD'] == int(stcd)]).values[0]
    time_list = ['0 days 00:04:00', '0 days 00:05:00', '0 days 00:06:00', '0 days 00:12:00', '0 days 00:15:00',
                 '0 days 00:30:00', '0 days 01:00:00']
    if time_str in time_list:
        freq = '1H'
        cut_df['TM'] = pd.to_datetime(cut_df['TM'])
        cut_df = cut_df.set_index('TM').resample(freq).interpolate().dropna()
        cut_df['Z'] = scaler.fit_transform(cut_df['Z'].values.reshape(-1, 1))
    elif time_str == '0 days 02:00:00':
        freq = '2H'
        cut_df['TM'] = pd.to_datetime(cut_df['TM'])
        cut_df = cut_df.set_index('TM').resample(freq).interpolate().dropna()
        cut_df['Z'] = scaler.fit_transform(cut_df['Z'].values.reshape(-1, 1))
    elif time_str == '0 days 06:00:00':
        freq = '6H'
        cut_df['TM'] = pd.to_datetime(cut_df['TM'])
        cut_df = cut_df.set_index('TM').resample(freq, origin='end').interpolate().dropna()
        cut_df['Z'] = scaler.fit_transform(cut_df['Z'].values.reshape(-1, 1))
    elif time_str == '1 days 00:00:00':
        freq = 'D'
        cut_df['TM'] = pd.to_datetime(cut_df['TM'])
        cut_df = cut_df.set_index('TM').resample(freq, origin='start').interpolate().dropna()
        cut_df['Z'] = scaler.fit_transform(cut_df['Z'].values.reshape(-1, 1))
    else:
        freq = '3D'
        scaler = StandardScaler()
        pass
    return cut_df, freq, scaler


# 生成输入数据，将每个时间点前72h或30d的归一化后水位数据组织到同一行，根据训练时间范围构建训练集或者测试集的X值
def gen_scale_X(date_times: DatetimeIndex, stcd, pre_index: int):
    train_test_dict = {}
    # 在这里将日期序列先转成6-10月汛期
    times = date_times.values
    nc_path = 'sta' + str(stcd) + '_' + str(date_times[0].year) + str(date_times[0].month) + str(
        date_times[0].day) + str(date_times[0].hour) + '_to_' + str(date_times[-1].year) + str(date_times[-1].month) + str(
        date_times[-1].day) + str(date_times[-1].hour) + '_' + str(pre_index) + 'h_scale.csv'
    # total_2020_csvs: 存放有20个水文站点的水位原始数据，均为2020年之后
    table_rsvr_path = os.path.join(relpath('datas/level_data/total_2020_csvs'), str(stcd) + '_rsvr_cut2020.csv')
    table_river_path = os.path.join(relpath('datas/level_data/total_2020_csvs'), str(stcd) + '_river_cut2020.csv')
    if os.path.exists(table_river_path):
        table = pd.read_csv(table_river_path, engine='c', parse_dates=True)
        table, freq, scaler = df_resample_cut(table, str(stcd))
    elif os.path.exists(table_rsvr_path):
        table = pd.read_csv(table_rsvr_path, engine='c', parse_dates=True)
        table, freq, scaler = df_resample_cut(table, str(stcd))
    else:
        table = pd.DataFrame()
        scaler = StandardScaler()
        freq = '3D'
    if os.path.exists(os.path.join('datas/level_data/train_test_scale_csvs', nc_path)):
        if os.path.getsize(os.path.join('datas/level_data/train_test_scale_csvs', nc_path)) > 256:
            train_df = pd.read_csv(os.path.join('datas/level_data/train_test_scale_csvs', nc_path), engine='c')
        else:
            train_df = pd.DataFrame()
    else:
        slice_array = table.index.values
        for predict_time in times.flatten():
            if predict_time not in slice_array:
                continue
            else:
                time_index = np.argwhere(slice_array == predict_time)[0][0]
                if time_index < pre_index:
                    if freq != 'D':
                        post_hour_array = table['Z'].values[0: time_index]
                        pre_hour_array = np.full(pre_index - time_index, post_hour_array[0])
                        hour_array = np.append(pre_hour_array, post_hour_array)
                    elif time_index < 30:
                        post_hour_array = table['Z'].values[0: time_index]
                        pre_hour_array = np.full(30 - time_index, post_hour_array[0])
                        hour_array = np.append(pre_hour_array, post_hour_array)
                    else:
                        hour_array = table['Z'].values[time_index - 30: time_index]
                else:
                    if freq != 'D':
                        hour_array = table['Z'].values[time_index - pre_index: time_index]
                    else:
                        hour_array = table['Z'].values[time_index - 30: time_index]
                train_test_dict[predict_time] = hour_array
        train_df = pd.DataFrame(train_test_dict).fillna(method='ffill')
        # 训练集/测试集X值表将被保存到如下路径中
        train_df.to_csv(os.path.join('datas/level_data/train_test_scale_csvs', nc_path), index=False)
    return train_df, scaler


# 生成输出数据，将每个时间点后72h或30d的归一化后水位数据组织到同一行，根据训练时间范围构建训练集或者测试集的Y值
def gen_scale_Y(date_times: DatetimeIndex, stcd, post_index: int):
    train_test_dict = {}
    times = date_times.values
    nc_path = 'sta' + str(stcd) + '_' + str(date_times[0].year) + str(date_times[0].month) + str(
        date_times[0].day) + str(date_times[0].hour) + '_to_' + str(date_times[-1].year) + str(date_times[-1].month) + str(
        date_times[-1].day) + str(date_times[-1].hour) + '_forward_' + str(post_index) + 'h_scale.csv'
    table_rsvr_path = os.path.join(relpath('datas/level_data/total_2020_csvs'), str(stcd) + '_rsvr_cut2020.csv')
    table_river_path = os.path.join(relpath('datas/level_data/total_2020_csvs'), str(stcd) + '_river_cut2020.csv')
    if os.path.exists(table_river_path):
        # 要在这里重采样
        table = pd.read_csv(table_river_path, engine='c', parse_dates=True)
        table, freq, scaler = df_resample_cut(table, str(stcd))
    elif os.path.exists(table_rsvr_path):
        table = pd.read_csv(table_rsvr_path, engine='c', parse_dates=True)
        table, freq, scaler = df_resample_cut(table, str(stcd))
    else:
        table = pd.DataFrame()
        scaler = StandardScaler()
        freq = '4D'
    if os.path.exists(os.path.join('datas/level_data/train_test_scale_csvs', nc_path)):
        if os.path.getsize(os.path.join('datas/level_data/train_test_scale_csvs', nc_path)) > 256:
            train_df = pd.read_csv(os.path.join('datas/level_data/train_test_scale_csvs', nc_path), engine='c')
        else:
            train_df = pd.DataFrame()
    # 此for循环最费时
    else:
        slice_array = table.index.values
        for predict_time in times.flatten():
            if predict_time not in slice_array:
                continue
            else:
                time_index = np.argwhere(slice_array == predict_time)[0][0]
                if time_index + post_index > len(slice_array):
                    if freq != 'D':
                        pre_hour_array = table['Z'].values[time_index: len(slice_array)]
                        post_hour_array = np.full(time_index + post_index - len(slice_array), pre_hour_array[-1])
                        hour_array = np.append(pre_hour_array, post_hour_array)
                    elif time_index + 30 > len(slice_array):
                        pre_hour_array = table['Z'].values[time_index: len(slice_array)]
                        post_hour_array = np.full(time_index + 30 - len(slice_array), pre_hour_array[-1])
                        hour_array = np.append(pre_hour_array, post_hour_array)
                    else:
                        hour_array = table['Z'].values[time_index: time_index + 30]
                else:
                    if freq != 'D':
                        hour_array = table['Z'].values[time_index: time_index + post_index]
                    else:
                        hour_array = table['Z'].values[time_index: time_index + 30]
                train_test_dict[predict_time] = hour_array
        train_df = pd.DataFrame(train_test_dict)
        # 训练集/测试集X值表将被保存到如下路径中
        train_df.to_csv(os.path.join('datas/level_data/train_test_scale_csvs', nc_path), index=False)
    return train_df, scaler
