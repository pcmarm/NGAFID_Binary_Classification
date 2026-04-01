
import gdown
import pandas as pd
import numpy as np
from sklearn import preprocessing
from tqdm.autonotebook import tqdm
import tensorflow as tf
from loguru import logger
from ngafiddataset.utils import shell_exec
import os
import tarfile
import shutil
import typing
from compress_pickle import load

from ngafiddataset.dataset.utils import *



class NGAFID_Dataset_Downloader:

    ngafid_urls = {
        "all_flights": "https://drive.google.com/uc?id=1-0pVPhwRQoifT_VuQyGDLXuzYPYySX-Y",
        "2days": "https://drive.google.com/uc?id=1-2pxwiQNhFnhTg7whosQoF_yztD5jOM2",
    }

    @classmethod
    def download(cls, name: str, destination: str = '', extract: bool = True):
        """
        下载并解压数据集文件。

        Args:
            name: 数据集名称 ('2days' 或 'all_flights')
            destination: 保存目录路径
            extract: 是否需要解压

        Returns:
            (name, destination): 数据集名称和目标目录
        """
        assert name in cls.ngafid_urls.keys()

        url = cls.ngafid_urls[name]
        output = os.path.join(destination, "%s.tar.gz" % name)
        extract_dir = os.path.join(destination, name)

        # 检查是否已解压
        if extract and cls._is_extracted(name, destination):
            logger.info(f'数据已解压，跳过解压步骤: {extract_dir}')
            return name, destination

        # 下载文件（如果不存在）
        if not os.path.exists(output):
            logger.info(f'正在下载数据集: {name}')
            gdown.download(url, output, quiet=False)
        else:
            logger.info(f'压缩文件已存在: {output}')

        # 解压文件
        if extract:
            logger.info(f'正在解压文件到: {extract_dir}')
            _ = tarfile.open(output).extractall(destination)
            logger.info('解压完成')

        return name, destination

    @classmethod
    def _is_extracted(cls, name: str, destination: str = '') -> bool:
        """
        检查数据集是否已解压。

        Args:
            name: 数据集名称
            destination: 目标目录路径

        Returns:
            bool: 如果已解压返回 True，否则返回 False
        """
        extract_dir = os.path.join(destination, name)

        # 2days 数据集：检查必要文件是否存在
        if name == '2days':
            required_files = ['flight_data.pkl', 'flight_header.csv', 'stats.csv']
            for file in required_files:
                file_path = os.path.join(extract_dir, file)
                if not os.path.exists(file_path):
                    return False
            return True

        # all_flights 数据集：检查 one_parq 目录是否存在
        elif name == 'all_flights':
            parquet_dir = os.path.join(extract_dir, 'one_parq')
            if os.path.exists(parquet_dir) and os.path.isdir(parquet_dir):
                # 检查目录是否为空
                if os.listdir(parquet_dir):
                    return True
            return False

        return False


class NGAFID_Dataset_Manager(NGAFID_Dataset_Downloader):

    channels = 23

    def __init__(self, name: str, destination: str = '', max_length: int = 4096, extract: bool = True, **kwargs):
        """
        初始化 NGAFID 数据集管理器。

        Args:
            name: 数据集名称 ('2days' 或 'all_flights')
            destination: 数据保存目录
            max_length: 时间序列最大长度
            extract: 是否解压（已解压会自动跳过）
            **kwargs: 其他参数
        """
        assert name in self.ngafid_urls.keys()

        if name == 'all_flights':
            logger.info('Downloading and extracting Parquet Files to %s\\one_parq. Please open them using dask dataframes' % destination)
            self.download(name, destination, extract=True)

        else:
            self.name = name
            self.max_length = max_length
            self.destination = destination

            # 定义必要文件路径
            self.files = ['flight_data.pkl', 'flight_header.csv', 'stats.csv']
            self.files = {file: os.path.join(destination, name, file) for file in self.files}

            # 下载/解压（已解压会自动跳过）
            self.download(name, destination, extract)

            # 加载数据文件
            logger.info(f'正在加载数据文件...')
            self.flight_header_df = pd.read_csv(self.files['flight_header.csv'], index_col='Master Index')
            self.flight_data_array = load(self.files['flight_data.pkl'])
            self.flight_stats_df = pd.read_csv(self.files['stats.csv'])
            logger.info(f'数据加载完成，共 {len(self.flight_header_df)} 条记录')

            # 提取归一化参数
            self.maxs = self.flight_stats_df.iloc[0, 1:24].to_numpy(dtype=np.float32)
            self.mins = self.flight_stats_df.iloc[1, 1:24].to_numpy(dtype=np.float32)

    def update_flight_header(self):
        '''
        Adds some things we forgot
        '''

        self.flight_header_df['number_flights_before'] = self.light_header_df.filename.apply(
            lambda x: x.split('_')[-2])

    def construct_data_dictionary(self, numpy = True):
        data_dict = []

        for index, row in tqdm(self.flight_header_df.iterrows(), total = len(self.flight_header_df)):

            # pad array
            arr = np.zeros((self.max_length, self.channels), dtype = np.float16)
            to_pad = self.flight_data_array[index][-self.max_length:, :]
            arr[:to_pad.shape[0], :] += to_pad

            if numpy:
                pass
            else:
                arr = tf.convert_to_tensor(arr, dtype = tf.bfloat16)

            data_dict.append({'id': index,
                              'data': arr,
                              'class': row['class'],
                              'fold': row['fold'],
                              'target_class': row['target_class'],
                              'before_after': row['before_after'],
                              'hclass': row['hclass']})

        return data_dict

    def get_tf_dataset(self, fold = 0, training = False, shuffle = False, batch_size = 64, repeat = False,
                        mode = 'before_after', ds = None):

        if ds is None:
            ds = tf.data.Dataset.from_tensor_slices(to_dict_of_list(get_slice(self.data_dict, fold = fold, reverse = training)))


        ds = ds.repeat() if repeat else ds
        ds = ds.shuffle(shuffle) if shuffle else ds

        ds = ds.map(get_dict_mod('data', get_scaler(self.maxs, self.mins)))
        ds = ds.map(get_dict_mod('data', replace_nan_w_zero))
        ds = ds.map(get_dict_mod('data', lambda x: tf.cast(x, tf.float32)))

        if mode == 'before_after':
            ds = ds.map(lambda x: (x['data'], x['before_after']))
        elif mode == 'classes':
            ds = ds.map(lambda x: (x['data'], x['target_class']))
        elif mode == 'both':
            ds = ds.map(lambda x: (
            {'data': x['data']}, {'before_after': x['before_after'], 'target_class': x['target_class']}))
        elif mode == 'hierarchy_basic':
            ds = ds.map(
                lambda x: ({'data': x['data']}, {'before_after': x['before_after'], 'target_class': x['hclass']}))
        else:
            raise KeyError

        ds = ds.batch(batch_size, drop_remainder = True) if batch_size else ds

        return ds


    def get_numpy_dataset(self, fold = 0, training = False):
        return to_dict_of_list(get_slice(self.data_dict, fold = fold, reverse = training))
