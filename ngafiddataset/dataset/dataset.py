
import os
import tarfile
import shutil
import typing
import hashlib
import time

import requests
import gdown
import pandas as pd
import numpy as np
from tqdm.autonotebook import tqdm
import tensorflow as tf
from loguru import logger
from compress_pickle import load

from ngafiddataset.utils import shell_exec
from ngafiddataset.dataset.utils import *


class NGAFID_Dataset_Downloader:

    ngafid_urls = {
        "all_flights": "https://zenodo.org/records/6624956/files/all_flight.tar.gz",
        "2days": "https://zenodo.org/records/6624956/files/2days.tar.gz",
    }

    ngafid_api_urls = {
        "all_flights": "https://zenodo.org/api/records/6624956/files/all_flight.tar.gz/content",
        "2days": "https://zenodo.org/api/records/6624956/files/2days.tar.gz/content",
    }

    ngafid_checksums = {
        "all_flights": "8d7501a4ea02eb113172938abdc76cf1",
        "2days": "83d70602c5cda57bba3512035c8abab0",
    }

    @classmethod
    def download(cls, name: str, destination: str = '', extract: bool = True, max_retries: int = 3):
        """
        下载并解压数据集文件。

        Args:
            name: 数据集名称 ('2days' 或 'all_flights')
            destination: 保存目录路径
            extract: 是否需要解压
            max_retries: 下载失败时的最大重试次数

        Returns:
            (name, destination): 数据集名称和目标目录
        """
        assert name in cls.ngafid_urls.keys()

        url = cls.ngafid_urls[name]
        api_url = cls.ngafid_api_urls[name]
        checksum = cls.ngafid_checksums[name]
        output = os.path.join(destination, "%s.tar.gz" % name)
        extract_dir = os.path.join(destination, name)

        # 检查是否已解压
        if extract and cls._is_extracted(name, destination):
            logger.info(f'数据已解压，跳过解压步骤: {extract_dir}')
            return name, destination

        # 删除损坏的或不完整的文件以便重新下载
        if os.path.exists(output):
            if not cls._verify_checksum(output, checksum):
                logger.warning(f'文件校验失败，将重新下载: {output}')
                os.remove(output)

        # 下载文件
        if not os.path.exists(output):
            logger.info(f'正在下载数据集: {name} (来源: {url})')
            success = cls._download_with_retry(api_url, output, name, max_retries)
            if not success:
                raise RuntimeError(
                    f"下载数据集 '{name}' 失败，已达到最大重试次数 ({max_retries})。"
                    f"请检查网络连接或手动下载文件: {url}"
                )

            # 校验文件完整性
            if not cls._verify_checksum(output, checksum):
                raise RuntimeError(
                    f"文件校验失败！MD5 不匹配。\n"
                    f"  文件: {output}\n"
                    f"  预期: {checksum}\n"
                    f"请删除该文件后重试，或手动下载: {url}"
                )
        else:
            logger.info(f'压缩文件已存在: {output}')

        # 解压文件
        if extract:
            logger.info(f'正在解压文件到: {extract_dir}')
            _ = tarfile.open(output).extractall(destination)
            logger.info('解压完成')

        return name, destination

    @classmethod
    def _download_with_retry(cls, url: str, output: str, name: str, max_retries: int = 3) -> bool:
        """
        使用 requests 流式下载文件，支持断点续传和重试。

        Args:
            url: 下载 URL (Zenodo API content URL)
            output: 保存路径
            name: 数据集名称
            max_retries: 最大重试次数

        Returns:
            bool: 下载是否成功
        """
        for attempt in range(1, max_retries + 1):
            try:
                logger.info(f'下载尝试 {attempt}/{max_retries}')
                cls._download_file(url, output, name)
                return True
            except requests.exceptions.RequestException as e:
                logger.warning(f'下载失败 (尝试 {attempt}/{max_retries}): {e}')
                if attempt < max_retries:
                    wait_time = 2 ** attempt  # 指数退避: 2s, 4s, 8s, ...
                    logger.info(f'{wait_time} 秒后重试...')
                    time.sleep(wait_time)
                else:
                    logger.error(f'下载失败，已达到最大重试次数')
                    return False
        return False

    @classmethod
    def _download_file(cls, url: str, output: str, name: str):
        """
        使用 requests 流式下载文件，支持断点续传，显示进度条。
        """
        headers = {}
        downloaded_size = 0

        # 断点续传：检查已下载的大小
        if os.path.exists(output):
            downloaded_size = os.path.getsize(output)
            headers['Range'] = f'bytes={downloaded_size}-'
            logger.info(f'检测到已下载 {downloaded_size} 字节，继续下载...')

        with requests.get(url, headers=headers, stream=True, timeout=300) as response:
            response.raise_for_status()

            # 获取文件总大小
            total_size = int(response.headers.get('Content-Length', 0))
            if total_size == 0:
                total_size = int(response.headers.get('content-length', 0))

            # 如果使用 Range 请求，服务器应返回 206 Partial Content
            # 否则从头开始下载
            if response.status_code == 206:
                total_size += downloaded_size
            else:
                downloaded_size = 0
                # 删除不完整的旧文件
                if os.path.exists(output):
                    os.remove(output)

            desc = f'下载 {name}'
            with open(output, 'ab' if downloaded_size > 0 else 'wb') as f, \
                    tqdm(total=total_size, initial=downloaded_size, unit='B',
                         unit_scale=True, unit_divisor=1024, desc=desc) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

    @classmethod
    def _verify_checksum(cls, filepath: str, expected_md5: str) -> bool:
        """
        验证文件的 MD5 校验和。

        Args:
            filepath: 文件路径
            expected_md5: 预期的 MD5 校验和 (32位十六进制字符串)

        Returns:
            bool: 校验是否通过
        """
        if not os.path.exists(filepath):
            return False

        md5_hash = hashlib.md5()
        with open(filepath, 'rb') as f:
            # 流式读取，避免大文件占用过多内存
            for chunk in iter(lambda: f.read(8192), b''):
                md5_hash.update(chunk)

        actual_md5 = md5_hash.hexdigest()
        match = actual_md5 == expected_md5.lower()
        if match:
            logger.info(f'文件校验通过: {filepath}')
        else:
            logger.warning(
                f'文件校验失败: {filepath}\n'
                f'  预期: {expected_md5}\n'
                f'  实际: {actual_md5}'
            )
        return match

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
