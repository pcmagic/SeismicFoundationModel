import os
import random
import concurrent.futures
from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map

def read_and_write_file(data_path, file_name, outfile):
    file_path = os.path.join(data_path, file_name)
    with open(file_path, 'rb') as infile:
        data = infile.read()
    outfile.write(data)

def merge_dat_files_parallel(data_path, output_filename):
    # 获取当前目录下所有的 .dat 文件
    dat_files = []
    for f in tqdm(os.listdir(data_path)):
        if f.endswith('.dat'):
            dat_files.append(f)
    # dat_files = [f for f in os.listdir(data_path) if f.endswith('.dat')]
    random.shuffle(dat_files)

    # 打开输出文件，准备写入
    with open(output_filename, 'wb') as outfile:
        # 使用线程池并行处理，并添加进度条
        thread_map(lambda file_name: read_and_write_file(data_path, file_name, outfile), dat_files, max_workers=None, desc='Merging files', unit='files')

# 调用函数，指定输出文件名
data_path = os.path.join(os.getcwd(), 'mae_data_more_small')
merge_dat_files_parallel(data_path, 'mae_data_more_small.dat')
