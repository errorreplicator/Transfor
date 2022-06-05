import tensorflow as tf
from pathlib import Path
import tensorflow_datasets as tfds

path = '/data/txtFiles/GutSubset/'
fileNames = []

for file in Path(path).iterdir():
    # print(file.name)
    if file.is_file() and file.suffix == '.txt':
        fileNames.append(path+file.name)

dataset = tf.data.TextLineDataset(filenames=fileNames,num_parallel_reads=2).filter(lambda line:tf.not_equal(tf.strings.length(line),0))# filter for removing 0 len sentences

for x in dataset.take(30):
    print(x.numpy())


data_tokens = dataset.map(lambda x: tf.strings.split(x))

for element in data_tokens.take(30):
    print(element.numpy())




# print(type(dataset))
#
# filepath_dataset = tf.data.Dataset.list_files(path, seed=42)
# n_readers = 5
# dataset = filepath_dataset.interleave(
#     lambda filepath: tf.data.TextLineDataset(filepath).skip(1),
#     cycle_length=n_readers)
#
# # print(type(dataset))
# import pandas as pd
# pd.set_option('display.max_rows', 500)
# pd.set_option('display.max_columns', 500)
# pd.set_option('display.width', 1000)
#
#
# # print(tfds.as_dataframe(dataset))
# ds = tfds.as_dataframe(dataset)
# # print(ds.iloc[58301])
# my_str = str(ds.iloc[58301])
# print(my_str)
# print(type(my_str))
# for char in my_str:
#     print(char)

# dataset = text_dataset_from_directory(directory=path,batch_size=128,labels="something_else")

