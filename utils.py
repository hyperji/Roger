import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import time
from contextlib import contextmanager
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error

@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f'[{name}] done in {time.time() - t0:.3f} s')

def molecule_fillna(df,select_col):
    scaler = MinMaxScaler(feature_range=(0,1))
    for col in select_col:
        df[col] = scaler.fit_transform(df[[col]].fillna(df[col].mean()))
    return df

def findall(sequence, word_length = 3, stride = 1):
    """
    :param sequence: 蛋白质分子序列, string
    :param word_length: 词长, int
    :param stride: 步长, int
    :return: a list, 以stride为步长，包含蛋白质分子序列的所有长度为word_length的序列
    """
    total_length = len(sequence)
    flag = (total_length - word_length)%stride == 0
    final_length = int(float(total_length - word_length) / stride + 1)
    indexs = np.array(range(final_length))*stride
    result = [sequence[i:i+word_length] for i in indexs]
    if flag:
        return result
    else:
        quekou = 2*word_length - (total_length - indexs[-1])
        return result + [sequence[indexs[-1]+word_length - quekou:]]


def reduce_dims_with_std(dataframe, features, std_threshold = 0.3):
    """
    用标准差作为阈值进行数据降维
    :param dataframe: pd.DataFrame实例
    :param features: 需要降维的特征, list or np.array
    :param std_threshold: 标准差的阈值（标准差小于这个值的特征将被抛弃）, float
    :return: 降维后的数据（pd.DataFrame实例）
    """
    features = np.array(features)
    stds = dataframe[features].std()
    masks = (stds>std_threshold).values
    reduced_featrues = features[masks]
    reduced_featrues = np.concatenate([["Protein_ID"], reduced_featrues])
    print("After reduce dims, the final dim is", len(reduced_featrues), "while the original dim is: ",len(features))
    return dataframe[reduced_featrues]


def reduce_dims_with_pca(dataframe, features, n_conponents = 200):
    """
    :param dataframe: pd.DataFrame实例
    :param features: 需要降维的特征, list or np.array
    :param n_conponents: 主成分个数, int
    :return: pca 降维后的数据（pd.DataFrame实例）
    """
    features = np.array(features)
    tag = int(dataframe.columns[0].split('_')[0])
    assert features.shape[0] >= n_conponents
    PID = dataframe.Protein_ID.values
    pca = PCA(n_components=n_conponents)
    final_feature_names = [str(tag)+'_pca_'+str(i) for i in range(n_conponents)]
    dataframe = pd.DataFrame(pca.fit_transform((dataframe[features].values)), columns=final_feature_names)
    dataframe["Protein_ID"] = PID
    print("After reduce dims, the final dim is", n_conponents, "while the original dim is: ", len(features))
    return dataframe


def preprocessing_data(data, features):
    """
    :param data:
    :param features:
    :return:
    """
    data[features] = StandardScaler(with_mean=True,with_std=True).fit_transform(data[features])
    return data


def rmse(y_true, y_pred):
    """
    :param y_true:
    :param y_pred:
    :return:
    """
    return np.sqrt(mean_squared_error(y_true=y_true, y_pred=y_pred))


def normalize_features(feature_matrix):
    norms = np.linalg.norm(feature_matrix, axis=0)
    normalized_features = feature_matrix / norms
    return (normalized_features, norms)

