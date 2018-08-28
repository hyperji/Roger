import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models import Word2Vec
from utils import findall, reduce_dims_with_std, reduce_dims_with_pca,molecule_fillna, timer
import gc
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestRegressor
from models import gbm_model, rf_model


def protein_embedding(protein_all, word_length=3, stride=1):
    """
    构建蛋白质词向量特征
    :param protein_all: 所有蛋白质词向量的序列, string
    :param word_length: 词长, int
    :param stride: 步长, int
    :return: 蛋白质词向量特征（pd.DataFrame实例）
    """

    texts_protein = list(protein_all["Sequence"].apply(lambda x: findall(x.upper(), word_length, stride)))

    n = 128

    model_protein = Word2Vec(texts_protein, size=n, window=4, min_count=1, negative=3,

                             sg=1, sample=0.001, hs=1, workers=4)

    vectors = pd.DataFrame([model_protein[word] for word in (model_protein.wv.vocab)])

    vectors['Word'] = list(model_protein.wv.vocab)

    vectors.columns = ["vec_{0}".format(i) for i in range(0, n)] + ["Word"]

    wide_vec = pd.DataFrame()

    result1 = []

    aa = list(protein_all['Protein_ID'])

    for i in range(len(texts_protein)):

        result2 = []

        for w in range(len(texts_protein[i])):
            result2.append(aa[i])

        result1.extend(result2)

    wide_vec['Id'] = result1

    result1 = []

    for i in range(len(texts_protein)):

        result2 = []

        for w in range(len(texts_protein[i])):
            result2.append(texts_protein[i][w])

        result1.extend(result2)

    wide_vec['Word'] = result1

    del result1

    wide_vec = wide_vec.merge(vectors, on='Word', how='left')

    wide_vec = wide_vec.drop('Word', axis=1)

    wide_vec.columns = ['Protein_ID'] + ["vec_{0}".format(i) for i in range(0, n)]

    del vectors

    name = ["vec_{0}".format(i) for i in range(0, n)]

    feat = pd.DataFrame(wide_vec.groupby(['Protein_ID'])[name].agg('mean')).reset_index()

    del wide_vec

    feat.columns = ["Protein_ID"] + [str(word_length) + "_mean_ci_{0}".format(i) for i in range(0, n)]

    return feat


def tfidf_and_wordcounts(protein_all, PID, word_length=2, stride=1):
    """
    构建蛋白质序列的tfidf和wordcount特征
    :param protein_all: 所有蛋白质词向量的序列, pd.DataFrame
    :param PID: 所有蛋白质ID
    :param word_length:　词长
    :param stride: 步长
    :return: tfidf特征和wordcount特征（pd.DataFrame实例）
    """
    # 用词长为word_length, 步长为stride来选择蛋白质文本信息
    texts_protein = list(protein_all["Sequence"].apply(lambda x: findall(x.upper(), word_length, stride)))

    # 合并＂蛋白质文本＂，并用空格隔开每个蛋白质序列的＂单词＂，构建＂文本＂
    corpus = list(map(lambda x: " ".join(i for i in x), texts_protein))
    # 计算每个＂单词＂的在蛋白质序列中的　term-frequence and inverse-document-frequence
    tfidf = TfidfVectorizer(token_pattern=u'(?u)\\b\\w+\\b')
    tfidf_vals = tfidf.fit_transform(corpus)
    tfidf_vals = tfidf_vals.toarray()

    # 计算每个＂单词＂在每个蛋白质序列出现的次数
    counts = CountVectorizer(token_pattern=u'(?u)\\b\\w+\\b')
    word_counts = counts.fit_transform(corpus)
    word_counts = word_counts.toarray()

    del corpus

    tfidf_vals = pd.DataFrame(tfidf_vals,
                              columns=[str(word_length) + "_ags_tfidfs_" + str(i) for i in range(tfidf_vals.shape[1])])
    word_counts = pd.DataFrame(word_counts, columns=[str(word_length) + "_ags_wordcounts_" + str(i) for i in
                                                     range(word_counts.shape[1])])

    tfidf_vals["Protein_ID"] = PID
    word_counts["Protein_ID"] = PID

    return tfidf_vals, word_counts


# 从原始竞赛数据中构建可用的数据
def build_useful_data():
    """
    #TODO 利用pca降维，或者LDA降维......方式构建特征文件
    构建可用的初始特征数据, 默认原始竞赛数据储存在当前文件夹中的datas文件夹中.
    :return: 可用数据（pd.DataFrame实例）
    """

    # 读取蛋白质数据
    with timer("Loading and merging data"):
        protein_train = pd.read_csv('datas/df_protein_train.csv')

        protein_test = pd.read_csv('datas/df_protein_test.csv')

        protein_all = pd.concat([protein_train, protein_test])

        # 添加蛋白质序列长度作为特征
        protein_all['seq_len'] = protein_all['Sequence'].apply(len)

        # 读取分子数据
        mol_train = pd.read_csv('datas/df_molecule.csv')

        aff_train = pd.read_csv('datas/df_affinity_train.csv')

        aff_test = pd.read_csv('datas/df_affinity_test_toBePredicted.csv')

        # 初始化待预测的Ki值为-11
        aff_test['Ki'] = -11

        aff_all = pd.concat([aff_train, aff_test])

        data = aff_all.merge(mol_train, on="Molecule_ID", how='left')
        data = data.merge(protein_all, on='Protein_ID', how='left')

        # 获取蛋白质ID
        PID = list(protein_all["Protein_ID"])
    with timer("Processing wordcount1"):
        # word_length = 1时的wordcount特征
        _, word_counts1 = tfidf_and_wordcounts(protein_all, PID, word_length=1, stride=1)

    # word_length = 2时的wordcount特征
    with timer("Processing wordcount2"):
        _, word_counts2 = tfidf_and_wordcounts(protein_all, PID, word_length=2, stride=1)

        word_counts1_2 = word_counts1.merge(word_counts2, on="Protein_ID", how="left")
        # 保存特征文件，以供后期训练
        word_counts1_2.to_csv("datas/1and2_1_421_protein_std.csv", index=False)

        del word_counts1_2, word_counts1, word_counts2

    with timer("Processing wordcount3"):
        _, word_count3 = tfidf_and_wordcounts(protein_all, PID, word_length=3, stride=1)

        word_count3_features = list(word_count3.columns)  # 8000维的数据，需要降维
        word_count3_features.remove("Protein_ID")

        # 利用标准差进行降维，设置标准差阈值为0.42，去掉标准差小于0.42的特征
        new_word_count3 = reduce_dims_with_std(word_count3, word_count3_features, std_threshold=0.3)
        # 保存特征文件，以供后期训练
        new_word_count3.to_csv("datas/3_1_protein_std_0.3.csv", index=False)
        del new_word_count3

        for i in range(len(word_count3_features) // 1000):
            # 每次划分1000个特征，并保存在特征文件里，以供后期训练
            file = word_count3[["Protein_ID"] + word_count3_features[i * 1000:(i + 1) * 1000]]
            file_name = "3_1_1000_protein_" + str(i)
            file.to_csv("datas/" + file_name + ".csv", index=False)

        del word_count3, word_count3_features

    with timer("Processing wordcount4"):
        gc.collect()
        _, word_count4 = tfidf_and_wordcounts(protein_all, PID, word_length=4, stride=1)

        word_count4_features = list(word_count4.columns)  # 140000+　维的数据，需要降维
        word_count4_features.remove("Protein_ID")

        new_word_count4 = reduce_dims_with_pca(word_count4, word_count4_features, n_conponents=1000)
        new_word_count4.to_csv("datas/wordcount4_pca.csv", index=False)

        # 利用标准差进行降维，设置标准差阈值为0.15，去掉标准差小于0.15的特征
        new_word_count4 = reduce_dims_with_std(word_count4, word_count4_features, std_threshold=0.15)
        new_word_count4.to_csv("datas/4_1_protein_std_0.15.csv", index=False)

        # 利用标准差进行降维，设置标准差阈值为0.12，去掉标准差小于0.12的特征
        new_word_count4 = reduce_dims_with_std(word_count4, word_count4_features, std_threshold=0.12)

        word_count4_features = list(new_word_count4.columns)
        word_count4_features.remove("Protein_ID")

        for i in range(len(word_count4_features) // 1000):
            # 每次划分500个特征，并保存在特征文件里，以供日后训练
            file = new_word_count4[["Protein_ID"] + word_count4_features[i * 1000:(i + 1) * 1000]]
            file_name = "4_1_1000_protein_" + str(i)
            file.to_csv("datas/" + file_name + ".csv", index=False)

        del new_word_count4, word_count4

    # 以下特征是蛋白质的词向量特征, 来自技术圈, 谢谢＂小武哥＂同学.但我们的最终提交版本没用这些特征
    "=====================================词向量特征==========================================="
    # feat2 = protein_embedding(protein_all, word_length = 2)
    # data = data.merge(feat2, on="Protein_ID", how="left")
    # del feat2
    # feat3 = protein_embedding(protein_all, word_length = 3)
    # data = data.merge(feat3, on="Protein_ID", how="left")
    # del feat3
    # feat4 = protein_embedding(protein_all, word_length = 4)
    # data = data.merge(feat4, on="Protein_ID", how="left")
    # del feat4
    "================================================================================"

    with timer("分子指纹展开"):
        mol_fingerprints = list(mol_train["Fingerprint"].apply(lambda x: list(np.array(x.split(',')).astype(int))))
        mol_fingerprints = pd.DataFrame(mol_fingerprints, columns=["Fingerprint_" + str(i) for i in range(167)])
        mol_fingerprints["Molecule_ID"] = mol_train["Molecule_ID"]

    del PID
    "=================================================================================================="

    with timer("加入分子指纹和描述符"):
        data = data.merge(mol_fingerprints, on="Molecule_ID", how='left')
        mol_ECFP4 = pd.read_csv("datas/df_mol_ECFP4s_1024.csv")
        data = data.merge(mol_ECFP4, on="Molecule_ID")
        del mol_fingerprints, mol_ECFP4
        del data["Sequence"], protein_train, protein_test, mol_train

        data.reset_index(drop=True, inplace=True)
        data.to_csv("datas/original_data.csv", index=False)

        del data
        print("Useful data have builded")


def feature_selection_with_rf(data, features, topk=100, epochs=10):
    """
    select features based on feature importance using random forest
    :param data:
    :param features:
    :param topk: topk featrues
    :param epochs:
    :return:
    """
    importance_sum = np.zeros(len(features))

    for epoch in range(epochs):
        print("epoch %d" % epoch)
        clf = RandomForestRegressor(n_estimators=100, random_state=0, n_jobs=-1)
        clf.fit(data[features].values, data.Ki.values)
        importance_sum += clf.feature_importances_
    return features[np.argsort(importance_sum)[-topk:]], importance_sum


def feature_selection_with_gbm(data, features, topk=100, epochs=10):
    """

    :param data:
    :param features:
    :param topk:
    :param epochs:
    :return:
    """
    importance_sum = np.zeros(len(features))
    for epoch in range(epochs):
        _, gbm = gbm_model(data, features, return_model=True)
        importance_sum += gbm.feature_importance()

    return np.array(features)[np.argsort(importance_sum)[-topk:]], importance_sum


if __name__ == "__main__":
    build_useful_data()

