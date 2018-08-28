import lightgbm as lgb
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.callbacks import LearningRateScheduler
from keras.callbacks import EarlyStopping, TensorBoard
from keras.layers.normalization import BatchNormalization


def gbm_model(train, features, test = None, return_model = False, num_round = 500):
    """
    #TODO grid search　以确定参数
    简单的lgb模型
    :param train: 训练数据，　pd.DataFrame
    :param test: 测试数据，　pd.DataFrame
    :param features: 特征，　list or np.array
    :return: prediction on test
    """
    dtrain = lgb.Dataset(train[features].values, train.Ki.values)

    valid = dtrain
    if test is not None:
        dtest = lgb.Dataset(test[features].values, test.Ki.values)
        valid = dtest

    params = {

        'boosting_type': 'gbdt',

        'objective': 'regression_l2',

        'metric': 'l2',

        'min_child_weight': 3,

        'num_leaves': 2 ** 9,

        'lambda_l2': 10,

        'subsample': 0.7,

        'colsample_bytree': 0.7,

        'learning_rate': 0.05,

        'seed': 2018,

        #'nthread': 12,

        #'device':'gpu'

    }
    #TODO(Whitemoon) 降低 num_round 减少过拟合
    num_round = num_round

    gbm = lgb.train(params,

                    dtrain,

                    num_round,

                    early_stopping_rounds=500,

                    verbose_eval=50,

                    valid_sets=[valid]

                    )

    # 结果保存

    del dtrain

    print("training finished!")
    preds = None
    if test is not None:
        preds = gbm.predict(test[features].values)
    if return_model:
        return preds, gbm
    else:
        return preds


def ridge_model(train, features, test=None, alpha = 1, return_model = False):
    """

    :param train: 训练数据，　pd.DataFrame
    :param test: 测试数据，　pd.DataFrame
    :param features: 特征，　list or np.array
    :return: prediction on test
    """
    ridge = Ridge(alpha=alpha, random_state=2018)
    ridge.fit(train[features].values, train.Ki.values)
    preds = None
    if test is not None:
        preds = ridge.predict(test[features].values)
    if return_model:
        return preds, ridge
    else:
        return preds


def lasso_model(train, features, test=None, alpha = 1, return_model = False):
    """

    :param train:
    :param test:
    :param features:
    :return: prediction on test
    """
    lasso = Lasso(alpha=alpha, random_state=2018)
    lasso.fit(train[features].values, train.Ki.values)
    preds = None
    if test is not None:
        preds = lasso.predict(test[features].values)
    if return_model:
        return preds, lasso
    else:
        return preds


def dnn_model(train, features, layer_dims, test = None, return_model = False):
    """

    :param train:
    :param test:
    :param features:
    :param layer_dims:
    :return:
    """
    early_stopping = EarlyStopping(monitor='val_loss', patience=2)
    tb_callback = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
    model = Sequential()
    model.add(Dense(units=layer_dims[0], input_dim=len(features), activation='relu'))
    for dim in layer_dims[1:-1]:
        model.add(Dense(units=dim, activation='relu'))
        model.add(BatchNormalization())
    model.add(Dense(units=layer_dims[-1]))
    model.compile(loss='mse',optimizer='adam',metrics=['mse'])
    model.fit(train[features].values, train.Ki.values, validation_split=0.1, callbacks=[early_stopping], epochs=10, batch_size=100)
    preds = None
    if test is not None:
        preds = model.predict(test[features].values)
    if return_model:
        return preds, model
    else:
        return preds



def rf_model(train, features, test = None,  return_model = False):
    """
    :param train:
    :param test:
    :param features:
    :return:
    """
    rf = RandomForestRegressor(n_estimators=100, random_state=2018)
    rf.fit(train[features].values, train.Ki.values)
    preds = None
    if test is not None:
        preds = rf.predict(test[features].values)
    if return_model:
        return preds, rf
    else:
        return preds
