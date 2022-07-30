import pandas as pd
import numpy as np
import time

from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, make_scorer, recall_score, f1_score
import matplotlib.pyplot as plt


def split_data(x, y, size, random_state):
    np.random.seed(random_state)
    number_of_instances = x.shape[0] * size
    x_values_selected = []
    y_values_selected = []
    already_selected = []
    count = 0
    while count < number_of_instances:
        index = np.random.randint(0, x.shape[0])
        if index not in already_selected:
            x_values_selected.append(x.iloc[index, :])
            y_values_selected.append(y.iloc[index])
            count = count + 1

    return pd.DataFrame(x_values_selected), pd.Series(y_values_selected)


# 朴素贝叶斯
def bay(X_train, y_train, X_test, y_test, metric, n_splits, random_state, num_features, print_info, dataset):
    alpha = [0.1, 0.5, 1.0]
    param_grid_bay = {'alpha': alpha}
    scorer_bay = make_scorer(metric)
    skf_bay = StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=True)
    bay = MultinomialNB()
    grid_search_model_bay = GridSearchCV(estimator=bay, param_grid=param_grid_bay, scoring=scorer_bay, cv=skf_bay)
    start_time = time.time()
    grid_search_model_bay.fit(X_train, y_train)
    end_time = time.time()
    y_pred_bay = grid_search_model_bay.predict(X_test)
    tn_bay, fp_bay, fn_bay, tp_bay = confusion_matrix(y_test, y_pred_bay).ravel()
    if print_info:
        print("Naive Bayes - Testing Data")
        print("Best paramaters = {}".format(grid_search_model_bay.best_params_))
        print("True negative = {}, False Positive = {}, False Negative = {}, True Positive = {}".format(tn_bay, fp_bay,
                                                                                                        fn_bay, tp_bay))
        print("Accuracy score = {}".format(accuracy_score(y_test, y_pred_bay)))
        print("Recall score = {}".format(recall_score(y_test, y_pred_bay)))
        print("AUC ROC score = {}".format(roc_auc_score(y_test, y_pred_bay)))
        print("F1 score = {}".format(f1_score(y_test, y_pred_bay)))
        print()
    if dataset == 1:
        return f1_score(y_test, y_pred_bay), end_time - start_time
    else:
        return roc_auc_score(y_test, y_pred_bay), end_time - start_time


# KNN
def KNN(X_train, y_train, X_test, y_test, metric, n_splits, random_state, num_features, print_info, dataset):
    neighbours = [i for i in range(3, 10)]  # 从3-10中找出最佳的n，要保证在这区间有出现峰值，否则扩大区间
    weights = ['uniform', 'distance']
    algorithm = ['auto', 'ball_tree', 'kd_tree', 'brute']
    param_grid_knn = {'n_neighbors': neighbours, 'weights': weights, 'algorithm': algorithm}
    scorer_knn = make_scorer(metric)
    # n_splits=K(折叠次数) —— 使用K折交叉验证模型调优，找到使得模型泛化性能最优的超参值
    # random_state: 控制随机状态，固定模式，使同一训练集重复的展现相同的结果（在shuffle==True时使用）
    # shuffle: 是否在每次分割之前打乱顺序
    skf_knn = StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=True)
    knn = KNeighborsClassifier()
    grid_search_model_knn = GridSearchCV(estimator=knn, param_grid=param_grid_knn, scoring=scorer_knn, cv=skf_knn)
    start_time = time.time()
    grid_search_model_knn.fit(X_train, y_train)
    end_time = time.time()
    y_pred_knn = grid_search_model_knn.predict(X_test)
    tn_knn, fp_knn, fn_knn, tp_knn = confusion_matrix(y_test, y_pred_knn).ravel()
    if print_info:
        print("KNN - Testing Data")
        print("Best paramaters = {}".format(grid_search_model_knn.best_params_))
        print("True negative = {}, False Positive = {}, False Negative = {}, True Positive = {}".format(tn_knn, fp_knn,
                                                                                                        fn_knn, tp_knn))
        print("Accuracy score = {}".format(accuracy_score(y_test, y_pred_knn)))
        print("Recall score = {}".format(recall_score(y_test, y_pred_knn)))
        print("AUC ROC score = {}".format(roc_auc_score(y_test, y_pred_knn)))
        print("F1 score = {}".format(f1_score(y_test, y_pred_knn)))
        print()
    if dataset == 1:
        return f1_score(y_test, y_pred_knn), end_time - start_time
    else:
        return roc_auc_score(y_test, y_pred_knn), end_time - start_time


# 决策树
def dt(X_train, y_train, X_test, y_test, metric, n_splits, random_state, num_features, print_info, dataset):
    min_samples_leaf = [i for i in range(3, 100)]
    max_depth = [i for i in range(5, 50, 5)]
    param_grid_dtl = {'min_samples_leaf': min_samples_leaf, 'max_depth': max_depth}
    scorer_dtl = make_scorer(metric)
    skf_dtl = StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=True)
    dtl = DecisionTreeClassifier()
    selector = SelectKBest(k=num_features)
    selector.fit(X_train, y_train)
    X_train = selector.transform(X_train)
    X_test = selector.transform(X_test)
    grid_search_model_dtl = GridSearchCV(estimator=dtl, param_grid=param_grid_dtl, scoring=scorer_dtl, cv=skf_dtl)
    start_time = time.time()
    grid_search_model_dtl.fit(X_train, y_train)
    end_time = time.time()
    y_pred_dtl = grid_search_model_dtl.predict(X_test)
    tn_dtl, fp_dtl, fn_dtl, tp_dtl = confusion_matrix(y_test, y_pred_dtl).ravel()
    if print_info:
        print("Decision Tree - Testing Data")
        print("Best paramaters = {}".format(grid_search_model_dtl.best_params_))
        print("True negative = {}, False Positive = {}, False Negative = {}, True Positive = {}".format(tn_dtl, fp_dtl,
                                                                                                        fn_dtl, tp_dtl))
        print("Accuracy score = {}".format(accuracy_score(y_test, y_pred_dtl)))
        print("Recall score = {}".format(recall_score(y_test, y_pred_dtl)))
        print("AUC ROC score = {}".format(roc_auc_score(y_test, y_pred_dtl)))
        print("F1 score = {}".format(f1_score(y_test, y_pred_dtl)))
        print()
    if dataset == 1:
        return f1_score(y_test, y_pred_dtl), end_time - start_time
    else:
        return roc_auc_score(y_test, y_pred_dtl), end_time - start_time


# SVM
def svc(X_train, y_train, X_test, y_test, metric, n_splits, random_state, kernels, num_features, print_info, dataset):
    gamma = [0.1, 1.0, 10]
    param_grid_svm = {'kernel': kernels, 'gamma': gamma}
    scorer_svm = make_scorer(metric)
    skf_svm = StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=True)
    svm = SVC()
    grid_search_model_svm = GridSearchCV(estimator=svm, param_grid=param_grid_svm, scoring=scorer_svm, cv=skf_svm)
    start_time = time.time()
    grid_search_model_svm.fit(X_train, y_train)
    end_time = time.time()
    y_pred_svm = grid_search_model_svm.predict(X_test)
    tn_svm, fp_svm, fn_svm, tp_svm = confusion_matrix(y_test, y_pred_svm).ravel()
    if print_info:
        print("SVM - Testing Data")
        print("Best paramaters = {}".format(grid_search_model_svm.best_params_))
        print("True negative = {}, False Positive = {}, False Negative = {}, True Positive = {}".format(tn_svm, fp_svm,
                                                                                                        fn_svm, tp_svm))
        print("Accuracy score = {}".format(accuracy_score(y_test, y_pred_svm)))
        print("Recall score = {}".format(recall_score(y_test, y_pred_svm)))
        print("AUC ROC score = {}".format(roc_auc_score(y_test, y_pred_svm)))
        print("F1 score = {}".format(f1_score(y_test, y_pred_svm)))
        print()
    if dataset == 1:
        return f1_score(y_test, y_pred_svm), end_time - start_time
    else:
        return roc_auc_score(y_test, y_pred_svm), end_time - start_time


# 神经网络
def nn(X_train, y_train, X_test, y_test, metric, n_splits, random_state, num_features, print_info, dataset):
    activation_functions = ['identity', 'logistic', 'tanh', 'relu']
    learning_rate = ['constant', 'invscaling', 'adaptive']
    hidden_layer_sizes = []
    for neurons in range(100, 250, 10):
        t = []
        val = neurons
        for size in range(0, 1):
            t.append(val)
            val = val + neurons

        hidden_layer_sizes.append(tuple(t))

    param_grid_nn = {'activation': activation_functions, 'hidden_layer_sizes': hidden_layer_sizes,
                     'learning_rate': learning_rate}
    scorer_nn = make_scorer(metric)
    skf_nn = StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=True)
    nn_model = MLPClassifier(max_iter=10000)
    grid_search_model_nn = GridSearchCV(estimator=nn_model, param_grid=param_grid_nn, scoring=scorer_nn, cv=skf_nn)
    start_time = time.time()
    grid_search_model_nn.fit(X_train, y_train)
    end_time = time.time()
    y_pred_nn = grid_search_model_nn.predict(X_test)
    tn_nn, fp_nn, fn_nn, tp_nn = confusion_matrix(y_test, y_pred_nn).ravel()
    if print_info:
        print("Neural Network - Testing Data")
        print("Best paramaters = {}".format(grid_search_model_nn.best_params_))
        print("True negative = {}, False Positive = {}, False Negative = {}, True Positive = {}".format(tn_nn, fp_nn,
                                                                                                        fn_nn, tp_nn))
        print("Accuracy score = {}".format(accuracy_score(y_test, y_pred_nn)))
        print("Recall score = {}".format(recall_score(y_test, y_pred_nn)))
        print("AUC ROC score = {}".format(roc_auc_score(y_test, y_pred_nn)))
        print("F1 score = {}".format(f1_score(y_test, y_pred_nn)))
        print()
    if dataset == 1:
        return f1_score(y_test, y_pred_nn), end_time - start_time
    else:
        return roc_auc_score(y_test, y_pred_nn), end_time - start_time


# ----------Main Code----------------------------------------------
if __name__ == '__main__':
        # 数据准备
    print()
    print("Baru-Credit-Data View")
    print("--------------------------------------------------------------")
    df_Baru = pd.read_csv("Data/barudata.csv", header=None)
    df_Baru.iloc[:, -1] = df_Baru.iloc[:, -1].replace( "Good", 1)
    df_Baru.iloc[:, -1] = df_Baru.iloc[:, -1].replace( "Bad", 0)
    fraud = df_Baru[df_Baru.iloc[:, 0:].iloc[:, -1] == 0]
    new_dataset = pd.concat([df_Baru[df_Baru.iloc[:, 0:].iloc[:, -1] == 1].sample(n=23521),
                         fraud], axis=0)
    print("Baru-Credit-Data : No. of positive samples = {}, No. of negative samples = {}".format(
        new_dataset.loc[new_dataset.iloc[:, -1] == 1, :].shape[0],
        new_dataset.loc[new_dataset.iloc[:, -1] == 0, :].shape[0]))
    print("--------------------------------------------------------------")
    print()
    X_train_Baru, X_test_Baru, y_train_Baru, y_test_Baru = train_test_split(
        new_dataset.iloc[:, 1:-1], new_dataset.iloc[:, -1], test_size=0.30, random_state=0)
    print()

    training_sizes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    algorithms = ['bay', 'knn', 'dtl', 'svm', 'nn']
  
    time_taken = {}
    print("Modeling")
    print("-------------------------------------------------------------")
    time_taken1 = {}
    for algorithm in algorithms:

        error_rate_training_data = []
        error_rate_testing_data = []
        time_taken_temp = []
        if algorithm == 'bay':
            for training_size in training_sizes:
                X_train_temp, y_train_temp = split_data(X_train_Baru, y_train_Baru, training_size, 6)
                if training_size == 1.0:
                    train_error, temp_time1 = bay(X_train_temp, y_train_temp, X_train_temp, y_train_temp, recall_score, 5,
                                              0, 10, 0, 1)
                    error_rate_training_data.append(train_error)
                    test_error, temp_time2 = bay(X_train_temp, y_train_temp, X_test_Baru, y_test_Baru, recall_score, 5,
                                             0, 10, 1, 1)
                    error_rate_testing_data.append(test_error)
                    time_taken_temp.append(temp_time1)
                else:
                    train_error, temp_time1 = bay(X_train_temp, y_train_temp, X_train_temp, y_train_temp, recall_score, 5,
                                              0, 10, 0, 1)
                    error_rate_training_data.append(train_error)
                    test_error, temp_time2 = bay(X_train_temp, y_train_temp, X_test_Baru, y_test_Baru, recall_score, 5,
                                             0, 10, 0, 1)
                    error_rate_testing_data.append(test_error)
                    time_taken_temp.append(temp_time1)

            time_taken1['Naive Bayes'] = time_taken_temp
            plt.plot(training_sizes, error_rate_training_data, label='Training')
            plt.plot(training_sizes, error_rate_testing_data, label='Testing')
            plt.legend()
            plt.xlabel("Training Sizes as a fraction of the original training data")
            plt.ylabel("F1 score")
            plt.title("Naive Bayes")
            plt.savefig("Baru-Credit-Data-NB.png")
            plt.close()

        elif algorithm == 'knn':
            for training_size in training_sizes:
                X_train_temp, y_train_temp = split_data(X_train_Baru, y_train_Baru, training_size, 6)
                if training_size == 1.0:
                    train_error, temp_time1 = KNN(X_train_temp, y_train_temp, X_train_temp, y_train_temp, accuracy_score, 5,
                                              0, 10, 0, 1)
                    error_rate_training_data.append(train_error)
                    test_error, temp_time2 = KNN(X_train_temp, y_train_temp, X_test_Baru, y_test_Baru,
                                             accuracy_score, 5, 0, 10, 1, 1)
                    error_rate_testing_data.append(test_error)
                    time_taken_temp.append(temp_time1)
                else:
                    train_error, temp_time1 = KNN(X_train_temp, y_train_temp, X_train_temp, y_train_temp, accuracy_score, 5,
                                              0, 10, 0, 1)
                    error_rate_training_data.append(train_error)
                    test_error, temp_time2 = KNN(X_train_temp, y_train_temp, X_test_Baru, y_test_Baru,
                                             accuracy_score, 5, 0, 10, 0, 1)
                    error_rate_testing_data.append(test_error)
                    time_taken_temp.append(temp_time1)

            time_taken1['KNN'] = time_taken_temp
            plt.plot(training_sizes, error_rate_training_data, label='Training')
            plt.plot(training_sizes, error_rate_testing_data, label='Testing')
            plt.legend()
            plt.xlabel("Training Sizes as a fraction of the original training data")
            plt.ylabel("F1 score")
            plt.title("KNN")
            plt.savefig("Baru-Credit-Data-KNN.png")
            plt.close()

        elif algorithm == 'dtl':
            for training_size in training_sizes:
                X_train_temp, y_train_temp = split_data(X_train_Baru, y_train_Baru, training_size, 6)
                if training_size == 1.0:
                    train_error, temp_time1 = dt(X_train_temp, y_train_temp, X_train_temp, y_train_temp, accuracy_score, 5,
                                             0, 10, 0, 1)
                    error_rate_training_data.append(train_error)
                    test_error, temp_time2 = dt(X_train_temp, y_train_temp, X_test_Baru, y_test_Baru,
                                            accuracy_score, 5, 0, 10, 1, 1)
                    error_rate_testing_data.append(test_error)
                    time_taken_temp.append(temp_time1)
                else:
                    train_error, temp_time1 = dt(X_train_temp, y_train_temp, X_train_temp, y_train_temp, accuracy_score, 5,
                                             0, 10, 0, 1)
                    error_rate_training_data.append(train_error)
                    test_error, temp_time2 = dt(X_train_temp, y_train_temp, X_test_Baru, y_test_Baru,
                                            accuracy_score, 5, 0, 10, 0, 1)
                    error_rate_testing_data.append(test_error)
                    time_taken_temp.append(temp_time1)

            time_taken1['Decision Tree'] = time_taken_temp
            plt.plot(training_sizes, error_rate_training_data, label='Training')
            plt.plot(training_sizes, error_rate_testing_data, label='Testing')
            plt.legend()
            plt.xlabel("Training Sizes as a fraction of the original training data")
            plt.ylabel("F1 score")
            plt.title("Decision Tree")
            plt.savefig("Baru-Credit-Data-DT.png")
            plt.close()


        elif  algorithm == 'svm':
            for training_size in training_sizes:
                X_train_temp, y_train_temp = split_data(X_train_Baru, y_train_Baru, training_size, 6)
                if training_size == 1.0:
                    train_error, temp_time1 = svc(X_train_temp, y_train_temp, X_train_temp, y_train_temp, accuracy_score, 5,
                                              0, ['linear'], 10, 0, 1)
                    error_rate_training_data.append(train_error)
                    test_error, temp_time2 = svc(X_train_temp, y_train_temp, X_test_Baru, y_test_Baru,
                                             accuracy_score, 5, 0, ['linear'], 10, 1, 1)
                    error_rate_testing_data.append(test_error)
                    time_taken_temp.append(temp_time1)
                else:
                    train_error, temp_time1 = svc(X_train_temp, y_train_temp, X_train_temp, y_train_temp, accuracy_score, 5,
                                              0, ['linear'], 10, 0, 1)
                    error_rate_training_data.append(train_error)
                    test_error, temp_time2 = svc(X_train_temp, y_train_temp, X_test_Baru, y_test_Baru,
                                             accuracy_score, 5, 0, ['linear'], 10, 0, 1)
                    error_rate_testing_data.append(test_error)
                    time_taken_temp.append(temp_time1)

            time_taken1['SVM'] = time_taken_temp
            plt.plot(training_sizes, error_rate_training_data, label='Training')
            plt.plot(training_sizes, error_rate_testing_data, label='Testing')
            plt.legend()
            plt.xlabel("Training Sizes as a fraction of the original training data")
            plt.ylabel("F1 score")
            plt.title("SVM")
            plt.savefig("Baru-Credit-Data-SVM.png")
            plt.close()

        elif algorithm == 'nn':
            for training_size in training_sizes:
                X_train_temp, y_train_temp = split_data(X_train_Baru, y_train_Baru, training_size, 6)
                if training_size == 1.0:
                    train_error, temp_time1 = nn(X_train_temp, y_train_temp, X_train_temp, y_train_temp, accuracy_score, 5,
                                             0, 10, 0, 1)
                    error_rate_training_data.append(train_error)
                    test_error, temp_time2 = nn(X_train_temp, y_train_temp, X_test_Baru, y_test_Baru,
                                            accuracy_score, 5, 0, 10, 1, 1)
                    error_rate_testing_data.append(test_error)
                    time_taken_temp.append(temp_time1)
                else:
                    train_error, temp_time1 = nn(X_train_temp, y_train_temp, X_train_temp, y_train_temp, accuracy_score, 5,
                                             0, 10, 0, 1)
                    error_rate_training_data.append(train_error)
                    test_error, temp_time2 = nn(X_train_temp, y_train_temp, X_test_Baru, y_test_Baru,
                                            accuracy_score, 5, 0, 10, 0, 1)
                    error_rate_testing_data.append(test_error)
                    time_taken_temp.append(temp_time1)

            time_taken1['Neural Network'] = time_taken_temp
            plt.plot(training_sizes, error_rate_training_data, label='Training')
            plt.plot(training_sizes, error_rate_testing_data, label='Testing')
            plt.legend()
            plt.xlabel("Training Sizes as a fraction of the original training data")
            plt.ylabel("F1 score")
            plt.title("Neural Network")
            plt.savefig("Baru-Credit-Data-NN.png")
            plt.close()

    for algorithm in time_taken1:
        plt.plot(training_sizes, time_taken1[algorithm], label=algorithm)
    plt.xlabel("Training Sizes as a fraction of the original training data")
    plt.ylabel("Training time (seconds)")
    plt.legend()
    plt.title("Training Time for Baru Credit Data")
    plt.savefig("Baru-Credit-Data-Time.png")
    plt.close()

    print("------------------------------------------------------------")
    print("------------------------------------------------------------")
    print()
