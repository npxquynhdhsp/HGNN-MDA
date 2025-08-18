# %%
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
from params import args

# %%
def calculate_score(true_list, predict_list):
    auc_list = []
    auprc_list = []
    accuracy_list = []
    precision_list = []
    recall_list = []
    f1_list = []
    for index, (true_fold, predict_fold) in enumerate(zip(true_list, predict_list)):
        auc = metrics.roc_auc_score(true_fold, predict_fold)
        precision, recall, thresholds = metrics.precision_recall_curve(true_fold, predict_fold)
        # auprc = metrics.auc(recall, precision)
        auprc = metrics.average_precision_score(true_fold, predict_fold)
        
        result_fold = [0 if j < 0.5 else 1 for j in predict_fold]
        accuracy = metrics.accuracy_score(true_fold, result_fold)
        precision = metrics.precision_score(true_fold, result_fold)
        recall = metrics.recall_score(true_fold, result_fold)
        f1 = metrics.f1_score(true_fold, result_fold)

        accuracy_list.append(accuracy)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
        auc_list.append(auc)
        auprc_list.append(auprc)
    print('AUC mean: %.4f, variance: %.4f \n' % (np.mean(auc_list), np.std(auc_list)),
          'AUPRC mean: %.4f, variance: %.4f \n' % (np.mean(auprc_list), np.std(auprc_list)),
          sep="")

    return np.mean(auc_list), np.mean(auprc_list), np.mean(accuracy_list), \
        np.mean(precision_list), np.mean(recall_list), np.mean(f1_list)

def models_eval(method_set_name, X_train_enc, X_test_enc, y_train, y_test, ix, loop_i, model_i):
    if method_set_name == 'RF':
        print('Random Forest')
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators=args.rf_ne, max_depth=None, n_jobs=-1)
    elif method_set_name == 'ETR':
        print('Extra trees regression')
        from sklearn.ensemble import ExtraTreesRegressor
        clf = ExtraTreesRegressor(n_estimators=args.etr_ne, n_jobs=-1)
    elif method_set_name == 'LR':
        print('Linear regression')
        from sklearn.linear_model import LinearRegression
        clf = LinearRegression()
    else:
        print('XGBoost')
        from xgboost import XGBClassifier
        clf = XGBClassifier(booster='gbtree', n_jobs=2, learning_rate=args.xg_lrr, n_estimators=args.xg_ne)

    clf.fit(X_train_enc, y_train)

    if (method_set_name == 'ETR') or (method_set_name == 'LR'):
        y_prob = clf.predict(X_test_enc)
    else:
        y_prob = clf.predict_proba(X_test_enc)[:,1]

    np.savetxt(args.fi_out + 'L' + str(loop_i) + '_M' + str(model_i) + '_yprob_' + method_set_name.lower() + str(ix) + '.csv', y_prob)
    calculate_score([y_test], [y_prob])
    return y_prob


