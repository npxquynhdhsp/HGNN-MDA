import numpy as np
from params import args
import random
import torch
import os
from sklearn.utils import shuffle
from utils.utils import calculate_score

def save_eval(method_set_name, true_set, prob_set):
    print(method_set_name,':')
    calculate_score(true_set, prob_set) # cal mean
    prob_set_join = np.concatenate(prob_set, axis = 0) # join
    np.savetxt(args.fi_out + 'prob_set_' + method_set_name.lower() + '.csv', prob_set_join)

    if method_set_name == 'XG':
        true_set_join = np.concatenate(true_set, axis = 0)
        np.savetxt(args.fi_out + 'true_set.csv', true_set_join, fmt='%d')

        #QX
    return

def save_eval2_all(method_set_name, true_set, prob_set):
    print(method_set_name,':')
    prob_set_join = np.concatenate(prob_set, axis = 0) # join
    np.savetxt(args.fi_out + 'prob_set' + method_set_name.lower() + '.csv', prob_set_join)

    true_set_join = np.concatenate(true_set, axis = 0)
    np.savetxt(args.fi_out + 'true_set.csv', true_set_join, fmt='%d')
    calculate_score([true_set_join], [prob_set_join])

def savekq(method_set_name, true_set, prob_set): # Q QT tuypp
    if args.db == "INDE_TEST":
        save_eval2_all(method_set_name, true_set, prob_set)
    else:
        if args.type_eval == "DENO_MI":
            save_eval2_all(method_set_name, true_set, prob_set)
        else:
            save_eval(method_set_name, true_set, prob_set)

def set_seed(seed):
    # Python random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Cố định tính toán
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Đặt biến môi trường cho reproducibility (đôi khi cần)
    os.environ["PYTHONHASHSEED"] = str(seed)

def balance_data(X, y, neg_rate):
    if (neg_rate == -1): #Q DU PHONG, or.. cmt
        print(X.shape)
        return X, y
    else:
        x_pos = X[y == 1]
        x_neg = X[y == 0]

        npos = x_pos.shape[0]

        x_neg_new = shuffle(x_neg, random_state=2022)
        x_neg_new = x_neg_new[:npos * neg_rate]

        X_new = np.vstack([x_pos, x_neg_new])
        y_new = np.hstack([np.ones(npos), np.zeros(npos * neg_rate)])

        X_balanced, y_balanced = shuffle(X_new, y_new, random_state=2022)

        return X_balanced, y_balanced