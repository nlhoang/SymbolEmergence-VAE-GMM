import sys
import time

import numpy as np
from sklearn import metrics

sys.path.append("../lib/")
import BoF
import datetime
import matplotlib.pyplot as plt

start_time = time.time()

dir_a = "../data/Fruit360/Categories10_BoF/histogramA_200.txt"
dir_b = "../data/Fruit360/Categories10_BoF/histogramB_200.txt"
label = "../data/Fruit360/Categories10_BoF/labelA_200.txt"
data_set_num = 1  # データセットの数

DATA_NUM = 4700  # 2350
WORD_DIM = 20  # sign
CONCEPT_DIM = 20  # category
iteration = 100  # イテレーション(反復回数)

# ハイパーパラメータ設定
beta_c_a_vision = 0.001
alpha_w_a = 0.01
beta_c_b_vision = 0.001
alpha_w_b = 0.01

# 重みの設定
w_a_v = 1.0
w_b_v = 1.0
w_a_w = 1.0
w_b_w = 1.0

# エージェントごとのカテゴリ数
concept_num_a = CONCEPT_DIM
concept_num_b = CONCEPT_DIM


# ===============================================================================

def softmax(arr, axis=0):
    arr = np.rollaxis(arr, axis)
    vmax = arr.max(axis=0)
    out = np.exp(arr - vmax) / np.sum(np.exp(arr - vmax), axis=0)
    return out


def new_feature_data_read(directory):
    all_feature = []
    f = directory
    feat = np.loadtxt(f)
    all_feature.append(feat)
    return all_feature


def Multi_prob(data, phi):
    phi_log = np.log(phi)
    prob = data.dot(phi_log.T)
    return prob


# ===============================================================================

# Metropolis-Hasting algorithm
def Metropolis_Hasting(iteration):
    # データ読み込み
    # Initialize
    print("Parameter Initialization 1")
    word_set_t = np.loadtxt(label)
    # ---------------------------------------------------------------------------

    # 各物体のカテゴリ決定(初期化) Category determination for each object (initialization)
    c_w_a = [1000 for n in range(DATA_NUM)]
    c_w_b = [1000 for n in range(DATA_NUM)]

    # 物体ごとの記号(サイン)当て Symbol (sign) assignment for each object
    word_a = np.zeros((DATA_NUM, WORD_DIM))
    word_b = np.zeros((DATA_NUM, WORD_DIM))

    initial_word_a = np.zeros((DATA_NUM, WORD_DIM))
    initial_word_b = np.zeros((DATA_NUM, WORD_DIM))

    # ARI, Kappaの結果
    ARI_a = np.zeros((iteration))
    ARI_b = np.zeros((iteration))
    ARI_ab = np.zeros((iteration))
    concidence = np.zeros((iteration))

    # ---------------------------------------------------------------------------

    # 特徴データ読み込み Load feature data
    print("Reading Data")
    feature_set_a_vision = new_feature_data_read(dir_a)
    feature_set_b_vision = new_feature_data_read(dir_b)

    # 各特徴の次元数 number of dimensions for each feature
    FEATURE_DIM_a_vision = len(feature_set_a_vision[0][0])
    FEATURE_DIM_b_vision = len(feature_set_b_vision[0][0])

    # ===========================================================================

    # 多項分布パラメータphi,theta設定 Set multinomial distribution parameters phi and theta
    print("Parameter Initialization 2")
    # 各カテゴリにどの特徴が振られるか Which features are assigned to each category
    phi_f_e_a_vision = np.array(
        [[[float(1.0) / FEATURE_DIM_a_vision for i in range(FEATURE_DIM_a_vision)] for j in range(concept_num_a)] for k
         in range(data_set_num)])

    phi_f_e_b_vision = np.array(
        [[[float(1.0) / FEATURE_DIM_b_vision for i in range(FEATURE_DIM_b_vision)] for j in range(concept_num_b)] for k
         in range(data_set_num)])

    # 各記号(サイン)にどのカテゴリが振られるか Which category is assigned to each symbol (sign)
    theta_w_e_a = np.array(
        [[[float(1.0) / concept_num_a for i in range(concept_num_a)] for j in range(WORD_DIM)] for k in
         range(data_set_num)])
    theta_w_e_b = np.array(
        [[[float(1.0) / concept_num_b for i in range(concept_num_b)] for j in range(WORD_DIM)] for k in
         range(data_set_num)])

    # カテゴリ番号 category number
    class_choice_a = [dc for dc in range(concept_num_a)]
    class_choice_b = [dc for dc in range(concept_num_b)]

    # サイン番号 sign number
    word_choice = [dc for dc in range(WORD_DIM)]  # [0,1,2,...,14]

    # 記号(サイン)に対する乱数ベクトルの初期化 Initialize Random Vector for Symbol (Sine)
    rand_set_a = []
    for d in range(DATA_NUM):
        rand_a = np.random.randint(0, WORD_DIM)
        rand_set_a.append(rand_a)

    rand_set_b = []
    for d in range(DATA_NUM):
        rand_b = np.random.randint(0, WORD_DIM)
        rand_set_b.append(rand_b)

    for d in range(DATA_NUM):
        word_a[d][rand_set_a[d]] = 1
        initial_word_a[d][rand_set_a[d]] = 1

        word_b[d][rand_set_b[d]] = 1
        initial_word_b[d][rand_set_b[d]] = 1

    # ===========================================================================
    # print(theta_w_e_a)
    for e in range(data_set_num):  # 1回
        # エージェントAの各データへのカテゴリの割り当て Assign categories to each data for agent A
        class_count_e_set_a = []
        theta_w_log_a = np.log(theta_w_e_a[0])
        for d in range(DATA_NUM):
            class_count_a = [0.0 for i in range(concept_num_a)]
            multi_prob_set_a = np.zeros((concept_num_a), dtype=float)

            if w_a_v > 0.0:
                for i in range(concept_num_a):
                    multi_prob_set_a[i] += w_a_v * Multi_prob(feature_set_a_vision[0][d], phi_f_e_a_vision[0][i])

            # 記号(サイン)情報 Symbol (sign) information
            for j in range(WORD_DIM):
                if word_a[d][j] == 1:
                    for k in range(concept_num_a):
                        multi_prob_set_a[k] += w_a_w * theta_w_log_a[j][k]

            multi_prob_set_a = softmax(multi_prob_set_a)
            # print("multi_prob_set_a : ", multi_prob_set_a)
            c_w_a[d] = np.random.choice(class_choice_a, p=multi_prob_set_a)
            # print(c_w_a[d])
            class_count_a[c_w_a[d]] += 1.0
            class_count_e_set_a.append(class_count_a)

        # データに割り当てられたカテゴリから, phi_f_e_a_* を計算 Calculate phi_f_e_a_* from the categories assigned to the data
        for c in range(concept_num_a):
            feat_e_c_a_vision = []

            for d in range(DATA_NUM):
                if c_w_a[d] == c:
                    feat_e_c_a_vision.append(feature_set_a_vision[0][d])

            total_feat_e_a_vision = BoF.bag_of_feature(feat_e_c_a_vision, FEATURE_DIM_a_vision)
            total_feat_e_a_vision = total_feat_e_a_vision + beta_c_a_vision

            phi_f_e_a_vision[0][c] = np.random.dirichlet(total_feat_e_a_vision) + 1e-100

        # データに割り当てられたカテゴリから, theta_w_a を計算 Calculate theta_w_a from the categories assigned to the data
        for w in range(WORD_DIM):
            c_e_w_a = []
            for d in range(DATA_NUM):
                if word_a[d][w] == 1:
                    c_e_w_a.append(class_count_e_set_a[d])
            total_c_e_a = BoF.bag_of_feature(c_e_w_a, concept_num_a)
            total_c_e_a = total_c_e_a + alpha_w_a

            theta_w_e_a[0][w] = np.random.dirichlet(total_c_e_a) + 1e-100

        # エージェントBの各データへのカテゴリの割り当て Assigning Categories to Each Data in Agent B
        class_count_e_set_b = []
        theta_w_log_b = np.log(theta_w_e_b[0])
        for d in range(DATA_NUM):
            class_count_b = [0.0 for i in range(concept_num_b)]
            multi_prob_set_b = np.zeros((concept_num_b), dtype=float)

            if w_b_v > 0.0:
                for i in range(concept_num_b):
                    multi_prob_set_b[i] += w_b_v * Multi_prob(feature_set_b_vision[0][d], phi_f_e_b_vision[0][i])

            # 記号(サイン)情報 Symbol (sign) information
            for j in range(WORD_DIM):
                if word_b[d][j] == 1:
                    for k in range(concept_num_b):
                        multi_prob_set_b[k] += w_b_w * theta_w_log_b[j][k]
            # print("multi_prob_set_b", multi_prob_set_b.shape)
            multi_prob_set_b = softmax(multi_prob_set_b)
            c_w_b[d] = np.random.choice(class_choice_b, p=multi_prob_set_b)
            class_count_b[c_w_b[d]] += 1.0
            class_count_e_set_b.append(class_count_b)

        # データに割り当てられたカテゴリから, phi_f_e_b_* を計算 Calculate phi_f_e_b_* from the categories assigned to the data
        for c in range(concept_num_b):
            feat_e_c_b_vision = []

            for d in range(DATA_NUM):
                if c_w_b[d] == c:
                    feat_e_c_b_vision.append(feature_set_b_vision[0][d])

            total_feat_e_b_vision = BoF.bag_of_feature(feat_e_c_b_vision, FEATURE_DIM_b_vision)
            total_feat_e_b_vision = total_feat_e_b_vision + beta_c_b_vision

            phi_f_e_b_vision[0][c] = np.random.dirichlet(total_feat_e_b_vision) + 1e-100

        # データ割り当てられたカテゴリから, theta_w_b を計算 theta_w_a を計算 Calculate theta_w_b from the categories assigned to the data
        for w in range(WORD_DIM):
            c_e_w_b = []
            for d in range(DATA_NUM):
                if word_b[d][w] == 1:
                    c_e_w_b.append(class_count_e_set_b[d])
            total_c_e_b = BoF.bag_of_feature(c_e_w_b, concept_num_b)
            total_c_e_b = total_c_e_b + alpha_w_b

            theta_w_e_b[0][w] = np.random.dirichlet(total_c_e_b) + 1e-100

    # ===========================================================================
    """
    以下M-H法
    """
    # パラメータ推定 parameter estimation
    for iter in range(iteration):
        # print("-------------iteration" + repr(iter) + "-------------")
        for e in range(data_set_num):
            # エージェントAから記号(サイン)のサンプリング Sampling of symbols (signs) from agent A
            new_word_a_set = []
            for d in range(DATA_NUM):
                word_multi_prob_set_a = np.zeros(WORD_DIM, dtype=float)
                total_phi_a = 0.0
                for w in range(WORD_DIM):
                    total_phi_a += theta_w_e_a[0][w][c_w_a[d]]
                    word_multi_prob_set_a[w] = theta_w_e_a[0][w][c_w_a[d]]

                word_multi_prob_set_a = word_multi_prob_set_a / total_phi_a
                new_word_a_set.append(np.random.choice(word_choice, p=word_multi_prob_set_a))

            # A提案の記号(サイン)の取捨選択 A Selection of symbols (signs) for proposals
            for d in range(DATA_NUM):
                word_multi_prob_b = np.zeros(1, dtype=float)
                word_multi_prob_b = theta_w_e_b[0][rand_set_b[d]][c_w_b[d]]
                # print("word_multi_prob_b : ", word_multi_prob_b)
                new_word_multi_prob_b = np.zeros(1, dtype=float)
                new_word_multi_prob_b = theta_w_e_b[0][new_word_a_set[d]][c_w_b[d]]
                judge_r = new_word_multi_prob_b / word_multi_prob_b
                # print("judge_r : ",judge_r)
                judge_r = min(1, judge_r)
                rand_u = np.random.rand()

                if (judge_r >= rand_u):
                    rand_set_b[d] = new_word_a_set[d]
                    for i in range(WORD_DIM):
                        word_b[d][i] = 0
                    word_b[d][new_word_a_set[d]] = 1

            # エージェントBの各データへのカテゴリの再割り当て Reassign categories to each data for agent B
            class_count_e_set_b = []
            theta_w_log_b = np.log(theta_w_e_b[0])
            # print("theta_w_log_b : ", theta_w_log_b)
            for d in range(DATA_NUM):
                class_count_b = [0.0 for i in range(concept_num_b)]
                multi_prob_set_b = np.zeros((concept_num_b), dtype=float)

                if w_b_v > 0.0:
                    for i in range(concept_num_b):
                        multi_prob_set_b[i] += w_b_v * Multi_prob(feature_set_b_vision[0][d], phi_f_e_b_vision[0][i])

                ##記号(サイン)情報
                for j in range(WORD_DIM):
                    if word_b[d][j] == 1:
                        for k in range(concept_num_b):
                            multi_prob_set_b[k] += w_b_w * theta_w_log_b[j][k]
                # print("multi_prob_set_b : ", multi_prob_set_b.shape)
                multi_prob_set_b = softmax(multi_prob_set_b)
                c_w_b[d] = np.random.choice(class_choice_b, p=multi_prob_set_b)
                class_count_b[c_w_b[d]] += 1.0
                class_count_e_set_b.append(class_count_b)

            ##データに割り当てられたカテゴリから, phi_f_b_* の再計算
            for c in range(concept_num_b):
                feat_e_c_b_vision = []

                for d in range(DATA_NUM):
                    if c_w_b[d] == c:
                        feat_e_c_b_vision.append(feature_set_b_vision[0][d])

                total_feat_e_b_vision = BoF.bag_of_feature(feat_e_c_b_vision, FEATURE_DIM_b_vision)
                total_feat_e_b_vision = total_feat_e_b_vision + beta_c_b_vision

                phi_f_e_b_vision[0][c] = np.random.dirichlet(total_feat_e_b_vision) + 1e-100

            ##データに割り当てられたカテゴリから, theta_w_b の再計算
            for w in range(WORD_DIM):
                c_e_w_b = []
                for d in range(DATA_NUM):
                    if word_b[d][w] == 1:
                        c_e_w_b.append(class_count_e_set_b[d])
                total_c_e_b = BoF.bag_of_feature(c_e_w_b, concept_num_b)
                total_c_e_b = total_c_e_b + alpha_w_b

                theta_w_e_b[0][w] = np.random.dirichlet(total_c_e_b) + 1e-100

            ##エージェントBから記号(サイン)のサンプリング
            new_word_b_set = []
            for d in range(DATA_NUM):
                word_multi_prob_set_b = np.zeros(WORD_DIM, dtype=float)
                total_phi_b = 0.0
                for w in range(WORD_DIM):
                    total_phi_b += theta_w_e_b[0][w][c_w_b[d]]
                    word_multi_prob_set_b[w] = theta_w_e_b[0][w][c_w_b[d]]

                word_multi_prob_set_b = word_multi_prob_set_b / total_phi_b

                new_word_b_set.append(np.random.choice(word_choice, p=word_multi_prob_set_b))

            ##B提案の記号(サイン)の取捨選択
            for d in range(DATA_NUM):
                word_multi_prob_a = np.zeros(1, dtype=float)
                word_multi_prob_a = theta_w_e_a[0][rand_set_a[d]][c_w_a[d]]

                new_word_multi_prob_a = np.zeros(1, dtype=float)
                new_word_multi_prob_a = theta_w_e_a[0][new_word_b_set[d]][c_w_a[d]]

                judge_r = new_word_multi_prob_a / word_multi_prob_a
                judge_r = min(1, judge_r)
                rand_u = np.random.rand()

                if (judge_r >= rand_u):
                    rand_set_a[d] = new_word_b_set[d]
                    for i in range(WORD_DIM):
                        word_a[d][i] = 0
                    word_a[d][new_word_b_set[d]] = 1

            ##エージェントAの各データへのカテゴリの再割り当て
            class_count_e_set_a = []
            theta_w_log_a = np.log(theta_w_e_a[0])
            for d in range(DATA_NUM):
                class_count_a = [0.0 for i in range(concept_num_a)]
                multi_prob_set_a = np.zeros((concept_num_a), dtype=float)

                ##視覚情報
                if w_a_v > 0.0:
                    for i in range(concept_num_a):
                        multi_prob_set_a[i] += w_a_v * Multi_prob(feature_set_a_vision[0][d], phi_f_e_a_vision[0][i])

                ##記号(サイン)情報
                for j in range(WORD_DIM):
                    if word_a[d][j] == 1:
                        for k in range(concept_num_a):
                            multi_prob_set_a[k] += w_a_w * theta_w_log_a[j][k]

                multi_prob_set_a = softmax(multi_prob_set_a)
                c_w_a[d] = np.random.choice(class_choice_a, p=multi_prob_set_a)
                class_count_a[c_w_a[d]] += 1.0
                class_count_e_set_a.append(class_count_a)

            ##データに割り当てられたカテゴリから, phi_f_a_* の再計算
            for c in range(concept_num_a):
                feat_e_c_a_vision = []

                for d in range(DATA_NUM):
                    if c_w_a[d] == c:
                        feat_e_c_a_vision.append(feature_set_a_vision[0][d])

                total_feat_e_a_vision = BoF.bag_of_feature(feat_e_c_a_vision, FEATURE_DIM_a_vision)
                total_feat_e_a_vision = total_feat_e_a_vision + beta_c_a_vision

                phi_f_e_a_vision[0][c] = np.random.dirichlet(total_feat_e_a_vision) + 1e-100

            ##データに割り当てられたカテゴリから, theta_w_a の再計算
            for w in range(WORD_DIM):
                c_e_w_a = []
                for d in range(DATA_NUM):
                    if word_a[d][w] == 1:
                        c_e_w_a.append(class_count_e_set_a[d])
                total_c_e_a = BoF.bag_of_feature(c_e_w_a, concept_num_a)
                total_c_e_a = total_c_e_a + alpha_w_a

                theta_w_e_a[0][w] = np.random.dirichlet(total_c_e_a) + 1e-100

            ##評価値計算
            sum_same_w = 0.0
            a_chance = 0.0
            prob_w = [0.0 for i in range(WORD_DIM)]
            w_count_a = [0.0 for i in range(WORD_DIM)]
            w_count_b = [0.0 for i in range(WORD_DIM)]

            for d in range(DATA_NUM):
                if rand_set_a[d] == rand_set_b[d]:
                    sum_same_w += 1

                for w in range(WORD_DIM):
                    if rand_set_a[d] == w:
                        w_count_a[w] += 1
                    if rand_set_b[d] == w:
                        w_count_b[w] += 1

            for w in range(WORD_DIM):
                prob_w[w] = (w_count_a[w] / DATA_NUM) * (w_count_b[w] / DATA_NUM)
                a_chance += prob_w[w]
            a_observed = (sum_same_w / DATA_NUM)

            ###Kappa係数の計算
            concidence[iter] = np.round((a_observed - a_chance) / (1 - a_chance), 3)

            ###ARIの計算
            ARI_a[iter] = np.round(metrics.adjusted_rand_score(word_set_t, c_w_a), 3)
            ARI_b[iter] = np.round(metrics.adjusted_rand_score(word_set_t, c_w_b), 3)
            ARI_ab[iter] = np.round(metrics.adjusted_rand_score(c_w_a, c_w_b), 3)
            # print("c_w_a : ", c_w_a)
            # print("c_w_b : ", len(c_w_b))

            ###confusion_matrixの計算
            # if iter == 0:
            #    initial_confusion_matrix_a = metrics.confusion_matrix(word_set_t, c_w_a)
            #    initial_confusion_matrix_b = metrics.confusion_matrix(word_set_t, c_w_b)
            # if iter == (iteration - 1):
            #    final_confusion_matrix_a = metrics.confusion_matrix(word_set_t, c_w_a)
            #    final_confusion_matrix_b = metrics.confusion_matrix(word_set_t, c_w_b)

            ###評価値の表示
            # print('ARI_a = ', ARI_a[iter])
            # print('ARI_b = ', ARI_b[iter])
            # print('ARI_ab = ', ARI_ab[iter])
            # print('concidence = ', concidence[iter])
            print(
                f"=> Epoch: {iter}, ARI_A:{ARI_a[iter]}, ARI_B:{ARI_b[iter]}, Concidence:{concidence[iter]}, ARI_ab:{ARI_ab[iter]}")

    # print(word_set_t)

    # ===========================================================================

    # データ保存
    today = datetime.date.today()
    todaydetail = datetime.datetime.today()

    ##出力ファイルパスの設定
    Out_put_dir = "./model/dm"  # 確認or練習用
    # Out_put_dir = "../result/Experiment/Data0/MH_S_iter_{}".format(iteration)
    # Out_put_dir = "../result/S_Experiment/Data0/MH_V_iter_{}".format(iteration)

    ##処理時間計算
    finish_time = time.time() - start_time
    f = open(Out_put_dir + "/time.txt", "w")
    f.write("time: " + repr(finish_time) + "seconds.")
    f.close()

    ##環境変数保存
    f = open(Out_put_dir + "/Parameter.txt", "w")
    f.write("Iteration: " + repr(iteration) +
            "\nDATA_NUM: " + repr(DATA_NUM) +
            "\nWORD_DIM: " + repr(WORD_DIM) +
            "\nbeta_c_a_vision" + repr(beta_c_a_vision) +
            "\nalpha_w_a" + repr(alpha_w_a) +
            "\nbeta_c_b_vision" + repr(beta_c_b_vision) +
            "\nalpha_w_b" + repr(alpha_w_b) +
            "\nconcept_num_a" + repr(concept_num_a) +
            "\nconcept_num_b" + repr(concept_num_b)
            )
    f.close()

    np.savetxt(Out_put_dir + "/ariA.txt", ARI_a)
    np.savetxt(Out_put_dir + "/ariB.txt", ARI_b)
    # np.savetxt(Out_put_dir + "/ARI_ab.txt", ARI_ab)
    np.savetxt(Out_put_dir + "/kappa.txt", concidence)

    # concidence
    plt.figure()
    plt.plot(range(0, iteration), concidence, marker="None")
    plt.xlabel('iteration');
    plt.ylabel('Concidence')
    plt.ylim(0, 1)
    plt.title('k')
    plt.savefig(Out_put_dir + "/conf.png")
    # plt.show()
    plt.close()

    # ARI
    plt.figure()
    plt.plot(range(0, iteration), ARI_a, marker="None", label="ARI_A")
    plt.plot(range(0, iteration), ARI_b, marker="None", label="ARI_B")
    plt.xlabel('iteration');
    plt.ylabel('ARI')
    plt.ylim(0, 1)
    plt.legend()
    plt.title('ARI')
    plt.savefig(Out_put_dir + "/ari.png")
    # plt.show()
    plt.close()


if __name__ == '__main__':
    Metropolis_Hasting(50)
