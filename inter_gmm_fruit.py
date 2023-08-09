import os
import numpy as np
from scipy.stats import wishart, multivariate_normal
import matplotlib.pyplot as plt
import argparse
from sklearn.metrics.cluster import adjusted_rand_score as ari
from sklearn.metrics import cohen_kappa_score


def new_feature_data_read(directory):
    all_feature = []
    f = directory
    feat = np.loadtxt(f)
    all_feature.append(feat)
    return all_feature


parser = argparse.ArgumentParser(description='M-H algorithm M-GMM Example')
parser.add_argument('--sign', type=int, default=10, metavar='K', help='Number of sign')
parser.add_argument('--mode', type=int, default=-1, metavar='M', help='0:All reject, 1:ALL accept')
parser.add_argument('--iteration', type=int, default=50, metavar='N', help='number of iteration for MGMM_MH')
args = parser.parse_args()

file_name = "gmm"; model_dir = "./model"; dir_name = "./model/"+file_name# debugフォルダに保存される
if not os.path.exists(model_dir):   os.mkdir(model_dir)
if not os.path.exists(dir_name):    os.mkdir(dir_name)

K = args.sign # number of category

# load dataset
dir_a = "../data/Fruit360/Categories10_BoF/histogramA_50.txt"
dir_b = "../data/Fruit360/Categories10_BoF/histogramB_50.txt"
label = "../data/Fruit360/Categories10_BoF/label.txt"
c_nd_A = np.loadtxt(dir_a)
c_nd_B = np.loadtxt(dir_b)
z_truth_n = np.loadtxt(label)

D = len(c_nd_A) # number of data
dim = len(c_nd_A[0]) # dimention of data

############################## Initializing parameters ##############################
# hyper parameter
beta = 1.0; m_d_A = np.repeat(0.0, dim)
m_d_B = np.repeat(0.0, dim)
w_dd_A = np.identity(dim) * 0.1
w_dd_B = np.identity(dim) * 0.1 
nu = dim

# Initializing \mu, \Lambda
mu_kd_A = np.empty((K, dim))
lambda_kdd_A = np.empty((K, dim, dim))
mu_kd_B = np.empty((K, dim))
lambda_kdd_B = np.empty((K, dim, dim))
for k in range(K):
    lambda_kdd_A[k] = wishart.rvs(df=nu, scale=w_dd_A, size=1)
    lambda_kdd_B[k] = wishart.rvs(df=nu, scale=w_dd_B, size=1)
    mu_kd_A[k] = np.random.multivariate_normal(mean=m_d_A, cov=np.linalg.inv(beta * lambda_kdd_A[k])).flatten()
    mu_kd_B[k] = np.random.multivariate_normal(mean=m_d_B, cov=np.linalg.inv(beta * lambda_kdd_B[k])).flatten()

# initializing w_dk
w_dk_A = np.random.multinomial(1, [1/K]*K, size=D)
w_dk_B = np.random.multinomial(1, [1/K]*K, size=D)

# initializing learning parameters
beta_hat_k_A = np.zeros(K)
beta_hat_k_B = np.zeros(K)
m_hat_kd_A = np.zeros((K, dim))
m_hat_kd_B = np.zeros((K, dim))
w_hat_kdd_A = np.zeros((K, dim, dim))
w_hat_kdd_B = np.zeros((K, dim, dim))
nu_hat_k_A = np.zeros(K)
nu_hat_k_B = np.zeros(K)
tmp_eta_nB = np.zeros((K, D))
eta_dkB = np.zeros((D, K))
tmp_eta_nA = np.zeros((K, D))
eta_dkA = np.zeros((D, K))
cat_liks_A = np.zeros(D)
cat_liks_B = np.zeros(D)

trace_w_in_A = [np.repeat(np.nan, D)]
trace_w_in_B = [np.repeat(np.nan, D)]
trace_mu_ikd_A = [mu_kd_A.copy()]
trace_mu_ikd_B = [mu_kd_B.copy()]
trace_lambda_ikdd_A = [lambda_kdd_A.copy()]
trace_lambda_ikdd_B = [lambda_kdd_B.copy()]
trace_beta_ik_A = [np.repeat(beta, K)]
trace_beta_ik_B = [np.repeat(beta, K)]
trace_m_ikd_A = [np.repeat(m_d_A.reshape((1, dim)), K, axis=0)]
trace_m_ikd_B = [np.repeat(m_d_B.reshape((1, dim)), K, axis=0)]
trace_w_ikdd_A = [np.repeat(w_dd_A.reshape((1, dim, dim)), K, axis=0)]
trace_w_ikdd_B = [np.repeat(w_dd_B.reshape((1, dim, dim)), K, axis=0)]
trace_nu_ik_A = [np.repeat(nu, K)]
trace_nu_ik_B = [np.repeat(nu, K)]

############################## Naming game using M-H algorithm ##############################
iteration = args.iteration # iteration
ARI_A = np.zeros((iteration)) # ARI
ARI_B = np.zeros((iteration)) 
concidence = np.zeros((iteration)) # Kappa score
accept_count_AtoB = np.zeros((iteration)) # number of acceptation
accept_count_BtoA = np.zeros((iteration)) 

for i in range(iteration):
    # categories estimated by the agent
    pred_label_A = []
    pred_label_B = []
    
    count_AtoB = 0
    count_BtoA = 0
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~Speaker:A->Listener:B~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    w_dk = np.zeros((D, K)); 
    for k in range(K): # Speaker:A：update w^A
        tmp_eta_nA[k] = np.diag(-0.5 * (c_nd_A - mu_kd_A[k]).dot(lambda_kdd_A[k]).dot((c_nd_A - mu_kd_A[k]).T)).copy() 
        tmp_eta_nA[k] += 0.5 * np.log(np.linalg.det(lambda_kdd_A[k]) + 1e-7)
        eta_dkA[:, k] = np.exp(tmp_eta_nA[k])
    eta_dkA /= np.sum(eta_dkA, axis=1, keepdims=True)

    for d in range(D):
        pvals = eta_dkA[d]
        if True in np.isnan(np.array(pvals)):
            pvals = [0.1] * 10
        w_dk_A[d] = np.random.multinomial(n=1, pvals=pvals, size=1).flatten()
        pred_label_A.append(np.argmax(w_dk_A[d]))

        if args.mode == 0:
            # no communication
            judge_r = -1 
        elif args.mode == 1:
            # all acceptance
            judge_r = 1000 
        else:
            # Communication (M-H algorithm)
            cat_liks_A[d] = multivariate_normal.pdf(c_nd_B[d], 
                            mean=mu_kd_B[np.argmax(w_dk_A[d])], 
                            cov=np.linalg.inv(lambda_kdd_B[np.argmax(w_dk_A[d])]),
                            )
            cat_liks_B[d] = multivariate_normal.pdf(c_nd_B[d], 
                            mean=mu_kd_B[np.argmax(w_dk_B[d])], 
                            cov=np.linalg.inv(lambda_kdd_B[np.argmax(w_dk_B[d])]),
                            )
            judge_r = cat_liks_A[d] / cat_liks_B[d] 
            judge_r = min(1, judge_r) # acceptance rate
        rand_u = np.random.rand() # sampling random variable
        if judge_r >= rand_u: 
            w_dk[d] = w_dk_A[d]
            count_AtoB = count_AtoB + 1 # count accept
        else: 
            w_dk[d] = w_dk_B[d]

    # update \mu, \lambda of Listener
    for k in range(K):
        # update \mu
        beta_hat_k_B[k] = np.sum(w_dk[:, k]) + beta
        m_hat_kd_B[k] = np.sum(w_dk[:, k] * c_nd_B.T, axis=1)
        m_hat_kd_B[k] += beta * m_d_B; m_hat_kd_B[k] /= beta_hat_k_B[k]
        # update \lambda
        tmp_w_dd_B = np.dot((w_dk[:, k] * c_nd_B.T), c_nd_B)
        tmp_w_dd_B += beta * np.dot(m_d_B.reshape(dim, 1), m_d_B.reshape(1, dim))
        tmp_w_dd_B -= beta_hat_k_B[k] * np.dot(m_hat_kd_B[k].reshape(dim, 1), m_hat_kd_B[k].reshape(1, dim))
        tmp_w_dd_B += np.linalg.inv(w_dd_B)
        w_hat_kdd_B[k] = np.linalg.inv(tmp_w_dd_B)
        nu_hat_k_B[k] = np.sum(w_dk[:, k]) + nu
        # sampling \lambda
        lambda_kdd_B[k] = wishart.rvs(size=1, df=nu_hat_k_B[k], scale=w_hat_kdd_B[k])
        # samplimg \mu
        mu_kd_B[k] = np.random.multivariate_normal(mean=m_hat_kd_B[k], cov=np.linalg.inv(beta_hat_k_B[k] * lambda_kdd_B[k]), size=1).flatten()

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~Speaker:B->Listener:A~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    w_dk = np.zeros((D, K)); 
    for k in range(K): 
        tmp_eta_nB[k] = np.diag(-0.5 * (c_nd_B - mu_kd_B[k]).dot(lambda_kdd_B[k]).dot((c_nd_B - mu_kd_B[k]).T)).copy() 
        tmp_eta_nB[k] += 0.5 * np.log(np.linalg.det(lambda_kdd_B[k]) + 1e-7)
        eta_dkB[:, k] = np.exp(tmp_eta_nB[k])
    eta_dkB /= np.sum(eta_dkB, axis=1, keepdims=True)

    for d in range(D):
        pvals = eta_dkB[d]
        if True in np.isnan(np.array(pvals)):
            pvals = [0.1] * 10
        w_dk_B[d] = np.random.multinomial(n=1, pvals=pvals, size=1).flatten()
        pred_label_B.append(np.argmax(w_dk_B[d]))

        if args.mode == 0:
            judge_r = -1 
        elif args.mode == 1:
            judge_r = 1000
        else:
            cat_liks_B[d] = multivariate_normal.pdf(c_nd_A[d], 
                            mean=mu_kd_A[np.argmax(w_dk_B[d])], 
                            cov=np.linalg.inv(lambda_kdd_A[np.argmax(w_dk_B[d])]),
                            )
            cat_liks_A[d] = multivariate_normal.pdf(c_nd_A[d], 
                    mean=mu_kd_A[np.argmax(w_dk_A[d])], 
                    cov=np.linalg.inv(lambda_kdd_A[np.argmax(w_dk_A[d])]),
                    )
            judge_r = cat_liks_B[d] / cat_liks_A[d]
            judge_r = min(1, judge_r) 
        rand_u = np.random.rand()
        if judge_r >= rand_u: 
            w_dk[d] = w_dk_B[d]
            count_BtoA = count_BtoA + 1
        else: 
            w_dk[d] = w_dk_A[d]

    for k in range(K):
        beta_hat_k_A[k] = np.sum(w_dk[:, k]) + beta; m_hat_kd_A[k] = np.sum(w_dk[:, k] * c_nd_A.T, axis=1)
        m_hat_kd_A[k] += beta * m_d_A; m_hat_kd_A[k] /= beta_hat_k_A[k]
        tmp_w_dd_A = np.dot((w_dk[:, k] * c_nd_A.T), c_nd_A)
        tmp_w_dd_A += beta * np.dot(m_d_A.reshape(dim, 1), m_d_A.reshape(1, dim))
        tmp_w_dd_A -= beta_hat_k_A[k] * np.dot(m_hat_kd_A[k].reshape(dim, 1), m_hat_kd_A[k].reshape(1, dim))
        tmp_w_dd_A += np.linalg.inv(w_dd_A)
        w_hat_kdd_A[k] = np.linalg.inv(tmp_w_dd_A)
        nu_hat_k_A[k] = np.sum(w_dk[:, k]) + nu
        lambda_kdd_A[k] = wishart.rvs(size=1, df=nu_hat_k_A[k], scale=w_hat_kdd_A[k])
        mu_kd_A[k] = np.random.multivariate_normal(mean=m_hat_kd_A[k], cov=np.linalg.inv(beta_hat_k_A[k] * lambda_kdd_A[k]), size=1).flatten()

    ############################## Evaluation index ##############################
    concidence[i] = np.round(cohen_kappa_score(pred_label_A,pred_label_B),3) # Kappa score
    ARI_A[i] = np.round(ari(z_truth_n,pred_label_A),3)
    ARI_B[i] = np.round(ari(z_truth_n,pred_label_B),3)
    accept_count_AtoB[i] = count_AtoB
    accept_count_BtoA[i] = count_BtoA

    print(f"=> Epoch: {i+1}, ARI_A:{ARI_A[i]}, ARI_B:{ARI_B[i]}, Concidence:{concidence[i]}, Accept_AtoB:{int(accept_count_AtoB[i])}, Accept_BtoA:{int(accept_count_BtoA[i])}")

    _, w_n_A = np.where(w_dk_A == 1)
    _, w_n_B = np.where(w_dk_B == 1)
    trace_w_in_A.append(w_n_A.copy())
    trace_w_in_B.append(w_n_B.copy())
    trace_mu_ikd_A.append(mu_kd_A.copy())
    trace_mu_ikd_B.append(mu_kd_B.copy())
    trace_lambda_ikdd_A.append(lambda_kdd_A.copy())
    trace_lambda_ikdd_B.append(lambda_kdd_B.copy())
    trace_beta_ik_A.append(beta_hat_k_A.copy())
    trace_beta_ik_B.append(beta_hat_k_B.copy())
    trace_m_ikd_A.append(m_hat_kd_A.copy())
    trace_m_ikd_B.append(m_hat_kd_B.copy())
    trace_w_ikdd_A.append(w_hat_kdd_A.copy())
    trace_w_ikdd_B.append(w_hat_kdd_B.copy())
    trace_nu_ik_A.append(nu_hat_k_A.copy())
    trace_nu_ik_B.append(nu_hat_k_B.copy())

np.savetxt(dir_name + "/ariA" + ".txt", ARI_A, fmt='%.3f');
np.savetxt(dir_name + "/ariB" + ".txt", ARI_B, fmt='%.2f');
np.savetxt(dir_name + "/kappa" + ".txt", concidence, fmt='%.2f')

# number of acceptation 
plt.figure()
#plt.ylim(0,)
plt.plot(range(0,iteration), accept_count_AtoB, marker="None", label="Accept_num:AtoB")
plt.plot(range(0,iteration), accept_count_BtoA, marker="None", label="Accept_num:BtoA")
plt.xlabel('iteration')
plt.ylabel('Number of acceptation')
plt.ylim(0,D)
plt.legend()
plt.savefig(dir_name+'/accept.png')
#plt.show()
plt.close()

# concidence
plt.plot(range(0,iteration), concidence, marker="None")
plt.xlabel('iteration')
plt.ylabel('Concidence')
plt.ylim(0,1)
plt.title('Kappa')
plt.savefig(dir_name+"/conf.png")
#plt.show()
plt.close()

# ARI
plt.plot(range(0,iteration), ARI_A, marker="None",label="ARI_A")
plt.plot(range(0,iteration), ARI_B, marker="None",label="ARI_B")
plt.xlabel('iteration')
plt.ylabel('ARI')
plt.ylim(0,1)
plt.legend()
plt.title('ARI')
plt.savefig(dir_name+"/ari.png")
#plt.show()
plt.close()
