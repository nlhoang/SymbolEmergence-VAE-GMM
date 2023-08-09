import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import wishart
from sklearn.metrics.cluster import adjusted_rand_score as ari
from torch.utils.data.dataset import Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder

import cnn_vae_module_fruit
from tool import calc_ari, cmx, save_toFile

parser = argparse.ArgumentParser(description='Symbol emergence based on VAE+GMM Example')
parser.add_argument('--dataset', type=str, default="mnist",  help='name of dataset, mnist or fruits360')
parser.add_argument('--dir-name', type=str,  help='name of directory for saving model')
parser.add_argument('--batch-size', type=int, default=256, metavar='B', help='input batch size for training')
parser.add_argument('--vae-iter', type=int, default=3, metavar='V', help='number of VAE iteration')
parser.add_argument('--mh-iter', type=int, default=2, metavar='M', help='number of M-H mgmm iteration')
parser.add_argument('--category', type=int, default=10, metavar='K', help='number of category for GMM module')
parser.add_argument('--W', type=float, default=0.05, metavar='W', help='hyperparameter for Wishart distribution')
parser.add_argument('--debug', type=bool, default=False, metavar='D', help='Debug mode')
parser.add_argument('--no-cuda', action='store_true', default=False, help='enables CUDA training')
parser.add_argument('--seed', type=int, default=2, metavar='S', help='random seed')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device('mps')
if args.debug is True: args.vae_iter=2; args.mh_iter=2


############################## Making directory ##############################
file_name = "debug"; model_dir = "./model"; dir_name = "./model/"+file_name
graphA_dir = "./model/" + file_name + "/graphA";
graphB_dir = "./model/" + file_name + "/graphB"
pth_dir = "./model/"+file_name+"/pth";npy_dir = "./model/"+file_name+"/npy/"
log_dir = model_dir+"/"+file_name+"/log"; result_dir = model_dir+"/"+file_name+"/result"
if not os.path.exists(model_dir):   os.mkdir(model_dir)
if not os.path.exists(dir_name):    os.mkdir(dir_name)
if not os.path.exists(graphA_dir):   os.mkdir(graphA_dir)
if not os.path.exists(graphB_dir):   os.mkdir(graphB_dir)
if not os.path.exists(pth_dir):    os.mkdir(pth_dir)
if not os.path.exists(npy_dir):    os.mkdir(npy_dir)
if not os.path.exists(log_dir):    os.mkdir(log_dir)
if not os.path.exists(result_dir):    os.mkdir(result_dir)

############################## Prepareing Dataset #############################
trained_root = "../data/Fruit360/Categories40_noise3"
data_root = "../data/Fruit360/Categories40"
angle_a = 0  # 回転角度
angle_b = 25  # 回転角度
image_size = 32

trans_ang1 = transforms.Compose(
    [transforms.RandomRotation(degrees=(angle_a, angle_a)),
     transforms.Resize((image_size, image_size)),
     transforms.ToTensor()])  # -angle度回転設定
trans_ang2 = transforms.Compose(
    [transforms.RandomRotation(degrees=(angle_b, angle_b)),
     transforms.Resize((image_size, image_size)),
     transforms.ToTensor()])  # angle度回転設定
obj_a_dataset_train = ImageFolder(trained_root, transform=trans_ang1)
obj_a_dataset = ImageFolder(data_root, transform=trans_ang1)
obj_b_dataset_train = ImageFolder(trained_root, transform=trans_ang2)
obj_b_dataset = ImageFolder(data_root, transform=trans_ang2)

n_samples = len(obj_a_dataset)
D = int(n_samples)

train_loader1 = torch.utils.data.DataLoader(obj_a_dataset_train, batch_size=args.batch_size, shuffle=False)
train_loader2 = torch.utils.data.DataLoader(obj_b_dataset_train, batch_size=args.batch_size, shuffle=False)
all_loader1 = torch.utils.data.DataLoader(obj_a_dataset, batch_size=D, shuffle=False)
all_loader2 = torch.utils.data.DataLoader(obj_b_dataset, batch_size=D, shuffle=False)


print(f"D={D}, Category:{args.category}")
print(f"VAE_iter:{args.vae_iter}, Batch_size:{args.batch_size}")
print(f"Gibbs Sampling")
K = args.category # サイン総数
mutual_iteration = 3
mu_d_A = np.zeros((D)); var_d_A = np.zeros((D)) 
mu_d_B = np.zeros((D)); var_d_B = np.zeros((D))

for it in range(mutual_iteration):
    print(f"------------------Mutual learning session {it} begins------------------")
    ############################## Training VAE ##############################

    c_nd_A, label, loss_list = cnn_vae_module_fruit.train(
            iteration=it, # Current iteration
            gmm_mu=torch.from_numpy(mu_d_A), gmm_var=torch.from_numpy(var_d_A), # mu and var estimated by Multimodal-GMM
            epoch=args.vae_iter, 
            train_loader=train_loader1, batch_size=args.batch_size, all_loader=all_loader1,
            model_dir=dir_name, agent="A"
        )

    c_nd_B, label, loss_list = cnn_vae_module_fruit.train(
            iteration=it, # Current iteration
            gmm_mu=torch.from_numpy(mu_d_B), gmm_var=torch.from_numpy(var_d_B), # mu and var estimated by Multimodal-GMM
            epoch=args.vae_iter, 
            train_loader=train_loader2, batch_size=args.batch_size, all_loader=all_loader2,
            model_dir=dir_name, agent="B"
        )
    
    z_truth_n = label # 真のカテゴリ
    save_toFile(npy_dir, file_name='z_truth_n', data_saved=z_truth_n, rows=0)
    save_toFile(npy_dir, file_name='latent_a'+str(it), data_saved=c_nd_A, rows=1)
    save_toFile(npy_dir, file_name='latent_b'+str(it), data_saved=c_nd_B, rows=1)
    dim = len(c_nd_A[0]) # VAEの潜在変数の次元数（分散表現のカテゴリ変数の次元数）

    ############################## Initializing parameters ##############################
    # Set hyperparameters
    beta = 1.0; m_d_A = np.repeat(0.0, dim); m_d_B = np.repeat(0.0, dim) # Hyperparameters for \mu^A, \mu^B
    w_dd_A = np.identity(dim) * args.W ; w_dd_B = np.identity(dim) * args.W    # Hyperparameters for \Lambda^A, \Lambda^B
    nu = dim

    # Initializing \mu, \Lambda
    mu_kd_A = np.empty((K, dim)); lambda_kdd_A = np.empty((K, dim, dim))
    mu_kd_B = np.empty((K, dim)); lambda_kdd_B = np.empty((K, dim, dim))
    for k in range(K):
        lambda_kdd_A[k] = wishart.rvs(df=nu, scale=w_dd_A, size=1); lambda_kdd_B[k] = wishart.rvs(df=nu, scale=w_dd_B, size=1)
        mu_kd_A[k] = np.random.multivariate_normal(mean=m_d_A, cov=np.linalg.inv(beta * lambda_kdd_A[k])).flatten()
        mu_kd_B[k] = np.random.multivariate_normal(mean=m_d_B, cov=np.linalg.inv(beta * lambda_kdd_B[k])).flatten()

    # Initializing unsampled \w
    w_dk = np.random.multinomial(1, [1/K]*K, size=D)

    # Initializing learning parameters
    beta_hat_k_A = np.zeros(K) ;beta_hat_k_B = np.zeros(K)
    m_hat_kd_A = np.zeros((K, dim)); m_hat_kd_B = np.zeros((K, dim))
    w_hat_kdd_A = np.zeros((K, dim, dim)); w_hat_kdd_B = np.zeros((K, dim, dim))
    nu_hat_k_A = np.zeros(K); nu_hat_k_B = np.zeros(K)
    tmp_eta_nB = np.zeros((K, D)); eta_dkB = np.zeros((D, K))
    tmp_eta_nA = np.zeros((K, D)); eta_dkA = np.zeros((D, K))
    eta_dk = np.zeros((D, K))
    cat_liks_A = np.zeros(D); cat_liks_B = np.zeros(D)
    mu_d_A = np.zeros((D,dim)); var_d_A = np.zeros((D,dim)) 
    mu_d_B = np.zeros((D,dim)); var_d_B = np.zeros((D,dim))
    w_dk_A = np.random.multinomial(1, [1/K]*K, size=D); w_dk_B = np.random.multinomial(1, [1/K]*K, size=D)

    iteration = args.mh_iter # M−H法のイテレーション数
    ARI_A = np.zeros((iteration)); ARI_B = np.zeros((iteration)); ARI = np.zeros((iteration)); concidence = np.zeros((iteration))
    accept_count_AtoB = np.zeros((iteration),dtype=np.int16); accept_count_BtoA = np.zeros((iteration),dtype=np.int16) # Number of acceptation
    ############################## M-H algorithm ##############################
    print(f"M-H algorithm Start({it}): Epoch:{iteration}")
    for i in range(iteration):
        pred_label_A = []; pred_label_B = []; pred_label = []
        count_AtoB = count_BtoA = 0 # 現在のイテレーションでの受容回数を保存する変数

        for k in range(K):
            tmp_eta_n = np.diag(-0.5 * (c_nd_A - mu_kd_A[k]).dot(lambda_kdd_A[k]).dot((c_nd_A - mu_kd_A[k]).T)).copy()
            tmp_eta_n += np.diag(-0.5 * (c_nd_B - mu_kd_B[k]).dot(lambda_kdd_B[k]).dot((c_nd_B - mu_kd_B[k]).T)).copy()
            tmp_eta_n += 0.5 * np.log(np.linalg.det(lambda_kdd_A[k]) + 1e-7)
            tmp_eta_n += 0.5 * np.log(np.linalg.det(lambda_kdd_B[k]) + 1e-7)
            # tmp_eta_n += np.log(pi_k[k] + 1e-7) # ベイズ推論式(4.92)の混合比に関する部分
            eta_dk[:, k] = np.exp(tmp_eta_n)
        eta_dk /= np.sum(eta_dk, axis=1, keepdims=True) # 正規化
        for d in range(D):
            pvals = eta_dk[d]
            if True in np.isnan(np.array(pvals)):
                pvals = [0.1] * 10
            w_dk[d] = np.random.multinomial(n=1, pvals=pvals, size=1).flatten()
            pred_label.append(np.argmax(w_dk[d]))

        for k in range(K):
            # muの事後分布のパラメータを計算
            beta_hat_k_A[k] = np.sum(w_dk[:, k]) + beta; m_hat_kd_A[k] = np.sum(w_dk[:, k] * c_nd_A.T, axis=1)
            m_hat_kd_A[k] += beta * m_d_A; m_hat_kd_A[k] /= beta_hat_k_A[k]
            # lambdaの事後分布のパラメータを計算
            tmp_w_dd_A = np.dot((w_dk[:, k] * c_nd_A.T), c_nd_A)
            tmp_w_dd_A += beta * np.dot(m_d_A.reshape(dim, 1), m_d_A.reshape(1, dim))
            tmp_w_dd_A -= beta_hat_k_A[k] * np.dot(m_hat_kd_A[k].reshape(dim, 1), m_hat_kd_A[k].reshape(1, dim))
            tmp_w_dd_A += np.linalg.inv(w_dd_A)
            w_hat_kdd_A[k] = np.linalg.inv(tmp_w_dd_A)
            nu_hat_k_A[k] = np.sum(w_dk[:, k]) + nu
            # 更新後のパラメータからlambdaをサンプル
            lambda_kdd_A[k] = wishart.rvs(size=1, df=nu_hat_k_A[k], scale=w_hat_kdd_A[k])
            # 更新後のパラメータからmuをサンプル
            mu_kd_A[k] = np.random.multivariate_normal(mean=m_hat_kd_A[k], cov=np.linalg.inv(beta_hat_k_A[k] * lambda_kdd_A[k]), size=1).flatten()

            # muの事後分布のパラメータを計算
            beta_hat_k_B[k] = np.sum(w_dk[:, k]) + beta; m_hat_kd_B[k] = np.sum(w_dk[:, k] * c_nd_B.T, axis=1)
            m_hat_kd_B[k] += beta * m_d_B; m_hat_kd_B[k] /= beta_hat_k_B[k]
            # lambdaの事後分布のパラメータを計算
            tmp_w_dd_B = np.dot((w_dk[:, k] * c_nd_B.T), c_nd_B)
            tmp_w_dd_B += beta * np.dot(m_d_B.reshape(dim, 1), m_d_B.reshape(1, dim))
            tmp_w_dd_B -= beta_hat_k_B[k] * np.dot(m_hat_kd_B[k].reshape(dim, 1), m_hat_kd_B[k].reshape(1, dim))
            tmp_w_dd_B += np.linalg.inv(w_dd_B)
            w_hat_kdd_B[k] = np.linalg.inv(tmp_w_dd_B)
            nu_hat_k_B[k] = np.sum(w_dk[:, k]) + nu
            # 更新後のパラメータからlambdaをサンプル
            lambda_kdd_B[k] = wishart.rvs(size=1, df=nu_hat_k_B[k], scale=w_hat_kdd_B[k])
            # 更新後のパラメータからmuをサンプル
            mu_kd_B[k] = np.random.multivariate_normal(mean=m_hat_kd_B[k], cov=np.linalg.inv(beta_hat_k_B[k] * lambda_kdd_B[k]), size=1).flatten()

        _, result = calc_ari(pred_label, z_truth_n)
        ARI[i] = np.round(ari(z_truth_n, result),3)
        
        if i == 0 or (i+1) % 10 == 0 or i == (iteration-1):
            print(f"=> Ep: {i+1}, ARI: {ARI[i]}")

    for d in range(D):
        mu_d_A[d] = mu_kd_A[np.argmax(w_dk[d])]
        var_d_A[d] = np.diag(np.linalg.inv(lambda_kdd_A[np.argmax(w_dk[d])]))
        mu_d_B[d] = mu_kd_B[np.argmax(w_dk[d])]
        var_d_B[d] = np.diag(np.linalg.inv(lambda_kdd_B[np.argmax(w_dk[d])]))

    np.save(npy_dir+'/c_nd_A_'+str(it)+'.npy', c_nd_A); np.save(npy_dir+'/c_nd_B_'+str(it)+'.npy', c_nd_B)
    np.save(npy_dir+'/mu_d_A_'+str(it)+'.npy', mu_d_A); np.save(npy_dir+'/mu_d_B_'+str(it)+'.npy', mu_d_B)
    np.save(npy_dir+'/var_d_A_'+str(it)+'.npy', var_d_A); np.save(npy_dir+'/var_d_B_'+str(it)+'.npy', var_d_B)    
    np.save(npy_dir+'/muA_'+str(it)+'.npy', mu_kd_A); np.save(npy_dir+'/muB_'+str(it)+'.npy', mu_kd_B)
    np.save(npy_dir+'/lambdaA_'+str(it)+'.npy', lambda_kdd_A); np.save(npy_dir+'/lambdaB_'+str(it)+'.npy', lambda_kdd_B)

    np.save(npy_dir+'/pred_label'+str(it)+'.npy', pred_label)
    np.save(npy_dir+'/result'+str(it)+'.npy', result)
    np.savetxt(log_dir+"/ari"+str(it)+".txt", ARI, fmt ='%.3f')

    # ARI
    plt.figure()
    plt.plot(range(0,iteration), ARI, marker="None",label="ARI")
    plt.xlabel('iteration'); plt.ylabel('ARI')
    plt.ylim(0,1)
    plt.legend()
    plt.title('ARI')
    plt.savefig(result_dir+"/ari"+str(it)+".png")
    #plt.show()
    plt.close()

    cmx(iteration=it, y_true=z_truth_n, y_pred=result, agent="Gibbs", save_dir=result_dir)
    print(f"Iteration:{it} Done:max_ARI: {max(ARI)}")
