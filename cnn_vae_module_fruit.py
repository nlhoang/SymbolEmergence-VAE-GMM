from __future__ import print_function
import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision.utils import save_image
import numpy as np
import matplotlib.pyplot as plt
from tool import visualize_ls, get_param

device = torch.device('mps')

x_dim = 12
h_dim = 1024
#h_dim = 2304
image_channels = 3
image_size = 64
data_rep = 4

img_dataSize = torch.Size([3, 32, 32])
img_imgChans = img_dataSize[0]
img_fBase = 32
const = 1e-6


class Encoder_Fruit32(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder_Fruit32, self).__init__()
        self.enc = nn.Sequential(
            # input size: 3 x 32 x 32
            nn.Conv2d(img_imgChans, img_fBase, 4, 2, 1, bias=False),
            nn.BatchNorm2d(img_fBase),
            nn.ReLU(True),
            # size: (fBase) x 16 x 16
            nn.Conv2d(img_fBase, img_fBase * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(img_fBase * 2),
            nn.ReLU(True),
            # size: (fBase * 2) x 8 x 8
            nn.Conv2d(img_fBase * 2, img_fBase * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(img_fBase * 4),
            nn.ReLU(True),
            # size: (fBase * 4) x 4 x 4
        )
        self.c1 = nn.Conv2d(img_fBase * 4, latent_dim, 4, 1, 0, bias=False)
        self.c2 = nn.Conv2d(img_fBase * 4, latent_dim, 4, 1, 0, bias=False)

    def forward(self, x):
        x = self.enc(x)
        mu = self.c1(x).squeeze() + const
        logvar = F.softplus(self.c2(x)).squeeze() + const
        return mu, logvar


class Decoder_Fruit32(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder_Fruit32, self).__init__()
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, img_fBase * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(img_fBase * 4),
            nn.ReLU(True),
            # size: (fBase * 4) x 4 x 4
            nn.ConvTranspose2d(img_fBase * 4, img_fBase * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(img_fBase * 2),
            nn.ReLU(True),
            # size: (fBase * 2) x 8 x 8
            nn.ConvTranspose2d(img_fBase * 2, img_fBase, 4, 2, 1, bias=False),
            nn.BatchNorm2d(img_fBase),
            nn.ReLU(True),
            # size: (fBase) x 16 x 16
            nn.ConvTranspose2d(img_fBase, img_imgChans, 4, 2, 1, bias=False),
            # Output size: 3 x 32 x 32 (no sigmoid)
        )

    def forward(self, z):
        z = z.unsqueeze(-1).unsqueeze(-1)  # fit deconv layers
        z = self.dec(z.view(-1, *z.size()[-3:]))
        z = torch.sigmoid(z)
        return z


class VAE(nn.Module):
    def __init__(self, latent_dim=x_dim):
        super(VAE, self).__init__()
        self.encoder = Encoder_Fruit32(latent_dim)
        self.decoder = Decoder_Fruit32(latent_dim)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        latent = self.reparameterize(mu, logvar)
        recon = self.decoder(latent)
        return recon, mu, logvar, latent

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = torch.empty_like(std).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

    def loss_function(self, recon_o, o_d, en_mu, en_logvar, gmm_mu, gmm_var, iteration):
        # print(f"o_d : {o_d.size()}  recon_o:{recon_o.size()}")
        # print(f"recon_o:{recon_o.view(o_d.size()).size()}")
        BCE = F.binary_cross_entropy(recon_o, o_d, reduction='sum')
        beta = 1.0
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        if iteration != 0:
            gmm_mu = nn.Parameter(gmm_mu)
            prior_mu = gmm_mu
            prior_mu.requires_grad = False
            prior_mu = prior_mu.float().expand_as(en_mu).to(device)
            gmm_var = nn.Parameter(gmm_var)
            prior_var = gmm_var
            prior_var.requires_grad = False
            prior_var = prior_var.float().expand_as(en_logvar).to(device)
            prior_logvar = nn.Parameter(prior_var.log())
            prior_logvar.requires_grad = False
            prior_logvar = prior_logvar.expand_as(en_logvar).to(device)

            var_division = en_logvar.exp() / prior_var  # Σ_0 / Σ_1
            diff = en_mu - prior_mu  # μ_１ - μ_0
            diff_term = diff * diff / prior_var  # (μ_1 - μ_0)(μ_1 - μ_0)/Σ_1
            logvar_division = prior_logvar - en_logvar  # log|Σ_1| - log|Σ_0| = log(|Σ_1|/|Σ_2|)
            KLD = 0.5 * ((var_division + diff_term + logvar_division).sum(1) - x_dim)
        else:
            KLD = -0.5 * torch.sum(1 + en_logvar - en_mu.pow(2) - en_logvar.exp())
        return BCE + KLD


class Flatten(nn.Module):
    def forward(self, input):
        #print("input.size(0)",input.size(0))
        #print("input.view(input.size(0),-1)",input.view(input.size(0), -1).size())
        return input.view(input.size(0), -1) # [10,1024]


class UnFlatten(nn.Module):
    def forward(self, input, size=h_dim):
        #print("input.view(input.size(0),size,1,1)",input.view(input.size(0), size,1,1).size())
        return input.view(input.size(0), size, 1, 1)


class VAE1(nn.Module):
    def __init__(self, image_channels=image_channels, h_dim=h_dim, x_dim=x_dim):
        super(VAE1, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=image_channels, out_channels=32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2),
            nn.ReLU(),
            Flatten()
        )

        self.fc1 = nn.Linear(h_dim, x_dim)
        self.fc2 = nn.Linear(h_dim, x_dim)
        self.fc3 = nn.Linear(x_dim, h_dim)
        
        self.decoder = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose2d(in_channels=h_dim, out_channels=128, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=6, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=32, out_channels=image_channels, kernel_size=6, stride=2),            
            nn.Sigmoid(),
        )
        self.fc1 = nn.Linear(h_dim, x_dim)
        self.fc2 = nn.Linear(h_dim, x_dim)
        self.fc3 = nn.Linear(x_dim, h_dim)
        # 事前分布のパラメータN(0,I)で初期化
        self.prior_var = nn.Parameter(torch.Tensor(1, x_dim).float().fill_(1.0))
        self.prior_logvar = nn.Parameter(self.prior_var.log())
        self.prior_var.requires_grad = False
        self.prior_logvar.requires_grad = False

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def encode(self, o_d):
        #print("o_d: ",o_d.size()) # [batch_size, channel, 縦, 横]
        h = self.encoder(o_d)
        #print("h: ",h.size())
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar

    def decode(self, z):
        z = self.fc3(z)
        z = self.decoder(z)
        return z

    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def forward(self, o_d):
        z, mu, logvar = self.encode(o_d)
        return self.decode(z), mu, logvar, z
     
    # Reconstruction + KL divergence losses summed over all elements and batch
    def loss_function(self, recon_o, o_d, en_mu, en_logvar, gmm_mu, gmm_var, iteration):
        #print(f"o_d : {o_d.size()}  recon_o:{recon_o.size()}")
        #print(f"recon_o:{recon_o.view(o_d.size()).size()}")
        BCE = F.binary_cross_entropy(recon_o, o_d, reduction='sum')
        beta = 1.0
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114 
        if iteration != 0: 
            gmm_mu = nn.Parameter(gmm_mu)
            prior_mu = gmm_mu
            prior_mu.requires_grad = False
            prior_mu = prior_mu.float().expand_as(en_mu).to(device)
            gmm_var = nn.Parameter(gmm_var)
            prior_var = gmm_var
            prior_var.requires_grad = False
            prior_var = prior_var.float().expand_as(en_logvar).to(device)
            prior_logvar = nn.Parameter(prior_var.log())
            prior_logvar.requires_grad = False
            prior_logvar = prior_logvar.expand_as(en_logvar).to(device)
            
            var_division = en_logvar.exp() / prior_var # Σ_0 / Σ_1
            diff = en_mu - prior_mu # μ_１ - μ_0
            diff_term = diff *diff / prior_var # (μ_1 - μ_0)(μ_1 - μ_0)/Σ_1
            logvar_division = prior_logvar - en_logvar # log|Σ_1| - log|Σ_0| = log(|Σ_1|/|Σ_2|)
            KLD = 0.5 * ((var_division + diff_term + logvar_division).sum(1) - x_dim)
        else:
            KLD = -0.5 * torch.sum(1 + en_logvar - en_mu.pow(2) - en_logvar.exp())
        return BCE + KLD


def train(iteration, gmm_mu, gmm_var, epoch, train_loader, batch_size, all_loader, model_dir, agent):
    prior_mean = torch.Tensor(len(train_loader), x_dim).float().fill_(0.0) # 最初のVAEの事前分布の\mu
    model = VAE().to(device)
    print(f"VAE_Agent {agent} Training Start({iteration}): Epoch:{epoch}, Batch_size:{batch_size}")
    #loss_list = []
    #epoch_list = np.arange(epoch)
    #model.load_state_dict(torch.load(save_dir+"/vae.pth"))
    #if first!=True:
        #print("前回の学習パラメータの読み込み")
        #model.load_state_dict(torch.load(save_dir+"/vae.pth"))

    gmm_mu_rep = gmm_mu.repeat_interleave(data_rep, dim=0)
    gmm_var_rep = gmm_var.repeat_interleave(data_rep, dim=0)

    #model.load_state_dict(torch.load(save_dir))
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_list = np.zeros((epoch))
    for i in range(epoch):
        model.train()
        train_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar, x_d = model(data)
            if iteration==0: # 最初の学習
                loss = model.loss_function(recon_batch, data, mu, logvar, gmm_mu=None, gmm_var=None, iteration=iteration)
            else:
                loss = model.loss_function(recon_batch, data, mu, logvar, gmm_mu_rep[batch_idx*batch_size:(batch_idx+1)*batch_size], gmm_var_rep[batch_idx*batch_size:(batch_idx+1)*batch_size], iteration=iteration)
            loss = loss.mean()
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        if i == 0 or (i+1) % 25 == 0 or i == (epoch-1):
            print('====> Epoch: {} Average loss: {:.4f}'.format(i+1, train_loss / len(train_loader.dataset)))
            
        loss_list[i] = -(train_loss / len(train_loader.dataset))
    
    #グラフ処理
    plt.figure()
    plt.plot(range(0,epoch), loss_list, color="blue", label="ELBO")
    if iteration!=0: 
        loss_0 = np.load(model_dir+'/npy/loss'+agent+'_0.npy')
        plt.plot(range(0,epoch), loss_0, color="red", label="ELBO_I0")
    plt.xlabel('epoch'); plt.ylabel('ELBO'); plt.legend(loc='lower right')
    plt.savefig(model_dir+'/graph'+agent+'/vae_loss_'+str(iteration)+'.png')
    plt.close()
    
    np.save(model_dir+'/npy/loss'+agent+'_'+str(iteration)+'.npy', np.array(loss_list))
    torch.save(model.state_dict(), model_dir+"/pth/vae"+agent+"_"+str(iteration)+".pth")
    
    x_d, label = send_all_z(iteration=iteration, all_loader=all_loader, model_dir=model_dir, agent=agent)

    # Send latent variable x_d to GMM
    return x_d, label, loss_list
    

def decode(iteration, decode_k, sample_num, sample_d, manual, model_dir, agent):
    print(f"Reconstruct image on Agent: {agent}, category: {decode_k}")
    model = VAE().to("cpu")
    
    model.load_state_dict(torch.load(str(model_dir)+"/pth/vae"+agent+"_"+str(iteration)+".pth",map_location=torch.device('cpu'))); model.eval()
    mu_gmm_kd, lambda_gmm_kdd, pi_gmm_k = get_param(iteration, model_dir=model_dir, agent=agent)
    sample_d = torch.from_numpy(sample_d.astype(np.float32)).clone()
    with torch.no_grad():
        sample_d = sample_d.to("cpu")
        sample_d = model.decode(sample_d).cpu()
        save_image(sample_d.view(sample_num, image_channels, image_size, image_size),model_dir+'/recon'+agent+'/random_'+str(decode_k)+'.png') if manual != True else save_image(sample_d.view(sample_num, image_channels, image_size, image_size),model_dir+'/recon'+agent+'/manual_'+str(decode_k)+'.png')
        

def plot_latent(iteration, all_loader, model_dir, agent): # VAEの潜在空間を可視化するメソッド
    print(f"Plot latent space on Agent: {agent}")
    #model = VAE().to(device)
    model = VAE().to("cpu")
    model.load_state_dict(torch.load(model_dir+"/pth/vae"+agent+"_"+str(iteration)+".pth", map_location=torch.device('cpu')))
    #model.load_state_dict(torch.load(model_dir+"/pth/vae"+agent+"_"+str(iteration)+".pth"))
    model.eval()
    for batch_idx, (data, label) in enumerate(all_loader):
        #data = data.to(device)
        data = data.to("cpu")
        recon_batch, mu, logvar, x_d = model(data)
        x_d = x_d.cpu()
        visualize_ls(iteration, x_d.detach().numpy(), label, model_dir,agent=agent)
        break


def send_all_z(iteration, all_loader, model_dir, agent): # gmmにvaeの潜在空間を送るメソッド
    model = VAE().to(device)
    model.load_state_dict(torch.load(model_dir+"/pth/vae"+agent+"_"+str(iteration)+".pth"))
    model.eval()
    for batch_idx, (data, label) in enumerate(all_loader):
        data = data.to(device)
        recon_batch, mu, logvar, x_d = model(data)
        x_d = x_d.cpu()
        label = label.cpu()
    return x_d.detach().numpy(), label.detach().numpy()


def send_all_z_batch(iteration, all_loader, model_dir, agent):
    model = VAE().to(device)
    model.load_state_dict(torch.load(model_dir + "/pth/vae" + agent + "_" + str(iteration) + ".pth"))
    model.eval()
    all_x_d = []
    all_label = []

    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(all_loader):
            data = data.to(device)
            recon_batch, mu, logvar, x_d = model(data)
            x_d = x_d.cpu()
            label = label.cpu()
            all_x_d.append(x_d)
            all_label.append(label)

    # concatenate all the batches
    all_x_d = torch.cat(all_x_d).detach().numpy()
    all_label = torch.cat(all_label).detach().numpy()
    return all_x_d, all_label
