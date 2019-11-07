import os, inspect, time, math

import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from torch.utils.tensorboard import SummaryWriter

PACK_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))+"/.."

def make_dir(path):

    try: os.mkdir(path)
    except: pass

def gray2rgb(gray):

    rgb = np.ones((gray.shape[0], gray.shape[1], 3)).astype(np.float32)
    rgb[:, :, 0] = gray[:, :, 0]
    rgb[:, :, 1] = gray[:, :, 0]
    rgb[:, :, 2] = gray[:, :, 0]

    return rgb

def dat2canvas(data):

    numd = math.ceil(np.sqrt(data.shape[0]))
    [dn, dh, dw, dc] = data.shape
    canvas = np.ones((dh*numd, dw*numd, dc)).astype(np.float32)

    for y in range(numd):
        for x in range(numd):
            try: tmp = data[x+(y*numd)]
            except: pass
            else: canvas[(y*dh):(y*dh)+28, (x*dw):(x*dw)+28, :] = tmp
    if(dc == 1):
        canvas = gray2rgb(gray=canvas)

    return canvas

def save_img(contents, names=["", "", ""], savename=""):

    num_cont = len(contents)
    plt.figure(figsize=(5*num_cont+2, 5))

    for i in range(num_cont):
        plt.subplot(1,num_cont,i+1)
        plt.title(names[i])
        plt.imshow(dat2canvas(data=contents[i]))

    plt.tight_layout()
    plt.savefig(savename)
    plt.close()

def discrete_cmap(N, base_cmap=None):

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)

    return base.from_list(cmap_name, color_list, N)

def latent_plot(latent, y, n, savename=""):

    plt.figure(figsize=(6, 5))
    plt.scatter(latent[:, 0], latent[:, 1], c=y, \
        marker='o', edgecolor='none', cmap=discrete_cmap(n, 'jet'))
    plt.colorbar(ticks=range(n))
    plt.grid()
    plt.tight_layout()
    plt.savefig(savename)
    plt.close()

def boxplot(contents, savename=""):

    data, label = [], []
    for cidx, content in enumerate(contents):
        data.append(content)
        label.append("class-%d" %(cidx))

    plt.clf()
    fig, ax1 = plt.subplots()
    bp = ax1.boxplot(data, showfliers=True, whis=3)
    ax1.set_xticklabels(label, rotation=45)

    plt.tight_layout()
    plt.savefig(savename)
    plt.close()

def histogram(contents, savename=""):

    n1, _, _ = plt.hist(contents[0], bins=100, alpha=0.5, label='Normal')
    n2, _, _ = plt.hist(contents[1], bins=100, alpha=0.5, label='Abnormal')
    h_inter = np.sum(np.minimum(n1, n2)) / np.sum(n1)
    plt.xlabel("MSE")
    plt.ylabel("Number of Data")
    xmax = max(contents[0].max(), contents[1].max())
    plt.xlim(0, xmax)
    plt.text(x=xmax*0.01, y=max(n1.max(), n2.max()), s="Histogram Intersection: %.3f" %(h_inter))
    plt.legend(loc='upper right')
    plt.savefig(savename)
    plt.close()

def save_graph(contents, xlabel, ylabel, savename):

    np.save(savename, np.asarray(contents))
    plt.clf()
    plt.rcParams['font.size'] = 15
    plt.plot(contents, color='blue', linestyle="-", label="loss")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout(pad=1, w_pad=1, h_pad=1)
    plt.savefig("%s.png" %(savename))
    plt.close()

def torch2npy(input):

    output = input.detach().numpy()
    return output

def loss(x, x_hat, mu, sigma):

    restore_error = -torch.sum(x * torch.log(x_hat + 1e-12) + (1 - x) * torch.log(1 - x_hat + 1e-12))
    kl_divergence = 0.5 * torch.sum(mu**2 + sigma**2 - torch.log(sigma**2 + 1e-12) - 1)

    return  restore_error + kl_divergence, restore_error, kl_divergence

def training(neuralnet, dataset, epochs, batch_size):

    print("\nTraining to %d epochs (%d of minibatch size)" %(epochs, batch_size))

    make_dir(path="results")
    result_list = ["tr_latent", "tr_resotring", "tr_latent_walk"]
    for result_name in result_list: make_dir(path=os.path.join("results", result_name))

    start_time = time.time()

    iteration = 0
    writer = SummaryWriter()

    test_sq = 20
    test_size = test_sq**2
    for epoch in range(epochs):

        x_tr, y_tr, _ = dataset.next_train(batch_size=test_size, fix=True) # Initial batch
        print(x_tr)
        z_enc, z_mu, z_sigma = neuralnet.encoder(x_tr.to(neuralnet.device))
        print(z_enc)
        x_restore = neuralnet.decoder(z_enc.to(neuralnet.device))
        print(x_restore)

        if(neuralnet.z_dim == 2):
            latent_plot(latent=torch2npy(z_enc), y=torch2npy(y_tr), n=dataset.num_class, \
                savename=os.path.join("results", "tr_latent", "%08d.png" %(epoch)))
        else:
            pca = PCA(n_components=2)
            pca_features = pca.fit_transform(torch2npy(z_enc))
            latent_plot(latent=pca_features, y=y_tr, n=dataset.num_class, \
                savename=os.path.join("results", "tr_latent", "%08d.png" %(epoch)))

        save_img(contents=[torch2npy(x_tr), torch2npy(x_restore), (torch2npy(x_tr)-torch2npy(x_restore))**2], \
            names=["Input\n(x)", "Restoration\n(x to x-hat)", "Difference"], \
            savename=os.path.join("results", "tr_resotring", "%08d.png" %(epoch)))

        if(neuralnet.z_dim == 2):
            x_values = np.linspace(-3, 3, test_sq)
            y_values = np.linspace(-3, 3, test_sq)
            z_latents = None
            for y_loc, y_val in enumerate(y_values):
                for x_loc, x_val in enumerate(x_values):
                    z_latent = np.reshape(np.array([y_val, x_val]), (1, neuralnet.z_dim))
                    if(z_latents is None): z_latents = z_latent
                    else: z_latents = np.append(z_latents, z_latent, axis=0)
            x_samples = neuralnet.decoder(torch.from_numpy(z_latents).to(neuralnet.device))
            plt.imsave(os.path.join("results", "tr_latent_walk", "%08d.png" %(epoch)), dat2canvas(data=torch2npy(x_samples)))

        while(True):
            x_tr, y_tr, terminator = dataset.next_train(batch_size) # y_tr does not used in this prj.

            z_enc, z_mu, z_sigma = neuralnet.encoder(x_tr.to(neuralnet.device))
            x_hat = neuralnet.decoder(z_enc.to(neuralnet.device))

            loss, restore_error, kl_divergence = loss(x=x_tr, x_hat=x_hat, mu=z_mu, sigma=z_sigma)
            loss.backward()
            neuralnet.optimizer.step()

            list_recon.append(restore_error.item())
            list_kld.append(kl_divergence.item())
            list_total.append(loss.item())

            writer.add_scalar('VAE/restore_error', restore_error, iteration)
            writer.add_scalar('VAE/kl_divergence', kl_divergence, iteration)
            writer.add_scalar('VAE/loss_total', loss, iteration)

            iteration += 1
            if(terminator): break

        print("Epoch [%d / %d] (%d iteration)  Restore:%.3f, KLD:%.3f, Total:%.3f" \
            %(epoch, epochs, iteration, restore_error, kl_divergence, loss))
        torch.save(neuralnet.model.state_dict(), PACK_PATH+"/runs/params")

    elapsed_time = time.time() - start_time
    print("Elapsed: "+str(elapsed_time))

    save_graph(contents=list_recon, xlabel="Iteration", ylabel="Reconstruction Error", savename="restore_error")
    save_graph(contents=list_kld, xlabel="Iteration", ylabel="KL-Divergence", savename="kl_divergence")
    save_graph(contents=list_total, xlabel="Iteration", ylabel="Total Loss", savename="loss_total")

def validation(neuralnet, dataset):

    if(os.path.exists(PACK_PATH+"/runs/params")):
        neuralnet.model.load_state_dict(torch.load(PACK_PATH+"/runs/params"))
        neuralnet.model.eval()

    makedir(PACK_PATH+"/test")
    makedir(PACK_PATH+"/test/reconstruction")

    start_time = time.time()
    print("\nValidation")
    for tidx in range(dataset.amount_te):

        X_te, Y_te, X_te_t, Y_te_t = dataset.next_test()
        if(X_te is None): break

        img_recon = neuralnet.model(X_te_t.to(neuralnet.device))
        tmp_psnr = psnr(input=X_te_t.to(neuralnet.device), target=img_recon.to(neuralnet.device)).item()
        img_recon = np.transpose(torch2npy(img_recon.cpu()), (0, 2, 3, 1))

        img_recon = np.squeeze(img_recon, axis=0)
        plt.imsave("%s/test/reconstruction/%d_psnr_%d.png" %(PACK_PATH, tidx, int(tmp_psnr)), img_recon)

        img_input = np.squeeze(X_te, axis=0)
        img_ground = np.squeeze(Y_te, axis=0)
        plt.imsave("%s/test/bicubic.png" %(PACK_PATH), img_input)
        plt.imsave("%s/test/high-resolution.png" %(PACK_PATH), img_ground)

    elapsed_time = time.time() - start_time
    print("Elapsed: "+str(elapsed_time))
