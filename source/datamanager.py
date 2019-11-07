import torch
import numpy as np
import tensorflow as tf

from sklearn.utils import shuffle

class Dataset(object):

    def __init__(self, normalize=True):

        print("\nInitializing Dataset...")

        self.normalize = normalize

        (x_tr, y_tr), (x_te, y_te) = tf.keras.datasets.mnist.load_data()
        self.x_tr, self.y_tr = x_tr, y_tr
        self.x_te, self.y_te = x_te, y_te

        self.x_tr = np.ndarray.astype(self.x_tr, np.float32)
        self.x_te = np.ndarray.astype(self.x_te, np.float32)

        self.split_dataset()

        self.num_tr, self.num_te = self.x_tr.shape[0], self.x_te.shape[0]
        self.idx_tr, self.idx_te = 0, 0

        print("Number of data\nTraining: %d, Test: %d\n" %(self.num_tr, self.num_te))

        x_sample, y_sample = self.x_te[0], self.y_te[0]
        self.height = x_sample.shape[0]
        self.width = x_sample.shape[1]
        try: self.channel = x_sample.shape[2]
        except: self.channel = 1

        self.min_val, self.max_val = x_sample.min(), x_sample.max()
        self.num_class = (y_te.max()+1)

        print("Information of data")
        print("Shape  Height: %d, Width: %d, Channel: %d" %(self.height, self.width, self.channel))
        print("Value  Min: %.3f, Max: %.3f" %(self.min_val, self.max_val))
        print("Class  %d" %(self.num_class))
        print("Normalization: %r" %(self.normalize))
        if(self.normalize): print("(from %.3f-%.3f to %.3f-%.3f)" %(self.min_val, self.max_val, 0, 1))

    def split_dataset(self):

        x_tot = np.append(self.x_tr, self.x_te, axis=0)
        y_tot = np.append(self.y_tr, self.y_te, axis=0)

        x_normal, y_normal = None, None
        x_abnormal, y_abnormal = None, None
        for yidx, y in enumerate(y_tot):

            x_tmp = np.expand_dims(x_tot[yidx], axis=0)
            y_tmp = np.expand_dims(y_tot[yidx], axis=0)

            if(y == 1): # as normal
                if(x_normal is None):
                    x_normal = x_tmp
                    y_normal = y_tmp
                else:
                    x_normal = np.append(x_normal, x_tmp, axis=0)
                    y_normal = np.append(y_normal, y_tmp, axis=0)

            else: # as abnormal
                if(x_abnormal is None):
                    x_abnormal = x_tmp
                    y_abnormal = y_tmp
                else:
                    if(x_abnormal.shape[0] < 1000):
                        x_abnormal = np.append(x_abnormal, x_tmp, axis=0)
                        y_abnormal = np.append(y_abnormal, y_tmp, axis=0)

            if(not(x_normal is None) and not(x_abnormal is None)):
                if((x_normal.shape[0] >= 2000) and x_abnormal.shape[0] >= 1000): break

        self.x_tr, self.y_tr = x_normal[:1000], y_normal[:1000]
        self.x_te, self.y_te = x_normal[1000:], y_normal[1000:]
        self.x_te = np.append(self.x_te, x_abnormal, axis=0)
        self.y_te = np.append(self.y_te, y_abnormal, axis=0)

    def reset_idx(self): self.idx_tr, self.idx_te = 0, 0

    def next_train(self, batch_size=1, fix=False):

        start, end = self.idx_tr, self.idx_tr+batch_size
        x_tr, y_tr = self.x_tr[start:end], self.y_tr[start:end]
        x_tr = np.expand_dims(x_tr, axis=3)

        terminator = False
        if(end >= self.num_tr):
            terminator = True
            self.idx_tr = 0
            self.x_tr, self.y_tr = shuffle(self.x_tr, self.y_tr)
        else: self.idx_tr = end

        if(fix): self.idx_tr = start

        if(x_tr.shape[0] != batch_size):
            x_tr, y_tr = self.x_tr[-1-batch_size:-1], self.y_tr[-1-batch_size:-1]
            x_tr = np.expand_dims(x_tr, axis=3)

        if(self.normalize):
            min_x, max_x = x_tr.min(), x_tr.max()
            x_tr = (x_tr - min_x) / (max_x - min_x)

        x_tr_torch = torch.from_numpy(np.transpose(x_tr, (0, 3, 1, 2)))
        y_tr_torch = torch.from_numpy(y_tr)

        return x_tr, x_tr_torch, y_tr, y_tr_torch, terminator

    def next_test(self, batch_size=1):

        start, end = self.idx_te, self.idx_te+batch_size
        x_te, y_te = self.x_te[start:end], self.y_te[start:end]
        x_te = np.expand_dims(x_te, axis=3)

        terminator = False
        if(end >= self.num_te):
            terminator = True
            self.idx_te = 0
        else: self.idx_te = end

        if(self.normalize):
            min_x, max_x = x_te.min(), x_te.max()
            x_te = (x_te - min_x) / (max_x - min_x)

        x_te_torch = torch.from_numpy(np.transpose(x_te, (0, 3, 1, 2)))
        y_te_torch = torch.from_numpy(y_te)

        return x_te, x_te_torch, y_te, y_te_torch, terminator
