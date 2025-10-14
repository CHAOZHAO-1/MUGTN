import torch
import numpy as np
from scipy.fftpack import fft
import scipy.io as scio


def zscore(Z):

    Zmax, Zmin = Z.max(axis=1), Z.min(axis=1)
    return (Z - Zmin.reshape(-1, 1)) / (Zmax.reshape(-1, 1) - Zmin.reshape(-1, 1) + 1e-8)


def min_max(Z):

    Zmin = Z.min(axis=1)
    return np.log(Z - Zmin.reshape(-1, 1) + 1)


def process_signal(data, key, fft_enabled, fft_length):

    X = data[key].T
    if fft_enabled:
        X = abs(fft(X))[:, :fft_length]
        X = min_max(X)
    return zscore(X)


def load_features(domainlist, fft_enabled, data_path, fft_len, mode):

    data = scio.loadmat(data_path)
    features = [process_signal(data, f'load{d}_{mode}', fft_enabled, fft_len) for d in domainlist]
    return np.vstack(features)


def make_labels(class_num, domains, samples_per_class, add_domain=False):

    labels = []
    for domain_id in range(domains):
        base = torch.repeat_interleave(torch.arange(class_num), samples_per_class)
        if add_domain:
            domain_col = torch.full((class_num * samples_per_class,), domain_id)
            labels.append(torch.stack([base, domain_col], dim=1))
        else:
            labels.append(base)
    return torch.cat(labels, dim=0)


def load_training(domainlist, fft1, batch_size, kwargs,
                  root_vib='/home/zhaochao/research/DTL/data/CG5.mat',  # vib data
                  root_asc='/home/zhaochao/research/DTL/data/CGaud5.mat', ## asc data
                  fft_vib_len=2048, fft_asc_len=19200, class_num=5):

    vib_features = load_features(domainlist, fft1, root_vib, fft_vib_len, 'train')
    asc_features = load_features(domainlist, fft1, root_asc, fft_asc_len, 'train')


    train_fea = np.concatenate((vib_features, asc_features), axis=1)


    train_label = make_labels(class_num, len(domainlist), 100, add_domain=True)


    train_fea = torch.tensor(train_fea, dtype=torch.float32)
    train_label = train_label.long()

    dataset = torch.utils.data.TensorDataset(train_fea, train_label)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)

    print(f"Training feature shape: {train_fea.shape}")
    return train_loader


def load_testing(domainlist, fft1, batch_size, kwargs,
                 root_vib='/home/zhaochao/research/DTL/data/CG5.mat',
                 root_asc='/home/zhaochao/research/DTL/data/CGaud5.mat',
                 fft_vib_len=2048, fft_asc_len=19200, class_num=5):

    vib_features = load_features(domainlist, fft1, root_vib, fft_vib_len, 'test')
    asc_features = load_features(domainlist, fft1, root_asc, fft_asc_len, 'test')
    test_fea = np.concatenate((vib_features, asc_features), axis=1)


    test_label = make_labels(class_num, len(domainlist), 50, add_domain=False)

    test_fea = torch.tensor(test_fea, dtype=torch.float32)
    test_label = test_label.long()

    dataset = torch.utils.data.TensorDataset(test_fea, test_label)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, **kwargs)

    print(f"Testing feature shape: {test_fea.shape}")
    return test_loader
