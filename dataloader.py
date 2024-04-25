import numpy as np
import torch


def augmented_data(data, label):
    augmented_data = []
    new_label = []

    shift_range = 10
    scale_range = (0.9, 1.1)
    noise_std = 0.1
            
    for sample, lbl in zip(data, label):
        sample = torch.tensor(sample, dtype=torch.float32)
        for _ in range(3):
            augmented_data.append(sample.unsqueeze(0))
            new_label.append(lbl)

            shift = np.random.randint(-shift_range, shift_range)
            shifted_sample = torch.tensor(np.roll(sample, shift, axis=-1), dtype=torch.float32)
            augmented_data.append(shifted_sample.unsqueeze(0))
            new_label.append(lbl)

            scale = np.random.uniform(*scale_range)
            scaled_sample = (sample * scale).unsqueeze(0)
            augmented_data.append(scaled_sample)
            new_label.append(lbl)

            noise = np.random.normal(loc=0, scale=noise_std, size=sample.shape)
            noisy_sample = (sample + noise).unsqueeze(0)
            augmented_data.append(noisy_sample)
            new_label.append(lbl)


            noise_freq = np.random.normal(loc=0, scale=noise_std, size=sample.shape)
            noise_freq = np.fft.fft(noise_freq)
            sample_freq = np.fft.fft(sample.numpy())
            noisy_sample_freq = sample_freq + noise_freq
            noisy_sample_time = np.fft.ifft(noisy_sample_freq).real
            noisy_sample_time = torch.tensor(noisy_sample_time, dtype=torch.float32)
            augmented_data.append(noisy_sample_time.unsqueeze(0))
            new_label.append(lbl)

    augmented_data = torch.cat(augmented_data, dim=0).float()
    new_label = np.array(new_label)

    return augmented_data, new_label



def read_bci_data():
    S4b_train = np.load('./data/S4b_train.npz')
    X11b_train = np.load('./data/X11b_train.npz')
    S4b_test = np.load('./data/S4b_test.npz')
    X11b_test = np.load('./data/X11b_test.npz')

    train_data = np.concatenate((S4b_train['signal'], X11b_train['signal']), axis=0)
    train_label = np.concatenate((S4b_train['label'], X11b_train['label']), axis=0)
    test_data = np.concatenate((S4b_test['signal'], X11b_test['signal']), axis=0)
    test_label = np.concatenate((S4b_test['label'], X11b_test['label']), axis=0)

    train_label = train_label - 1
    test_label = test_label -1
    train_data = np.transpose(np.expand_dims(train_data, axis=1), (0, 1, 3, 2))
    test_data = np.transpose(np.expand_dims(test_data, axis=1), (0, 1, 3, 2))

    mask = np.where(np.isnan(train_data))
    train_data[mask] = np.nanmean(train_data)

    mask = np.where(np.isnan(test_data))
    test_data[mask] = np.nanmean(test_data)

    train_data, train_label = augmented_data(train_data, train_label)
    

    print(train_data.shape, train_label.shape, test_data.shape, test_label.shape)

    return train_data, train_label, test_data, test_label
