import torch


def batched_packed_loader(dataset, *args, **kwargs):
    kwargs.setdefault("pin_memory", True)
    kwargs.setdefault("batch_size", 8)
    kwargs.setdefault("shuffle", True)
    # kwargs.setdefault("collate_fn", collate_variable_size_samples)
    return torch.utils.data.DataLoader(dataset, *args, **kwargs)


def collate_variable_size_samples(samples):
    torch.nn.utils.rnn.pack_padded_sequence(samples,)
