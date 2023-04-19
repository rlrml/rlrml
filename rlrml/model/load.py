import torch


def batched_packed_loader(dataset, *args, **kwargs):
    kwargs.setdefault("pin_memory", False)
    kwargs.setdefault("batch_size", 32)
    kwargs.setdefault("shuffle", True)
    kwargs.setdefault("collate_fn", collate_variable_size_samples)
    return torch.utils.data.DataLoader(dataset, *args, **kwargs)


def collate_variable_size_samples(samples):
    padded = torch.nn.utils.rnn.pad_sequence(
        (s[0] for s in samples), batch_first=True
    )
    arg = list(zip(padded, [s[1] for s in samples]))
    result = torch.utils.data.default_collate(arg)
    return result
