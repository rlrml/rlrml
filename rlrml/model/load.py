import torch


def batched_packed_loader(dataset, *args, **kwargs) -> torch.utils.data.DataLoader:
    kwargs.setdefault("pin_memory", False)
    kwargs.setdefault("batch_size", 32)
    kwargs.setdefault("shuffle", True)
    kwargs.setdefault("collate_fn", collate_variable_size_samples)
    return torch.utils.data.DataLoader(dataset, *args, **kwargs)


def collate_variable_size_samples(samples, truncate_to=4000):
    padded = torch.nn.utils.rnn.pad_sequence(
        (s[0][:truncate_to] for s in samples), batch_first=True
    )

    zip_args = [padded]
    for i in range(1, len(samples[0])):
        zip_args.append([s[i] for s in samples])
    result = torch.utils.data.default_collate(list(zip(*zip_args)))
    return result
