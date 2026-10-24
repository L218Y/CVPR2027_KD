"""The data stream must not depend on how much RNG an arm consumed before training.

Building a teacher or a probe draws from the global RNG, so arms that build
different auxiliary modules would otherwise iterate different batches despite the
same --seed. tools/train_kd.py passes a dedicated generator to both the sampler
and the DataLoader; this test pins that behaviour.
"""

import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler


class _Rows(Dataset):
    def __init__(self, n=64):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return torch.tensor([i], dtype=torch.float32)


def _first_epoch_indices(prior_draws, seed=3407):
    torch.manual_seed(seed)
    for _ in range(prior_draws):        # stand-in for building a teacher/probe
        torch.rand(1)
    gen = torch.Generator()
    gen.manual_seed(seed)
    ds = _Rows()
    sampler = RandomSampler(ds, generator=gen)
    loader = DataLoader(ds, batch_size=8, sampler=sampler, num_workers=0, drop_last=True)
    return [int(x.item()) for batch in loader for x in batch]


def test_data_order_is_independent_of_prior_rng_use():
    a = _first_epoch_indices(0)
    b = _first_epoch_indices(1)
    c = _first_epoch_indices(37)
    assert a == b == c, (a[:8], b[:8], c[:8])


def test_data_order_still_changes_with_seed():
    assert _first_epoch_indices(0, seed=3407) != _first_epoch_indices(0, seed=3408)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
