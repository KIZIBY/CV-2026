import statistics

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def prepare_data() -> TensorDataset:
    X = torch.randn(10000, 128)
    y = torch.randint(0, 2, (10000,))
    dataset = TensorDataset(X, y)
    return dataset


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"

    dataloader = DataLoader(
        prepare_data(),
        batch_size=256,
        shuffle=True,
        pin_memory=use_cuda,  # enables faster async CPU -> GPU copies
    )

    model = nn.Sequential(
        nn.Linear(128, 512), nn.ReLU(),
        nn.Linear(512, 128), nn.ReLU(),
        nn.Linear(128, 2),
    ).to(device).train()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    losses_history = []
    forward_events = []
    backward_events = []

    for batch_idx, (data, target) in enumerate(dataloader):
        # non_blocking=True can overlap data transfer with GPU work when memory is pinned.
        data = data.to(device, non_blocking=use_cuda)
        target = target.to(device, non_blocking=use_cuda)

        # Create noise directly on GPU instead of creating it on CPU and copying it.
        noise = torch.randn_like(data)
        data = data + noise

        optimizer.zero_grad(set_to_none=True)

        if use_cuda:
            fwd_start = torch.cuda.Event(enable_timing=True)
            fwd_end = torch.cuda.Event(enable_timing=True)
            bwd_start = torch.cuda.Event(enable_timing=True)
            bwd_end = torch.cuda.Event(enable_timing=True)

            fwd_start.record()

        output = model(data)
        loss = criterion(output, target)

        if use_cuda:
            fwd_end.record()
            bwd_start.record()

        loss.backward()

        if use_cuda:
            bwd_end.record()
            forward_events.append((fwd_start, fwd_end))
            backward_events.append((bwd_start, bwd_end))

        optimizer.step()

        # Store only a plain Python number, not the tensor with its computation graph.
        losses_history.append(loss.detach().cpu().item())

        # Avoid synchronizing GPU with CPU on every batch; log less often.
        if batch_idx % 10 == 0:
            print(f"Batch {batch_idx} loss: {losses_history[-1]:.4f}")

    if use_cuda:
        torch.cuda.synchronize()
        forward_times = [
            start.elapsed_time(end) / 1000
            for start, end in forward_events
        ]
        backward_times = [
            start.elapsed_time(end) / 1000
            for start, end in backward_events
        ]

        print(
            f"Epoch finished, avg forward time is {statistics.mean(forward_times)}, "
            f"avg backward time is {statistics.mean(backward_times)}"
        )

    print(f"Avg loss is {statistics.mean(losses_history):.4f}")


if __name__ == "__main__":
    train()
