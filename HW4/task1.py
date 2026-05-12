import statistics
import time

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def prepare_data() -> TensorDataset:
    X = torch.randn(10000, 128)
    y = torch.randint(0, 2, (10000,))
    dataset = TensorDataset(X, y)
    return dataset


def train():
    dataloader = DataLoader(prepare_data(), batch_size=256, shuffle=True, pin_memory=True) # включим pin_memory 
    device = torch.device('cuda')


    model = nn.Sequential(
        nn.Linear(128, 512), nn.ReLU(),
        nn.Linear(512, 128), nn.ReLU(),
        nn.Linear(128, 2)
    ).to(device).train()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    losses_history = []
    forward_times = []
    backward_times = []

    for batch_idx, (data, target) in enumerate(dataloader):

        
        data = data.to(device, non_blocking=True)
        
        noise = torch.randn_like(data)
        data = data + noise # изначально шум создавался на ЦПУ, из-за чего это замедляло обучение
        target = target.to(device, non_blocking=True)


        optimizer.zero_grad(set_to_none=True) # обычное занулени медленное и держит лишнюю память, будем задавать None

        # Изначально при записи времени мы использовали time.time() это некорректно, так как так мы замеряем постановку операций в очередь, а не время выполнения. 
        # Cuda работает асинхронно, поэтому время постановки операции не равно времени когда начались вычисления
        time_start = torch.cuda.Event(enable_timing=True) 
        time_end = torch.cuda.Event(enable_timing=True)
        time_start.record()
        output = model(data)
        loss = criterion(output, target)
        time_end.record()
        forward_times.append((time_start, time_end))

        time_start_bwd = torch.cuda.Event(enable_timing=True)
        time_end_bwd = torch.cuda.Event(enable_timing=True)
        time_start_bwd.record()
        loss.backward()
        time_end_bwd.record()
        backward_times.append((time_start_bwd, time_end_bwd))

        optimizer.step()


        # Мы сохраняли лосс вместе с графом вычислений, что вызовет OOM при большом датасете
        losses_history.append(loss.detach()) 


        # при использовании loss.item() CPU синхронится с GPU, что ломает асинхронность
        print(f"Batch {batch_idx} loss: {loss.detach().cpu().item():.4f}")


        # эта строчка замедляет обучение и ломает caching allocator
        # torch.cuda.empty_cache()
    

    torch.cuda.synchronize()
    forward_times = [start.elapsed_time(end) for start, end in forward_times]
    backward_times = [start.elapsed_time(end) for start, end in backward_times]

    print(f"Epoch finished, avg forward time is {statistics.mean(forward_times):.4f} ms, "
          f"avg backward time is {statistics.mean(backward_times):.4f} ms")

if __name__ == '__main__':
    train()