import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, DistributedSampler
import timm
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import os
from models import ViT


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def evaluate(rank, world_size):
    setup(rank, world_size)
    device = torch.device(f"cuda:{rank}")

    # model_name = 'vit_base_patch16_224'
    model_name = 'B_16_imagenet1k'
    # model = timm.create_model(model_name, pretrained=True)
    model = ViT(name=model_name, pretrained=True)

    transform = transforms.Compose([
        transforms.Resize(model.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Adjust DataLoader for DDP
   
    val_dataset = ImageFolder(root='/home/c3-0/datasets/ImageNet/validation', transform=transform)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, sampler=val_sampler)

    model.to(device)
    ddp_model = DDP(model, device_ids=[rank])
    ddp_model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            if rank == 0:
                print(f"labels: {labels}")
            outputs = ddp_model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    # Gather results from all processes
    correct_tensor = torch.tensor(correct).to(device)
    total_tensor = torch.tensor(total).to(device)
    dist.reduce(correct_tensor, dst=0)
    dist.reduce(total_tensor, dst=0)
    if rank == 0:  # Only print on the main process
        print(f'Accuracy of the model on the validation images: {100 * correct_tensor.item() / total_tensor.item()}%')

    cleanup()

def run_demo(world_size):
    mp.spawn(evaluate,
             args=(world_size,),
             nprocs=world_size,
             join=True)

if __name__ == "__main__":
    world_size = 1 
    run_demo(world_size)
