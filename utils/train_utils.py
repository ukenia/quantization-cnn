import torch

def get_num_workers():
    # Check if cuda is available

    cuda = torch.cuda.is_available()
    num_workers = 4 if cuda else 0
    print("Cuda = "+str(cuda)+" with num_workers = "+str(num_workers))
    return num_workers


def get_device():
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device
