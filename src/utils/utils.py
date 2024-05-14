import torch


def del_tensor_rows(arr, index, n):
    """
    arr: Input tensor
    index: Index of the location to be deleted
    n: Number of rows to be deleted from :attr:`index`
    """
    arr1 = arr[0:index]
    arr2 = arr[index + n:]
    return torch.cat((arr1, arr2), dim=0)
