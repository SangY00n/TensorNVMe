import os
import torch
import uuid
from typing import Callable, Optional, List, Dict, Tuple
from tensornvme._C import Offloader, get_backends
import numpy as np

class ArrayInfo():
    def __init__(self, array: np.ndarray) -> None:
        # self.data_ptr: int = array.__array_interface__['data'][0]
        self.nbytes: int = array.nbytes
        # self.array_id = id(array)
        self.shape: Tuple = array.shape
        # self.numel = array.size # == np.prod(array.shape)

    # def __hash__(self) -> int:
    #     return self.array_id

    # def __eq__(self, other: 'ArrayInfo') -> bool:
    #     return self.array_id == other.array_id

    # def __ne__(self, other: 'ArrayInfo') -> bool:
    #     return self.array_id != other.array_id

    def __str__(self) -> str:
        return f'ArrayInfo(nbytes={self.nbytes}, shape={self.shape})'
        # return f'ArrayInfo(data_ptr={self.data_ptr}, nbytes={self.nbytes}, array_id={self.array_id})'

    def __repr__(self) -> str:
        return self.__str__()

    # def __del__(self) -> None:
    #     print(f'ArrayInfo: {self.array_id} is deleted')

class DiskOffloader(Offloader):
    
    def __init__(self, dir_name: str, n_entries: int = 16, backend: str = 'uring') -> None:
        assert backend in get_backends(
        ), f'Unsupported backend: {backend}, please install tensornvme with this backend'
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
        assert os.path.isdir(dir_name)
        filename = os.path.join(dir_name, f'offload-{uuid.uuid4().hex}')
        while os.path.exists(filename):
            filename = os.path.join(dir_name, f'offload-{uuid.uuid4().hex}')
            
        self.array_id_info_dict: Dict[int, ArrayInfo] = {}
            
        super().__init__(filename, n_entries, backend)

    def async_write(self, array: np.ndarray, callback: Optional[Callable[[], None]] = None) -> np.ndarray:
        # assert tensor.storage().size() > 0
        assert array.size > 0
        data_ptr = array.__array_interface__['data'][0]
        array_info = ArrayInfo(array)
        self.array_id_info_dict[id(array)] = array_info

        def callback_fn():
            # tensor.storage().resize_(0)
            array.resize(0, refcheck=False)
            if callback is not None:
                callback()
        super().async_write(data_ptr, array_info.nbytes, str(id(array)), callback_fn)
        
        return array

    def async_read(self, array: np.ndarray, callback: Optional[Callable[[], None]] = None) -> np.ndarray:
        # if tensor.storage().size() == 0:
        #     tensor.storage().resize_(tensor.numel())
        
        array_info = self.array_id_info_dict[id(array)]
        if array.size == 0:
            array.resize(array_info.shape, refcheck=False)
            
        data_ptr = array.__array_interface__['data'][0]
            
        super().async_read(data_ptr, array_info.nbytes, str(id(array)), callback)
        
        return array

    def sync_write(self, array: np.ndarray) -> np.ndarray:
        # assert tensor.storage().size() > 0
        
        assert array.size > 0
        data_ptr = array.__array_interface__['data'][0]
        array_info = ArrayInfo(array)
        self.array_id_info_dict[id(array)] = array_info
        
        super().sync_write(data_ptr, array_info.nbytes, str(id(array)))
        # tensor.storage().resize_(0)
        array.resize(0, refcheck=False)
        
        return array

    def sync_read(self, array: np.ndarray) -> np.ndarray:
        # if tensor.storage().size() == 0:
        #     tensor.storage().resize_(tensor.numel())
        
        array_info = self.array_id_info_dict[id(array)]
        if array.size == 0:
            array.resize(array_info.shape, refcheck=False)
        
        data_ptr = array.__array_interface__['data'][0]
            
        super().sync_read(data_ptr, array_info.nbytes, str(id(array)))
        
        return array

    def async_writev(self, tensors: List[torch.Tensor], callback: Optional[Callable[[], None]] = None) -> None:
        for tensor in tensors:
            assert tensor.storage().size() > 0
        key = str(hash(tuple(tensors)))

        def callback_fn():
            for tensor in tensors:
                tensor.storage().resize_(0)
            if callback is not None:
                callback()
        super().async_writev(tensors, key, callback_fn)

    def async_readv(self, tensors: List[torch.Tensor], callback: Optional[Callable[[], None]] = None) -> None:
        for tensor in tensors:
            if tensor.storage().size() == 0:
                tensor.storage().resize_(tensor.numel())
        key = str(hash(tuple(tensors)))
        super().async_readv(tensors, key, callback)

    def sync_writev(self, tensors: List[torch.Tensor]) -> None:
        for tensor in tensors:
            assert tensor.storage().size() > 0
        key = str(hash(tuple(tensors)))
        super().sync_writev(tensors, key)
        for tensor in tensors:
            tensor.storage().resize_(0)

    def sync_readv(self, tensors: List[torch.Tensor]) -> None:
        for tensor in tensors:
            if tensor.storage().size() == 0:
                tensor.storage().resize_(tensor.numel())
        key = str(hash(tuple(tensors)))
        super().sync_readv(tensors, key)