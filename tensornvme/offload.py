import os
import torch # import 안하면 ImportError: libc10.so: cannot open shared object file: No such file or directory 오류 발생
import uuid
from typing import Callable, Optional, List, Dict, Tuple, Union
from tensornvme._C import Offloader, get_backends
import numpy as np
from jax import Array as jaxarray

class ArrayInfo():
    def __init__(self, array: Union[np.ndarray, jaxarray]) -> None:
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
            
        
        self._key_counter = 0
        self.array_id_info_dict: Dict[int, ArrayInfo] = {}
        self.arrays_key_infos_dict: Dict[str, List[ArrayInfo]] = {}
            
        super().__init__(filename, n_entries, backend)
        
    def key_generator(self) -> int:
        new_key = self._key_counter
        self._key_counter += 1
        return new_key

    def async_write(self, array: Union[np.ndarray, jaxarray], callback: Optional[Callable[[], None]] = None) -> int:
        assert array.size > 0
        
        if not isinstance(array, np.ndarray): # if isinstance(array, jaxarray)
            array = np.asarray(array)
            # array가 jax array인 경우 np.ndarray 로 바꿔줌 (copy x, 버퍼 공유)
            # jax array에 대한 메모리 할당 해제 및 garbage collection은 외부에서 사용자가 직접 해줘야 함.
            
        data_ptr = array.__array_interface__['data'][0]
        array_info = ArrayInfo(array)
        
        
        key = self.key_generator()
        self.array_id_info_dict[key] = array_info

        def callback_fn():
            if callback is not None:
                callback()
                
        super().async_write(data_ptr, array_info.nbytes, str(key), callback_fn)
        
        return key

    def async_read(self, key: int, callback: Optional[Callable[[], None]] = None) -> np.ndarray:        
        array_info = self.array_id_info_dict[key]
        
        def callback_fn():
            if callback is not None:
                callback()
            
        array = super().async_read(array_info.nbytes, str(key), callback_fn)
        array.resize(array_info.shape, refcheck=False)
        
        return array

    def sync_write(self, array: Union[np.ndarray, jaxarray]) -> int:        
        assert array.size > 0
        
        if not isinstance(array, np.ndarray): # if isinstance(array, jaxarray)
            array = np.asarray(array)
            # array가 jax array인 경우 np.ndarray 로 바꿔줌 (copy x, 버퍼 공유)
            # jax array에 대한 메모리 할당 해제 및 garbage collection은 외부에서 사용자가 직접 해줘야 함.
        
        data_ptr = array.__array_interface__['data'][0]
        array_info = ArrayInfo(array)
        
        
        key = self.key_generator()
        self.array_id_info_dict[key] = array_info
        
        super().sync_write(data_ptr, array_info.nbytes, str(key))
        
        return key
            

    def sync_read(self, key: int) -> np.ndarray:        
        array_info = self.array_id_info_dict[key]
        
        array = super().sync_read(array_info.nbytes, str(key))
        array.resize(array_info.shape, refcheck=False)
        
        return array

    def async_writev(self, arrays: List[Union[np.ndarray, jaxarray]], callback: Optional[Callable[[], None]] = None) -> List[np.ndarray]:
        # for tensor in tensors:
        #     assert tensor.storage().size() > 0
        
        is_readonly_array_list = list([False] * len(arrays))
        for i, array in enumerate(arrays):
            assert array.size > 0
            
            if not isinstance(array, np.ndarray):
                arrays[i] = np.asarray(array) # 이렇게 해도 되나? for문 안에서 요소 값 변경하면..
                is_readonly_array_list[i] = arrays[i].__array_interface__['data'][1]
        
        data_ptr_list = list(map(lambda array: array.__array_interface__['data'][0], arrays))
        array_info_list = list(map(lambda array: ArrayInfo(array), arrays))
        nbytes_list = list(map(lambda array_info: array_info.nbytes, array_info_list))
        
        # data pointer 랑 nbytes 다 추출했으므로 key 생성을 위해 arrays 내 jaxarray를 empty np.ndarray로 바꿔줌.
        for i in range(len(arrays)):
            if is_readonly_array_list[i]: # readonly가 True면 jaxarray이므로 empty array로 바꿔줌.
                empty_array = np.empty(0, dtype=arrays[i].dtype)
                arrays[i] = empty_array
        # key = str(hash(tuple(arrays))) # np.ndarray는 unhashable type 이므로 이렇게 하면 안됨.
        key = str(hash(tuple(map(lambda array: id(array), arrays)))) # 의외로 얼마 안걸림. 원래꺼보다 빠름.
        self.arrays_key_infos_dict[key] = array_info_list

        def callback_fn():
            # for tensor in tensors:
            #     tensor.storage().resize_(0)
            for i, array in enumerate(arrays):
                if not is_readonly_array_list[i]:
                    array.resize(0, refcheck=False)
                else:
                    pass
            if callback is not None:
                callback()
                
        super().async_writev(data_ptr_list, nbytes_list, key, callback_fn)
        
        return arrays

    def async_readv(self, arrays: List[np.ndarray], callback: Optional[Callable[[], None]] = None) -> List[np.ndarray]:
        # for tensor in tensors:
        #     if tensor.storage().size() == 0:
        #         tensor.storage().resize_(tensor.numel())
        
        key = str(hash(tuple(map(lambda array: id(array), arrays))))
        array_info_list = self.arrays_key_infos_dict[key]
        
        nbytes_list = list(map(lambda array_info: array_info.nbytes, array_info_list))

        for array, array_info in zip(arrays, array_info_list):
            if array.size == 0:
                array.resize(array_info.shape, refcheck=False)
        
        data_ptr_list = list(map(lambda array: array.__array_interface__['data'][0], arrays))
                
        super().async_readv(data_ptr_list, nbytes_list, key, callback)
        
        return arrays

    def sync_writev(self, arrays: List[np.ndarray]) -> List[np.ndarray]:
        # for tensor in tensors:
        #     assert tensor.storage().size() > 0
        
        is_readonly_array_list = list([False] * len(arrays))
        for i, array in enumerate(arrays):
            assert array.size > 0
            
            if not isinstance(array, np.ndarray):
                arrays[i] = np.asarray(array) # 이렇게 해도 되나? for문 안에서 요소 값 변경하면..
                is_readonly_array_list[i] = arrays[i].__array_interface__['data'][1]
            
        data_ptr_list = list(map(lambda array: array.__array_interface__['data'][0], arrays))
        array_info_list = list(map(lambda array: ArrayInfo(array), arrays))
        nbytes_list = list(map(lambda array_info: array_info.nbytes, array_info_list))


        # data pointer 랑 nbytes 다 추출했으므로 key 생성을 위해 arrays 내 jaxarray를 empty np.ndarray로 바꿔줌.
        for i in range(len(arrays)):
            if is_readonly_array_list[i]: # readonly가 True면 jaxarray이므로 empty array로 바꿔줌.
                empty_array = np.empty(0, dtype=arrays[i].dtype)
                arrays[i] = empty_array
        key = str(hash(tuple(map(lambda array: id(array), arrays))))
        self.arrays_key_infos_dict[key] = array_info_list
        
        super().sync_writev(data_ptr_list, nbytes_list, key)
        # for tensor in tensors:
        #     tensor.storage().resize_(0)
        
        for i, array in enumerate(arrays):
            if not is_readonly_array_list[i]:
                array.resize(0, refcheck=False)
            else:
                pass
            
        return arrays

    def sync_readv(self, arrays: List[np.ndarray]) -> List[np.ndarray]:
        # for tensor in tensors:
        #     if tensor.storage().size() == 0:
        #         tensor.storage().resize_(tensor.numel())
        
        key = str(hash(tuple(map(lambda array: id(array), arrays))))
        array_info_list = self.arrays_key_infos_dict[key]
        
        nbytes_list = list(map(lambda array_info: array_info.nbytes, array_info_list))

        for array, array_info in zip(arrays, array_info_list):
            if array.size == 0:
                array.resize(array_info.shape, refcheck=False)
        
        data_ptr_list = list(map(lambda array: array.__array_interface__['data'][0], arrays))

        super().sync_readv(data_ptr_list, nbytes_list, key)
        
        return arrays