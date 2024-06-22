
import quickcluster._C as _bindings
import os

# Used to store the handle for the GPU
__device_handle = None

class Device:
    """Represents a GPU device on the host computer"""

    def __init__(self, id: int, name: str):
        self.id = id
        self.name = name

    # String representation on the device
    def __str__(self) -> str:
        return self.name


def findDevice() -> Device | None:
    """Attempt to find the GPU on the current hardware"""

    # Returned as a python dictionary or None
    device_attr = _bindings.device_retrieve_gpu()

    # No device found
    if device_attr == None:
        return None
    
    return Device(device_attr['device_id'], device_attr['device_name'])


def deviceEnabled() -> bool:
    """Checks to see if the GPU is enabled"""

    return findDevice() != None


def useDevice() -> None:
    """Signify that the KMeans implementation should use the GPU"""

    # Get the path to the library path for python
    file_path = os.path.abspath(__file__)
    lib_path = os.path.dirname(file_path)
    lib_path = os.path.join(lib_path, 'lib')

    # Globally set the device handle
    global __device_handle
    __device_handle = _bindings.device_create_handle(lib_path)

def retrieve_device_handle():
    """Internal function to pass the device handle around"""

    return __device_handle