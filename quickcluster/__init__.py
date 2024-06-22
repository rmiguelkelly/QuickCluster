

from quickcluster._kmeans import KMeans

from quickcluster._device import Device
from quickcluster._device import findDevice
from quickcluster._device import useDevice

__all__ = [

    # KMeans algorithm class
    'KMeans',

    # GPU related stuff
    'Device',
    'findDevice',
    'deviceEnabled',
    'useDevice',
]