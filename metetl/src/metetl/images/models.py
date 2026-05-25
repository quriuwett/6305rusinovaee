import numpy as np
from typing import Dict
from numpy.typing import NDArray

class Artwork:
    __slots__ = ['__image', '__metadata', '__order_index']

    def __init__(self, metadata: Dict[str, str], image: NDArray[np.uint8], order_index: int):
        self.__metadata = metadata
        self.__image = image
        self.__order_index = order_index

    @property
    def metadata(self) -> Dict[str, str]: return self.__metadata

    @property
    def image(self) -> NDArray[np.uint8]: return self.__image
    
    @property
    def order_index(self) -> int: return self.__order_index

    def halftone(self) -> NDArray[np.uint8]:
        res = (np.array(0.2126 * self.__image[:, :, 0]
                       + 0.7152 * self.__image[:, :, 1]
                       + 0.0722 * self.__image[:, :, 2])
        .astype(np.uint8))
        return res