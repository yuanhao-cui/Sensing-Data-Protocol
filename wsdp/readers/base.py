from abc import ABC, abstractmethod
from wsdp.structure import CSIData


class BaseReader(ABC):
    """
    Base class for Readers
    One reader handles specified type of file
    """

    @abstractmethod
    def read_file(self, file_path: str) -> CSIData:
        pass
