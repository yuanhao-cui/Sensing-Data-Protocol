from abc import abstractmethod
from typing import Any, Dict, List

from wsdp.interfaces import Reader
from wsdp.structure import CSIData


class BaseReader(Reader):
    """
    Base class for Readers
    One reader handles specified type of file
    """

    @abstractmethod
    def read_file(self, file_path: str) -> CSIData | List[CSIData]:
        pass

    def sniff(self, file_path: str) -> bool:
        """
        Check if a file matches this reader's expected format.
        Override in subclasses for format-specific detection.

        Args:
            file_path: Path to the file

        Returns:
            True if the file can be read by this reader, False to skip.
        """
        return True

    def get_metadata(self) -> Dict[str, Any]:
        """
        Return metadata about this reader and its supported format.

        Returns:
            dict: Metadata including reader name, supported format, etc.
        """
        return {
            "reader": self.__class__.__name__,
            "format": self.__class__.__name__.replace("Reader", "").lower(),
        }
