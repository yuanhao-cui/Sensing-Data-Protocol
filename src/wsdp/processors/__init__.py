from wsdp.interfaces import Processor as Processor

from .base_processor import BaseProcessor as BaseProcessor
from .configurable_processor import ConfigurableProcessor as ConfigurableProcessor
from .modular_processor import ModularProcessor as ModularProcessor

__all__ = ["BaseProcessor", "ConfigurableProcessor", "ModularProcessor", "Processor"]
