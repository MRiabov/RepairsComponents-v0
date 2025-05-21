from abc import ABC, abstractmethod
import numpy as np

class ElectricalComponent(ABC):
    def __init__(self, name: str = None):
        # Use a list for connections; handle vectorization at the simulation level
        self.connected_to: list[ElectricalComponent] = [] 
        self.name = name
        self.state = None  # Optional: can be used for output, logic state, etc.
    
    def connect(self, other: 'ElectricalComponent'):
        self.connected_to.append(other)
    
    @abstractmethod
    def modify_current(self, voltage: float, current: float) -> float:
        pass
    
class ElectricalConsumer(ElectricalComponent):
    @abstractmethod
    def modify_current(self, voltage: float, current: float) -> float:
        pass
    
    @abstractmethod
    def use_current(self, voltage: float, current: float) -> float:
        pass
    
class ElectricalGate(ElectricalComponent):
    
    
    @abstractmethod
    def modify_current(self, voltage: float, current: float, property) -> float:
        pass
        