"""Base component class for Genesis-based components."""
from abc import ABC, abstractmethod
from typing import Dict, Any
import genesis as gs  # type: ignore

class GenesisComponent(ABC):
    """Base class for all Genesis repair components.
    
    This class defines the common interface and functionality for all repair components
    using the Genesis simulator.
    """
    
    def __init__(self, name: str = "") -> None:
        """Initialize the component.
        
        Args:
            name: Optional name for the component. If None, a default name will be generated.
        """
        self.name = name or f"{self.__class__.__name__}_{id(self)}"
        self._scene = None
        self._initialized = False
    
    @abstractmethod
    def to_mjcf(self) -> str:
        """Convert the component to MJCF XML string.
        
        Returns:
            str: MJCF XML string representation of the component.
        """
        pass
    
    def attach_to_scene(self, scene: 'gs.Scene') -> None:
        """Attach the component to a Genesis scene.
        
        Args:
            scene: The Genesis scene to attach to.
        """
        self._scene = scene
        self._initialized = True
    
    def step(self) -> None:
        """Perform a simulation step for this component.
        
        This method is called once per simulation step. Subclasses can override this
        to implement custom behavior.
        """
        pass
    
    def reset(self) -> None:
        """Reset the component to its initial state.
        
        This method is called when the simulation is reset. Subclasses can override this
        to implement custom reset behavior.
        """
        pass
    
    def get_state(self) -> Dict[str, Any]:
        """Get the current state of the component.
        
        Returns:
            Dictionary containing the component's state.
        """
        return {}
    
    def set_state(self, state: Dict[str, Any]) -> None:
        """Set the state of the component.
        
        Args:
            state: Dictionary containing the component's state.
        """
        pass
