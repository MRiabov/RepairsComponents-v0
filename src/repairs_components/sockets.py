"""Socket components for connecting and fastening."""
from typing import Dict, Any, Optional, Tuple
import numpy as np
import mujoco
from .base import Component


class BasicSocket(Component):
    """A basic socket that can be plugged and unplugged."""
    
    def __init__(
        self,
        size: float = 8.0,  # Standard size in mm
        depth: float = 20.0,
        wall_thickness: float = 2.0,
        name: str = None,
    ):
        """Initialize a basic socket.
        
        Args:
            size: Inner size (across flats) of the socket in mm.
            depth: Depth of the socket in mm.
            wall_thickness: Thickness of the socket walls in mm.
            name: Optional name for the socket.
        """
        super().__init__(name=name)
        self.size = size
        self.depth = depth
        self.wall_thickness = wall_thickness
        self._connected = False
        self._connection_force = 0.0
    
    def to_mjcf(self) -> str:
        """Convert the socket to MJCF XML string."""
        # Hexagonal socket geometry
        outer_radius = (self.size + 2 * self.wall_thickness) / 2 / np.cos(np.pi/6)
        inner_radius = self.size / 2 / np.cos(np.pi/6)
        
        return f"""
        <body name="{self.name}">
            <freejoint/>
            <!-- Outer shell -->
            <geom type="cylinder" size="{outer_radius/1000} {self.depth/2000}" 
                  rgba="0.3 0.3 0.3 1"/>
            <!-- Inner socket (hollow) -->
            <geom type="cylinder" size="{inner_radius/1000} {self.depth/2000}" 
                  pos="0 0 0" rgba="0.8 0.8 0.8 0.5" contype="0" conaffinity="0"/>
        </body>
        """
    
    def connect(self, force: float = 10.0) -> bool:
        """Attempt to connect the socket.
        
        Args:
            force: Connection force to apply.
            
        Returns:
            bool: True if connection was successful, False otherwise.
        """
        self._connection_force = force
        self._connected = True
        return True
    
    def disconnect(self) -> bool:
        """Disconnect the socket.
        
        Returns:
            bool: True if disconnection was successful.
        """
        self._connection_force = 0.0
        self._connected = False
        return True
    
    @property
    def is_connected(self) -> bool:
        """Check if the socket is connected."""
        return self._connected
    
    @property
    def state(self) -> Dict[str, Any]:
        """Get the current state of the socket."""
        base_state = super().state
        base_state.update({
            'size': self.size,
            'depth': self.depth,
            'connected': self._connected,
            'connection_force': self._connection_force,
        })
        return base_state


class LockingSocket(BasicSocket):
    """A socket with a locking mechanism that requires a release to disconnect."""
    
    def __init__(
        self,
        size: float = 10.0,
        depth: float = 25.0,
        wall_thickness: float = 2.5,
        requires_release: bool = True,
        release_force: float = 5.0,
        name: str = None,
    ):
        """Initialize a locking socket.
        
        Args:
            size: Inner size (across flats) of the socket in mm.
            depth: Depth of the socket in mm.
            wall_thickness: Thickness of the socket walls in mm.
            requires_release: Whether the release mechanism must be activated to disconnect.
            release_force: Force required to activate the release mechanism in Newtons.
            name: Optional name for the socket.
        """
        super().__init__(size, depth, wall_thickness, name)
        self.requires_release = requires_release
        self.release_force = release_force
        self._release_activated = False
    
    def to_mjcf(self) -> str:
        """Convert the locking socket to MJCF XML string."""
        base_xml = super().to_mjcf()
        # Add release mechanism (simplified)
        release_xml = f"""
            <!-- Release mechanism -->
            <site name="{self.name}_release" pos="0 0 {-(self.depth/2 + 2)/1000}" size="0.005"/>
            <geom type="box" size="0.01 0.01 0.002" 
                  pos="0 0 {-(self.depth/2 + 1)/1000}" 
                  rgba="1 0 0 0.5"
                  group="3"/>
        """
        return base_xml.replace("</body>", f"{release_xml}\n        </body>")
    
    def activate_release(self, force: float) -> bool:
        """Activate the release mechanism.
        
        Args:
            force: Force applied to the release mechanism.
            
        Returns:
            bool: True if release was successfully activated.
        """
        if force >= self.release_force:
            self._release_activated = True
            return True
        return False
    
    def deactivate_release(self) -> None:
        """Deactivate the release mechanism."""
        self._release_activated = False
    
    def disconnect(self) -> bool:
        """Disconnect the socket if release is activated or not required."""
        if not self.requires_release or self._release_activated:
            self._release_activated = False
            return super().disconnect()
        return False
    
    @property
    def state(self) -> Dict[str, Any]:
        """Get the current state of the locking socket."""
        base_state = super().state
        base_state.update({
            'requires_release': self.requires_release,
            'release_activated': self._release_activated,
            'release_force': self.release_force,
        })
        return base_state
    
    def reset(self) -> None:
        """Reset the locking socket to its initial state."""
        super().reset()
        self._release_activated = False
