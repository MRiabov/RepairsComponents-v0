"""Fastener components like screws, bolts, and nuts."""
from typing import Dict, Any, Optional
import numpy as np
#import mujoco #TODO replace mujoco with Genesis!
from .base import Component


class Screw(Component):
    """A screw component that can be fastened and released through rotation."""
    
    def __init__(
        self,
        thread_pitch: float = 0.5,
        length: float = 10.0,
        diameter: float = 3.0,
        head_diameter: float = 5.5,
        head_height: float = 2.0,
        name: Optional[str] = None,
        fastened: bool = False
    ):
        """Initialize a screw component.
        
        Args:
            thread_pitch: Distance between threads in mm.
            length: Total length of the screw in mm.
            diameter: Outer diameter of the screw thread in mm.
            head_diameter: Diameter of the screw head in mm.
            head_height: Height of the screw head in mm.
            name: Optional name for the screw.
        """
        super().__init__(name=name)
        self.thread_pitch = thread_pitch
        self.length = length
        self.diameter = diameter
        self.head_diameter = head_diameter
        self.head_height = head_height
        
        # State variables
        self._position = 0.0  # Current position along the screw axis
        self._rotation = 0.0  # Current rotation in radians
        self._fastened = fastened
    
    def to_mjcf(self) -> str:
        """Convert the screw to MJCF XML string.
        
        Returns:
            str: MJCF XML representation of the screw.
        """
        # This is a simplified representation. In a real implementation, you would
        # generate the complete MJCF for the screw including geometry and joints.
        return f"""
        <body name="{self.name}">
            <freejoint/>
            <geom type="cylinder" size="{self.diameter/2000} {self.length/2000}" rgba="0.8 0.8 0.8 1"/>
            <geom type="cylinder" size="{self.head_diameter/2000} {self.head_height/2000}" 
                  pos="0 0 {-(self.length + self.head_height)/2000}" rgba="0.5 0.5 0.5 1"/>
        </body>
        """
    
    def step(self) -> None:
        """Update the screw state based on current simulation state."""
        super().step()
        if self._data is not None:
            # Update position and rotation from simulation
            body_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, self.name)
            if body_id != -1:
                self._position = self._data.xpos[body_id][2]  # Z-axis position
                quat = self._data.xquat[body_id]
                self._rotation = 2 * np.arccos(quat[0])  # Simple conversion from quat to angle
    
    def fasten(self, rotation_angle: float) -> float:
        """Apply a rotation to fasten or loosen the screw.
        
        Args:
            rotation_angle: Angle to rotate the screw in radians.
            
        Returns:
            float: The actual rotation applied after considering constraints.
        """
        if not self._initialized:
            raise RuntimeError("Screw must be attached to a model before fastening")
            
        # Calculate linear movement based on thread pitch
        linear_movement = (rotation_angle / (2 * np.pi)) * self.thread_pitch
        
        # Update rotation
        self._rotation = (self._rotation + rotation_angle) % (2 * np.pi)
        
        # Update position
        new_position = self._position + linear_movement
        
        # Check if screw is fully fastened or released
        if new_position <= 0:
            new_position = 0
            self._fastened = False
        elif new_position >= self.length:
            new_position = self.length
            self._fastened = True
        else:
            self._fastened = False
            
        self._position = new_position
        
        # Update simulation state if attached to a model
        if self._model is not None and self._data is not None:
            body_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, self.name)
            if body_id != -1:
                self._data.qpos[body_id * 7 + 2] = self._position / 1000.0  # Convert to meters
                # Update rotation (simplified)
                self._data.qpos[body_id * 7 + 3:body_id * 7 + 7] = [
                    np.cos(self._rotation / 2),
                    0,
                    0,
                    np.sin(self._rotation / 2)
                ]
        
        return rotation_angle
    
    @property
    def is_fastened(self) -> bool:
        """Check if the screw is fully fastened."""
        return self._fastened
    
    @property
    def state(self) -> Dict[str, Any]:
        """Get the current state of the screw."""
        base_state = super().state
        base_state.update({
            'position': self._position,
            'rotation': self._rotation,
            'fastened': self._fastened,
            'thread_pitch': self.thread_pitch,
            'length': self.length,
            'diameter': self.diameter,
        })
        return base_state
    
    def reset(self) -> None:
        """Reset the screw to its initial state."""
        super().reset()
        self._position = 0.0
        self._rotation = 0.0
        self._fastened = False


# Export the Screw class
__all__ = ['Screw']
