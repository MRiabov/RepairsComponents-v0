"""Control components like buttons and switches."""
from typing import Dict, Any, Optional, Callable, Union
#import mujoco #TODO replace mujoco with Genesis!
from .base import Component


class Button(Component):
    """A button component that can be toggled on/off and triggers a callback.
    
    The button can be pressed to toggle its state between on and off. When the state
    changes, an optional callback function is called with the new state.
    """
    
    def __init__(
        self,
        on_press: Optional[Callable[[bool], None]] = None,
        initial_state: bool = False,
        press_force: float = 1.0,
        size: float = 1.0,
        name: str = None,
    ):
        """Initialize a button component.
        
        Args:
            on_press: Optional callback function that is called when the button is pressed.
                     The callback receives a single boolean argument (the new state).
            initial_state: Initial state of the button (True = on, False = off).
            press_force: Force required to press the button in Newtons.
            size: Size of the button in mm.
            name: Optional name for the button.
        """
        super().__init__(name=name)
        self._on_press = on_press
        self._state = initial_state
        self._press_force = press_force
        self._size = size
        self._prev_state = initial_state
    
    def to_mjcf(self) -> str:
        """Convert the button to MJCF XML string.
        
        Returns:
            str: MJCF XML representation of the button.
        """
        # Visual representation of the button
        button_color = "0 0.8 0 1" if self._state else "0.8 0 0 1"  # Green when on, red when off
        
        return f"""
        <body name="{self.name}">
            <freejoint/>
            <geom type="sphere" size="{self._size/2000}" rgba="{button_color}"/>
            <site name="{self.name}_press_sensor" size="0.01"/>
        </body>
        """
    
    def press(self, force: float = 1.0) -> bool:
        """Press the button with a given force.
        
        Args:
            force: Force applied to press the button in Newtons.
            
        Returns:
            bool: True if the button state changed, False otherwise.
        """
        if force >= self._press_force:
            self._prev_state = self._state
            self._state = not self._state
            
            # Call the callback if provided
            if self._on_press is not None:
                self._on_press(self._state)
                
            return True
        return False
    
    def step(self) -> None:
        """Update the button state based on current simulation state."""
        super().step()
        
        # In a real implementation, you would check for contact forces
        # with the button to determine if it's being pressed
        if self._model is not None and self._data is not None:
            # This is a simplified example - in a real implementation, you would
            # check contact forces on the button's geom
            pass
    
    @property
    def state(self) -> bool:
        """Get the current state of the button.
        
        Returns:
            bool: True if the button is pressed/on, False otherwise.
        """
        return self._state
    
    @property
    def state_changed(self) -> bool:
        """Check if the button state changed in the last step.
        
        Returns:
            bool: True if the state changed, False otherwise.
        """
        return self._state != self._prev_state
    
    def reset(self) -> None:
        """Reset the button to its initial state."""
        super().reset()
        self._state = False
        self._prev_state = False
    
    @property
    def state_dict(self) -> Dict[str, Any]:
        """Get the current state of the button as a dictionary.
        
        Returns:
            Dict containing the button's state information.
        """
        base_state = super().state
        base_state.update({
            'state': self._state,
            'press_force': self._press_force,
            'size': self._size,
        })
        return base_state
