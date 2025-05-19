"""Tests for the controls module."""

import pytest
from unittest.mock import Mock

from repairs_components.controls import Button


def test_button_initialization():
    """Test button initialization."""
    button = Button(name="test_button")
    
    assert button.name == "test_button"
    assert not button.state
    assert not button.state_changed


def test_button_press():
    """Test button press functionality."""
    # Create a mock callback
    mock_callback = Mock()
    
    # Create button with mock callback
    button = Button(on_press=mock_callback, press_force=1.0)
    
    # Press with insufficient force (should not trigger)
    assert not button.press(force=0.5)
    assert not button.state
    mock_callback.assert_not_called()
    
    # Press with sufficient force (should trigger)
    assert button.press(force=1.0)
    assert button.state
    mock_callback.assert_called_once_with(True)
    
    # Press again to toggle off
    mock_callback.reset_mock()
    assert button.press(force=1.0)
    assert not button.state
    mock_callback.assert_called_once_with(False)


def test_button_state_changed():
    """Test button state change detection."""
    button = Button(press_force=1.0)
    
    # Initial state
    assert not button.state_changed
    
    # Press the button
    button.press(force=1.0)
    assert button.state_changed
    
    # State changed flag should be reset after checking
    assert not button.state_changed
    
    # Press again
    button.press(force=1.0)
    assert button.state_changed


def test_button_reset():
    """Test button reset functionality."""
    button = Button(initial_state=True, press_force=1.0)
    
    # Change state
    button.press(force=1.0)
    assert not button.state
    
    # Reset should return to initial state
    button.reset()
    assert button.state is False
    assert not button.state_changed


def test_button_state_dict():
    """Test button state dictionary."""
    button = Button(
        on_press=None,
        initial_state=True,
        press_force=2.5,
        size=10.0,
        name="test_button"
    )
    
    state = button.state_dict
    assert state["name"] == "test_button"
    assert state["state"] is True
    assert state["press_force"] == 2.5
    assert state["size"] == 10.0
