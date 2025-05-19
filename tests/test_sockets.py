"""Tests for the sockets module."""

import pytest

from repairs_components.sockets import BasicSocket, LockingSocket


def test_basic_socket_initialization():
    """Test basic socket initialization."""
    socket = BasicSocket(size=8.0, name="test_socket")
    
    assert socket.name == "test_socket"
    assert socket.size == 8.0
    assert not socket.is_connected


def test_basic_socket_connect_disconnect():
    """Test basic socket connection and disconnection."""
    socket = BasicSocket(size=8.0)
    
    # Connect the socket
    assert socket.connect(force=15.0)
    assert socket.is_connected
    
    # Disconnect the socket
    assert socket.disconnect()
    assert not socket.is_connected


def test_locking_socket_initialization():
    """Test locking socket initialization."""
    socket = LockingSocket(size=10.0, requires_release=True, name="test_locking_socket")
    
    assert socket.name == "test_locking_socket"
    assert socket.size == 10.0
    assert socket.requires_release
    assert not socket.is_connected
    assert not socket._release_activated


def test_locking_socket_connect_disconnect():
    """Test locking socket connection and disconnection with release mechanism."""
    socket = LockingSocket(size=10.0, requires_release=True)
    
    # Connect the socket
    assert socket.connect(force=15.0)
    assert socket.is_connected
    
    # Try to disconnect without activating release (should fail)
    assert not socket.disconnect()
    assert socket.is_connected
    
    # Activate release and try again
    assert socket.activate_release(force=6.0)  # Above release force
    assert socket.disconnect()
    assert not socket.is_connected


def test_locking_socket_no_release_needed():
    """Test locking socket when no release is needed."""
    socket = LockingSocket(size=10.0, requires_release=False)
    
    # Connect the socket
    assert socket.connect(force=15.0)
    assert socket.is_connected
    
    # Should be able to disconnect without activating release
    assert socket.disconnect()
    assert not socket.is_connected
