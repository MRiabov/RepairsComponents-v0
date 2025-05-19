# Sockets

## BaseSocket

A base class for socket components that can connect and disconnect.

```{eval-rst}
.. autoclass:: repairs_components.sockets.BaseSocket
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource
```

### Initialization

```python
def __init__(
    self,
    size: float = 8.0,
    depth: float = 10.0,
    wall_thickness: float = 1.0,
    friction: float = 0.1,
    name: Optional[str] = None
)
```

**Parameters:**
- `size`: Inner diameter of the socket in mm
- `depth`: Depth of the socket in mm
- `wall_thickness`: Thickness of the socket walls in mm
- `friction`: Friction coefficient for the socket
- `name`: Optional name for the socket

### Methods

#### `connect`

```python
def connect(self, force: float = 1.0) -> bool
```

Attempt to connect the socket.

**Parameters:**
- `force`: The force applied during connection in Newtons

**Returns:**
- `True` if connection was successful, `False` otherwise

#### `disconnect`

```python
def disconnect(self) -> bool
```

Disconnect the socket.

**Returns:**
- `True` if disconnection was successful, `False` otherwise

### Properties

#### `is_connected`

```python
@property
def is_connected(self) -> bool
```

Check if the socket is connected.

## LockingSocket

A socket with a locking mechanism that requires a release action.

```{eval-rst}
.. autoclass:: repairs_components.sockets.LockingSocket
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource
```

### Initialization

```python
def __init__(
    self,
    size: float = 8.0,
    depth: float = 10.0,
    wall_thickness: float = 1.0,
    requires_release: bool = True,
    release_force: float = 5.0,
    friction: float = 0.1,
    name: Optional[str] = None
)
```

**Parameters:**
- `size`: Inner diameter of the socket in mm
- `depth`: Depth of the socket in mm
- `wall_thickness`: Thickness of the socket walls in mm
- `requires_release`: Whether the socket requires a release action to disconnect
- `release_force`: Force required to activate the release in Newtons
- `friction`: Friction coefficient for the socket
- `name`: Optional name for the socket

### Methods

#### `activate_release`

```python
def activate_release(self, force: float) -> bool
```

Activate the release mechanism.

**Parameters:**
- `force`: The force applied to activate the release in Newtons

**Returns:**
- `True` if release was activated, `False` otherwise

### Example

```python
from repairs_components import LockingSocket

# Create a locking socket
socket = LockingSocket(
    size=10.0,           # 10mm inner diameter
    depth=15.0,          # 15mm deep
    wall_thickness=1.5,  # 1.5mm wall thickness
    requires_release=True,
    release_force=5.0,   # 5N to release
    name="power_socket"
)

# Connect the socket
if socket.connect(force=3.0):
    print("Socket connected!")


# Try to disconnect (will fail without releasing)
if not socket.disconnect():
    print("Cannot disconnect: release not activated")

# Activate release and disconnect
if socket.activate_release(force=6.0):  # Must be >= release_force
    if socket.disconnect():
        print("Successfully disconnected")
```

## Exceptions

### SocketError

Base class for all socket-related exceptions.

```{eval-rst}
.. autoexception:: repairs_components.sockets.SocketError
   :members:
```

### SocketConnectionError

Raised when there is an error connecting or disconnecting a socket.

```{eval-rst}
.. autoexception:: repairs_components.sockets.SocketConnectionError
   :members:
```

## Constants

### MIN_SOCKET_SIZE

Minimum allowed socket size in mm.

### MAX_SOCKET_DEPTH

Maximum allowed socket depth in mm.
