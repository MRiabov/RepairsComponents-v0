# API Reference

This section provides detailed documentation for the RepairsComponents API.

## Base Classes

```{toctree}
:maxdepth: 1

base
```

## Components

```{toctree}
:maxdepth: 1

fasteners
sockets
controls
```

## API Conventions

### Naming

- Class names use `CamelCase`
- Method and function names use `snake_case`
- Private methods and attributes start with `_`

### Parameters

- All distances are in millimeters (mm)
- All angles are in radians unless otherwise specified
- Forces are in Newtons (N)
- Torques are in Newton-meters (N·m)

### Return Values

- Methods that modify state typically return `None`
- Methods that check conditions return `bool`
- Getters return the requested value
- Setters typically return `None`

## Type Hints

All functions and methods use Python type hints. For example:

```python
def fasten(self, angle: float) -> None:
    """Fasten the screw by rotating it.
    
    Args:
        angle: The angle to rotate (in radians)
    """
    pass
```

## Error Handling

- Invalid operations raise appropriate exceptions
- Methods document the exceptions they may raise
- Error messages are descriptive and include suggestions for resolution

## Thread Safety

- Components are not thread-safe by default
- For multi-threaded applications, use appropriate synchronization

## Performance Considerations

- Minimize object creation in performance-critical code
- Reuse objects when possible
- Use vectorized operations for better performance

## Extending the API

To extend the API, create a new class that inherits from `Component` or one of its subclasses. Override methods as needed to implement custom behavior.
