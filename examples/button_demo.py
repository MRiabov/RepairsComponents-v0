"""Example demonstrating the Button component in a simple Genesis environment."""
import genesis as gs

def main():
    """Run a simple Genesis simulation with a button and a box."""
    # Initialize Genesis with CPU backend
    gs.init(backend='cpu', logging_level='info')
    
    # Create a scene
    scene = gs.Scene(
        show_viewer=False,
        renderer=gs.renderers.Rasterizer(),
    )
    
    # Add a floor (underscore indicates this is intentionally unused)
    _ = scene.add_entity(
        gs.morphs.Plane(size=(2.0, 2.0)),
        material=gs.materials.Rigid(),
        surface=gs.surfaces.Plastic(
            color=(0.9, 0.9, 0.9, 1.0),
        ),
    )
    
    # Add a box that can interact with the button
    box = scene.add_entity(
        gs.morphs.Box(size=(0.2, 0.2, 0.2), pos=(0, 0, 0.5)),
        material=gs.materials.Rigid(),
        surface=gs.surfaces.Plastic(
            color=(0.2, 0.6, 0.8, 1.0),
        ),
    )
    
    # Add a button (small cylinder)
    # Store the button reference but don't use it (prefix with _)
    _button = scene.add_entity(
        gs.morphs.Cylinder(radius=0.1, height=0.05, pos=(0.5, 0, 0.025)),
        material=gs.materials.Rigid(static=True),  # Make it static
        surface=gs.surfaces.Plastic(
            color=(1.0, 0.0, 0.0, 1.0),  # Red
        ),
    )
    
    # Add a camera
    camera = scene.add_camera(
        res=(1280, 720),
        pos=(2.0, -2.0, 1.5),
        lookat=(0.25, 0, 0.5),
        fov=45,
        gui=True  # Enable GUI for interactive viewing
    )
    
    # Build the scene
    scene.build()
    
    # Apply initial velocity to the box (using a small impulse)
    box.add_impulse(impulse=(0.5, 0, 0))
    
    # Run the simulation
    print("Starting simulation. Press Ctrl+C to stop.")
    try:
        while True:
            scene.step()
            camera.render()
    except KeyboardInterrupt:
        print("Simulation stopped by user")

if __name__ == "__main__":
    main()
