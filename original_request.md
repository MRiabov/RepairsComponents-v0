Okay, I want to make a bit of a pivot in my project. The current project deals with specifically car repairs. Fact is though, car repairs are very difficult and have too many parts for me to handle.

I want to create another RepairsComponents library which would serve as a reusable library for insertion of various repair components which we can train RL policies on. For now, it's only fasteners, and sockets. Fasteners (screws) should be able to be fastened and released (by turning), and would be more or less physically precise. Sockets can be plugged and unplugged, but some sockets will be more challenging, because some can only be unplugged when a small release hatch is pressed on them.

To stay faithful to the Repairs-v0, we are modelling everything with Mujoco XML description; or mujoco's modelling lib, either way. Start with generating docs for the project (in RepairsComponents-v0 repo)
