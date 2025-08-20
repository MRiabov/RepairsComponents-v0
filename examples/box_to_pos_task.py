from build123d import *

from repairs_components.training_utils.env_setup import EnvSetup


class MoveBoxSetup(EnvSetup):
    "Simplest env, only for basic debug."

    def desired_state_geom(self) -> Compound:
        with BuildPart() as box:
            with Locations((10, 0, 10)):
                Box(25, 25, 25)

        box.part.label = "box@solid"

        compound = Compound(children=[box.part]).move(Pos(self.BOTTOM_CENTER_OF_PLATE))

        return compound


# the better workflow would be to create desired state, then create many environments for different tasks from the desired state. wouldn't it?
# it would lessen the need for recompilation, too.
