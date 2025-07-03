def get_robot_cfg(robot_cfg: str = "franka"):
    match robot_cfg:
        case "franka":
            return {
                "pos": (0.3, -(0.64 / 2 + 0.2 / 2), 0),
                "file": "xml/franka_emika_panda/panda.xml",
                "plane_height": -0.2,
                "tooling_stand_height": -0.1,
            }
        case "humanoid":
            raise NotImplementedError("Humanoid not implemented yet.")

    # TODO if any interest in the project, add more robots. Like K-scale robot (an open-source humanoid.)
