import numpy as np
from build123d import *
from ocp_vscode import show

with BuildPart() as build_part:
    Box(10, 10, 10)

with BuildPart() as ignored_intersecting_part:
    Box(10, 10, 10)

with BuildPart() as build_part2:
    Box(10, 10, 10)

build_part.part.label = "in_compound@solid"
ignored_intersecting_part.part.label = "intersects_with_other"
build_part2.part.label = "intersects"

ignored_intersecting_part.part = ignored_intersecting_part.part.located(Pos(20, 20, 20))
build_part2.part = build_part2.part.located(Pos(20, 20, 20))

compound_with_child = Compound(
    children=[build_part.part, ignored_intersecting_part.part],
    label="child_compound",
)

intersecting_compound = Compound(
    children=[compound_with_child, build_part2.part], label="top_compound"
)

intersection, intersecting_parts, volume = intersecting_compound.do_children_intersect()
print(intersecting_compound.show_topology())
print([p.label for p in intersecting_parts])
assert intersection
assert set(component.label for component in intersecting_parts) == {
    "intersects_with_other",
    "intersects",
}
assert np.isclose(volume, 10e3, atol=1e-3)

show(intersecting_compound)
