# Plan of work for two days:
1. Translation of electronics and fasteners is not implemented. (WIP)
  - Persistence of mjcf singletons is not implemented on electronics; not modular on fasteners too. (DONE)
2. Electronics collision has distance checks implemented (untested), but it does not have actual "connect" logic implemented. `check_connections` outputs (presumably) correct labels of connectors (connector_defs?), but then we need to `connect` the items manually (DONE, untested.)
3. Collision detection of electronics during simulation is untested. does it work? (WIP)
4. Collision detection of fasteners during simulation is untested. does it work? 
5. Diff code may be unupdated. (DONE, untested.)  
- (next task) Additionally, they need hints (untested.)
6. Electronics graph collection can not be disabled although it should be. 
7. Only one connector type implemented.
8. Buttons, LEDs, and switches are not implemented.
9. Fasteners may not constrain two parts properly. (this is blocked by Genesis bug.)
10. (related) fastener insertion hint (next task)
11. Europlug connector_pos_relative_to_center_male and female are not implemented. (DONE, tested.)
12. connector_pos_relative_to_center_male and female are not used properly. Links are most likely gotten by the center of the part (although I think I fixed it?). Possibly, may not work at all (DONE, untested).
13. Fasteners and electronics rely on MJCF and MJCF does not support links and glb mesh imports. WON'T DO: better to use MJCF and just export to obj. And calculate fixed frames manually.
- MJCF caused unexpected problems/bugs. It isn't necessary either. Deprecate mjcf and use meshes instead.
~~14. Simulating my small objects on meter scale is not stable (as indicated by Genesis). Need to change settings of Genesis to milimeter including exports.~~ (Reverted)
15. (related but global) get_pos in translation returned NaN (Done, tested.)
16. Somehow fixed bodies are getting moved... OR desired state is not updated during reset, which is less likely. 
17. I'm blocked by Genesis CUDA bug. So I need to make a reproducible example, and then get back to fixing fasteners.
18. does picking up screwdriver actually work? (DONE, tested)
- debug render the buffer prefill steps, will tell. (DONE, yes, works)
- Screwdriver must be repositioned to hand position when close enough.

19. Fastener functionality must be done:
- Fasteners are currently not picked up. (WIP/done, untested)
- Fasteners are are incorrectly inserted. (logic is unfinished)
- Fasteners are not released. (WIP/done, untested.)
  - Bug found: when screwdriver was released, fastener it held was not released. (done)
- Parts are not constrained. (DONE, untested).
  - I suspect it's impossible to constrain two parts at the same time with collision detection logic. Screw in is happening at one position, the fastener is later snapped into place, released from the screwdriver, and then how is the second part constrained?
  A solution would be to not remove the fastener from the screwdriver, and remove it only when the ML decides to release it. The realistic solution pattern would be then move to one hole -> screw in (snap) -> move to another hole -> screw in too. However the attachment should not happen twice. (untested - will work second part?)

  - When fasteners are constrained, the are constrained how they currently are and not in the hole that they should be. (rel 21) (WIP: activate_fastener_to_hand_connection- done, *activate_part_to_fastener_connection not*.) 
  - Fasteners constrain parts but do not align them to the hole.
    - how to avoid extreme, instant snaps?
- Fastener quat/pos is not updated in tranlsation from genesis. (DONE, untested?)
- (duplicate) build123d joints are, as I understand not used to create relative hole positions.(done, untested at all)
- During insertion fasteners should be at least roughly close to a hole - set max angle difference to 30deg.

20. Hole positions and quats are not translated from initial state in perturb. (necessary for fastener collision detection&alignment)
21. Hole positions are not recalculated to allow for collision detection and alignment. (WIP)
- bug found: positions are calculated as global positions, not local to part.
~~22. (ML) Fastener pickup/release is not done.~~(duplicate 19a)
~~23. Screwdriver grip positions (tool grip and fastener grip) is not updated~~ (done, transiently.)
- Would be convenient to store grip positions in Screwdriver class, although current approach is fine.
22. picked_up_fastener_tip_position are irrelevant if a tool or fastener is dropped, however it is still passed to step_screw_in_or_out
18. (minor?) - there is a major snap of fasteners/parts that hold them when fasteners are inserted.




# Hearbeat 21.07.25: Publish the "Teaching Robots to Repair" paper.

UNDONE:
1,3,4,6,7,9,16,(17)

### Next:
Run and debug.

(after):
(18) - test whether a screwdriver works.


It turns out the fasteners functionality was commented out, and all fastener functionality is actually untested. Test and fix it.
%% Note: if everything BUT step_repairs functionality worked, it is probably there.
%%wait, but if I tested today and hit it (once?) was it the cause?



## More detailed:
- Persistence of mjcf singletons is not implemented on electronics; not modular on fasteners too.


### Other bugs: 
1. Perturb does not take into account fixed parts.
2. Perturb does not take into account linked parts. 
3. Translate does not take into account linked parts.
4. It seems perturb tries to move parts by their bounding box, not by their position. Center!=position. (fixed)
5. Does perturb move out of bounds? (probably fixed)
6. Perturb initial state sometimes puts two parts in overlapping position.
7. BUG: Perturb definitely currently aligns all parts in one direction.
8. Optim: voxel export does not take into account shared parts. i.e. europlug_2_male and europlug_3_male are exported to voxel and stl many times even though they are equal parts.
9. (ML) for whichever reason, feature to fastener encoder fasteners are passed as 8 and not 9. Buffers?
10. (ML) Quat action is not normalized. Squares of quat values should sum to 1. (0.25^2+0.25^2+0.25^2+0.25^2=1) (not 0.25 but in that range.)
11. actions are sampled twice in train loop
? - motion planning actions - relative to current pos or absolute? I think the absolute is the standard.
12. (ML) Hole positions are not encoded.
13. (ML) Fastener pickup/release has no action index associated to it.

? 14. (ML) Fastener pickup/release is not encoded. (hmm... but for real?)
15. ! are parts not rotated in genesis scene at creation? 
- Part quats are currently stored incorrectly in persist meshes (the recentering does not adjust quats!)
16. (?) Fasteners are not encoded at translation? I have 0,0,0 for fastener position. (DONE, untested)
17. There may be a clash between tools when both are within pickup distance.



and yet I need to fix the perturb instead of electronics and fasteners...

after:
1. Machine learning isn't polished up. It may not work at all.
2. Electronics graphs should be easily disabled in encoding.
3. (obviously) expand all graph nets to encode positional hints (perhaps even vision and voxels?) <- perhaps encode desired change into vision? It could be a little too troublesome though. But would work?
%note: will my paper contribute anything meaningful, considering that Nvidia does robotic assembly too? I may need to add electronics testing to make this more interesting.


#### (minor) notes:
%%Note to self: when implementing something significant (e.g. a paper) Try to always have a list of at least 5-7 tasks that need to be done.
