# Plan of work for two days:

1. Translation of electronics and fasteners is not implemented. (DONE)
  - Persistence of mjcf singletons is not implemented on electronics; not modular on fasteners too. (DONE)
2. Electronics collision has distance checks implemented (untested), but it does not have actual "connect" logic implemented. `check_connections` outputs (presumably) correct labels of connectors (terminal_defs?), but then we need to `connect` the items manually (DONE, untested.)
3. Collision detection of electronics during simulation is untested. does it work? (WIP)
4. Collision detection of fasteners during simulation is untested. does it work? 
5. Diff code may be unupdated. (DONE, untested.)  
- (next task) Additionally, they need hints (untested.)
6. Electronics graph collection can not be disabled although it should be. 
7. Only one connector type implemented.
8. Buttons, LEDs, and switches are not implemented.
9. Fasteners may not constrain two parts properly. (untested)
10. (related) fastener insertion hint (next task)
11. Europlug terminal_pos_relative_to_center_male and female are not implemented. (DONE, tested.)
12. terminal_pos_relative_to_center_male and female are not used properly. Links are most likely gotten by the center of the part (although I think I fixed it?). Possibly, may not work at all (DONE, untested).
13. ~~Fasteners and electronics rely on MJCF and MJCF does not support links and glb mesh imports. WON'T DO: better to use MJCF and just export to obj. And calculate fixed frames manually.~~
- MJCF caused unexpected problems/bugs. It isn't necessary either. Deprecate mjcf and use meshes instead. (DONE)
~~14. Simulating my small objects on meter scale is not stable (as indicated by Genesis). Need to change settings of Genesis to milimeter including exports.~~ (Reverted)
15. (related but global) get_pos in translation returned NaN (Done, tested.)
16. Somehow fixed bodies are getting moved... OR desired state is not updated during reset, which is less likely. 
17. I'm blocked by Genesis CUDA bug. So I need to make a reproducible example, and then get back to fixing fasteners. DONE
18. does picking up screwdriver actually work? (DONE, tested)
- debug render the buffer prefill steps, will tell. (DONE, yes, works)
- bug: Screwdriver must be repositioned to hand position when picked up.
  - screwdriver is not repositioned.
    - **!** it is possible that the screwdriver is not repositioned because the constraint does not use the pos from set_pos.

19. Fastener functionality:
- Fasteners are currently not picked up. (WIP/done, tested)
- Fasteners are are incorrectly inserted. (logic is unfinished)
- Fasteners are not released. (WIP/done, tested.)
  - Bug found: when screwdriver was released, fastener it held was not released. (done)
- Parts are not constrained. (DONE, untested).
  - I suspect it's impossible to constrain two parts at the same time with collision detection logic. Screw in is happening at one position, the fastener is later snapped into place, released from the screwdriver, and then how is the second part constrained?
  A solution would be to not remove the fastener from the screwdriver, and remove it only when the ML decides to release it. The realistic solution pattern would be then move to one hole -> screw in (snap) -> move to another hole -> screw in too. However the attachment should not happen twice. (untested - will work second part?)

  - When fasteners are constrained, they are constrained how they currently are and not in the hole that they should be. (rel 21) (WIP: activate_fastener_to_hand_connection- done, *activate_part_to_fastener_connection not*.) (DONE)
  - Fasteners constrain parts but do not align them to the hole. (DONE)
    - how to avoid extreme, instant snaps?
- Fastener quat/pos is not updated in tranlsation from genesis. (DONE, untested?)
- (duplicate) build123d joints are, as I understand not used to create relative hole positions.(done, untested at all)
- During insertion fasteners should be at least roughly close to a hole - set max angle difference to 30deg. (DONE)
20. Hole positions and quats are not translated from initial state in perturb. (necessary for fastener collision detection&alignment) (DONE)
21. Hole positions are not recalculated to allow for collision detection and alignment. (DONE)
- bug found: positions are calculated as global positions, not local to part.
~~22. (ML) Fastener pickup/release is not done.~~(duplicate 19a)
~~23. Screwdriver grip positions (tool grip and fastener grip) is not updated~~ (done, transiently.)
- Would be convenient to store grip positions in Screwdriver class, although current approach is fine.
24. (minor?) - there is a major snap of fasteners/parts that hold them when fasteners are inserted.

## tests:
### tool_genesis.py
1. test_attach_tool_to_arm (DONE, untested)
2. test_detach_tool_from_arm (DONE, untested)
3. test_attach_and_detach_tool_to_arm_with_fastener (DONE, untested)
### fasteners.py
1. attach_fastener_to_screwdriver (WIP)
2. deactivate_fastener_to_screwdriver_connection
3. attach_fastener_to_part
  - test that they could attach two parts.
4. detach_fastener_from_part
### connectors
5. check_connections
### repairs_sim_step (maybe)
(all suspicious methods. Not all because too much time to test them.)





# Hearbeat 21.07.25: Publish the "Teaching Robots to Repair" paper.

UNDONE:
3,4,6,7,9,16,18b
untested:
3,4,9

### Next:
Test and debug fasteners attach/detach.
<!-- note: need to finish testing tool pickup when batched motion planning is fixed in Genesis.-->

<!-- Debug screwdriver not being repositioned to hand position when picked up. (18b) -->
<!-- >
Make a test suite using Genesis: create a single scene and run all `repairs_sim_step` on it with assertions. -->

<!-- Fix CUDA index error; do a test for it. -->

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
9. (ML) for whichever reason, feature to fastener encoder fasteners are passed as 8 and not 9. Buffers? (DONE)
10. (ML) Quat action is not normalized. Squares of quat values should sum to 1. (0.25^2+0.25^2+0.25^2+0.25^2=1) (not 0.25 but in that range.) (DONE)
11. actions are sampled twice in train loop
? - motion planning actions - relative to current pos or absolute? I think the absolute is the standard.
12. (ML) Hole positions are not encoded.
13. (ML) Fastener pickup/release has no action index associated to it.

? 14. (ML) Fastener pickup/release is not encoded. (hmm... but for real?)
15. ! are parts not rotated in genesis scene at creation? 
- Part quats are currently stored incorrectly in persist meshes (the recentering does not adjust quats!)
16. (?) Fasteners are not encoded at translation? I have 0,0,0 for fastener position. (DONE, untested)
17. There may be a clash between tools when both are within pickup distance.
- at the moment, picking up only one tool is supported.
18. (minor) Twisting movement is not prevented/penalized when already inserted on max depth.
19. (maybe; major?) instead of `set_pos`, why wouldn't I make IK to this position first? For 10-15, at least we'll get closer. 
- That would require making extra `scene.step()` calls.
20. (significant optim) Genesis is much more optimal when stepped through in large batches e.g. 5-10k. During training, set up a batch of 5-10k and step them at once, then split by smaller ML batches and make updates on them in those smaller batches.
21. Fasteners are 

## other tests:
1. Assert that after perturb fasteners are oriented(!) as they are expected.

and yet I need to fix the perturb instead of electronics and fasteners...

after:
1. Machine learning isn't polished up. It may not work at all.
2. Electronics graphs should be easily disabled in encoding.
3. (obviously) expand all graph nets to encode positional hints (perhaps even vision and voxels?) <- perhaps encode desired change into vision? It could be a little too troublesome though. But would work?
%note: will my paper contribute anything meaningful, considering that Nvidia does robotic assembly too? I may need to add electronics testing to make this more interesting.





## Electronics repair task list (if will do)
1. Components are attachable and detachable
2. More component types are added
- Motors
- LEDs
- Buttons (?)
- Resistors
- Switches
%%^done?
3. Component visualization
- Motors visualization (Spin)
- LEDs visualization (Light)
- Buttons visualization (Press)
- Switches visualization (flip)
- CAD for all the above
4. Components electronics sim
5. Multimeter tool
- Simulation of all electronics above (measure voltage, current, resistance)
- Being able to add current and/or voltage to a component.
- Perturb of components connection/disconnection; working/not working.
6. (ML) encoding of the electronics types (done?)
- Encoding of wires (not done)
7. `EnvSetup`s for electronics repair
8. More robust collision detection for electronics.


### ideas:
Interesting, if motion planning and perfect positional informa is fully available, it may be reasonable to teach a model that would simply predict the order of items in which the assembly needs to be assembled, instead of raw actions.
E.g. in building of a chair, predict that it would be reasonable to move element a to element b in its position.
Or, better yet, teach a model that would pick two items that need to be joined and join them via motion planning. The rest is, of course, on the programmer.



#### (minor) notes:
%%Note to self: when implementing something significant (e.g. a paper) Try to always have a list of at least 5-7 tasks that need to be done.
