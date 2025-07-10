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
11. Europlug connector_pos_relative_to_center_male and female are not implemented.
12. connector_pos_relative_to_center_male and female are not used properly. Links are most likely gotten by the center of the part (although I think I fixed it?). Possibly, may not work at all (untested).
13. Fasteners and electronics rely on MJCF and MJCF does not support links and glb mesh imports. WON'T DO: better to use MJCF and just export to obj. And calculate fixed frames manually.
14. Simulating my small objects on meter scale is not stable (as indicated by Genesis). Need to change settings of Genesis to milimeter including exports. (WIP)


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



and yet I need to fix the perturb instead of electronics and fasteners...

after:
1. Machine learning isn't polished up. It may not work at all.
2. Electronics graphs should be easily disabled in encoding.
3. (obviously) expand all graph nets to encode positional hints (perhaps even vision and voxels?) <- perhaps encode desired change into vision? It could be a little too troublesome though. But would work?


#### (minor) notes:
%%Note to self: when implementing something significant (e.g. a paper) Try to always have a list of at least 5-7 tasks that need to be done.
