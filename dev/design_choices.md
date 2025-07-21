Technically offline dataloading should be the first-class citizen. Online dataloading is much slower.

###

So how do we connect and disconnect fasteners when we don't have unique ids of them?
How do we distinguish between multiple equal fasteners connecting two parts?
If using integer IDs, how do we get the correct ones later?

forgoing strings is good because it allows for less manual code work.
but. how to connect and disconnect and distinguish? how to encode? #<- encoding is done by graph edge attributes.

Fastener insertion protocol:
given a fastener that is picked up, and that there is twisting movement
1. find one closest hole *except not the one it's already connected to* (one per env idx per step) #<-edit
2. check if the hole is close enough 
3. insert the fastener - connect the fastener to the body's hole.
 - insert the fastener: rotate and relocate the fastener into the hole, constrain.

Fastener removal protocol:
given a fastener that is picked up, and that there is twisting movement counterclockwise
1. remove connection from the body's hole / disconnect the fastener from the body's hole (I pick second option.)
2. Pick up fastener from the body's hole
 - deconstrain the fastener from the body, rotate and relocate the fastener into the hand, constrain.

To allow ML to constrain two parts, do not remove the fastener from the screwdriver during screw in, and remove it only when the ML decides to release it. The realistic solution pattern would be then move to one hole -> screw in (snap) -> move to another hole -> screw in too. (However the attachment should not happen twice.)


but how to know which fastener is picked up?
have a graph tensor of ALL fasteners and return global feats as only unconnected ones?
how to know which fasteners need connection checks?

**choice**: edge_index does not exist at all before export, because it does nothing and only complicates handling.
**choice**: instead, graph global feats store connected/unconnected fasteners. Where both are !=-1, it's an edge.


%%shit, how many bugs are left to fix?


### electronics connectors encoding:
1. electronics connectors @connector are encoded as solids in mechanical graph with type "connector".
2. when they are connected with other connectors, add edges to electronics graph.
3. the connector def positions will have to be observed from voxels (although I do understand this is a deficiency. Possibly mitigated by 

%% this would mean that an assembly with only electronics connectors would have an empty electronics graph. ... 
%% which is fair? there is no flow of electronics through those connectors anyway.
%%note: when connecting other components, e.g. buttons, how would I encode their connection defs?


# Encoding hints to the model
Due to recent very strong advances in computer vision and scene reconstruction, I do believe it is fair to pass current mechanical graph to the model with "hints" in which direction and distance the components should be moved to finish assembly. This is to speed up learning. It is a strong signal that would allow the model to see what needs to be done to finish assembly.
#### electronics connectors encoding as hint
The model would get electronics XYZ direction as hint relative to it's **connector def** position
## initial and desired positions 
Consequently, the desired position would have the hint as (0,0,0) because it's already at the desired position and initial position would show exactly how much we need to move. 


%% so the way connector def is used is XYZ and quat hint... but why bother?
%% but definitely for the *"electronics connectors encoding"* - connectors are encoded as solids.

%% NOTE: put this into tasks. ("add hint encoding to mechanical graph.")

## Modelling:
Since their export from desired_state_geom, all models should be centered at the origin. It makes downstream work easier.

# ML
ML is done via inverse kinematics and can move in only -0.5< x <0.5, -0.5< y <0.5, 0< z <0.8 ranges.

%% genesis note: quats are in WXYZ convention.

%% note to self: RL is highly sample inefficient. It's stupid to use plain RL when VLAs are available. this is why, when looking for jobs, most people are asking for LLMs or VLMs and not RL.