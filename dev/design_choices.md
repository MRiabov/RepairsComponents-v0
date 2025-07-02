Technically offline dataloading should be the first-class citizen. Online dataloading is much slower.

###

So how do we connect and disconnect fasteners when we don't have unique ids of them?
How do we distinguish between multiple equal fasteners connecting two parts?
If using integer IDs, how do we get the correct ones later?

forgoing strings is good because it allows for less manual code work.
but. how to connect and disconnect and distinguish? how to encode? #<- encoding is done by graph edge attributes.

Fastener insertion protocol:
given a fastener that is picked up,
1. find closest hole
2. check if the hole is close enough
3. check if there is twisting movement
4. insert the fastener - connect the fastener to the body's hole.

Fastener removal protocol:
1. get picked up fastener
2. check if there is twisting movement counterclockwise
3. remove connection from the body's hole



but how to know which fastener is picked up?
have a graph tensor of ALL fasteners and return global feats as only unconnected ones?
how to know which fasteners need connection checks?

**choice**: edge_index does not exist at all before export, because it does nothing and only complicates handling.
**choice**: instead, graph global feats store connected/unconnected fasteners. Where both are !=-1, it's an edge.

%%note: found possible bug: twisting movement is not prevented/penalized when already inserted on max depth.

%%shit, how many bugs are left to fix? 
