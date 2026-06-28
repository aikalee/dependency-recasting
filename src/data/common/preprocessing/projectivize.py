from copy import deepcopy
from collections import Counter, defaultdict

def is_non_proj(arcs) -> bool:

    for i, (h1, d1) in enumerate(arcs):
        
        l1, r1 = sorted([h1, d1]) # direction does not matter

        for h2, d2 in arcs[i+1:]: # symmetric along diagonal

            l2, r2 = sorted([h2, d2])

            if (l1 < l2 < r1 < r2) or (l2 < l1 < r2 < r1):
                return True
            
    return False

def get_non_proj_arcs(arcs, symmetric_counting=False, return_intersecting_arcs=False, dlookup=None):

    if symmetric_counting and not dlookup:
        raise ValueError("Provide dlookup to enable symmetric_counting.")
    
    non_proj_arcs = Counter() # cross counts of all non-projective arcs
    hlookup = dict(arcs) # a lookup dictionary mapping dependent to head
    intersecting_arcs = []
    
    # loop through every arc in a sentence
    for d, h in arcs:
        
        if h == 0: # root arc must be projective
            continue
        
        l, r = sorted([h, d])

        for t in range(l+1, r):

            head_of_t = hlookup[t]

            if symmetric_counting:
                dep_of_t = dlookup[t]
                for dt in dep_of_t:
                    if dt < l or dt > r:
                        non_proj_arcs.update([(d, h)])

            if head_of_t < l or head_of_t > r:
                non_proj_arcs.update([(d, h)])
                if return_intersecting_arcs:
                    intersecting_arcs.append(tuple(sorted([(t, head_of_t), (d, h)])))

    if return_intersecting_arcs:
        return non_proj_arcs, intersecting_arcs
               
    return non_proj_arcs

def projectivize(arcs, symmetric_counting=False, dlookup=None, return_all=False, count_steps=False):

    arcs = deepcopy(arcs) # no root token
    hlookup = dict(arcs)
    lifted_tokens = []
    lifted_arcs = []
         
    while is_non_proj(arcs):
        non_proj_arcs = get_non_proj_arcs(arcs, symmetric_counting=symmetric_counting, dlookup=dlookup)
        # print(non_proj_arcs)
        sorted_non_proj_arcs = sorted(list(non_proj_arcs.keys()), key=lambda x: x[0])
        smallest_distance = float('inf')
        smallest_arc = None

        for dep, head in sorted_non_proj_arcs:
            distance = abs(int(head) - int(dep))
            if smallest_distance > distance and hlookup[head] != 0:
                smallest_distance = distance
                smallest_arc = (dep, head)
            # else:


                # print(f"head: {head}, dep: {dep}, distance: {distance}, smallest_distance: {smallest_distance}")

        smallest_dep, smallest_head = smallest_arc
        new_head = hlookup[smallest_head]
        arcs[smallest_dep-1] = (smallest_dep, new_head)
        lifted_tokens.append(smallest_dep)
    
    for token in set(lifted_tokens):
        if token == arcs[token-1][0]:
            lifted_arcs.append(arcs[token-1])
        else:
            raise ValueError("Token ids do not match.")
    
    return lifted_arcs

def relabel(orig_deprels, projz_arcs, head: bool = False, path: bool = True) -> dict:

    if not (head or path):
        raise ValueError("At least head or path must be True.")
    
    hlookup = dict(orig_deprels.keys())
    projz_deprels = {}
    changed_tokens = []

    # print(projz_arcs)

    for d, goal_h in projz_arcs:

        h = hlookup[d]
        orig_deprel = orig_deprels[(d, h)]

       
        head_of_h = hlookup[h]
        if orig_deprels.get((h, head_of_h), None):
            parent_deprel = orig_deprels[(h, head_of_h)]
        else:
            parent_deprel = projz_deprels[(h, head_of_h)]
        
        projz_deprel = orig_deprel + "↑"

        if head:
            projz_deprel += parent_deprel 

        projz_deprels[(d, goal_h)] = projz_deprel
        changed_tokens.append(d) 
        # print(projz_deprels)
        orig_deprels.pop((d, h))
        hlookup[d] = goal_h
        

        head_of_h = hlookup[h]

        if path:
            
            # if d not in changed_tokens:
            new_parent_deprel = parent_deprel + "↓"
            projz_deprels[(h, head_of_h)] = new_parent_deprel
            # else:
            #     new_parent_deprel = projz_deprels.get((d, goal_h)) + "↓"
            #     projz_deprels[(d, goal_h)] = new_parent_deprel
       

    return projz_deprels