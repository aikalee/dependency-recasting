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

    arcs = arcs.copy()
    hlookup = dict(arcs)

    lifted_indices = []
    lift_counts = {}   
         
    while is_non_proj(arcs):
  
        non_proj_arcs = get_non_proj_arcs(arcs, symmetric_counting=symmetric_counting, dlookup=dlookup)

        for d, h in non_proj_arcs:
            i = int(d)-1
            lift_counts.setdefault(i, 0)

    
        min_value = min(lift_counts.values())

        # find the least lifted arcs
        while True:
            candidate_arcs = [
                arc 
                for k, v in lift_counts.items()
                if v == min_value 
                and (arc := arcs[k]) in non_proj_arcs 
                and hlookup[arc[1]] != 0
            ]
            if candidate_arcs:
                break
            
            min_value += 1
            
        cross_counts = {k: non_proj_arcs[k] for k in candidate_arcs} # cross counts of candidate arcs
       

        # If there are more than one cross_counts, select the leftmost arc among the most crossed arcs
        # Otherwise, select the leftmost arc

        if len(set(cross_counts)) > 1:
            # most_crossed_arcs = max(cross_counts, key=non_proj_arcs.get) 
            max_cross_count = max(cross_counts.values())
            most_crossed_arcs = [arc for arc, cross_count in cross_counts.items() if cross_count == max_cross_count]

            if len(most_crossed_arcs) > 1 :
                selected = min(most_crossed_arcs, key=lambda t: min(t[0], t[1])) 
            else: 
                selected = most_crossed_arcs[0]

        else:
            selected = min(cross_counts, key=lambda t: min(t[0], t[1]))

        d, h = selected
        idx = d-1
        
        head_of_h = hlookup[h]
        projectivized = (d, head_of_h)
        arcs[idx] = projectivized

        lifted_indices.append(idx)
        lift_counts[idx] += 1
    
    # Turn indices into arcs
    lifted_arcs = []
    
    for idx in lifted_indices:

        lifted_arc = arcs[idx]
        lifted_arcs.append(lifted_arc)


    if return_all:
        results = [arcs, lifted_arcs]
    else:
        results = [lifted_arcs]
    
    if count_steps:
        results += [sum(lift_counts.values())]

    if len(results) > 1:
        results = tuple(results)
    else:
        results = results[0]
    
    return results

def relabel(orig_deprels, projz_arcs, head: bool = False, path: bool = True) -> dict:

    if not (head or path):
        raise ValueError("At least head or path must be True.")
    
    hlookup = dict(orig_deprels.keys())
    projz_deprels = {}


    for d, goal_h in projz_arcs:

        h = hlookup[d]
        orig_deprel = orig_deprels[(d, h)]

       
        head_of_h = hlookup[h]
        parent_deprel = orig_deprels[(h, head_of_h)]
        
        projz_deprel = orig_deprel + "↑"

        if head:
            projz_deprel += parent_deprel 

        projz_deprels[(d, goal_h)] = projz_deprel

        if path:
            new_parent_deprel = parent_deprel + "↓"
            projz_deprels[(h, head_of_h)] = new_parent_deprel
       

    return projz_deprels