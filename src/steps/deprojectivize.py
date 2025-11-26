from collections import defaultdict, deque 
from functools import reduce
from itertools import product
from operator import mul

from src.steps.projectivize import get_non_proj_arcs

def is_projz(deprels):

    for name in deprels.values():
        
        if ("↑" or "↓") in name:
            return True
        
    return False

def is_valid_tree(arcs, num_tokens):

    # Return False if multiple heads for a dependent 
    parents = {}
    for d, h in arcs:
        if d in parents:
            return False 
        parents[d] = h

    # Return false if there are more than one root
    root_count = sum(1 for d, h in arcs if h == 0)
    if root_count != 1:
        return False  

    # All tokens must be reachable
    children = defaultdict(list)
    for d, h in arcs:
        children[h].append(d)

    visited = set()
    queue = deque([0])  # Start from root

    while queue:

        current = queue.popleft()
        visited.add(current)
        
        for child in children[current]:
            if child not in visited:
                queue.append(child)

    return len(visited) == num_tokens 

def deprojectivize_by_head(sentencedata, head=False, path=True):
    """
    Simply experimental
    """

    MAX_COMBINATIONS = 100
   
    memory = {}
    seen = set()

    # all_candidate_deprels = []
    all_candidate_deprels = {}
    max_crossings = 0
    max_combo = {}
    
    deprojz_deprels = {}

    
    arcs = sentencedata.arcs
    num_tokens = sentencedata.num_tokens
    deprels = sentencedata.deprels
    dlookup = sentencedata.dlookup
    head_candidate_lookup = sentencedata.head_candidate_lookup

    stack = sentencedata.stack
    parent_stack = sentencedata.parent_stack
    print(stack)
    print(parent_stack)
    
    def update(d, h, orig_h, orig_deprel):

        nonlocal arcs, head_candidate_lookup, deprojz_deprels, dlookup, stack

        arcs += [(d, orig_h)]
        head_candidate_lookup[(d, orig_deprel)] += [orig_h]
        dlookup[orig_h] += [d]
        deprojz_deprels[(d, orig_h)] = orig_deprel
       
        
        stack.popleft()
    
    def find_candidates():
        """
        First, find candidates among direct children.
        If no candidates found among direct children, search among the descendents.
        Return the candidates of the dependent's head.
        Returns: 
            A list of integers.
        """
        nonlocal head, path
        # head = True
        # path = False
        
    
        candidates = head_candidate_lookup[(h, parent_deprel)]

        curr = [h]  
        
        if not candidates:
            candidates = []
            while curr:
                prev = curr
                curr = []
                
                for p in prev:
                    children = dlookup[p]
                    curr += children
                    
                    for child in children:
                        candidates += head_candidate_lookup[(child, parent_deprel)]
        
        return candidates
    
        
        

    def check_word_order():

        """
        If the dependency relation is "flat" or "conj", check if the candidate prescede the child.
        If not, remove the candidate from candidate list.
        """
        nonlocal candidates

        if orig_deprel in ["flat", "conj"]:
           
            for cand in candidates:
                if cand > d:
                    candidates.remove(cand)
    
    def find_most_crossed_combination():
        """
        For the dependents left with more than one candidates of their original head,
        try all the combinations to find the most crossed combinations of the possible arcs.
        """

        nonlocal deprojz_deprels, max_combo, max_crossings, stack

        if len(all_candidate_deprels) == len(stack):

            # predict the number of combinations in all_combo
            total_combinations = reduce(mul, [len(lst) for lst in all_candidate_deprels.values()])
            print(all_candidate_deprels, stack)
            if total_combinations < MAX_COMBINATIONS:
               
                all_combo = list(product(*all_candidate_deprels.values()))
                
                for combo in all_combo:
                    
                    new_arcs = arcs + list(combo)

                    # Ensure the combination forms a valid tree
                    if not is_valid_tree(new_arcs, num_tokens):
                        continue 

                    arc_crossings = get_non_proj_arcs(new_arcs)
                    total_crossings = sum(arc_crossings.values())
                    
                    if total_crossings > max_crossings:
                        print(max_crossings)
                        max_crossings = total_crossings
                        max_combo = combo
            
            if not max_combo:
                max_combo = [lst[0] for lst in all_candidate_deprels.values()]
            
            for d1, h1 in list(max_combo):

                orig_deprel = memory[d1] 
                deprojz_deprels[(d1, h1)] = orig_deprel
        
            stack = deque()
            print(stack)


    while stack:
    # for i in range(60):
        print(stack)
        print(parent_stack)    
        d, h = stack[0]
        orig_deprel, parent_deprel = deprels[(d, h)].split("↑")
       


        candidates = find_candidates()
        check_word_order()
                      
        candidate_deprels = list(product([d], candidates))

        if len(candidates) == 1:
            orig_h = candidates[0]
            update(d, h, orig_h, orig_deprel)
            # continue

        elif len(candidates) == 0:

            # === Final strategy ===
            # assign the main verb to the dependent 
            # if the dependent lost connection to its head
            # (i.e., when the stack does not change)
            state = tuple(stack) 

            if state not in seen:
                seen.add(state) 
                stack.rotate(-1)
            else:
                if len(head_candidate_lookup[(0, "root")]) == 1:
                    main_verb = head_candidate_lookup[(0, "root")][0]  
                    update(d, h, main_verb, orig_deprel)
                    # continue
                else:
                    raise ValueError("One root only in each sentence.")

            # continue

        
        elif len(candidates) > 1:

            # add candidate_deprels to all_candidate_deprels
            # after all the dependents are handled
            if d not in memory: 
                memory[d] = orig_deprel
            else:
                all_candidate_deprels[d] = candidate_deprels
                
            stack.rotate(-1)

        # the stack might be empty after running the above lines
        if stack:
            find_most_crossed_combination()
    
    return deprojz_deprels


def deprojectivize_by_path(sentencedata, closeness=True):
    arcs = sentencedata.arcs
    hlookup = dict(arcs)
    deprels = sentencedata.deprels
    path_candidate_lookup = sentencedata.path_candidate_lookup
    stack = sentencedata.stack
    
    deprojz_deprels = {}
    global_path_tracking = []

    def _find_deepest(head):
        """Recursively follow ↓-marked arcs downward from head."""
        local_children = []
        
        for candidate in path_candidate_lookup.get(head, []):
            local_children.append(candidate)
            deeper = _find_deepest(candidate)
            if deeper:
                local_children.extend(deeper)
        return local_children
    
    def _find_closest(head):
        candidates = path_candidate_lookup.get(head, [])
        if candidates:
            return min(candidates)
    
    def _remove_arrows(deprojz_deprels):
        for parent in path_candidate_lookup.values():
            for p in parent:
                head_of_parent = hlookup[p]
                new_deprels = deprels[(p, head_of_parent)].replace("↓", "")
                deprojz_deprels[(p, head_of_parent)] = new_deprels
        return deprojz_deprels
 
    if closeness:
        while stack:
            d, h = stack.popleft()
            original_head = _find_closest(h)
            if original_head:
                deprojz_deprels[(d, original_head)] = deprels[(d, h)].replace("↑", "")
            else:
                stack.rotate(-1)
        deprojz_deprels = _remove_arrows(deprojz_deprels)

    else:
        while stack:
            d, h = stack.popleft()
            paths = _find_deepest(h)
            if paths:
                global_path_tracking.append((d, paths))
                original_head = paths[-1]
        all_seen_parent = set([item for _, sublist in global_path_tracking for item in sublist])
        if not all_seen_parent.issubset(path_candidate_lookup):
            deprojz_deprels = _remove_arrows(deprojz_deprels)

    # print(deprojz_deprels)
    return deprojz_deprels


                   