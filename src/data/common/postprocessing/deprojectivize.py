from collections import defaultdict, deque
from functools import reduce
from itertools import product
from operator import mul

from src.data.common.preprocessing.projectivize import get_non_proj_arcs

def is_projz(deprels):

    for name in deprels.values():
        
        if "↑" in name or "↓" in name:
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

def deprojectivize_by_path(sentencedata):
    deprels = sentencedata.deprels
    path_candidate_lookup = sentencedata.path_candidate_lookup
    tokens_with_arrows = sentencedata.tokens_with_arrows
    stack = sentencedata.stack
    deprojz_arcs = {}
    
    def _find_closest(head, path_candidate_lookup):
        candidates = path_candidate_lookup.get(head, [])
        if candidates:
            return min(candidates)
    
    def remove_arrows_in_deprels(tokens_with_arrows):
        replaced_deprels = {}
        for d, h in tokens_with_arrows:
            replaced_deprels[(d, h)] = deprels[(d, h)].replace("↓", "").replace("↑", "")
        return replaced_deprels      
 
    while stack:
        d, h = stack.popleft()
        found_original_head = _find_closest(h, path_candidate_lookup)
        # deprojectivizing the lifted children
        if found_original_head:
            deprojz_arcs[(d, found_original_head)] = deprels[(d, h)].replace("↑", "")
    # removing leftover arrows
    only_d_in_deprojz_arcs = [d for d, _ in deprojz_arcs]
    remaining_tokens_with_arrows = [(d, h) for d, h in tokens_with_arrows if d not in only_d_in_deprojz_arcs]
    
    removed_arrows = remove_arrows_in_deprels(remaining_tokens_with_arrows)
    updated_deprels = deprojz_arcs | removed_arrows

    return updated_deprels, deprojz_arcs



                   