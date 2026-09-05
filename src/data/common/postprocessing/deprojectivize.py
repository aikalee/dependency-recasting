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

def get_parent_label(deprel):
    return deprel.split("↑")

def deprojectivize_by_head(sentencedata):

    def search_until_match(head, dlookup, deprels, target_label, possible_parents=None):
        if possible_parents is None:
            possible_parents = []
       
        children_of_current_parent = dlookup.get(head, [])
        if len(children_of_current_parent) == 0:
            return possible_parents
       
        for child in children_of_current_parent:
            prt_label = deprels[(child, head)]
            if prt_label == target_label:
                possible_parents.append(child)
            search_until_match(child, dlookup, deprels, target_label, possible_parents)
        return possible_parents
        
            
    deprels = sentencedata.deprels
    tokens_with_arrows = sentencedata.tokens_with_arrows
    dlookup = sentencedata.dlookup
    stack = sentencedata.stack
    deprojz_arcs = {}
   
   

    while stack:
        possible_parents = []
        d, h = stack.popleft()
        child_deprel, orig_prt_label = get_parent_label(deprels[(d, h)])

        # Original parent is the child of current parent
        # If lifted more than once, there will be a mismatch
        possible_parents = search_until_match(h, dlookup=dlookup, deprels=deprels, target_label=orig_prt_label)

        if len(possible_parents) > 1:
            smallest_dist = float('inf')
            selected_prt = None
            for prt in possible_parents:
                curr_dist = abs(prt-d)
                if curr_dist < smallest_dist:
                    smallest_dist = curr_dist
                    selected_prt = prt
        elif len(possible_parents) == 1:
            selected_prt = possible_parents[0]
        else:
            selected_prt = h

        deprojz_arcs[(d, selected_prt)] = child_deprel
        

    return deprojz_arcs


def search_until_match(head, path_candidate_lookup, dlookup, deprels, target_label):
    for prt in path_candidate_lookup.get(head, []):
        prt_label = deprels[(prt, head)].replace("↓", "")
        if prt_label == target_label:
            return prt
        else:
            children = dlookup[prt]
            for child in children:
                search_until_match(child, path_candidate_lookup, dlookup, deprels, target_label)


def deprojectivize_by_head_path(sentencedata):

    # def _find_match(head, path_candidate_lookup, dlookup):
    #     if path_candidate_lookup.get(head, [])

    deprels = sentencedata.deprels
    path_candidate_lookup = sentencedata.path_candidate_lookup
    tokens_with_arrows = sentencedata
    dlookup = sentencedata.dlookup
    stack = sentencedata.stack
    # possible_parents = []
    deprojz_arcs = {}

    while stack:
        # possible_parents = []
        d, h = stack.popleft()
    
        child_label, orig_parent_label = get_parent_label(deprels[(d, h)])
        
        prt = search_until_match(h, path_candidate_lookup, dlookup, deprels, orig_parent_label)

        
        deprojz_arcs[(d, prt)] = child_label
       
       
    return deprojz_arcs
    
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
        # deprojectivizing the lifted children (they can have downward arrows!!!)
        if found_original_head:
            new_deprel = deprels[(d, h)].replace("↑", "")
            deprojz_arcs[(d, found_original_head)] = new_deprel
           
    # removing leftover arrows
    only_d_in_deprojz_arcs = [d for d, _ in deprojz_arcs]
    remaining_tokens_with_arrows = [(d, h) for d, h in tokens_with_arrows if d not in only_d_in_deprojz_arcs]
    
    removed_arrows = remove_arrows_in_deprels(remaining_tokens_with_arrows)
    updated_deprels = deprojz_arcs | removed_arrows

    return updated_deprels, deprojz_arcs
