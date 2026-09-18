from collections import defaultdict, deque
from functools import reduce
from itertools import product
from operator import mul

from src.data.common.preprocessing.projectivize import get_non_proj_arcs

def get_parent_label(deprel):
    return deprel.split("↑")

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


def get_all_descendants(node, dlookup):
    descendants = set()
    stack = list(dlookup.get(node, []))

    while stack:
        current = stack.pop()

        if current in descendants:
            continue

        descendants.add(current)
        stack.extend(dlookup.get(current, []))

    return descendants


def remove_arrows_in_deprels(tokens_with_arrows, deprels):
        replaced_deprels = {}
        for d, h in tokens_with_arrows:
            replaced_deprels[(d, h)] = deprels[(d, h)].replace("↓", "").replace("↑", "")
        return replaced_deprels  


def search_until_match_by_head(head, dlookup, deprels, target_label, forbidden_nodes=None, possible_parents=None):
    if possible_parents is None:
        possible_parents = []

    if forbidden_nodes is None:
        forbidden_nodes = set()
    
    children_of_current_parent = dlookup.get(head, [])
    if len(children_of_current_parent) == 0:
        return possible_parents
    
    for child in children_of_current_parent:
        if child in forbidden_nodes:
            continue
        prt_label = deprels[(child, head)]
        if prt_label == target_label:
            possible_parents.append(child)
        search_until_match_by_head(child, dlookup, deprels, target_label, possible_parents)
    return possible_parents 

def deprojectivize_by_head(sentencedata):

    

        
            
    deprels = sentencedata.deprels
    dlookup = sentencedata.dlookup
    stack = sentencedata.stack
    deprojz_arcs = {}
   

    while stack:
        possible_parents = []
        d, h = stack.popleft()
        child_deprel, orig_prt_label = get_parent_label(deprels[(d, h)])

        # Original parent is the child of current parent
        # If lifted more than once, there will be a mismatch
        forbidden_nodes = get_all_descendants(d, dlookup)
        possible_parents = search_until_match(h, dlookup=dlookup, deprels=deprels, target_label=orig_prt_label, forbidden_nodes=forbidden_nodes)

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


def search_until_match_by_head_path(head, path_candidate_lookup, dlookup, deprels, orig_head, target_label):

    for prt in path_candidate_lookup.get(head, []):
        prt_label = deprels[(prt, head)].replace("↓", "")
        if prt_label == target_label:
            return prt
        
        result = search_until_match_by_head_path(prt, path_candidate_lookup, dlookup, deprels, orig_head, target_label)

        if result is not None:
            return result
        
    if head == orig_head:
        return orig_head
    
    return None

       

def deprojectivize_by_head_path(sentencedata):

    deprels = sentencedata.deprels
    path_candidate_lookup = sentencedata.path_candidate_lookup
    tokens_with_arrows = sentencedata.tokens_with_arrows
    dlookup = sentencedata.dlookup
    stack = sentencedata.stack
    deprojz_arcs = {}
  

    while stack:
        # possible_parents = []
        d, h = stack.popleft()
    
        child_label, orig_parent_label = get_parent_label(deprels[(d, h)])
        prt = search_until_match_by_head_path(h, path_candidate_lookup, dlookup, deprels, h, orig_parent_label)
        deprojz_arcs[(d, prt)] = child_label

    deprojz_dependents = [d for d, _ in deprojz_arcs]
    remaining_tokens_with_arrows = [(d, h) for d, h in tokens_with_arrows if d not in deprojz_dependents]
        
    removed_arrows = remove_arrows_in_deprels(remaining_tokens_with_arrows, deprels)
    updated_deprels = deprojz_arcs | removed_arrows
       
    return updated_deprels, deprojz_arcs

def find_closest(head, path_candidate_lookup):
    candidates = path_candidate_lookup.get(head, [])
    if candidates:
        return min(candidates)
    
def deprojectivize_by_path(sentencedata):
    deprels = sentencedata.deprels
    path_candidate_lookup = sentencedata.path_candidate_lookup
    tokens_with_arrows = sentencedata.tokens_with_arrows
    stack = sentencedata.stack
    deprojz_arcs = {}
    

    while stack:
        d, h = stack.popleft()
        found_original_head = find_closest(h, path_candidate_lookup)
        # deprojectivizing the lifted children (they can have downward arrows!!!)
        if found_original_head:
            new_deprel = deprels[(d, h)].replace("↑", "")
            deprojz_arcs[(d, found_original_head)] = new_deprel
           
    # removing leftover arrows
    deprojz_dependents = [d for d, _ in deprojz_arcs]
    remaining_tokens_with_arrows = [(d, h) for d, h in tokens_with_arrows if d not in deprojz_dependents]
    
    removed_arrows = remove_arrows_in_deprels(remaining_tokens_with_arrows, deprels)
    updated_deprels = deprojz_arcs | removed_arrows

    return updated_deprels, deprojz_arcs
