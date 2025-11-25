from data_loader import read_conllu_file_from_tgz, read_conllu_file
from copy import deepcopy
from collections import defaultdict


DELIMITER_DEP = "↑" # for lifted arcs
DELIMITER_HEAD = "↓" # for lifted path, indicating there are some lifted arcs underneth in the original sentence

def is_non_projective_sentence(sentence, return_arcs=False):
    """
    Determines if a sentence contains non-projective dependencies. Multiple crosses in one sentence is considered as one count.
    Input: sentence - list of token dictionaries from a CoNLL-U file
    Output: 
        - True if non-projective, False otherwise
        - (optional output) list of non-projective arcs [(head, dep), ] if return_arcs is True
    """
    
    non_projective_arcs = []
    for _, token in sentence.items():
        head = token['head']
        dep = token['id']
        if is_non_projective_arc(sentence, dep):
            if return_arcs:
                non_projective_arcs.append((head, dep))
            else:
                return True

    if return_arcs:
        if len(non_projective_arcs) == 0:
            return False, []
        else:
            return True, non_projective_arcs
    else:
        return False
    
    # if return_arcs is False:
    #     # Check for crossing arcs
    #     for i, (a1, a2) in enumerate(arcs):
    #         for b1, b2 in arcs[i+1:]:
    #             (a1, a2) = sorted([a1, a2])
    #             (b1, b2) = sorted([b1, b2])
    #             if (a1 < b1 < a2 < b2) or (b1 < a1 < b2 < a2):
    #                 return True
    #     return False
    
    # cross_arcs = {}
    # for i, (head1, dep1) in enumerate(arcs):
    #     for head2, dep2 in arcs[i+1:]:
    #         (a1, a2) = sorted([head1, dep1])
    #         (b1, b2) = sorted([head2, dep2])
    #         if (a1 < b1 < a2 < b2) or (b1 < a1 < b2 < a2):
    #             cross_arcs[dep1] = head1
    #             cross_arcs[dep2] = head2
    # if len(cross_arcs) == 0:
    #     return False, []
    
    # non_proj_arcs = []
    # for dep, head in cross_arcs.items():
    #     if is_non_projective_arc(sentence, dep):
    #         non_proj_arcs.append((head, dep))

    # return True, non_proj_arcs
        




def is_non_projective_arc(sentence, token_id, return_cross_count=False):
    """
    Determines if a specific dependency arc is non-projective.
    Input: sentence - list of token dictionaries from a CoNLL-U file
           dep_id - the ID of the dependency to check
    Output: True if non-projective, False otherwise
            (optional output) cross_count - number of crossing arcs, only count incoming arcs
    """
    
    cross_count = 0
    token_head = sentence[token_id]['head']
    if token_head == 0:
        if return_cross_count:
            return False, 0
        else:
            return False
    
    (left, right) = sorted([token_head, token_id])
    
    for dep_id in range(left+1, right):
        if not (left <= sentence[dep_id]['head'] <= right):
            if return_cross_count:
                cross_count += 1
            else:
                return True
    
    if return_cross_count:
        if cross_count > 0:
            return True, cross_count
        else:
            return False, cross_count
    else:
        return False

    
    # arcs = {}
    # for _, token in sentence.items():
    #     head = token['head']
    #     dep = token['id']
    #     arcs[dep] = head

    # token_head = arcs[token_id]
    # if token_head == 0:
    #     return False

    # (left, right) = sorted([token_head, token_id])
    # # print(f"\ntoken_id: {token_id}, token_head: {token_head}")

    # for dep in range(left+1, right):
    #     # print(f"Checking {dep}")
    #     head = dep
    #     while head != token_head:
    #         head = arcs[head]
    #         # print(f"head: {head}")
    #         if head< left or head>right:
    #             return True

    # return False


def count_non_projective_sentences(folder_path, file_path):
    """
    Counts the number of non-projective sentences in a CoNLL-U file within a .tgz archive. Multiple crosses in one sentence is considered as one count.
    Input: folder_path - path to the .tgz archive
           file_path - path to the CoNLL-U file within the archive
    Output: Tuple (number of non-projective sentences, total number of sentences)
    """
    total = 0
    non_proj = 0
    for sentence in read_conllu_file_from_tgz(folder_path, file_path).items():
        total += 1
        if is_non_projective_sentence(sentence):
            non_proj += 1
            # lst = []
            # for token in sentence:
            #     lst.append(token['form'])
            # print(lst)
    return non_proj, total

def example_non_projective_sentences(folder_path, file_path, limit=float('inf'), from_tgz=True):
    """
    Returns a list of non-projective sentences from a CoNLL-U file within a .tgz archive.
    Input: folder_path - path to the .tgz archive
           file_path - path to the CoNLL-U file within the archive
           limit - maximum number of sentences to return (default is no limit)
    Output: List of non-projective sentences (each sentence is a list of token dictionaries)
    """
    lst = []
    limit_count = 0
    if from_tgz:
        read_func = read_conllu_file_from_tgz(folder_path, file_path)
    else:
        read_func = read_conllu_file(file_path)
    for sentence in read_func:
        if is_non_projective_sentence(sentence):
            lst.append(sentence)
            limit_count += 1
            if limit_count >= limit:
                break
    return lst

def sentence_to_latex_tree(sentence, message=""):
    """
    Converts a sentence to a LaTeX tree structure.
    Input: sentence - list of token dictionaries from a CoNLL-U file
    Output: LaTeX tree structure as a string
    """
    words = []
    ids = []
    edges = []
    root_index = None

    for _, token in sentence.items():
        idx = int(token['id'])
        word = token['form'].replace('_', '\_')  # escape underscores for LaTeX
        head = int(token['head'])
        deprel = token.get('deprel', '')

        words.append(word)
        ids.append(str(idx))

        if head == 0:
            root_index = idx
        else:
            edges.append((head, idx, deprel))
    
    # Start LaTeX output
    output = ["\\begin{frame}{", message, "}", "\\resizebox{\\textwidth}{!}", "{", "\\begin{dependency}[edge horizontal padding=10pt]", "\\begin{deptext}[column sep=0.7cm]"]
    output.append(" \\& ".join(words) + " \\\\")
    output.append(" \\& ".join(ids) + " \\\\")
    output.append("\\end{deptext}")

    # Add edges
    for head, dep, label in edges:
        output.append(f"   \\depedge[]{{{head}}}{{{dep}}}{{{label}}}")

    if root_index is not None:
        output.append(f"   \\deproot[edge below]{{{root_index}}}{{root}}")

    output.append("\\end{dependency}")
    output.append("}")
    output.append("\\end{frame}")

    return "\n".join(output)

def projectivize(sentence, labeling="head", left_first=True, smallest_first=True, return_step_count=False, VERBOSE=False, print_lifting_steps_latex=False):
    """
    Projectivize sentences by lifting the original head to the head's head, loop until there are no non-projective arcs,
    Inputs
        - Sentence 
            - list of token dictionaries from a CoNLL-U file
        - (optional) labeling
            - "None" for no labeling
            - "head"
            - "path"
            - "head_path"
        - (optional) left_first
            - bool of whether to prefer leftmost arcs
        - (optional) smallest_first
            - bool of whether to prefer smallest distance arcs
        - (optional) return_step_count
            - bool of whether to return the number of steps in lifting
        - (optional) VERBOSE
        - (optional) print_lifting_steps_latex
            - bool of whether to print the LaTeX steps of lifting
    Outputs 
        - reconstructed sentence
        - (optional) the number of steps
    """
    
    if labeling not in [None, "head", "path", "head_path", "head_all"]:
        raise ValueError("labeling must be one of 'None', 'head', 'path', 'head_path', 'head_all'")

    steps_latex = []
    sentence_reconstructed = deepcopy(sentence)
    
    flag_non_proj, non_proj_arcs = is_non_projective_sentence(sentence_reconstructed, return_arcs=True)
    
    if VERBOSE:
        print("sentence\n", sentence_reconstructed)
        print(flag_non_proj, non_proj_arcs)
    if print_lifting_steps_latex:
        steps_latex.append(sentence_to_latex_tree(sentence_reconstructed, message="Sentence step 0"))

    step_count = 0
    lifted_paths_head2dep = defaultdict(list)    # to store the deps that was lifted through this head
    lifted_paths_dep2head = defaultdict(list)    # to store the heads of the lifting dep

    while flag_non_proj:
        step_count += 1
        
        #1 choose an arc to lift
        chosen_np_head, chosen_np_dep = choose_arc_to_lift(non_proj_arcs, sentence_reconstructed, left_first=left_first, smallest_first=smallest_first, VERBOSE=VERBOSE)
        
        if chosen_np_head is None or chosen_np_dep is None:
            print("NON_PROJECTIVIZABLE")
            if print_lifting_steps_latex:
                return sentence_reconstructed, steps_latex
            else:
                return sentence_reconstructed
        
        if VERBOSE:
            print(f"Chosen non-projective arc: ({chosen_np_head}, {chosen_np_dep})")
        



        #2 lift the arc
        sentence_reconstructed[chosen_np_dep]['head'] = sentence_reconstructed[chosen_np_head]['head']
        if VERBOSE:
            print("sentence reconstructed \n", sentence_reconstructed)
        
        #3 append path list for labeling
        lifted_paths_head2dep[chosen_np_head].append(chosen_np_dep)
        lifted_paths_dep2head[chosen_np_dep].append(chosen_np_head)
        
        # 4 check if there are still non-projective arcs
        flag_non_proj, non_proj_arcs = is_non_projective_sentence(sentence_reconstructed, return_arcs=True)

        if VERBOSE:
            print(flag_non_proj, non_proj_arcs)
        if print_lifting_steps_latex:
            steps_latex.append(sentence_to_latex_tree(sentence_reconstructed, f"Sentence step {step_count}"))

    # laebling
    sentence_reconstructed = decorate_projectivized_sentence(sentence, sentence_reconstructed, lifted_paths_head2dep, lifted_paths_dep2head, labeling=labeling)
    
    if print_lifting_steps_latex:
        for step in steps_latex:
            print(step)
            print("\n")
    
    if return_step_count:
        return sentence_reconstructed, step_count
    
    return sentence_reconstructed
        
def filter_unliftable_arcs(arcs, sentence):
    valid_arcs = []
    for (head, dep) in arcs:
        if sentence[head]['head'] != 0:
            valid_arcs.append((head, dep))
    return valid_arcs

def get_smallest_arc(arcs, VERBOSE=False):
    smallest_distance = float('inf')
    smallest_arc = []
    for (head, dep) in arcs:
        distance = abs(int(head) - int(dep))
        if distance < smallest_distance:
            smallest_distance = distance
            smallest_arc = [(head, dep)]
        elif distance == smallest_distance:
            smallest_arc.append((head, dep))
        if VERBOSE:
            print(f"head: {head}, dep: {dep}, distance: {distance}, smallest_distance: {smallest_distance}")
    return smallest_arc

def get_largest_arc(arcs, VERBOSE=False):
    largest_distance = 0
    largest_arc = []
    for (head, dep) in arcs:
        distance = abs(int(head) - int(dep))
        if distance > largest_distance:
            largest_distance = distance
            largest_arc = [(head, dep)]
        elif distance == largest_distance:
            largest_arc.append((head, dep))
        if VERBOSE:
            print(f"head: {head}, dep: {dep}, distance: {distance}, largest_distance: {largest_distance}")
    return largest_arc

def get_leftmost_arc(arcs, VERBOSE=False):
    leftmost_arc = arcs[0]
    leftmost_id = sorted(arcs[0])[0]
    for (head, dep) in arcs:
        if sorted([head, dep])[0] < leftmost_id:
            leftmost_id = sorted([head, dep])[0]
            leftmost_arc = (head, dep)
    return leftmost_arc

def get_rightmost_arc(arcs, VERBOSE=False):
    rightmost_arc = arcs[0]
    rightmost_id = sorted(arcs[0])[1]
    for (head, dep) in arcs:
        if sorted([head, dep])[1] > rightmost_id:
            rightmost_id = sorted([head, dep])[1]
            rightmost_arc = (head, dep)
    return rightmost_arc

def choose_arc_to_lift(arcs, sentence, left_first=True, smallest_first=True, VERBOSE=False):
    """
    Choose an arc to lift based on the specified criteria.
    Inputs:
        - arcs: list of non-projective arcs [(head, dep), ]
        - sentence: list of token dictionaries from a CoNLL-U file
        - left_first: if True, prefer leftmost arcs
        - smallest_first: if True, prefer smallest distance arcs
    Output:
        - (head, dep) tuple of the chosen arc to lift
    """
    
    arcs = deepcopy(arcs)
    arcs = filter_unliftable_arcs(arcs, sentence)

    if len(arcs) == 0:
        return None, None
    
    if smallest_first:
        arcs = get_smallest_arc(arcs, VERBOSE)
    else:
        arcs = get_largest_arc(arcs, VERBOSE)
    
    if left_first:
        chosen_arc = get_leftmost_arc(arcs, VERBOSE)
    else:
        chosen_arc = get_rightmost_arc(arcs, VERBOSE)

    return chosen_arc

def decorate_projectivized_sentence(original_sentence, reconstructed_sentence, lifted_paths_head2dep=None, lifted_paths_dep2head=None, labeling="head"):
    """
    Decorate the reconstructed sentence with labels based on the original sentence.
    Inputs:
        - original_sentence: list of token dictionaries from a CoNLL-U file
        - reconstructed_sentence: list of token dictionaries from a CoNLL-U file
        - lifted_paths_head2dep: dictionary of lifted paths {head_id: [dep_id1, dep_id2, ...]}
        - labeling: "None", "head", "path", "head_path", "head_all"
    Output:
        - decorated reconstructed sentence
    """
    
    if labeling == "None":
        return reconstructed_sentence
    
    if labeling in ["head", "head_path"]:
        for id, _ in reconstructed_sentence.items():
            original_head = original_sentence[id]['head']
            reconstructed_head = reconstructed_sentence[id]['head']
            if original_head != reconstructed_head:
                reconstructed_sentence[id]['deprel'] = f"{original_sentence[id]['deprel']}{DELIMITER_DEP}{original_sentence[original_head]['deprel']}"

    if labeling in ["path", "head_path"]:
        for head_id, dep_ids in lifted_paths_head2dep.items():
            if len(dep_ids) > 0:
                reconstructed_sentence[head_id]['deprel'] = f"{reconstructed_sentence[head_id]['deprel']}{DELIMITER_HEAD}"
    
    if labeling == "head_all":
        for id, head_id_lst in lifted_paths_dep2head.items():
            head_name_lst = [original_sentence[head_id]["deprel"] for head_id in head_id_lst]
            reconstructed_sentence[id]['deprel'] = f"{reconstructed_sentence[id]['deprel']}{DELIMITER_DEP}{DELIMITER_DEP.join(head_name_lst)}"

    
    return reconstructed_sentence

def deprojectivize(sentence, labeling="head", VERBOSE=False, print_lifting_steps_latex=False, left_first=True):
    """
    Deprojectivize sentences by lifting the original head to the head's head, loop until there are no non-projective arcs,
    Inputs
        - Sentence 
            - list of token dictionaries from a CoNLL-U file
        - (optional) labeling
            - "None" for no labeling
            - "head"
            - "path"
            - "head_path"
        - (optional) VERBOSE
        - (optional) print_lifting_steps_latex
            - bool of whether to print the LaTeX steps of lifting
    Outputs 
        - reconstructed sentence
    """
    
    if labeling not in [None, "head", "path", "head_path"]:
        raise ValueError("labeling must be one of None, 'head', 'path', 'head_path'")

    steps_latex = []
    sentence_reconstructed = deepcopy(sentence)

    head_to_dep = defaultdict(list)
    for _, token in sentence_reconstructed.items():
        head = token['head']
        dep = token['id']
        head_to_dep[head].append(dep)

    flag_lifted , lifted_arcs = is_lifted_sentence(sentence_reconstructed, return_lifted_arcs=True)

    if VERBOSE:
        print("\nsentence\n", sentence_reconstructed)
        print(head_to_dep)
        
        print(flag_lifted, lifted_arcs)
    if print_lifting_steps_latex:
        steps_latex.append(sentence_to_latex_tree(sentence_reconstructed, message="Sentence step 0"))

    step_count = 0
    
    while flag_lifted:
        step_count += 1
        
        #1 choose arc to delift
        if left_first:
            chosen_lifted_head, chosen_lifted_dep = get_leftmost_arc(lifted_arcs, VERBOSE)
        else:
            chosen_lifted_head, chosen_lifted_dep = get_rightmost_arc(lifted_arcs, VERBOSE)
        if VERBOSE:
            print(f"Chosen lifted arc: ({chosen_lifted_head}, {chosen_lifted_dep})")
        
        #2 delift the arc
        #for "head" method
        original_dep_label, original_head_label = sentence_reconstructed[chosen_lifted_dep]['deprel'].split(DELIMITER_DEP)
        
        original_head_id = None
        serach_lst = [chosen_lifted_head]
        head_to_dep[chosen_lifted_head].remove(chosen_lifted_dep)  # remove the lifted arc from the head to dep

        if VERBOSE:
            print(f"Search head: {serach_lst[0]}")
        
        while original_head_id is None:
            serach_lst = [head_to_dep.get(id, None) for id in serach_lst]
            serach_lst = [x for x in serach_lst if x is not None]  # filter out empty lists
            if VERBOSE:
                print(f"Search lists: {serach_lst}")
            
            if serach_lst == []:
                break
            serach_lst = [x for xs in serach_lst for x in xs]  # flatten the list
            original_head_id = find_original_head(sentence_reconstructed, serach_lst, original_head_label)

        if original_head_id is None:
            original_head_id = head_to_dep[0][0]  # if not found, use the root as the original head

        if VERBOSE:
            print(f"Original head ID: {original_head_id}, Original head label: {sentence_reconstructed[original_head_id]['deprel']}, Target Original head label: {original_head_label}")

        #3 reconstruct the sentence
        sentence_reconstructed[chosen_lifted_dep]['head'] = original_head_id
        sentence_reconstructed[chosen_lifted_dep]['deprel'] = original_dep_label
        
        head_to_dep[original_head_id].append(chosen_lifted_dep)  # add the delifted arc to the head to dep mapping
        
        if VERBOSE:
            print("\nsentence reconstructed \n", sentence_reconstructed)
            print(head_to_dep)

        
        #4 check if there are still lifted arcs
        flag_lifted, lifted_arcs = is_lifted_sentence(sentence_reconstructed, return_lifted_arcs=True)

        if VERBOSE:
            print(flag_lifted, lifted_arcs)
        if print_lifting_steps_latex:
            steps_latex.append(sentence_to_latex_tree(sentence_reconstructed, f"Sentence step {step_count}"))

    return sentence_reconstructed

def find_original_head(sentence, token_id_lst, original_head_label):
    for id in token_id_lst:
        if sentence[id]['deprel'].startswith(original_head_label):
            return id
    return None


def is_lifted_sentence(sentence, return_lifted_arcs=False):
    """
    Determines if a sentence contains lifted arcs.
    Input: sentence - list of token dictionaries from a CoNLL-U file
    Output: 
        - True if lifted arcs are present, False otherwise
        - (optional output) list of lifted arcs [(head, dep), ] if return_lifted_arcs is True
    """
    
    lifted_arcs = []
    for id, _ in sentence.items():
        if is_lifted_arc(sentence, id):
            head = sentence[id]['head']
            lifted_arcs.append((head, id))
    
    if return_lifted_arcs:
        if len(lifted_arcs) == 0:
            return False, []
        else:
            return True, lifted_arcs
    else:
        return len(lifted_arcs) > 0

def is_lifted_arc(sentence, token_id):
    if DELIMITER_DEP in sentence[token_id]["deprel"]:
        return True
    return False

    

def main():

    folder_path = "../../data/ud-treebanks-v2.15.tgz"
    file_path = "ud-treebanks-v2.15/UD_Korean-GSD/ko_gsd-ud-test.conllu"
    non_proj_sentences = example_non_projective_sentences(folder_path, file_path, limit=2)
    print(non_proj_sentences[1])
    
    non_proj_sentence = non_proj_sentences[1]
    projectivized_sentence, step_count= projectivize(non_proj_sentence, return_step_count=True, left_first=False)
    print(projectivized_sentence)
    print(step_count)


    deprojectivized_sentence = deprojectivize(projectivized_sentence, VERBOSE=True)

    print("revertable:", deprojectivized_sentence== non_proj_sentence)


    # for sentence in non_proj_sentences:
    #     _, non_proj_arc = is_non_projective_sentence(sentence, return_arcs=True)
    #     print(f"Sentence: {sentence}")
    #     print(f"Non-projective arcs: {non_proj_arc}")
    
    # for token in sentence:
    #     if '-' in token['id'] or '.' in token['id']:
    #         continue  # Skip multiword tokens and empty nodes
    #     head = int(token['head'])
    #     dep = int(token['id'])

    #     if is_non_projective_arc(sentence, dep):
    #         print(f"Dependent token {token['form']} (ID: {dep}) is a non-projective arc.")

    # print(sentence_to_latex_tree(sentence))


if __name__ == "__main__":
    main()