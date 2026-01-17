import fileinput

word_index = 0

class PhraseTree(object):
# adapted from https://github.com/nikitakit/self-attentive-parser/blob/master/data/common/strip_functional.py

    def __init__(
        self,
        symbol=None,
        word=None,
        children=[],
        parent=None,
    ):
        self.symbol = symbol        # label at top node
        self.word= word             # word at leaft level, else None
        self.children = children    # list of PhraseTree objects
        self.parent = None
        for child in children:
            child.parent = self

        self._str = None
        self._left_span = None
        self._right_span = None

    def __str__(self):
        if self._str is None:
            if len(self.children) != 0:
                childstr = ' '.join(str(c) for c in self.children)
                self._str = '({} {})'.format(self.symbol, childstr)
            else:
                self._str = '({} {})'.format(
                    self.symbol,
                    self.word,
                )
        return self._str

    @staticmethod
    def parse(line):
        """
        Loads a tree from a tree in PTB parenthetical format
        """
        line += " "
        ix, t = PhraseTree._parse(line, 0)

        return t
    
    @staticmethod
    def _parse(line, index):
        # assert line[index] == '(', "Invalid tree string {} at index {}".format(line, index)
        index += 1
        symbol = None
        word = None
        children = []
        while index < len(line) and line[index] != ')':
            if line[index] == '(':
                index, t = PhraseTree._parse(line, index)
                if t is not None:
                    children.append(t)
            else:
                if symbol == None:
                    # symbol is here!
                    rpos = min(line.find(' ', index), line.find(')', index))
                    # see above N.B. (find could return -1)

                    symbol = line[index:rpos] # (word, tag) string pair

                    index = rpos
                else:
                    lpos = index
                    rpos = index
                    while rpos < len(line) and line[rpos] != ')':
                        if line[lpos] == '(':
                            lpos, t = PhraseTree._parse(line, lpos)
                            if t is not None:
                                children.append(t)
                        rpos = line.find(' ', lpos)
                        child_symbol = line[lpos:rpos]
                        child = None
                        if '*' in child_symbol:
                            child = PhraseTree(
                                symbol='-NONE-',
                                children=[],
                                word=child_symbol
                            )
                        elif child_symbol == 'SFN':
                            child = PhraseTree(
                                symbol='.',
                                children=[],
                                word='.'
                            )
                        elif child_symbol == 'SCM':
                            child = PhraseTree(
                                symbol=',',
                                children=[],
                                word=','
                            )
                        elif child_symbol == 'SLQ':
                            child = PhraseTree(
                                symbol='``',
                                children=[],
                                word='``'
                            )
                        elif child_symbol == 'SRQ':
                            child = PhraseTree(
                                symbol='\'\'',
                                children=[],
                                word='\'\''
                            )
                        elif child_symbol == '':
                            pass
                        else:
                            global word_index
                            child = PhraseTree(
                                symbol=child_symbol,
                                children=[],
                                word='word' + str(word_index)
                            )
                            word_index += 1
                        if child != None:
                            children.append(child)
                        rpos += 1
                        lpos = rpos
                    index = rpos
            if index >= len(line) or index == -1:
                break
            if line[index] == " ":
                index += 1
        
        # assert line[index] == ')', "Invalid tree string %s at %d" % (line, index)

        t = PhraseTree(
            symbol=symbol,
            children=children,
            word=word
        )
        
        index = line.find(' ', index)

        return index, t

if __name__ == "__main__":
    help_message = """
    Script to recover tree structure from linearized data. Used to allow the evaluation of linearized data.
    """

    for line in fileinput.input():
        line = line.strip()

        word_index = 0
        tree = PhraseTree.parse(line)

        print(tree)