import argparse

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
                self._str = '({} {} ){}'.format(self.symbol, childstr, self.symbol)
            else:
                if '*' in self.word:
                    self._str = '{}'.format(
                        self.word,
                    )
                else:
                    self._str = '{}'.format(
                        self.symbol,
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
        assert line[index] == '(', "Invalid tree string {} at index {}".format(line, index)
        index += 1
        symbol = None
        word = None
        children = []
        while line[index] != ')':
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
                    rpos = line.find(')', index)
                    word = line[index:rpos]
                    index = rpos
            if line[index] == " ":
                index += 1
        
        assert line[index] == ')', "Invalid tree string %s at %d" % (line, index)

        t = PhraseTree(
            symbol=symbol,
            children=children,
            word=word
        )

        return (index + 1), t
    
def linearize(tree, isKorean):
    nodes = [tree]              # list of PhraseTree objects
    while (len(nodes) != 0):
        curr = nodes.pop(0)
        nodes.extend(curr.children)
        if (not isKorean and curr.symbol != "-NONE-"):
            curr.word = ""

    return tree

if __name__ == "__main__":
    help_message = """
    Script to linearize PTB trees. Used to allow sequence-to-equence neural network parsing.
    """
    parser = argparse.ArgumentParser(prog="Linearize PTB", description=help_message)
    parser.add_argument("-k", "--korean", action="store_true", help="sets language to korean (default works for english and chinese)")
    parser.add_argument("input", help="input PTB tree")
    
    args = parser.parse_args()

    for line in open(args.input):
        line = line.strip()

        linearized = PhraseTree.parse(line)
        tree = linearize(linearized, args.korean)

        print(linearized)
        break