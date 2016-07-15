"""
@file
@brief About completion
"""


class CompletionTrieNode(object):
    """
    node representation in a trie
    """

    __slots__ = ["value", "children", "weight", "leave", "stat"]

    def __init__(self, value, leave, weight=1.0):
        """
        @param      value       value (a character)
        """
        self.value = value
        self.children = None
        self.weight = weight
        self.leave = leave
        self.stat = None

    def __str__(self):
        """
        usual
        """
        return "{2}:{0}:{1}".format(self.value, self.weight, "#" if self.leave else "-")

    def add(self, key, child):
        """
        add a child

        @param      child       child
        @return                 self
        """
        if self.children is None:
            self.children = {key: child}
        elif key in self.children:
            raise KeyError("'{0}' already added".format(key))
        else:
            self.children[key] = child

    def items_list(self):
        """
        all nodes in a list
        """
        res = [self]
        if self.children is not None:
            for k, v in sorted(self.children.items()):
                r = v.items_list()
                res.extend(r)
        return res

    def items(self):
        """
        iterates on childen, iterates on weight, key, child
        """
        if self.children is not None:
            for k, v in self.children.items():
                yield v.weight, k, v

    def iter_leaves(self, max_weight=None):
        """
        iterators on leaves sorted per weight, yield weight, value

        @param      max_weight   keep all value under this threshold or None for all
        """
        def iter_local(node):
            if node.leave and (max_weight is None or node.weight <= max_weight):
                yield node.weight, None, node.value
            for w, k, v in sorted(node.items()):
                for w_, k_, v_ in iter_local(v):
                    yield w_, k_, v_

        for w, k, v in sorted(iter_local(self)):
            yield w, v

    @staticmethod
    def build(words):
        """
        builds a trie

        @param  words       list of (weight, word)
        @return             root of the trie
        """
        root = CompletionTrieNode('', False)
        nb = 0
        minw = None
        for w, word in words:
            if w is None:
                w = nb
            if minw is None or minw > w:
                minw = w
            node = root
            for c in word:
                if node.children is not None and c in node.children:
                    new_node = node.children[c]
                    if not node.leave:
                        node.weight = min(node.weight, w)
                    node = new_node
                else:
                    new_node = CompletionTrieNode(
                        node.value + c, False, weight=w)
                    node.add(c, new_node)
                    node = new_node
            new_node.leave = True
            new_node.weight = w
            nb += 1
        root.weight = minw
        return root

    def find(self, prefix):
        """
        returns the node which holds all suggestions starting with prefix

        @param      prefix      prefix
        @return                 node or None for no result
        """
        node = self
        for c in prefix:
            if node.children is not None and c in node.children:
                node = node.children[c]
            else:
                return None
        return node

    def min_keystroke(self, word):
        """
        return the minimum keystrokes for a word

        @param      word        word
        @return                 number, length of best prefix
        """
        nodes = [self]
        node = self
        for c in word:
            if node.children is not None and c in node.children:
                node = node.children[c]
                nodes.append(node)
            else:
                # not found
                return len(word), -1
        nodes.reverse()
        metric = len(word)
        best = len(word)
        for node in nodes[1:]:
            res = list(n[1] for n in node.iter_leaves())
            ind = res.index(word)
            m = len(node.value) + ind + 1
            if m < metric:
                metric = m
                best = len(node.value)
            if ind >= len(word):
                # no need to go further, the position will increase
                break
        return metric, best
