"""
@file
@brief About completion
"""
from typing import Tuple, List, Iterator
from collections import deque


class CompletionTrieNode(object):
    """
    node definition in a trie used to do completion,
    see :ref:`l-completion0`.
    """

    __slots__ = ["value", "children", "weight", "leave", "stat", "parent"]

    def __init__(self, value, leave, weight=1.0):
        """
        @param      value       value (a character)
        """
        self.value = value
        self.children = None
        self.weight = weight
        self.leave = leave
        self.stat = None
        self.parent = None

    def __str__(self):
        """
        usual
        """
        return "[{2}:{0}:w={1}]".format(self.value, self.weight, "#" if self.leave else "-")

    def _add(self, key, child):
        """
        add a child

        @param      key         one letter of the word
        @param      child       child
        @return                 self
        """
        if self.children is None:
            self.children = {key: child}
            child.parent = self
        elif key in self.children:
            raise KeyError("'{0}' already added".format(key))
        else:
            self.children[key] = child
            child.parent = self
        return self

    def items_list(self) -> List['CompletionTrieNode']:
        """
        all children nodes inluding itself in a list

        @return          list[
        """
        res = [self]
        if self.children is not None:
            for k, v in sorted(self.children.items()):
                r = v.items_list()
                res.extend(r)
        return res

    def __iter__(self):
        """
        iterates on all nodes
        """
        yield self
        if self.children is not None:
            for v in self.children.values():
                for _ in v:
                    yield _

    def items(self) -> Iterator[Tuple[float, str, 'CompletionTrieNode']]:
        """
        iterates on children, iterates on weight, key, child
        """
        if self.children is not None:
            for k, v in self.children.items():
                yield v.weight, k, v

    def iter_leaves(self, max_weight=None) -> Iterator[Tuple[float, str]]:
        """
        iterators on leaves sorted per weight, yield weight, value

        @param      max_weight  keep all value under this threshold or None for all
        """
        def iter_local(node):
            if node.leave and (max_weight is None or node.weight <= max_weight):
                yield node.weight, None, node.value
            for w, k, v in sorted(node.items()):
                for w_, k_, v_ in iter_local(v):
                    yield w_, k_, v_

        for w, k, v in sorted(iter_local(self)):
            yield w, v

    def leaves(self) -> Iterator['CompletionTrieNode']:
        """
        iterators on leaves
        """
        stack = [self]
        while len(stack) > 0:
            pop = stack.pop()
            if pop.leave:
                yield pop
            if pop.children:
                stack.extend(pop.children.values())

    @staticmethod
    def build(words) -> 'CompletionTrieNode':
        """
        builds a trie

        @param  words       list of (weight, word)
        @return             root of the trie (CompletionTrieNode)
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
                    node._add(c, new_node)
                    node = new_node
            new_node.leave = True
            new_node.weight = w
            nb += 1
        root.weight = minw
        return root

    def find(self, prefix: str) -> 'CompletionTrieNode':
        """
        returns the node which holds all suggestions starting with a given prefix

        @param      prefix      prefix
        @return                 node or None for no result
        """
        if len(prefix) == 0:
            if not self.value:
                return self
            else:
                raise ValueError(
                    "find '{0}' but node is not empty '{1}'".format(prefix, self.value))
        node = self
        for c in prefix:
            if node.children is not None and c in node.children:
                node = node.children[c]
            else:
                return None
        return node

    def min_keystroke(self, word: str) -> Tuple[int, int]:
        """
        returns the minimum keystrokes for a word

        @param      word        word
        @return                 number, length of best prefix

        See :ref:`l-completion-optim`.

        .. math::
            :nowrap:

            \\begin{eqnarray*}
            K(q, k, S) &=& \\min\\acc{ i | s_i \\succ q[1..k], s_i \\in S } \\\\
            M(q, S) &=& \\min_{0 \\infegal k \\infegal l(q)}  k + K(q, k, S)
            \\end{eqnarray*}
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

    def min_dynamic_keystroke(self, word: str) -> Tuple[int, int]:
        """
        returns the dynamic minimum keystrokes for a word,

        @param      word        word
        @return                 number, length of best prefix, iteration it stops moving

        This function must be called after @see me precompute_stat
        and @see update_stat_dynamic.
        See :ref:`Dynamic Minimum Keystroke <def-mks2>`.

        .. math::
            :nowrap:

            \\begin{eqnarray*}
            K(q, k, S) &=& \\min\\acc{ i | s_i \\succ q[1..k], s_i \\in S } \\\\
            M'(q, S) &=& \\min_{0 \\infegal k \\infegal l(q)} \\acc{ M'(q[1..k], S) + K(q, k, S) | q[1..k] \\in S }
            \\end{eqnarray*}
        """
        node = self.find(word)
        if not hasattr(node, "stat"):
            raise AttributeError("run precompute_stat and update_stat_dynamic")
        if not hasattr(node.stat, "mks"):
            raise AttributeError("run precompute_stat and update_stat_dynamic\nnode={0}\n{1}".format(
                self, "\n".join(sorted(self.stat.__dict__.keys()))))
        return node.stat.mks, node.stat.mks_, node.stat.mksi_

    def precompute_stat(self):
        """
        computes and stores list of suggestions for each node,
        computes mks

        @param      clean   clean stat
        """
        stack = deque()
        stack.extend(self.leaves())
        while len(stack) > 0:
            pop = stack.popleft()
            if pop.stat is not None:
                continue
            if not pop.children:
                pop.stat = CompletionTrieNode._Stat()
                pop.stat.suggestions = []
                pop.stat.mks0 = len(pop.value)
                pop.stat.mks0_ = len(pop.value)
                if pop.parent is not None:
                    stack.append(pop.parent)
            elif all(v.stat is not None for v in pop.children.values()):
                pop.stat = CompletionTrieNode._Stat()
                if pop.leave:
                    pop.stat.mks0 = len(pop.value)
                    pop.stat.mks0_ = len(pop.value)
                stack.extend(pop.children.values())
                pop.stat.merge_suggestions(pop.value, pop.children.values())
                pop.stat.update_minimum_keystroke(len(pop.value))
                if pop.parent is not None:
                    stack.append(pop.parent)
            else:
                # we'll do it again later
                stack.append(pop)

    def update_stat_dynamic(self):
        """
        must be called after @see me precompute_stat
        and computes dynamic mks (see :ref:`Dynamic Minimum Keystroke <def-mks2>`)

        @return         number of iterations to converge
        """
        for node in self:
            if node.leave:
                node.stat.init_dynamic_minimum_keystroke()
            node.stat.iter_ = 0
        updates = 1
        iter = 0
        while updates > 0:
            updates = 0
            done = {}
            stack = []
            stack.append(self)
            while len(stack) > 0:
                pop = stack.pop()
                if pop.stat.iter_ > iter:
                    continue
                if pop.leave:
                    updates += pop.stat.update_dynamic_minimum_keystroke(
                        len(pop.value))
                if pop.children:
                    stack.extend(pop.children.values())
                pop.stat.iter_ += 1
            iter += 1
        return iter

    class _Stat:
        """
        stores statistics and intermediate data about the compuation
        the metrics
        """

        def merge_suggestions(self, prefix: int, nodes: 'CompletionTrieNode'):
            """
            merges list of suggestions and cut the list, we assume
            given list are sorted
            """
            class Fake:
                pass
            res = []
            indexes = [0 for _ in nodes]
            indexes.append(0)
            last = Fake()
            last.stat = CompletionTrieNode._Stat()
            last.stat.suggestions = list(
                sorted((_.weight, _) for _ in nodes if _.leave))
            nodes = list(nodes)
            nodes.append(last)

            maxl = 0
            while True:
                en = [(_.stat.suggestions[indexes[i]][0], i, _.stat.suggestions[indexes[i]][1])
                      for i, _ in enumerate(nodes) if indexes[i] < len(_.stat.suggestions)]
                if not en:
                    break
                e = min(en)
                i = e[1]
                res.append((e[0], e[2]))
                indexes[i] += 1
                maxl = max(maxl, len(res[-1][1].value))
            if len(res) > maxl - len(prefix):
                self.suggestions = res[:maxl - len(prefix)]
            else:
                self.suggestions = res

        def update_minimum_keystroke(self, lw):
            """
            update minimum keystroke for the suggestions

            @param      lw      prefix length
            """
            for i, wsug in enumerate(self.suggestions):
                sug = wsug[1]
                nl = lw + i + 1
                if not hasattr(sug.stat, "mks0") or sug.stat.mks0 > nl:
                    sug.stat.mks0 = nl
                    sug.stat.mks0_ = lw

        def update_dynamic_minimum_keystroke(self, lw):
            """
            update dynamic minimum keystroke for the suggestions

            @param      lw      prefix length
            @return             number of updates
            """
            self.mks_iter += 1
            update = 0
            for i, wsug in enumerate(self.suggestions):
                sug = wsug[1]
                if not sug.leave:
                    continue
                nl = self.mks + i + 1
                if sug.stat.mks > nl:
                    sug.stat.mks = nl
                    sug.stat.mks_ = lw
                    sug.stat.mksi_ = self.mks_iter
                    update += 1
            return update

        def init_dynamic_minimum_keystroke(self):
            """
            initializes mks from mks0
            """
            self.mks = self.mks0
            self.mks_ = self.mks0_
            self.mks_iter = 0
            self.mksi_ = 0

        def str_mks0(self) -> str:
            """
            return a string with metric information
            """
            if hasattr(self, "mks0"):
                return "MKS0={0} *={1} l={2}".format(self.mks0, self.mks0_, len(self.suggestions))
            else:
                return "-"

        def str_mks(self) -> str:
            """
            return a string with metric information
            """
            s0 = self.str_mks0()
            if hasattr(self, "mks"):
                return s0 + " MKS={0} *={1} i={2}".format(self.mks, self.mks_, self.mksi_)
            else:
                return s0
