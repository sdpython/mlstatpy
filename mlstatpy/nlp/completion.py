from typing import Tuple, List, Iterator
from collections import deque


class CompletionTrieNode:
    """
    Node definition in a trie used to do completion, see :ref:`l-completion0`.
    This implementation is not very efficient about memmory consumption,
    it does not hold above 200.000 words.
    It should be done another way (:epkg:`cython`, :epkg:`C++`).
    """

    __slots__ = (  # noqa: RUF023
        "value",
        "children",
        "weight",
        "leave",
        "stat",
        "parent",
        "disp",
    )

    def __init__(self, value, leave, weight=1.0, disp=None):
        """
        @param      value       value (a character)
        @param      leave       boolean (is it a completion)
        @param      weight      ordering (the lower, the first)
        @param      disp        original string, use this to identify the node
        """
        if not isinstance(value, str):
            raise TypeError(f"value must be str not '{value}' - type={type(value)}")
        self.value = value
        self.children = None
        self.weight = weight
        self.leave = leave
        self.stat = None
        self.parent = None
        self.disp = disp

    @property
    def root(self):
        """
        Returns the initial node with no parent.
        """
        node = self
        while node.parent is not None:
            node = node.parent
        return node

    def __str__(self):
        """
        usual
        """
        return f"[{'#' if self.leave else '-'}:{self.value}:w={self.weight}]"

    def _add(self, key, child):
        """
        Adds a child.

        @param      key         one letter of the word
        @param      child       child
        @return                 self
        """
        if self.children is None:
            self.children = {key: child}
            child.parent = self
        elif key in self.children:
            raise KeyError(f"'{key}' already added")
        else:
            self.children[key] = child
            child.parent = self
        return self

    def items_list(self) -> List["CompletionTrieNode"]:
        """
        All children nodes inluding itself in a list.

        @return          list[
        """
        res = [self]
        if self.children is not None:
            for _, v in sorted(self.children.items()):
                r = v.items_list()
                res.extend(r)
        return res

    def __iter__(self):
        """
        Iterates on all nodes (sorted).
        """
        stack = [self]
        while len(stack) > 0:
            node = stack.pop()
            yield node
            if node.children:
                stack.extend(v for k, v in sorted(node.children.items(), reverse=True))

    def unsorted_iter(self):
        """
        Iterates on all nodes.
        """
        stack = [self]
        while len(stack) > 0:
            node = stack.pop()
            yield node
            if node.children:
                stack.extend(node.children.values())

    def items(self) -> Iterator[Tuple[float, str, "CompletionTrieNode"]]:
        """
        Iterates on children, iterates on weight, key, child.
        """
        if self.children is not None:
            for k, v in self.children.items():
                yield v.weight, k, v

    def iter_leaves(self, max_weight=None) -> Iterator[Tuple[float, str]]:
        """
        Iterators on leaves sorted per weight, yield weight, value.

        @param      max_weight  keep all value under this threshold or None for all
        """

        def iter_local(node):
            if node.leave and (max_weight is None or node.weight <= max_weight):
                yield node.weight, None, node.value
            for _w, _k, v in sorted(node.items()):
                for w_, k_, v_ in iter_local(v):  # noqa: UP028
                    yield w_, k_, v_

        for w, _, v in sorted(iter_local(self)):
            yield w, v

    def leaves(self) -> Iterator["CompletionTrieNode"]:
        """
        Iterators on leaves.
        """
        stack = [self]
        while len(stack) > 0:
            pop = stack.pop()
            if pop.leave:
                yield pop
            if pop.children:
                stack.extend(pop.children.values())

    def all_completions(self) -> List[Tuple["CompletionTrieNode", List[str]]]:
        """
        Retrieves all completions for a node,
        the method does not need @see me precompute_stat to be run first.
        """
        word = self.value
        nodes = [self.root]
        node = nodes[0]
        for c in word:
            if node.children is not None and c in node.children:
                node = node.children[c]
                nodes.append(node)
        nodes.reverse()
        all_res = []
        for node in nodes:
            res = [n[1] for n in node.iter_leaves()]
            all_res.append((node, res))
        all_res.reverse()
        return all_res

    def all_mks_completions(
        self,
    ) -> List[Tuple["CompletionTrieNode", List["CompletionTrieNode"]]]:
        """
        Retrieves all completions for a node,
        the method assumes @see me precompute_stat was run.
        """
        res = []
        node = self
        while True:
            res.append((node, node.stat.completions))
            if node.parent is None:
                break
            node = node.parent
        res.reverse()
        return res

    def str_all_completions(self, maxn=10, use_precompute=True) -> str:
        """
        Builds a string with all completions for all
        prefixes along the paths.

        :param maxn: maximum number of completions to show
        :param use_precompute: use intermediate results
            built by @see me precompute_stat
        :return: str
        """
        res = self.all_mks_completions() if use_precompute else self.all_completions()
        rows = []
        for node, sug in res:
            rows.append(
                "l={3} p='{0}' {1} {2}".format(
                    node.value,
                    "-" * 10,
                    node.stat.str_mks(),
                    "+" if node.leave else "-",
                )
            )
            for i, s in enumerate(sug):
                if isinstance(s, str):
                    rows.append(f"  {i + 1}-'{s}'")
                else:
                    rows.append(f"  {i + 1}-w{s[0]}-'{s[1].value}'")
                if maxn is not None and i > maxn:
                    break
        return "\n".join(rows)

    @staticmethod
    def build(words) -> "CompletionTrieNode":
        """
        Builds a trie.

        :param words: list of ``(word)`` or ``(weight, word)``
            or ``(weight, word, display string)``
        :return: root of the trie (CompletionTrieNode)
        """
        root = CompletionTrieNode("", False)
        nb = 0
        minw = None
        for wword in words:
            if isinstance(wword, tuple):
                if len(wword) == 2:
                    w, word = wword
                    disp = None
                elif len(wword) == 3:
                    w, word, disp = wword
                else:
                    raise ValueError(
                        f"Unexpected number of values, it should be "
                        f"(weight, word) or (weight, word, "
                        f"dispplay string): {wword}"
                    )
            else:
                w = 1.0
                word = wword
                disp = None
            if w is None:
                w = nb
            if minw is None or minw > w:
                minw = w
            node = root
            new_node = None
            for c in word:
                if node.children is not None and c in node.children:
                    if not node.leave:
                        node.weight = min(node.weight, w)
                    node = node.children[c]
                else:
                    new_node = CompletionTrieNode(node.value + c, False, weight=w)
                    node._add(c, new_node)
                    node = new_node
            if new_node is None:
                if node.leave:
                    raise ValueError(
                        f"Value '{word}' appears twice in the input list (not allowed)."
                    )
                new_node = node
            new_node.leave = True
            new_node.weight = w
            if disp is not None:
                new_node.disp = disp
            nb += 1  # noqa: SIM113
        root.weight = minw
        return root

    def find(self, prefix: str) -> "CompletionTrieNode":
        """
        Returns the node which holds all completions starting with a given prefix.

        :param prefix: prefix
        :return: node or None for no result
        """
        if not prefix:
            if not self.value:
                return self
            else:
                raise ValueError(
                    f"find '{prefix}' but node is not empty '{self.value}'"
                )
        node = self
        for c in prefix:
            if node.children is not None and c in node.children:
                node = node.children[c]
            else:
                return None
        return node

    def min_keystroke(self, word: str) -> Tuple[int, int]:
        """
        Returns the minimum keystrokes for a word without optimisation,
        this function should be used if you only have a couple of values to
        computes. You shoud use @see me min_keystroke0 to compute all of them.

        @param      word        word
        @return                 number, length of best prefix

        See :ref:`l-completion-optim`.

        .. math::
            :nowrap:

            \\begin{eqnarray*}
            K(q, k, S) &=& \\min\\acc{ i | s_i \\succ q[1..k], s_i \\in S } \\\\
            M(q, S) &=& \\min_{0 \\leqslant k \\leqslant l(q)}  k + K(q, k, S)
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
            res = [n[1] for n in node.iter_leaves()]
            ind = res.index(word)
            m = len(node.value) + ind + 1
            if m < metric:
                metric = m
                best = len(node.value)
            if ind >= len(word):
                # no need to go further, the position will increase
                break
        return metric, best

    def min_keystroke0(self, word: str) -> Tuple[int, int]:
        """
        Returns the minimum keystrokes for a word.

        :param word: word
        :return: number, length of best prefix, iteration it stops moving

        This function must be called after :meth:`precompute_stat`
        and :meth:`update_stat_dynamic`.

        See :ref:`l-completion-optim`.

        .. math::
            :nowrap:

            \\begin{eqnarray*}
            K(q, k, S) &=& \\min\\acc{ i | s_i \\succ q[1..k], s_i \\in S } \\\\
            M(q, S) &=& \\min_{0 \\leqslant k \\leqslant l(q)}  k + K(q, k, S)
            \\end{eqnarray*}
        """
        node = self.find(word)
        if node is None:
            raise NotImplementedError(
                f"this metric is not yet computed for a "
                f"query outside the trie: '{word}'"
            )
        if not hasattr(node, "stat"):
            raise AttributeError("run precompute_stat and update_stat_dynamic")
        if not hasattr(node.stat, "mks1"):
            raise AttributeError(
                "run precompute_stat and update_stat_dynamic\nnode={0}\n{1}".format(
                    self, "\n".join(sorted(self.stat.__dict__.keys()))
                )
            )
        return node.stat.mks0, node.stat.mks0_, 0

    def min_dynamic_keystroke(self, word: str) -> Tuple[int, int]:
        """
        Returns the dynamic minimum keystrokes for a word.

        @param      word        word
        @return                 number, length of best prefix, iteration it stops moving

        This function must be called after @see me precompute_stat
        and @see me update_stat_dynamic.
        See :ref:`Dynamic Minimum Keystroke <def-mks2>`.

        .. math::
            :nowrap:

            \\begin{eqnarray*}
            K(q, k, S) &=& \\min\\acc{ i | s_i \\succ q[1..k], s_i \\in S } \\\\
            M'(q, S) &=& \\min_{0 \\leqslant k \\leqslant l(q)}
            \\acc{ M'(q[1..k], S) + K(q, k, S) | q[1..k] \\in S }
            \\end{eqnarray*}
        """
        node = self.find(word)
        if node is None:
            raise NotImplementedError(
                f"this metric is not yet computed for "
                f"a query outside the trie: '{word}'"
            )
        if not hasattr(node, "stat"):
            raise AttributeError("run precompute_stat and update_stat_dynamic")
        if not hasattr(node.stat, "mks1"):
            raise AttributeError(
                "run precompute_stat and update_stat_dynamic\nnode={0}\n{1}".format(
                    self, "\n".join(sorted(self.stat.__dict__.keys()))
                )
            )
        return node.stat.mks1, node.stat.mks1_, node.stat.mks1i_

    def min_dynamic_keystroke2(self, word: str) -> Tuple[int, int]:
        """
        Returns the modified dynamic minimum keystrokes for a word.

        @param      word        word
        @return                 number, length of best prefix, iteration it stops moving

        This function must be called after @see me precompute_stat
        and :meth`update_stat_dynamic`.
        See :ref:`Modified Dynamic Minimum Keystroke <def-mks3>`.

        .. math::
            :nowrap:

            \\begin{eqnarray*}
            K(q, k, S) &=& \\min\\acc{ i | s_i \\succ q[1..k], s_i \\in S } \\\\
            M"(q, S) &=& \\min \\left\\{ \\begin{array}{l}
                            \\min_{1 \\leqslant k \\leqslant l(q)}
                            \\acc{ M"(q[1..k-1], S) + 1 + K(q, k, S) | q[1..k]
                            \\in S } \\\\
                            \\min_{0 \\leqslant k \\leqslant l(q)}
                            \\acc{ M"(q[1..k], S) + \\delta + K(q, k, S) | q[1..k]
                            \\in S }
                            \\end{array} \\right .
            \\end{eqnarray*}
        """
        node = self.find(word)
        if node is None:
            raise NotImplementedError(
                f"this metric is not yet computed for a "
                f"query outside the trie: '{word}'"
            )
        if not hasattr(node, "stat"):
            raise AttributeError("run precompute_stat and update_stat_dynamic")
        if not hasattr(node.stat, "mks2"):
            raise AttributeError(
                "run precompute_stat and update_stat_dynamic\nnode={0}\n{1}".format(
                    self, "\n".join(sorted(self.stat.__dict__.keys()))
                )
            )
        return node.stat.mks2, node.stat.mks2_, node.stat.mks2i_

    def precompute_stat(self):
        """
        Computes and stores list of completions for each node,
        computes *mks*.

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
                pop.stat.completions = []
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
                pop.stat.merge_completions(pop.value, pop.children.values())
                pop.stat.next_nodes = pop.children
                pop.stat.update_minimum_keystroke(len(pop.value))
                if pop.parent is not None:
                    stack.append(pop.parent)
            else:
                # we'll do it again later
                stack.append(pop)

    def update_stat_dynamic(self, delta=0.8):
        """
        Must be called after @see me precompute_stat
        and computes dynamic mks (see :ref:`Dynamic Minimum Keystroke <def-mks2>`).

        :param delta: parameter :math:`\\delta` in defintion
            :ref:`Modified Dynamic KeyStroke <def-mks3>`
        :return: number of iterations to converge
        """
        for node in self.unsorted_iter():
            node.stat.init_dynamic_minimum_keystroke(len(node.value))
            node.stat.iter_ = 0
        updates = 1
        itera = 0
        while updates > 0:
            updates = 0
            stack = []
            stack.append(self)
            while len(stack) > 0:
                pop = stack.pop()
                if pop.stat.iter_ > itera:
                    continue
                updates += pop.stat.update_dynamic_minimum_keystroke(
                    len(pop.value), delta
                )
                if pop.children:
                    stack.extend(pop.children.values())
                pop.stat.iter_ += 1
            itera += 1
        return itera

    ##
    # end of methods, beginning of subclasses
    ##

    class _Stat:
        """
        Stores statistics and intermediate data about the compuation the metrics.

        It contains the following members:

        * mks0*: value of minimum keystroke
        * mks0_*: length of the prefix to obtain *mks0*
        * *mks_iter*: current iteration during the computation of mks

        * *mks1*: value of dynamic minimum keystroke
        * *mks1_*: length of the prefix to obtain *mks*
        * *mks1i_*: iteration when it was obtained

        * *mks2*: value of modified dynamic minimum keystroke
        * *mks2_*: length of the prefix to obtain *mks2*
        * *mks2i*: iteration when it converged
        """

        def merge_completions(self, prefix: int, nodes: "[CompletionTrieNode]"):
            """
            Merges list of completions and cut the list, we assume
            given lists are sorted.
            """

            class Fake:
                pass

            res = []
            indexes = [0 for _ in nodes]
            indexes.append(0)
            last = Fake()
            last.value = None
            last.stat = CompletionTrieNode._Stat()
            last.stat.completions = list(
                sorted((_.weight, _) for _ in nodes if _.leave)
            )
            nodes = list(nodes)
            nodes.append(last)

            maxl = 0
            while True:
                en = [
                    (
                        _.stat.completions[indexes[i]][0],
                        i,
                        _.stat.completions[indexes[i]][1],
                    )
                    for i, _ in enumerate(nodes)
                    if indexes[i] < len(_.stat.completions)
                ]
                if not en:
                    break
                e = min(en)
                i = e[1]
                res.append((e[0], e[2]))
                indexes[i] += 1
                maxl = max(maxl, len(res[-1][1].value))

            # maxl - len(prefix) represents the longest list
            # which reduces the number of keystrokes
            # however, as the method aggregates completions at a lower lovel,
            # we must keep longer completions for lower levels
            ind = maxl
            if len(res) > ind:
                self.completions = res[:ind]
            else:
                self.completions = res

        def update_minimum_keystroke(self, lw):
            """
            Updates minimum keystroke for the completions.

            @param      lw      prefix length
            """
            for i, wsug in enumerate(self.completions):
                sug = wsug[1]
                nl = lw + i + 1
                if not hasattr(sug.stat, "mks0") or sug.stat.mks0 > nl:
                    sug.stat.mks0 = nl
                    sug.stat.mks0_ = lw

        def update_dynamic_minimum_keystroke(self, lw, delta):
            """
            Updates dynamic minimum keystroke for the completions.

            @param      lw      prefix length
            @param      delta   parameter :math:`\\delta` in defintion
                                :ref:`Modified Dynamic KeyStroke <def-mks3>`
            @return             number of updates
            """
            self.mks_iter += 1
            update = 0
            for i, wsug in enumerate(self.completions):
                sug = wsug[1]
                if sug.leave:
                    # this is a leave so we consider the completion being part
                    # of the list of completions
                    nl = self.mks1 + i + 1
                    if sug.stat.mks1 > nl:
                        sug.stat.mks1 = nl
                        sug.stat.mks1_ = lw
                        sug.stat.mks1i_ = self.mks_iter
                        update += 1
                    nl = self.mks2 + i + 1 + delta
                    if sug.stat.mks2 > nl:
                        sug.stat.mks2 = nl
                        sug.stat.mks2_ = lw
                        sug.stat.mks2i_ = self.mks_iter
                        update += 1
                else:
                    raise RuntimeError("this case should not happen")

            # optimisation of second case of modified metric
            # in a separate function for profiling
            def second_step(update):
                if hasattr(self, "next_nodes"):
                    for _, child in self.next_nodes.items():
                        for i, wsug in enumerate(child.stat.completions):
                            sug = wsug[1]
                            if not sug.leave:
                                continue
                            nl = self.mks2 + i + 2
                            if sug.stat.mks2 > nl:
                                sug.stat.mks2 = nl
                                sug.stat.mks2_ = lw
                                sug.stat.mks2i_ = self.mks_iter
                                update += 1
                return update

            update = second_step(update)

            # finally we need to update mks, mks2 for every prefix
            # this is not necessary a leave so it does not
            # appear in the list of completions
            # but we need to update mks for these strings, we assume it just
            # requires an extra character, somehow, we propagate the values
            if hasattr(self, "next_nodes"):
                for _, n in self.next_nodes.items():
                    if not hasattr(n.stat, "mks1") or n.stat.mks1 > self.mks1 + 1:
                        n.stat.mks1 = self.mks1 + 1
                        n.stat.mks1_ = self.mks1_
                        n.stat.mks1i_ = self.mks_iter
                        update += 1
                    if not hasattr(n.stat, "mks2") or n.stat.mks2 > self.mks2 + 1:
                        n.stat.mks2 = self.mks2 + 1
                        n.stat.mks2_ = self.mks2_
                        n.stat.mks2i_ = self.mks_iter
                        update += 1

            return update

        def init_dynamic_minimum_keystroke(self, lw):
            """
            Initializes *mks* and *mks2* from from *mks0*.

            @param      lw      length of the prefix
            """
            if hasattr(self, "mks0"):
                self.mks1 = self.mks0
                self.mks1_ = self.mks0_
                self.mks_iter = 0
                self.mks1i_ = 0
                self.mks2 = self.mks0
                self.mks2_ = self.mks0_
                self.mks2i_ = 0
            else:
                self.mks0 = lw
                self.mks0_ = 0
                self.mks1 = lw
                self.mks1_ = lw
                self.mks_iter = 0
                self.mks1i_ = 0
                self.mks2 = lw
                self.mks2_ = lw
                self.mks2i_ = 0

        def str_mks0(self) -> str:
            """
            Returns a string with metric information.
            """
            if hasattr(self, "mks0"):
                return f"MKS={self.mks0} *={self.mks0_}"
            else:
                return "-"

        def str_mks(self) -> str:
            """
            Returns a string with metric information.
            """
            s0 = self.str_mks0()
            if hasattr(self, "mks1"):
                return s0 + " |'={0} *={1},{2} |\"={3} *={4},{5} |nn={6}".format(
                    self.mks1,
                    self.mks1_,
                    self.mks1i_,
                    self.mks2,
                    self.mks2i_,
                    self.mks2i_,
                    "+" if hasattr(self, "next_nodes") else "-",
                )
            else:
                return s0
