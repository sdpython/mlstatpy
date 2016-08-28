"""
@file
@brief About completion, simple algorithm
"""
from typing import Tuple, List, Iterator
from pyquickhelper.loghelper import noLOG
from .completion import CompletionTrieNode


class CompletionElement(object):
    """
    Element definition in a completion system,
    it contains the following members:

    * *value*: the completion
    * *weight*: a weight or a position, we assume a completion with
      a lower weight is shown at a lower position
    * *disp*: display string (no impact on the algorithm)

    * mks0*: value of minimum keystroke
    * mks0_*: length of the prefix to obtain *mks0*

    * *mks1*: value of dynamic minimum keystroke
    * *mks1_*: length of the prefix to obtain *mks1*

    * *mks2*: value of modified dynamic minimum keystroke
    * *mks2_*: length of the prefix to obtain *mks2*
    """

    __slots__ = ["value", "weight", "disp",
                 'mks0', 'mks0_',
                 'mks1', 'mks1_',
                 'mks2', 'mks2_']

    def __init__(self, value: str, weight: float=1.0, disp=None):
        """
        @param      value       value (a character)
        @param      weight      ordering (the lower, the first)
        @param      disp        original string, use this to identify the node
        """
        if not isinstance(value, str):
            raise TypeError(
                "value must be str not '{0}' - type={1}".format(value, type(value)))
        self.value = value
        self.weight = weight
        self.disp = disp

    def str_mks0(self) -> str:
        """
        return a string with metric information
        """
        if hasattr(self, "mks0"):
            return "MKS={0} *={1}".format(self.mks0, self.mks0_)
        else:
            return "-"

    def str_mks(self) -> str:
        """
        return a string with metric information
        """
        s0 = self.str_mks0()
        if hasattr(self, "mks1"):
            return s0 + " |'={0} *={1} |\"={2} *={3}".format(
                self.mks1, self.mks1_, self.mks2, self.mks2_)
        else:
            return s0

    def init_metrics(self, position: int):
        """
        initiate the metrics

        @param      position    position in the completion system when prefix is null,
                                *position starting from 0*
        """
        position += 1
        if len(self.value) <= position:
            self.mks0 = len(self.value)
            self.mks1 = len(self.value)
            self.mks2 = len(self.value)
            self.mks0_ = len(self.value)
            self.mks1_ = len(self.value)
            self.mks2_ = len(self.value)
        else:
            self.mks0 = position
            self.mks1 = position
            self.mks2 = position
            self.mks0_ = 0
            self.mks1_ = 0
            self.mks2_ = 0

    def update_metrics(self, prefix: str, position: int, improved: dict, delta: float):
        """
        update the metrics

        @param      prefix      prefix
        @param      position    position in the completion system when prefix has length k,
                                *position starting from 0*
        @param      improved    if one metrics is < to the completion length, it means
                                it can be used to improve others queries
        @param      delta       delta in the dynamic modified mks
        """
        k = len(prefix)
        pos = position + 1
        mks = k + pos
        check = False
        if mks < self.mks0:
            self.mks0 = mks
            self.mks0_ = k
            check = True
        if mks < self.mks1:
            self.mks1 = mks
            self.mks1_ = k
            check = True
        if mks < self.mks2:
            self.mks2 = mks
            self.mks2_ = k
            check = True

        if prefix in improved:
            v = improved[prefix]
            mks = v.mks1 + min(len(self.value) - len(prefix), pos)
            if mks < self.mks1:
                self.mks1 = mks
                self.mks1_ = k
                check = True
            mks = v.mks2 + min(len(self.value) - len(prefix), pos + delta)
            if mks < self.mks2:
                self.mks2 = mks
                self.mks2_ = k
                check = True
        prefix = prefix[:-1]
        if prefix in improved:
            v = improved[prefix]
            mks = v.mks2 + min(len(self.value) - len(prefix), pos + 1)
            if mks < self.mks2:
                self.mks2 = mks
                self.mks2_ = k - 1
                check = True
        if check and self.value not in improved:
            improved[self.value] = self


class CompletionSystem:
    """
    define a completion system
    """

    def __init__(self, elements: List[CompletionElement]):
        """
        fill the completion system
        """
        self._elements = [(e if isinstance(e, CompletionElement)
                           else CompletionElement(e[1], e[0])) for e in elements]

    def find(self, value: str, is_sorted=False) -> CompletionElement:
        """
        not very efficient, find an item in a the list

        @param      value       string to find
        @param      is_sorted   the function will assume the elements are sorted by
                                alphabetical order
        @return                 element or None
        """
        if is_sorted:
            raise NotImplementedError()
        else:
            for e in self:
                if e.value == value:
                    return e
            return None

    def items(self) ->Iterator[Tuple[str, CompletionElement]]:
        """
        iterate on ``(e.value, e)``
        """
        for e in self._elements:
            yield e.value, e

    def tuples(self) ->Iterator[Tuple[float, str]]:
        """
        iterate on ``(e.weight, e.value)``
        """
        for e in self._elements:
            yield e.weight, e.value

    def __len__(self) -> int:
        """
        number of elements
        """
        return len(self._elements)

    def __iter__(self) -> Iterator[CompletionElement]:
        """
        iterates over elements
        """
        for e in self._elements:
            yield e

    def sort_values(self):
        """
        sort the elements by value
        """
        self._elements = list(
            _[-1] for _ in sorted((e.value, e.weight, e) for e in self))

    def sort_weight(self):
        """
        sort the elements by value
        """
        self._elements = list(
            _[-1] for _ in sorted((e.weight, e.value, e) for e in self))

    def compare_with_trie(self, delta: float=0.8):
        """
        compare the results with the other implementation

        @param      delta       parameter *delta* in the dynamic modified mks
        @return                 None or differences
        """
        def format_diff(el, f, diff):
            s = "VALUE={0}\nS=[{1}]\nTRIE=[{2}]".format(
                el.value, el.str_mks(), f.stat.str_mks())
            if diff:
                return "-------\n{0}\n-------".format(s)
            else:
                return s

        trie = CompletionTrieNode.build(self.tuples())
        self.compute_metrics(delta=delta)
        trie.precompute_stat()
        trie.update_stat_dynamic(delta=delta)
        diffs = []
        for el in self:
            f = trie.find(el.value)
            d0 = el.mks0 - f.stat.mks0
            d1 = el.mks1 - f.stat.mks1
            d2 = el.mks2 - f.stat.mks2
            d4 = el.mks0_ - f.stat.mks0_
            if d0 != 0 or d1 != 0 or d2 != 0 or d4 != 0:
                diffs.append((el, f, format_diff(el, f, True)))
        return diffs if diffs else None

    def compute_metrics(self, filter: 'func'=None, delta: float=0.8, fLOG: 'func'=noLOG):
        """
        compute the metric for the completion itself

        @param      filter      filter function
        @param      delta       parameter *delta* in the dynamic modified mks
        @param      fLOG        logging function
        """
        self.sort_weight()
        if filter is not None:
            raise NotImplementedError("filter not None is not implemented")
        # max_length = max(len(e.value) for e in self)
        improved = {}
        displayed = {}
        for i, el in enumerate(self._elements):
            el.init_metrics(i)
            for k in range(1, len(el.value)):
                prefix = el.value[:k]
                if prefix not in displayed:
                    displayed[prefix] = 0
                else:
                    displayed[prefix] += 1
                el.update_metrics(prefix, displayed[prefix], improved, delta)
