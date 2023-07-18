"""
@file
@brief About completion, simple algorithm
"""
import time
from typing import Tuple, List, Iterator, Dict
from pyquickhelper.loghelper import noLOG
from .completion import CompletionTrieNode


class CompletionElement:
    """
    Definition of an element in a completion system,
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

    __slots__ = (
        "value",
        "weight",
        "disp",
        "mks0",
        "mks0_",
        "mks1",
        "mks1_",
        "mks2",
        "mks2_",
        "prefix",
        "_info",
    )

    def __init__(self, value: str, weight=1.0, disp=None):
        """
        constructor

        @param      value       value (a character)
        @param      weight      ordering (the lower, the first)
        @param      disp        original string, use this to identify the node
        """
        if not isinstance(value, str):
            raise TypeError(f"value must be str not '{value}' - type={type(value)}")
        self.value = value
        self.weight = weight
        self.disp = disp
        self.prefix = None
        self._info = None

    @staticmethod
    def empty_prefix():
        """
        return an instance filled with an empty prefix
        """
        if not hasattr(CompletionElement, "static_empty_prefix"):
            res = CompletionElement("", None)
            res.mks0 = res.mks1 = res.mks2 = 0
            res.mks0_ = res.mks1_ = res.mks2_ = 0
            CompletionElement.static_empty_prefix = res
            return res
        else:
            return CompletionElement.static_empty_prefix

    def __repr__(self):
        """
        usual
        """
        if self._info:
            return "CompletionElementInfo('{0}'{1}{2})".format(
                self.value,
                ", {0}".format(self.weight) if self.weight != 1 else "",
                ", disp='{0}'" if self.disp else "",
            )
        else:
            return "CompletionElement('{0}'{1}{2})".format(
                self.value,
                ", {0}".format(self.weight) if self.weight != 1 else "",
                ", disp='{0}'" if self.disp else "",
            )

    def str_mks0(self) -> str:
        """
        return a string with metric information
        """
        if hasattr(self, "mks0"):
            return f"MKS={self.mks0} *={self.mks0_}"
        else:
            return "-"

    def str_mks(self) -> str:
        """
        return a string with metric information
        """
        s0 = self.str_mks0()
        if hasattr(self, "mks1"):
            return s0 + " |'={0} *={1} |\"={2} *={3}".format(
                self.mks1, self.mks1_, self.mks2, self.mks2_
            )
        else:
            return s0

    def str_all_completions(self, maxn=10, use_precompute=True) -> str:
        """
        builds a string with all completions for all
        prefixes along the paths, this is only available
        if parameter *completions* was used when calling
        method @see me update_metrics.

        :param maxn: maximum number of completions to show
        :param use_precompute: use intermediate results built
            by @see me precompute_stat
        :return: str
        """
        rows = [f"{self.weight} -- {self.value} -- {self.str_mks()}"]
        if self._info is not None:
            rows.append("------------------")
            for el in self._info._log_imp:
                rows.append(str(el))
            for i in range(len(self.value)):
                prefix = self.value[:i]
                rows.append("------------------")
                rows.append(f"i={i} - {prefix}")
                completions = self._info._completions.get(prefix, [])
                for i2, el in enumerate(completions):
                    ar = "   " if el.value != self.value else "-> "
                    add = "{5}{0}:{1} -- {2}{4}-- {3}".format(
                        i2,
                        el.weight,
                        el.value,
                        el.str_mks(),
                        " " * (20 - len(el.value)),
                        ar,
                    )
                    rows.append(add)
        else:
            rows.append("NO INFO")
        return "\n".join(rows)

    def init_metrics(
        self, position: int, completions: List["CompletionElement"] = None
    ):
        """
        initiate the metrics

        @param      position    position in the completion system when prefix is null,
                                *position starting from 0*
        @param      completions displayed completions, if not None, the method will
                                store them in member *_completions*
        @return                 boolean which indicates there was an update
        """
        if completions is not None:
            log_imp = True

            class c:
                def __str__(self):
                    return f"{self._completions}-{self._log_imp}"

            self._info = c()
            self._info._completions = {}
            self._info._log_imp = []
            if "" not in self._info._completions:
                cut = min(15, max(10, len(self.value)), len(completions[""]))
                if len(completions[""]) >= cut:
                    self._info._completions[""] = completions[""][:cut]
                else:
                    self._info._completions[""] = completions[""].copy()
        else:
            log_imp = False

        self.prefix = CompletionElement.empty_prefix()
        position += 1
        if len(self.value) <= position:
            self.mks0 = len(self.value)
            self.mks1 = len(self.value)
            self.mks2 = len(self.value)
            self.mks0_ = len(self.value)
            self.mks1_ = len(self.value)
            self.mks2_ = len(self.value)
            return False
        else:
            self.mks0 = position
            self.mks1 = position
            self.mks2 = position
            self.mks0_ = 0
            self.mks1_ = 0
            self.mks2_ = 0
            if log_imp:
                self._info._log_imp.append(
                    (0, "mks0", position, "", f"k={0}", f"p={position}", f"it={0}")
                )
            return True

    def update_metrics(
        self,
        prefix: str,
        position: int,
        improved: dict,
        delta: float,
        completions: List["CompletionElement"] = None,
        iteration=-1,
    ):
        """
        Updates the metrics.

        :param prefix: prefix
        :param position: position in the completion system
            when prefix has length k, *position starting from 0*
        :param improved: if one metrics is < to the completion length, it means
            it can be used to improve others queries
        :param delta: delta in the dynamic modified mks
        :param completions: displayed completions, if not None, the method will
            store them in member *_completions*
        :param iteration: for debugging purpose, indicates
            when this improvment was detected
        :return: boolean which indicates there was an update
        """
        if self.prefix is not None and len(prefix) < len(self.prefix.value):
            # no need to look into it
            return False

        if completions is not None:
            log_imp = True
            if prefix not in self._info._completions:
                cut = min(15, max(10, len(self.value)), len(completions[prefix]))
                if len(completions[prefix]) >= cut:
                    self._info._completions[prefix] = completions[prefix][:cut]
                else:
                    self._info._completions[prefix] = completions[prefix].copy()
        else:
            log_imp = False

        k = len(prefix)
        pos = position + 1
        mks = k + pos
        check = False
        if mks < self.mks0:
            self.mks0 = mks
            self.mks0_ = k
            check = True
            if log_imp:
                self._info._log_imp.append(
                    (
                        1,
                        "mks0",
                        mks,
                        prefix,
                        f"k={k}",
                        f"p={position}",
                        f"it={iteration}",
                        f"last={self.prefix.value}",
                    )
                )
        elif mks == self.mks0 and self.mks0_ < k:
            self.mks0_ = k
        if mks < self.mks1:
            self.mks1 = mks
            self.mks1_ = k
            check = True
        if mks < self.mks2:
            self.mks2 = mks
            self.mks2_ = k
            check = True

        if self.prefix and len(self.prefix.value) < len(prefix):
            # we use the latest prefix available
            v = self.prefix
            dd = len(prefix) - len(v.value) + pos
            mks = v.mks1 + dd
            if mks < self.mks1:
                self.mks1 = mks
                self.mks1_ = k
                check = True
                if log_imp:
                    self._info._log_imp.append(
                        (
                            4,
                            "mks1",
                            mks,
                            prefix,
                            f"k={k}",
                            f"p={position}",
                            f"it={iteration}",
                            f"last={self.prefix.value}",
                        )
                    )
            mks = v.mks2 + dd
            if mks < self.mks2:
                self.mks2 = mks
                self.mks2_ = k
                check = True
                if log_imp:
                    self._info._log_imp.append(
                        (
                            5,
                            "mks2",
                            mks,
                            prefix,
                            f"k={k}",
                            f"p={position}",
                            f"it={iteration}",
                            f"last={self.prefix.value}",
                        )
                    )
        if prefix in improved:
            v = improved[prefix]
            self.prefix = v
            mks = v.mks1 + min(len(self.value) - len(prefix), pos)
            if mks < self.mks1:
                self.mks1 = mks
                self.mks1_ = k
                check = True
                if log_imp:
                    self._info._log_imp.append(
                        (
                            2,
                            "mks1",
                            mks,
                            prefix,
                            f"k={k}",
                            f"p={position}",
                            f"it={iteration}",
                            f"last={self.prefix.value}",
                        )
                    )
            mks = v.mks2 + min(len(self.value) - len(prefix), pos + delta)
            if mks < self.mks2:
                self.mks2 = mks
                self.mks2_ = k
                check = True
                if log_imp:
                    self._info._log_imp.append(
                        (
                            3,
                            "mks2",
                            mks,
                            prefix,
                            f"k={k}",
                            f"p={position}",
                            f"it={iteration}",
                            f"last={self.prefix.value}",
                        )
                    )

        if log_imp and self.prefix and self.prefix.value != "":
            self._info._log_imp.append(self.prefix)
        return check


class CompletionSystem:
    """
    define a completion system
    """

    def __init__(self, elements: List[CompletionElement]):
        """
        fill the completion system
        """

        def create_element(i, e):
            if isinstance(e, CompletionElement):
                return e
            if isinstance(e, tuple):
                return CompletionElement(e[1], e[0] if e[0] else i)
            return CompletionElement(e, i)

        self._elements = [create_element(i, e) for i, e in enumerate(elements)]

    def __getitem__(self, i):
        """
        Returns ``elements[i]``.
        """
        return self._elements[i]

    def find(self, value: str, is_sorted=False) -> CompletionElement:
        """
        Not very efficient, finds an item in a the list.

        :param value: string to find
        :param is_sorted: the function will assume the elements are sorted by
            alphabetical order
        :return: element or None
        """
        if is_sorted:
            raise NotImplementedError(  # pragma: no cover
                "No optimisation for the sorted case."
            )
        for e in self:
            if e.value == value:
                return e
        return None

    def items(self) -> Iterator[Tuple[str, CompletionElement]]:
        """
        Iterates on ``(e.value, e)``.
        """
        for e in self._elements:
            yield e.value, e

    def tuples(self) -> Iterator[Tuple[float, str]]:
        """
        Iterates on ``(e.weight, e.value)``.
        """
        for e in self._elements:
            yield e.weight, e.value

    def __len__(self) -> int:
        """
        Number of elements.
        """
        return len(self._elements)

    def __iter__(self) -> Iterator[CompletionElement]:
        """
        Iterates over elements.
        """
        for e in self._elements:
            yield e

    def sort_values(self):
        """
        sort the elements by value
        """
        self._elements = list(
            _[-1] for _ in sorted((e.value, e.weight, e) for e in self)
        )

    def sort_weight(self):
        """
        Sorts the elements by value.
        """
        self._elements = list(
            _[-1] for _ in sorted((e.weight, e.value, e) for e in self)
        )

    def compare_with_trie(self, delta=0.8, fLOG=noLOG):
        """
        Compares the results with the other implementation.

        @param      delta       parameter *delta* in the dynamic modified mks
        @param      fLOG        logging function
        @return                 None or differences
        """

        def format_diff(el, f, diff):
            s = (
                "VALUE={0}\nSYST=[{1}]\nTRIE=[{2}]\nMORE SYSTEM:"
                "\n{3}\n######\nMORE TRIE:\n{4}"
            ).format(
                el.value,
                el.str_mks(),
                f.stat.str_mks(),
                el.str_all_completions(),
                f.str_all_completions(),
            )
            if diff:
                return f"-------\n{s}\n-------"
            return s

        trie = CompletionTrieNode.build(self.tuples())
        self.compute_metrics(delta=delta, fLOG=fLOG, details=True)
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
                diffs.append((d0, d1, d2, d4, el, f, format_diff(el, f, True)))
        if diffs:
            diffs.sort(key=str)
            return diffs
        else:
            return None

    def to_dict(self) -> Dict[str, CompletionElement]:
        """
        Returns a dictionary.
        """
        return {el.value: el for el in self}

    def compute_metrics(
        self, ffilter=None, delta=0.8, details=False, fLOG=noLOG
    ) -> int:
        """
        Computes the metric for the completion itself.

        @param      ffilter     filter function
        @param      delta       parameter *delta* in the dynamic modified mks
        @param      details     log more details about displayed completions
        @param      fLOG        logging function
        @return                 number of iterations

        The function ends by sorting the set of completion by alphabetical order.
        """
        self.sort_weight()
        if ffilter is not None:
            raise NotImplementedError(  # pragma: no cover
                "ffilter not None is not implemented"
            )
        if details:
            store_completions = {"": []}

        improved = {}
        to = time.perf_counter()
        fLOG("init_metrics:", len(self))
        for i, el in enumerate(self._elements):
            if details:
                store_completions[""].append(el)
                r = el.init_metrics(i, store_completions)
            else:
                r = el.init_metrics(i)
            if r and el.value not in improved:
                improved[el.value] = el
        t = time.perf_counter()
        fLOG(f"interation 0: #={len(self)} dt={t - to} - log details={details}")

        updates = 1
        it = 1
        while updates > 0:
            displayed = {}
            updates = 0
            for i, el in enumerate(self._elements):
                for k in range(0, len(el.value)):
                    prefix = el.value[:k]
                    if prefix not in displayed:
                        displayed[prefix] = 0
                        if details:
                            store_completions[prefix] = [el]
                    else:
                        displayed[prefix] += 1
                        if details:
                            store_completions[prefix].append(el)
                    r = el.update_metrics(
                        prefix,
                        displayed[prefix],
                        improved,
                        delta,
                        completions=(store_completions if details else None),
                        iteration=it,
                    )
                    if r:
                        if el.value not in improved:
                            improved[el.value] = el
                        updates += 1
            t = time.perf_counter()
            fLOG(f"interation {it}: updates={updates} dt={t - to}")
            it += 1

        self.sort_values()
        return it - 1

    def enumerate_test_metric(
        self, qset: Iterator[Tuple[str, float]]
    ) -> Iterator[Tuple[CompletionElement, CompletionElement]]:
        """
        Evaluates the completion set on a set of queries,
        the function returns a list of @see cl CompletionElement
        with the three metrics :math:`M`, :math:`M'`, :math:`M"`
        for these particular queries.

        :param qset: list of tuple(str, float) = (query, weight)
        :return: list of tuple of @see cl CompletionElement,
            the first one is the query, the second one is the None or
            the matching completion

        The method @see me compute_metric needs to be called first.
        """
        qset = sorted(qset)
        current = 0
        for query, weight in qset:
            while current < len(self) and self[current].value <= query:
                current += 1
            ind = current - 1
            el = CompletionElement(query, weight)
            if ind >= 0:
                inset = self[ind]
                le = len(inset.value)
                if le <= len(query) and inset.value == query[:le]:
                    if le == len(query):
                        found = inset
                        el.mks0 = inset.mks0
                        el.mks1 = inset.mks1
                        el.mks2 = inset.mks2
                        el.mks0_ = len(query)
                        el.mks1_ = len(query)
                        el.mks2_ = len(query)
                    else:
                        found = None
                        el.mks0 = 0
                        el.mks0_ = 0
                        el.mks1 = inset.mks1 + len(query) - le
                        el.mks1_ = le
                        el.mks2 = inset.mks2 + len(query) - le
                        el.mks2_ = le
                else:
                    found = None
                    el.mks0 = len(query)
                    el.mks1 = len(query)
                    el.mks2 = len(query)
                    el.mks0_ = len(query)
                    el.mks1_ = len(query)
                    el.mks2_ = len(query)
            else:
                found = None
                el.mks0 = len(query)
                el.mks1 = len(query)
                el.mks2 = len(query)
                el.mks0_ = len(query)
                el.mks1_ = len(query)
                el.mks2_ = len(query)

            yield el, found

    def test_metric(self, qset: Iterator[Tuple[str, float]]) -> Dict[str, float]:
        """
        Evaluates the completion set on a set of queries,
        the function returns a dictionary with the aggregated metrics and
        some statistics about them.

        @param      qset        list of tuple(str, float) = (query, weight)
        @return                 list of @see cl CompletionElement

        The method @see me compute_metric needs to be called first.
        It then calls @see me enumerate_metric.
        """
        res = dict(mks0=0.0, mks1=0.0, mks2=0.0, sum_weights=0.0, sum_wlen=0.0, n=0)
        hist = {k: {} for k in {"mks0", "mks1", "mks2", "l"}}  # pylint: disable=C0208
        wei = {k: {} for k in hist}
        res["hist"] = hist
        res["histnow"] = wei

        for el, _ in self.enumerate_test_metric(qset):
            le = len(el.value)
            w = el.weight
            res["mks0"] += w * el.mks0
            res["mks1"] += w * el.mks1
            res["mks2"] += w * el.mks2
            res["sum_weights"] += w
            res["sum_wlen"] += w * le
            res["n"] += 1

            if el.mks0 not in hist["mks0"]:
                hist["mks0"][el.mks0] = w
                wei["mks0"][el.mks0] = 1
            else:
                hist["mks0"][el.mks0] += w
                wei["mks0"][el.mks0] += 1
            if el.mks1 not in hist["mks1"]:
                hist["mks1"][el.mks1] = w
                wei["mks1"][el.mks1] = 1
            else:
                hist["mks1"][el.mks1] += w
                wei["mks1"][el.mks1] += 1
            if el.mks2 not in hist["mks2"]:
                hist["mks2"][el.mks2] = w
                wei["mks2"][el.mks2] = 1
            else:
                hist["mks2"][el.mks2] += w
                wei["mks2"][el.mks2] += 1
            if le not in hist["l"]:
                hist["l"][le] = w
                wei["l"][le] = 1
            else:
                hist["l"][le] += w
                wei["l"][le] += 1
        return res
