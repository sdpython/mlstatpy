import copy
import re


class _Vertex:
    __slots__ = ("nb", "label", "weight")

    def __init__(self, nb, label, weight):
        self.nb = nb  # kind of id
        self.label = label  # label
        self.weight = weight


class Vertex(_Vertex):
    """
    Defines a vertex of a graph.
    """

    def __init__(self, nb, label, weight):
        """
        constructor
        @param      nb      (int) index of the vertex
        @param      label   (str) label
        @para       weight  (float)
        """
        _Vertex.__init__(self, nb, label, weight)
        self.pair = (None, None)
        self.edges = {}
        self.predE = {}
        self.succE = {}

    def __str__(self):
        """
        usual
        """
        return f"{self.Label}"

    def __repr__(self):
        """
        usual
        """
        return f"Vertex({repr(self.nb)}, {repr(self.Label)}, {self.weight})"

    def is_vertex(self):
        """
        returns True
        """
        return True

    def is_edge(self):
        """
        returns False
        """
        return False

    @property
    def Label(self):
        """
        returns the label
        """
        return self.label


class _Edge:
    __slots__ = ("from_", "to", "label", "weight", "nb")

    def __init__(self, from_, to, label, weight):
        self.from_, self.to = from_, to
        self.nb = from_, to
        self.label = label


class Edge(_Edge):
    """
    Defines an edge of a graph.
    """

    def __init__(self, from_, to, label, weight):
        """
        @param  from_       (int)
        @param  to          (int)
        @param  label       (str)
        @param  weight      (float)

        ``'00'`` means the beginning of a graph, ``'11'`` the end.
        """
        _Edge.__init__(self, from_, to, label, weight)
        self.pair = (None, None)
        self.weight = weight
        if self.from_ == "00" and self.to == "00":
            raise AssertionError("should not happen")
        if self.from_ == "11" and self.to == "11":
            raise AssertionError("should not happen")

    def __str__(self):
        """
        usual
        """
        return f"{self.nb[0]} -> {self.nb[1]} [{self.Label}]"

    def __repr__(self):
        """
        usual
        """
        return "Edge({}, {}, {}, {})".format(
            repr(self.nb[0]),
            repr(self.nb[1]),
            repr(self.Label),
            self.weight,
        )

    def is_vertex(self):
        """
        returns False
        """
        return False

    def is_edge(self):
        """
        returns True
        """
        return True

    @property
    def Label(self):
        """
        returns the label
        """
        return self.label


class GraphDistance:
    """
    Defines a graph to compute a distance between two graphs.

    .. exref::
        :title: Compute a distance between two graphs.

        See :ref:`l-graph_distance`.

        .. runpython::
            :showcode:

            import copy
            from mlstatpy.graph import GraphDistance

            # We define two graphs as list of edges.
            graph1 = [("a", "b"), ("b", "c"), ("b", "X"), ("X", "c"),
                      ("c", "d"), ("d", "e"), ("0", "b")]
            graph2 = [("a", "b"), ("b", "c"), ("b", "X"), ("X", "c"),
                      ("c", "t"), ("t", "d"), ("d", "e"), ("d", "g")]

            # We convert them into objects GraphDistance.
            graph1 = GraphDistance(graph1)
            graph2 = GraphDistance(graph2)

            distance, graph = graph1.distance_matching_graphs_paths(
                graph2, use_min=False)

            print("distance", distance)
            print("common paths:", graph)
    """

    # graph is a dictionary
    @staticmethod
    def get_list_of_vertices(graph):
        edges = [edge[:2] for edge in graph]
        unique = {}
        for i, j in edges:
            unique[i] = unique[j] = 1
        vertices = list(unique.keys())
        vertices.sort()
        return vertices

    def __init__(
        self,
        edge_list,
        vertex_label=None,
        add_loop=False,
        weight_vertex=1.0,
        weight_edge=1.0,
    ):
        """
        constructor

        @param      edge_list       list of edges
        @param      add_loop        automatically add a loop on
                                    each vertex (an edge from a vertex to itself)
        @param      weight_vertex   weight for every vertex
        @param      weight_edge     weight for every edge
        """
        if vertex_label is None:
            vertex_label = {}
        if isinstance(edge_list, str):
            self.load_from_file(edge_list, add_loop)
        else:
            self.vertices = {}
            self.edges = {}
            self.labelBegin = "00"
            self.labelEnd = "11"
            vid = GraphDistance.get_list_of_vertices(edge_list)
            for u in vid:
                self.vertices[u] = Vertex(u, vertex_label.get(u, str(u)), weight_vertex)
            for e in edge_list:
                i, j = e[:2]
                ls = "" if len(e) < 3 else e[2]
                self.edges[i, j] = Edge(i, j, str(ls), weight_edge)
            self._private__init__(add_loop, weight_vertex, weight_edge)

    def __getitem__(self, index):
        """
        returns a vertex or an edge if no vertex with the given index was found
        @param      index   id (index) to look for
        @return             Vertex or Edge
        """
        if isinstance(index, str):
            return self.vertices[index]
        if isinstance(index, tuple):
            return self.edges[index]
        raise KeyError("unable to get element " + str(index))

    @staticmethod
    def load_from_file(filename, add_loop):
        """
        loads a graph from a file
        @param      filename        file name
        @param      add_loop         @see me __init__
        """
        with open(filename, "r") as f:
            lines = f.readlines()
        regV = re.compile('\\"?([a-z0-9_]+)\\"? *[[]label=\\"(.*)\\"[]]')
        regE = re.compile(
            '\\"?([a-z0-9_]+)\\"? *-> *\\"?([a-z0-9_]+)\\"? *[[]label=\\"(.*)\\"[]]'
        )
        edge_list = []
        vertex_label = {}
        for line in lines:
            line = line.strip("\r\n ;")
            ed = regE.search(line)
            ve = regV.search(line)
            if ed:
                g = ed.groups()
                edge_list.append((g[0], g[1], g[2]))
            elif ve:
                g = ve.groups()
                vertex_label[g[0]] = g[1]
        if not vertex_label or not edge_list:
            raise OSError(f"Unable to parse file {filename!r}.")
        return GraphDistance(edge_list, vertex_label, add_loop)

    def _private__init__(self, add_loop, weight_vertex, weight_edge):
        if add_loop:
            for k in self.vertices:
                if k not in (self.labelBegin, self.labelEnd):
                    self.edges[k, k] = Edge(k, k, "", weight_edge)
        self.connect_root_and_leave(weight_vertex, weight_edge)
        self.compute_predecessor()
        self.compute_successor()

    def connect_root_and_leave(self, weight_vertex, weight_edge):
        order = self.get_order_vertices()
        roots = [v for v, k in order.items() if k == 0]
        vert = {}
        for o in order:
            vert[o] = 0
        for k in self.edges:
            if k[0] != k[1]:
                vert[k[0]] += 1
        for r in roots:
            if self.labelBegin not in self.vertices:
                self.vertices[self.labelBegin] = Vertex(
                    self.labelBegin, self.labelBegin, weight_vertex
                )
            if r != self.labelBegin:
                self.edges[self.labelBegin, r] = Edge(
                    self.labelBegin, r, "", weight_edge
                )

        leaves = [k for k, v in vert.items() if v == 0]
        for r in leaves:
            if self.labelEnd not in self.vertices:
                self.vertices[self.labelEnd] = Vertex(
                    self.labelEnd, self.labelEnd, weight_vertex
                )
            if r != self.labelEnd:
                self.edges[r, self.labelEnd] = Edge(r, self.labelEnd, "", weight_edge)

    def get_order_vertices(self):
        edges = self.edges
        order = {}
        for k in edges:
            order[k[0]] = 0
            order[k[1]] = 0

        modif = 1
        while modif > 0:
            modif = 0
            for k in edges:
                i, j = k
                if i != j and order[j] <= order[i]:
                    order[j] = order[i] + 1
                    modif += 1

        return order

    def __str__(self):
        """
        usual
        """
        li = []
        for v in self.vertices.values():
            li.append(str(v))
        for _k, e in self.edges.items():
            li.append(str(e))
        return "\n".join(li)

    def __repr__(self):
        """
        usual
        """
        edges = ", ".join(repr(v) for _, v in sorted(self.edges.items()))
        vertices = ", ".join(
            f"'{k}': {repr(v)}" for k, v in sorted(self.vertices.items())
        )
        return f"GraphDistance(\n    [{edges}],\n    {{{vertices}}})"

    def compute_predecessor(self):
        """
        usual
        """
        pred = {}
        for i, j in self.edges:
            if j not in pred:
                pred[j] = {}
            pred[j][i, j] = 0
        for k, v in pred.items():
            for n in v:
                self.vertices[k].predE[n] = self.edges[n]

    def compute_successor(self):
        succ = {}
        for i, j in self.edges:
            if i not in succ:
                succ[i] = {}
            succ[i][i, j] = i, j
        for k, v in succ.items():
            for n in v:
                self.vertices[k].succE[n] = self.edges[n]

    def get_matching_functions(
        self, function_mach_vertices, function_match_edges, cost=False
    ):
        """
        Returns default matching functions between two vertices and two edges.

        :param function_mach_vertices: if not None, this function
            is returned, othewise, it returns a new fonction. See below.
        :param function_match_edges: if not None, this function is returned,
            othewise, it returns a new fonction. See below.
        :param cost: if True, the returned function should
            return a float, otherwise a boolean
        :return: a pair of functions

        Example for * if cost is False:

        ::

            lambda v1,v2,g1,g2,w1,w2 : v1.label == v2.label

        Example for *function_mach_vertices* if cost is True:

        ::

            def tempF1 (v1,v2,g1,g2,w1,w2) :
                if v1 is not None and not v1.is_vertex():
                    raise TypeError("should be a vertex")
                if v2 is not None and not v2.is_vertex():
                    raise TypeError("should be a vertex")
                if v1 is None and v2 is None : return 0
                elif v1 is None or v2 is None :
                    return v2.weight*w2 if v1 is None else v1.weight*w1
                else :
                    return 0 if v1.label == v2.label else (
                        0.5*(v1.weight*w1 + v2.weight*w2))

        Example for *function_match_edges* if cost is False:

        ::

            lambda e1,e2,g1,g2,w1,w2 : e1.label == e2.label and
                        (e1.from_ != e1.to or e2.from_ != e2.to) and
                        (e1.from_ != self.labelBegin or e1.to != self.labelBegin) and
                        (e1.from_ != self.labelEnd or e1.to != self.labelEnd)

        Example if cost is True:

        ::

            def tempF2 (e1,e2,g1,g2,w1,w2) :
                if e1 is not None and not e1.is_edge():
                    raise TypeError("should be an edge")
                if e2 is not None and not e2.is_edge():
                    raise TypeError("should be an edge")
                if e1 is None and e2 is None: return 0
                elif e1 is None or e2 is None:
                    return e2.weight*w2 if e1 is None else e1.weight*w1
                elif e1.label != e2.label: return 0.5*(e1.weight*w1 + e2.weight*w2)
                else :
                    lab1 = g1.vertices [e1.from_].label == g2.vertices [e2.from_].label
                    lab2 = g1.vertices [e1.to].label == g2.vertices [e2.to].label
                    if lab1 and lab2: return 0
                    else: return e1.weight*w1 + e2.weight*w2

        """
        if cost:
            if function_mach_vertices is None:

                def tempF1_vertex(v1, v2, g1, g2, w1, w2):
                    if v1 is None:
                        if v2 is None:
                            return 0.0
                        if not v2.is_vertex():
                            raise TypeError("v2 should be a vertex")
                        return v2.weight * w2
                    elif v2 is None:
                        if not v1.is_vertex():
                            raise TypeError("v1 should be a vertex")
                        if not v1.is_vertex():
                            raise TypeError("v1 should be a vertex")
                        return v1.weight * w1
                    else:
                        if not v1.is_vertex():
                            raise TypeError("v1 should be a vertex")
                        if not v2.is_vertex():
                            raise TypeError("v2 should be a vertex")
                        return (
                            0
                            if v1.label == v2.label
                            else 0.5 * (v1.weight * w1 + v2.weight * w2)
                        )

                function_mach_vertices = tempF1_vertex

            if function_match_edges is None:

                def tempF2_edge(e1, e2, g1, g2, w1, w2):
                    if e1 is not None and not e1.is_edge():
                        raise TypeError("should be an edge")
                    if e2 is not None and not e2.is_edge():
                        raise TypeError("should be an edge")
                    if e1 is None and e2 is None:
                        return 0
                    elif e1 is None or e2 is None:
                        return e2.weight * w2 if e1 is None else e1.weight * w1
                    elif e1.label != e2.label:
                        return 0.5 * (e1.weight * w1 + e2.weight * w2)
                    else:
                        lab1 = (
                            g1.vertices[e1.from_].label == g2.vertices[e2.from_].label
                        )
                        lab2 = g1.vertices[e1.to].label == g2.vertices[e2.to].label
                        if lab1 and lab2:
                            return 0
                        else:
                            return e1.weight * w1 + e2.weight * w2

                function_match_edges = tempF2_edge
        else:
            if function_mach_vertices is None:
                function_mach_vertices = (
                    lambda v1, v2, g1, g2, w1, w2: v1.label == v2.label
                )
            if function_match_edges is None:
                function_match_edges = (
                    lambda e1, e2, g1, g2, w1, w2: e1.label == e2.label
                    and (e1.from_ != e1.to or e2.from_ != e2.to)
                    and (e1.from_ != self.labelBegin or e1.to != self.labelBegin)
                    and (e1.from_ != self.labelEnd or e1.to != self.labelEnd)
                )
        return function_mach_vertices, function_match_edges

    def common_paths(
        self,
        graph2,
        function_mach_vertices=None,
        function_match_edges=None,
        noClean=False,
    ):
        function_mach_vertices, function_match_edges = self.get_matching_functions(
            function_mach_vertices, function_match_edges
        )
        g = GraphDistance([])
        vfirst = Vertex(
            self.labelBegin,
            f"{self.labelBegin}-{self.labelBegin}",
            (
                self.vertices[self.labelBegin].weight
                + graph2.vertices[self.labelBegin].weight
            )
            / 2,
        )
        g.vertices[self.labelBegin] = vfirst
        vfirst.pair = self.vertices[self.labelBegin], graph2.vertices[self.labelBegin]

        modif = 1
        while modif > 0:
            modif = 0
            add = {}
            for _k, v in g.vertices.items():
                v1, v2 = v.pair
                if not v.succE:
                    for e1 in v1.succE:
                        for e2 in v2.succE:
                            oe1 = self.edges[e1]
                            oe2 = graph2.edges[e2]
                            if function_match_edges(oe1, oe2, self, graph2, 1.0, 1.0):
                                tv1 = self.vertices[oe1.to]
                                tv2 = graph2.vertices[oe2.to]
                                if function_mach_vertices(
                                    tv1, tv2, self, graph2, 1.0, 1.0
                                ):
                                    # we have a match
                                    ii = f"{tv1.nb}-{tv2.nb}"
                                    if (
                                        tv1.nb == self.labelEnd
                                        and tv2.nb == self.labelEnd
                                    ):
                                        ii = self.labelEnd
                                    lab = (
                                        f"{tv1.label}-{tv2.label}"
                                        if tv1.label != tv2.label
                                        else tv1.label
                                    )
                                    tv = Vertex(ii, lab, (tv1.weight + tv2.weight) / 2)
                                    lab = (
                                        f"{oe1.label}-{oe2.label}"
                                        if oe1.label != oe2.label
                                        else oe1.label
                                    )
                                    ne = Edge(
                                        v.nb, tv.nb, lab, (oe1.weight + oe2.weight) / 2
                                    )
                                    add[tv.nb] = tv
                                    g.edges[ne.from_, ne.to] = ne
                                    ne.pair = oe1, oe2
                                    tv.pair = tv1, tv2
                                    v.succE[ne.from_, ne.to] = ne
                                    modif += 1
            for k, v in add.items():
                g.vertices[k] = v

        if not noClean:
            # g.connect_root_and_leave()
            g.compute_predecessor()
            g.clean_dead_ends()
        return g

    def clean_dead_ends(self):
        edgesToKeep = {}
        verticesToKeep = {}
        if self.labelEnd in self.vertices:
            v = self.vertices[self.labelEnd]
            verticesToKeep[v.nb] = False

            modif = 1
            while modif > 0:
                modif = 0
                add = {}
                for k, v in verticesToKeep.items():
                    if v:
                        continue
                    modif += 1
                    verticesToKeep[k] = True
                    for pred, vv in self.vertices[k].predE.items():
                        edgesToKeep[pred] = True
                        add[vv.from_] = verticesToKeep.get(vv.from_, False)
                for k, v in add.items():
                    verticesToKeep[k] = v

            remove = {}
            for k in self.vertices:
                if k not in verticesToKeep:
                    remove[k] = True
            for k in remove:
                del self.vertices[k]

            remove = {}
            for k in self.edges:
                if k not in edgesToKeep:
                    remove[k] = True
            for k in remove:
                del self.edges[k]
        else:
            self.vertices = {}
            self.edges = {}

    def enumerate_all_paths(self, edges_and_vertices, begin=None):
        if begin is None:
            begin = []
        if self.vertices and self.edges:
            if edges_and_vertices:
                last = begin[-1] if begin else self.vertices[self.labelBegin]
            else:
                last = (
                    self.vertices[begin[-1].to]
                    if begin
                    else self.vertices[self.labelBegin]
                )

            if edges_and_vertices and not begin:
                begin = [last]

            for ef in last.succE:
                e = self.edges[ef]
                path = copy.copy(begin)
                v = self.vertices[e.to]
                if e.to == e.from_:
                    # cycle
                    continue
                path.append(e)
                if edges_and_vertices:
                    path.append(v)
                if v.label == self.labelEnd:
                    yield path
                else:
                    yield from self.enumerate_all_paths(edges_and_vertices, path)

    def edit_distance_path(
        self,
        p1,
        p2,
        g1,
        g2,
        function_mach_vertices=None,
        function_match_edges=None,
        use_min=False,
        debug=False,
        cache=None,
    ):
        """
        Tries to align two paths from two graphs.

        :param p1: path 1 (from g1)
        :param p2: path 2 (from g2)
        :param g1: graph 1
        :param g2: graph 2
        :param function_mach_vertices: function which gives a
            distance bewteen two vertices, if None, it take the output of
            :meth:`get_matching_functions`
        :param function_match_edges: function which gives a distance bewteen
            two edges, if None, it take the output of
            :meth:`get_matching_functions`
        :param use_min: the returned is based on a edit distance,
            if this parameter is True, the returned value will be:

            ::

                if use_min :
                    n = min (len(p1), len(p2))
                    d = d*1.0 / n if n > 0 else 0

        :param debug: unused
        :param cache: to cache the costs
        :return: 2-uple: distance, aligned path
        """

        def edge_vertex_match(x, y, g1, g2, w1, w2):
            if x is None:
                if y is None:
                    raise RuntimeError("Both x and y are None.")
                return y.weight * w2
            elif y is None:
                return x.weight * w1
            return (x.weight * w1 + y.weight * w2) / 2

        def cache_cost(func, a, b, g1, g2, w1, w2):
            if cache is None:
                return func(a, b, g1, g2, w1, w2)
            key = (id(a), id(b), w1, w2)
            if key in cache:
                return cache[key]
            cost = func(a, b, g1, g2, w1, w2)
            cache[key] = cost
            return cost

        function_mach_vertices, function_match_edges = self.get_matching_functions(
            function_mach_vertices, function_match_edges, True
        )
        dist = {(-1, -1): (0, None, None)}

        if use_min:
            w1 = 1.0 / len(p1)
            w2 = 1.0 / len(p2)
        else:
            w1 = 1.0
            w2 = 1.0

        p2l = list(enumerate(p2))
        for i1, eorv1 in enumerate(p1):
            ed1 = eorv1.is_edge()
            ve1 = eorv1.is_vertex()
            for i2, eorv2 in p2l:
                np = i1, i2
                posit = [
                    ((i1 - 1, i2 - 1), (eorv1, eorv2)),
                    ((i1 - 1, i2), (eorv1, None)),
                    ((i1, i2 - 1), (None, eorv2)),
                ]

                if ed1 and eorv2.is_edge():
                    func = function_match_edges
                elif ve1 and eorv2.is_vertex():
                    func = function_mach_vertices
                else:
                    func = edge_vertex_match

                for p, co in posit:
                    if p in dist:
                        c0 = dist[p][0]
                        c1 = cache_cost(func, co[0], co[1], g1, g2, w1, w2)
                        c = c0 + c1
                        if np not in dist or c < dist[np][0]:
                            dist[np] = (c, p, co, (c0, c1))

        last = dist[len(p1) - 1, len(p2) - 1]
        path = []
        while last[1] is not None:
            path.append(last)
            last = dist[last[1]]

        path.reverse()

        d = dist[len(p1) - 1, len(p2) - 1][0]
        if use_min:
            n = min(len(p1), len(p2))
            d = d * 1.0 / n if n > 0 else 0
        return d, path

    def private_count_left_right(self, valuesInList):
        countLeft = {}
        countRight = {}
        for _k, v in valuesInList:
            i, j = v
            if i not in countRight:
                countRight[i] = {}
            countRight[i][j] = countRight[i].get(j, 0) + 1
            if j not in countLeft:
                countLeft[j] = {}
            countLeft[j][i] = countLeft[j].get(i, 0) + 1
        return countLeft, countRight

    def private_kruskal_matrix(self, matrix, reverse):
        countLeft, countRight = self.private_count_left_right(matrix)
        cleft, cright = len(countLeft), len(countRight)
        matrix.sort(reverse=reverse)
        count = max(
            max(sum(_.values()) for _ in countRight.values()),
            max(sum(_.values()) for _ in countLeft.values()),
        )
        while count > 1:
            k, v = matrix.pop()
            i, j = v
            countRight[i][j] -= 1
            countLeft[j][i] -= 1
            count = max(
                max(max(_.values()) for _ in countRight.values()),
                max(max(_.values()) for _ in countLeft.values()),
            )

        mini = min(cleft, cright)
        if len(matrix) < mini:
            raise RuntimeError(
                "impossible: the smallest set should get all "
                "its element associated to at least one coming "
                "from the other set"
            )

    def _private_string_path_matching(self, path, skipEdge=False):
        temp = []
        for p in path:
            u, v = p[2]
            if skipEdge and (
                (u is not None and u.is_edge()) or (v is not None and v.is_edge())
            ):
                continue
            su = "-" if u is None else str(u.nb)
            sv = "-" if v is None else str(v.nb)
            s = f"({su},{sv})"
            temp.append(s)
        return " ".join(temp)

    def distance_matching_graphs_paths(
        self,
        graph2,
        function_mach_vertices=None,
        function_match_edges=None,
        noClean=False,
        store=None,
        use_min=True,
        weight_vertex=1.0,
        weight_edge=1.0,
        verbose=0,
    ):
        """
        Computes an alignment between two graphs.

        :param graph2: the other graph
        :param function_mach_vertices: function which gives a distance
            between two vertices, if None, it take the output of
            :meth:`get_matching_functions`
        :param function_match_edges: function which gives a distance
            bewteen two edges, if None, it take the output of
            :meth:`get_matching_functions`
        :param noClean: if True, clean unmatched vertices and edges
        :param store: if None, does nothing, if it is a
            dictionary, the function will store
            here various information about how
            the matching was operated
        :param use_min: @see me edit_distance_path
        :param weight_vertex: a weight for every vertex
        :param weight_edge: a weight for every edge
        :param verbose: display some progress with :epkg:`tqdm`
        :return: 2 tuple (a distance, a graph containing
            the aligned paths between the two graphs)

        See :ref:`l-graph_distance`.
        """

        function_mach_vertices, function_match_edges = self.get_matching_functions(
            function_mach_vertices, function_match_edges, True
        )

        paths1 = list(self.enumerate_all_paths(True))
        paths2 = list(graph2.enumerate_all_paths(True))

        if store is not None:
            store["nbpath1"] = len(paths1)
            store["nbpath2"] = len(paths2)

        matrix_distance = {}
        if verbose > 0:
            print("[distance_matching_graphs_paths] builds matrix_distance")
            from tqdm import tqdm

            loop1 = tqdm(list(enumerate(paths1)))
        else:
            loop1 = enumerate(paths1)
        loop2 = list(enumerate(paths2))
        if verbose > 0:
            print(
                "[distance_matching_graphs_paths] len(loop1)=%d"
                % len(list(enumerate(paths1)))
            )
            print(f"[distance_matching_graphs_paths] len(loop2)={len(loop2)}")
        cache = {}
        for i1, p1 in loop1:
            for i2, p2 in loop2:
                matrix_distance[i1, i2] = self.edit_distance_path(
                    p1,
                    p2,
                    self,
                    graph2,
                    function_mach_vertices,
                    function_match_edges,
                    use_min=use_min,
                    cache=cache,
                )
        if verbose > 0:
            print(f"[distance_matching_graphs_paths] len(cache)={len(cache)}")

        if store is not None:
            store["matrix_distance"] = matrix_distance
        reduction = [(v[0], k) for k, v in matrix_distance.items()]
        if store is not None:
            store["path_mat1"] = copy.deepcopy(reduction)
        if verbose > 0:
            print("[distance_matching_graphs_paths] private_kruskal_matrix")
        self.private_kruskal_matrix(reduction, False)
        if store is not None:
            store["path_mat2"] = copy.deepcopy(reduction)

        if verbose > 0:
            print("[distance_matching_graphs_paths] pair_count_vertex")
        pair_count_edge = {}
        pair_count_vertex = {}
        for _k, v in reduction:
            path = matrix_distance[v][1]
            for el in path:
                n1, n2 = el[2]
                if n1 is not None and n2 is not None:
                    if n1.is_edge() and n2.is_edge():
                        add = n1.nb, n2.nb
                        pair_count_edge[add] = pair_count_edge.get(add, 0) + 1
                    elif n1.is_vertex() and n2.is_vertex():
                        add = n1.nb, n2.nb
                        pair_count_vertex[add] = pair_count_vertex.get(add, 0) + 1

        if store is not None:
            store["pair_count_vertex"] = pair_count_vertex
            store["pair_count_edge"] = pair_count_edge

        reduction_edge = [(v, k) for k, v in pair_count_edge.items()]
        if store is not None:
            store["edge_mat1"] = copy.copy(reduction_edge)
        self.private_kruskal_matrix(reduction_edge, True)
        if store is not None:
            store["edge_mat2"] = copy.copy(reduction_edge)

        reduction_vertex = [(v, k) for k, v in pair_count_vertex.items()]
        if store is not None:
            store["vertex_mat1"] = copy.copy(reduction_vertex)
        self.private_kruskal_matrix(reduction_vertex, True)
        if store is not None:
            store["vertex_mat2"] = copy.copy(reduction_vertex)

        if verbose > 0:
            print("[distance_matching_graphs_paths] private_count_left_right")
        count_edge_left, count_edge_right = self.private_count_left_right(
            reduction_edge
        )
        count_vertex_left, count_vertex_right = self.private_count_left_right(
            reduction_vertex
        )

        res_graph = GraphDistance([])
        doneVertex = {}
        done_edge = {}

        if verbose > 0:
            print("[distance_matching_graphs_paths] builds merged graph")
        for k, v in self.vertices.items():
            newv = Vertex(v.nb, v.label, weight_vertex)
            res_graph.vertices[k] = newv
            if v.nb in count_vertex_right:
                ind = list(count_vertex_right[v.nb].keys())[0]  # noqa: RUF015
                newv.pair = (v, graph2.vertices[ind])
                doneVertex[ind] = newv
                if newv.pair[0].label != newv.pair[1].label:
                    newv.label = f"{newv.pair[0].label}|{newv.pair[1].label}"
            else:
                newv.pair = (v, None)

        for k, v in graph2.vertices.items():
            if k in doneVertex:
                continue
            newv = Vertex(f"2a.{v.nb}", v.label, weight_vertex)
            res_graph.vertices[newv.nb] = newv
            newv.pair = (None, v)

        for k, e in self.edges.items():
            newe = Edge(e.from_, e.to, e.label, weight_edge)
            res_graph.edges[k] = newe
            if e.nb in count_edge_right:
                ind = list(count_edge_right[e.nb].keys())[0]  # noqa: RUF015
                newe.pair = (e, graph2.edges[ind])
                done_edge[ind] = newe
            else:
                newe.pair = (e, None)

        for k, e in graph2.edges.items():
            if k in done_edge:
                continue
            from_ = (
                list(count_vertex_left[e.from_].keys())[0]  # noqa: RUF015
                if e.from_ in count_vertex_left
                else f"2a.{e.from_}"
            )
            to = (
                list(count_vertex_left[e.to].keys())[0]  # noqa: RUF015
                if e.to in count_vertex_left
                else f"2a.{e.to}"
            )
            if from_ not in res_graph.vertices:
                raise RuntimeError("should not happen " + from_)
            if to not in res_graph.vertices:
                raise RuntimeError("should not happen " + to)
            newe = Edge(from_, to, e.label, weight_edge)
            res_graph.edges[newe.nb] = newe
            newe.pair = (None, e)

        if verbose > 0:
            print(
                "[distance_matching_graphs_paths] "
                "compute_predecessor, compute_successor"
            )
        res_graph.compute_predecessor()
        res_graph.compute_successor()

        allPaths = list(res_graph.enumerate_all_paths(True))

        temp = [
            sum(0 if None in _.pair else 1 for _ in p) * 1.0 / len(p) for p in allPaths
        ]
        distance = 1.0 - 1.0 * sum(temp) / len(allPaths)

        return distance, res_graph

    def draw_vertices_edges(self):
        vertices = []
        edges = []
        for k, v in self.vertices.items():
            if v.pair == (None, None) or (
                v.pair[0] is not None and v.pair[1] is not None
            ):
                vertices.append((k, v.label))
            elif v.pair[1] is None:
                vertices.append((k, "-" + v.label, "red"))
            elif v.pair[0] is None:
                vertices.append((k, "+" + v.label, "green"))
            else:
                raise RuntimeError("?")

        for _k, v in self.edges.items():
            if v.pair == (None, None) or (
                v.pair[0] is not None and v.pair[1] is not None
            ):
                edges.append((v.from_, v.to, v.label))
            elif v.pair[1] is None:
                edges.append((v.from_, v.to, "-" + v.label, "red"))
            elif v.pair[0] is None:
                edges.append((v.from_, v.to, "+" + v.label, "green"))
            else:
                raise RuntimeError("?")

        return vertices, edges
