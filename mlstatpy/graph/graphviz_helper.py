"""
@file
@brief graphviz helper
"""
import os
import sys
from pyquickhelper.loghelper import run_cmd
from pyquickhelper.helpgen.conf_path_tools import find_graphviz_dot


def run_graphviz(filename, image, engine="dot"):
    """
    Run :epkg:`GraphViz`.

    @param      filename        filename which contains the graph definition
    @param      image           output image
    @param      engine          *dot* or *neato*
    @return                     output of graphviz
    """
    ext = os.path.splitext(image)[-1]
    if ext != ".png":
        raise RuntimeError("extension should be .png not " + str(ext))
    if sys.platform.startswith("win"):
        bin_ = os.path.dirname(find_graphviz_dot())
        # if bin not in os.environ["PATH"]:
        #    os.environ["PATH"] = os.environ["PATH"] + ";" + bin
        cmd = f'"{bin_}\\{engine}" -Tpng "{filename}" -o "{image}"'
    else:
        cmd = f'"{engine}" -Tpng "{filename}" -o "{image}"'
    out, err = run_cmd(cmd, wait=True)
    if len(err) > 0:
        raise RuntimeError(
            f"Unable to run Graphviz\nCMD:\n{cmd}\nOUT:\n{out}\nERR:\n{err}"
        )
    return out


def edges2gv(vertices, edges):
    """
    Converts a graph into a :epkg:`GraphViz` file format.

    @param      edges           see below
    @param      vertices        see below
    @return                     gv format

    The function creates a file ``<image>.gv``.

    .. runpython::
        :showcode:

        from mlstatpy.graph.graphviz_helper import edges2gv
        gv = edges2gv([(1, "eee", "red")],
                      [(1, 2, "blue"), (3, 4), (1, 3)])
        print(gv)

    """
    memovertex = {}
    for v in vertices:
        if isinstance(v, tuple):
            if len(v) == 1:
                memovertex[v[0]] = None
            else:
                memovertex[v[0]] = v[1:]
        else:
            memovertex[v] = None
    for edge in edges:
        i, j = edge[:2]
        if i not in memovertex:
            memovertex[i] = None
        if j not in memovertex:
            memovertex[j] = None

    li = ["digraph{"]
    for k, v in memovertex.items():
        if v is None:
            li.append(f"{k} ;")
        elif len(v) == 1:
            li.append(f'"{k}" [label="{v[0]}"];')
        elif len(v) == 2:
            li.append(f'"{k}" [label="{v[0]}",fillcolor={v[1]},color={v[1]}];')
        else:
            raise ValueError("unable to understand " + str(v))

    for edge in edges:
        i, j = edge[:2]
        if len(edge) == 2:
            li.append(f'"{i}" -> "{j}";')
        elif len(edge) == 3:
            li.append(f'"{i}" -> "{j}" [label="{edge[2]}"];')
        elif len(edge) == 4:
            li.append(f'"{i}" -> "{j}" [label="{edge[2]}",color={edge[3]}];')
        else:
            raise ValueError("unable to understand " + str(edge))
    li.append("}")

    text = "\n".join(li)
    return text


def draw_graph_graphviz(vertices, edges, image=None, engine="dot"):
    """
    Draws a graph using :epkg:`Graphviz`.

    @param      edges           see below
    @param      vertices        see below
    @param      image           output image, None, just returns the output
    @param      engine          *dot* or *neato*
    @return                     :epkg:`Graphviz` output or
                                the dot text if *image* is None

    The function creates a file ``<image>.gv`` if *image* is not None.
    ::

        edges    = [ (1,2, label, color), (3,4), (1,3), ... ]  , liste d'arcs
        vertices = [ (1, label, color), (2), ... ]  , liste de noeuds
        image = nom d'image (format png)

    """
    text = edges2gv(vertices, edges)
    if image is None:
        return text
    filename = image + ".gv"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(text)

    out = run_graphviz(filename, image, engine=engine)
    if not os.path.exists(image):
        raise FileNotFoundError(f"GraphViz failed with no reason. '{image}' not found.")
    return out
