import math
import numpy
import numpy.random as nprnd


def random_noise_image(size, ratio=0.1):
    """
    Construit une image blanche de taille *size*,
    noircit aléatoirement *ratio x nb pixels*
    pixels.

    @param      size        taille de l'image
    @param      ratio       proportion de pixels à noircir
    @return                 :epkg:`numpy:array`
    """
    img = numpy.zeros((size[1], size[0]), dtype=numpy.float32)
    nb = int(ratio * size[0] * size[1])
    xr = nprnd.randint(0, size[0] - 1, nb)
    yr = nprnd.randint(0, size[1] - 1, nb)
    img[yr, xr] = 1
    return img


def random_segment_image(image, lmin=0.1, lmax=1.0, noise=0.01, density=1.0):
    """
    Ajoute un segment aléatoire à une image.
    Génère des points le long d'un segment aléatoire.

    @param      image       :epkg:`numpy:array` (modifié par la fonction)
    @param      lmin        taille minimal du segment
    @param      lmax        taille maximam du segment
    @param      density     nombre de pixel à tirer le long de l'axe
    @param      noise       bruit
    @return                 dictionary with *size, angle, x1, y1, x2, y2, nbpoints*
    """

    def move_coordinate(x1, y1, x2, y2, X, Y):
        if x2 < 0:
            x1 -= x2
            x2 = 0
        x1 = min(max(x1, 0), X - 1)
        x2 = min(max(x2, 0), X - 1)
        y1 = min(max(y1, 0), Y - 1)
        y2 = min(max(y2, 0), Y - 1)
        size = int(((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5)
        return x1, y1, x2, y2, size

    mind = min(image.shape)
    lmin = int(mind * lmin)
    lmax = int(mind * lmax)
    size = nprnd.randint(lmin, lmax)
    angle = nprnd.random() * math.pi
    x1 = nprnd.randint(image.shape[1] - int(size * abs(math.cos(angle)) - 1))
    y1 = nprnd.randint(image.shape[0] - int(size * math.sin(angle) - 1))
    x2 = x1 + size * math.cos(angle)
    y2 = y1 + size * math.sin(angle)
    x1, y1, x2, y2, size = move_coordinate(
        x1, y1, x2, y2, image.shape[1], image.shape[0]
    )
    t = nprnd.randint(0, size, int(size * density))
    xs = t * math.cos(angle) + x1
    ys = t * math.sin(angle) + x2
    noise = nprnd.randn(xs.shape[0] * 2).reshape(xs.shape[0], 2) * noise * mind
    xs += noise[:, 0]
    ys += noise[:, 1]
    xs = numpy.maximum(xs, numpy.zeros(xs.shape[0]))
    ys = numpy.maximum(ys, numpy.zeros(xs.shape[0]))
    xs = numpy.minimum(
        xs,
        numpy.zeros(xs.shape[0]) + image.shape[1] - 1,
    )
    ys = numpy.minimum(
        ys,
        numpy.zeros(xs.shape[0]) + image.shape[0] - 1,
    )
    xs = xs.astype(numpy.int32)
    ys = ys.astype(numpy.int32)
    image[ys, xs] = 1
    res = dict(size=size, angle=angle, x1=x1, y1=y1, x2=x2, y2=y2, nbpoints=xs.shape[0])
    return res
