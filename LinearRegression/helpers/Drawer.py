import numpy as np

import plotly.plotly as py
from plotly.graph_objs import Surface, Scatter3d, Scene, XAxis, YAxis, ZAxis
from plotly.graph_objs import Data, Figure, Layout


def draw(x, y, w, y0):
    def ff(x, y):
        return w[0] + w[1] * x + w[2] * y

    yt = (y)[:, np.newaxis]
    z = ff(x, yt)
    trace1 = Surface(x=x, y=y, z=z)

    trace0 = Scatter3d(x=x, y=y, z=y0)
    data = Data([trace0, trace1])

    unique_url = py.plot(data, filename='basic-line')
