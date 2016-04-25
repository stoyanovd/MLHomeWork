import numpy as np
import pickle
import plotly.plotly as py
import plotly.graph_objs as go

CIRCLE_SIZE = 10
BORDER_WIDTH = 3


def draw(train, test, corrects):
    x = train[train[:, 2] >= 0]
    trace0 = go.Scatter(x=x[:, 0], y=x[:, 1], name='Train' + '_' + 'True', mode='markers',
                        marker=dict(
                            size=CIRCLE_SIZE,
                            color='rgba(0, 0, 152, .8)',
                            line=dict(
                                width=BORDER_WIDTH,
                                color='rgb(0, 0, 0)'
                            )
                        ))
    x = train[train[:, 2] < 0]
    trace1 = go.Scatter(x=x[:, 0], y=x[:, 1], name='Train' + '_' + 'False', mode='markers',
                        marker=dict(
                            size=CIRCLE_SIZE,
                            color='rgba(152, 0, 0, .8)',
                            line=dict(
                                width=BORDER_WIDTH,
                                color='rgb(0, 0, 0)'
                            )
                        ))
    trace = [0 for _ in range(4)]

    def f(isPositive, isTrue):
        rx, ry = [], []
        for i, x in enumerate(test):
            if (x[2] >= 0) == isPositive and corrects[i] == isTrue:
                rx.append(x[0])
                ry.append(x[1])
        trace[2 * isPositive + isTrue] = go.Scatter(x=rx, y=ry, name='Test' + '_' + str(isPositive) + '_' + str(isTrue),
                                                    mode='markers',
                                                    marker=dict(
                                                        size=CIRCLE_SIZE,
                                                        color='rgba(' + str(152 * (not isPositive)) + ', 0, ' + str(
                                                            152 * isPositive) + ', .8)',
                                                        line=dict(
                                                            width=BORDER_WIDTH,
                                                            color='rgb(' + str(152 * (not isTrue)) + ', 0, ' + str(
                                                                152 * isTrue) + ')'
                                                        )
                                                    ))

    for a1 in [True, False]:
        for a2 in [True, False]:
            f(a1, a2)
    data = go.Data([trace0, trace1] + trace)

    unique_url = py.plot(data, filename='basic-line')
