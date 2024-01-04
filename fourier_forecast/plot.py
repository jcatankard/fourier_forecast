from plotly.subplots import make_subplots
from numpy.typing import NDArray
from typing import Any, Optional
import plotly.graph_objs as go
import numpy as np


FIGSIZE = (900, 300)
COMPONENTS_TYPE = dict[str, dict[str, Any]]


def plot_seasonality_components(m, figsize: tuple[int, int] = FIGSIZE) -> go.Figure:
    components = get_seasonality_components(m)
    fig = create_fig(components, figsize)
    return fig


def plot_regressor_components(m,
                              regressor_names: Optional[list[str]] = None,
                              figsize: tuple[int, int] = FIGSIZE
                              ) -> go.Figure:
    regressor_names = list(map(str, range(m.n_regressors))) if regressor_names is None else regressor_names
    components = get_regressor_components(m, regressor_names)
    fig = create_fig(components, figsize)
    return fig


def plot_components(m, figsize: tuple[int, int] = FIGSIZE) -> go.Figure:
    trend = get_trend_component(m)
    seasonality = get_seasonality_components(m)
    extra_regressors = get_extra_regressors_component(m)
    components = {
        **trend, **seasonality, **extra_regressors
    }
    fig = create_fig(components, figsize)
    return fig


def get_trend_component(m) -> COMPONENTS_TYPE:
    trend = m.x_[:, 0] * m.params_[0] + m.x_[:, 1] * m.params_[1]
    name = 'trend'
    return {name: {
        'trace': get_trace(name, m.ds, trend),
        'xaxis': go.layout.XAxis(range=[min(m.ds), max(m.ds)]),
        'yaxis': go.layout.YAxis(title=go.layout.yaxis.Title(text=name))
    }
    }


def get_extra_regressors_component(m) -> COMPONENTS_TYPE:
    regs = m.x_[:, -m.n_regressors:] @ m.params_[-m.n_regressors:]
    name = 'extra_regressors'
    return {name: {
        'trace': get_trace(name, m.ds, regs),
        'xaxis': go.layout.XAxis(range=[min(m.ds), max(m.ds)]),
        'yaxis': go.layout.YAxis(title=go.layout.yaxis.Title(text=name))
    }
    }


def get_seasonality_components(m) -> COMPONENTS_TYPE:
    start_col = 2
    components = {}
    for periods, n_terms in m.seasonality_terms.items():
        n_to_plot = np.ceil(periods).astype(np.int64)

        end_col = start_col + n_terms * 2
        y = m.x_[: n_to_plot, start_col: end_col] @ m.params_[start_col: end_col]

        name = f'seasonality: periods={periods}'

        components[name] = {
            'trace': get_trace(name, np.arange(n_to_plot), y),
            'xaxis': go.layout.XAxis(range=[0, n_to_plot - 1]),
            'yaxis': go.layout.YAxis(title=go.layout.yaxis.Title(text=name))
        }

        start_col = end_col

    return components


def get_regressor_components(m, regressor_names: list[str]) -> COMPONENTS_TYPE:
    start_col = m.x_.shape[1] - m.n_regressors
    components = {}
    for i, name in enumerate(regressor_names):
        col = start_col + i
        components[name] = {
            'trace': get_trace(name, m.ds, m.x_[:, col] * m.params_[col]),
            'xaxis': go.layout.XAxis(range=[min(m.ds), max(m.ds)]),
            'yaxis': go.layout.YAxis(title=go.layout.yaxis.Title(text=name))
        }
    return components


def get_trace(name: str, x: NDArray, y: NDArray) -> go.Scatter:
    return go.Scatter(name=name, x=x, y=y, mode='lines', line=go.scatter.Line(width=2))


def create_fig(components: dict[str, Any], figsize: tuple[int, int] = FIGSIZE) -> go.Figure:
    fig = make_subplots(rows=len(components), cols=1, print_grid=False)
    fig['layout'].update(go.Layout(
        showlegend=False,
        width=figsize[0],
        height=figsize[1] * len(components)
    ))
    for i, name in enumerate(components):
        if i == 0:
            xaxis = fig['layout']['xaxis']
            yaxis = fig['layout']['yaxis']
        else:
            xaxis = fig['layout']['xaxis{}'.format(i + 1)]
            yaxis = fig['layout']['yaxis{}'.format(i + 1)]
        xaxis.update(components[name]['xaxis'])
        yaxis.update(components[name]['yaxis'])

        fig.add_trace(components[name]['trace'], i + 1, 1)
    return fig
