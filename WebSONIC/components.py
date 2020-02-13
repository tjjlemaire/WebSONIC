# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2018-08-23 08:26:27
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2020-02-13 21:37:08

''' Extension of dash components. '''

import numpy as np
import dash_html_components as html
import dash_core_components as dcc
import dash_daq as daq

from PySONIC.utils import si_format


def separator():
    return html.Hr(className='separator')


def unorderedList(items):
    return dcc.Markdown(children=['''* {}'''.format('\r\n* '.join(items))])


def slider(id, bounds, n, value, disabled=False, scale='lin'):
    ''' Return linearly spaced slider. '''
    if scale == 'log':
        bounds = [np.log10(x) for x in bounds]
        value = np.log10(value)
    xmin, xmax = bounds
    return dcc.Slider(id=id, className='slider', min=xmin, max=xmax, step=(xmax - xmin) / n,
                      value=value, disabled=disabled)


def labeledSliderRow(label, id, bounds, n, value, disabled=False, scale='lin'):
    ''' Return a label:slider table row. '''
    return html.Tr(className='table-row', children=[
        html.Td(label, className='row-label'),
        html.Td(className='row-slider', children=[
            slider(id, bounds, n, value, disabled=disabled, scale=scale)]),
        html.Td(id=f'{id}-value', className='row-value')
    ])


def labeledSlidersTable(id, labels, ids, bounds, n, values, scales, disabled):
    ''' Return a table of labeled sliders. '''
    return html.Table(id=id, className='table', children=[
        labeledSliderRow(labels[i], ids[i], bounds[i], n[i], values[i],
                         scale=scales[i], disabled=disabled[i])
        for i in range(len(labels))]
    )


def panel(children):
    ''' Return a panel with contents. '''
    return html.Div(className='panel', children=children)


def collapsablePanel(title, children):
    ''' Return a collapsable panel with title and contents. '''
    if title is None:
        title = ''
    return html.Details(open=True, className='panel', children=[
        html.Summary(title, className='panel-title'), *children])


def dataRows(labels, values, units):
    ''' Return a list of label:data table rows. '''
    rows = []
    for label, (value, unit) in zip(labels, zip(values, units)):
        if value is not None:
            if unit is not None:
                datastr = '{}{}'.format(si_format(value, space=' '), unit)
            else:
                datastr = str(value)
        else:
            datastr = '---'
        rows.append(html.Tr([
            html.Td(label, className='row-label'),
            html.Td(datastr, className='row-data')]))
    return rows


def ddGraph(id, labels, values, default=None):
    ''' Return div with variable selection dropdown list and graph object. '''

    # Construct dropdown list
    dd = dcc.Dropdown(
        className='ddlist',
        id=f'graph{id}-dropdown',
        options=[{'label': label, 'value': value} for label, value in zip(labels, values)],
        value=default if default is not None else values[0]
    )

    # Construct graph object
    graph = dcc.Graph(
        id=f'graph{id}',
        className='graph',
        animate=False,
        config={
            'editable': False,
            'modeBarButtonsToRemove': ['sendDataToCloud', 'displaylogo', 'toggleSpikelines']
        },
        figure={'data': [], 'layout': {}}
    )

    # If not 1st graph, add loading state wrapper
    if id != '1':
        graph = dcc.Loading(id=f'loading-graph{id}', type='default', children=[graph])

    # Return div with dropdown and graph
    return html.Div(id=f'ddgraph{id}', className='graph-div', children=[dd, graph])
