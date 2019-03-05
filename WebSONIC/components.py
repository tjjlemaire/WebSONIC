# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2018-08-23 08:26:27
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-03-05 18:23:39

''' Extension of dash components. '''

import dash_html_components as html
import dash_core_components as dcc
import dash_daq as daq

from PySONIC.utils import si_format


def separator():
    return html.Hr(className='separator')


def unorderedList(items):
    return dcc.Markdown(className='ul', children=['''* {}'''.format('\r\n* '.join(items))])


def linearSlider(id, noptions, value=0, disabled=False):
    ''' Return linearly spaced slider. '''
    return dcc.Slider(id=id, min=0, max=noptions - 1, step=1, value=value, disabled=disabled)


def labeledSliderRow(label, id, noptions, value=0, disabled=False):
    ''' Return a label:slider table row. '''
    return html.Tr(className='slider-row', children=[
        html.Td(label, className='row-label'),
        html.Td(className='row-data', children=[linearSlider(id, noptions, value, disabled)])
    ])


def labeledSlidersTable(id, labels, ids, sizes, values=None):
    ''' Return a table of labeled sliders. '''
    if values is None:
        values = [0] * len(labels)
    children = []
    for i in range(len(labels)):
        children.append(labeledSliderRow(labels[i], ids[i], sizes[i], value=values[i]))
    return html.Table(id=id, className='table', children=children)


def numInputBox(id, min, max, value=None):
    ''' Return numerical input box with lwoer and upper bounds. '''
    if value is None:
        value = (min + max) / 2
    return dcc.Input(id=id, className='input-box', type='number', min=min, max=max, value=value)


def labeledInputRow(label, id, min, max, value=None):
    ''' Return a label:input table row. '''
    return html.Tr(className='input-row', children=[
        html.Td(label, className='row-label'),
        html.Td(className='row-data', children=[numInputBox(id, min, max, value)])
    ])


def labeledInputsTable(id, labels, ids, mins, maxs, values=None):
    ''' Return a table of labeled inputs. '''
    if values is None:
        values = [None] * len(labels)
    children = []
    for i in range(len(labels)):
        children.append(labeledInputRow(labels[i], ids[i], mins[i], maxs[i], values[i]))
    return html.Table(id=id, className='table', children=children)


def labeledToggleSwitch(id, labelLeft='Left', labelRight='Right', value=False, boldLabels=False):
    ''' Return a labelLeft - toggleSwitch - labelRight html div element. '''

    return html.Div(className='toggle-switch-container', children=[
        html.Span(children=[
            labelLeft,
            daq.ToggleSwitch(id=id, className='toggle-switch', value=value),
            labelRight
        ], style={'fontWeight': 'bold' if boldLabels else 'normal'})
    ])


def panel(children):
    ''' Return a panel with contents. '''
    return html.Div(className='panel', children=children)


def collapsablePanel(title, children):
    ''' Return a collapsable panel with title and contents. '''
    if title is None:
        title = ''
    return html.Details(open=1, className='panel', children=[
        html.Summary(title, className='panel-title'), *children])


def dataRows(labels, values, units):
    ''' Return a list of label:data table rows. '''
    rows = []
    for label, (value, unit) in zip(labels, zip(values, units)):
        if value is not None:
            if unit is not None:
                datastr = '{}{}'.format(si_format(value, space=' '), unit)
            else:
                datastr = '{}'.format(value)
        else:
            datastr = '---'
        rows.append(html.Tr([
            html.Td(label, className='row-label'),
            html.Td(datastr, className='row-data')]))
    return rows


def ddGraph(id, labels, values, default=None, sep=False):
    ''' Return div with variable selection dropdown list and graph object. '''
    return html.Div(id='{}-ddgraph'.format(id), className='graph-div', children=[
        # Optional separator
        html.Hr(className='graph-separator') if sep else None,

        # Dropdown list
        dcc.Dropdown(
            id='{}-dropdown'.format(id),
            options=[{'label': label, 'value': value} for label, value in zip(labels, values)],
            value=default if default is not None else values[0]),

        # Graph
        dcc.Graph(
            id='{}-graph'.format(id),
            className='graph',
            animate=False,
            config={
                'editable': False,
                'modeBarButtonsToRemove': ['sendDataToCloud', 'displaylogo', 'toggleSpikelines']
            })
    ])
