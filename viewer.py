#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2017-06-22 16:57:14
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2018-01-19 15:26:54

''' Layout and callbacks of the web app. '''

import os
import time
import pickle
import urllib
import numpy as np
import dash
from dash.dependencies import Input, Output, State
import dash_auth
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import colorlover as cl
import pandas as pd

from server import server, static_route, stylesheets
from sftp import channel, remoteroot
from login import VALID_USERNAME_PASSWORD_PAIRS

from PointNICE.solvers import SolverElec, SolverUS, detectSpikes, runEStim, runAStim
from PointNICE.utils import getNeuronsDict
from PointNICE.plt import getPatchesLoc
from PointNICE.constants import SPIKE_MIN_DT, SPIKE_MIN_QAMP, SPIKE_MIN_VAMP


# -------------------------------- PLOT VARIABLES --------------------------------

# Define output variables
charge = {'names': ['Qm'], 'desc': 'charge density', 'label': 'charge', 'unit': 'nC/cm2',
          'factor': 1e5, 'min': -90, 'max': 50}
potential = {'names': ['Vm'], 'desc': 'membrane potential', 'label': 'potential', 'unit': 'mV',
             'factor': 1e0, 'min': -150, 'max': 50}
deflection = {'names': ['Z'], 'desc': 'leaflets deflection', 'label': 'deflection', 'unit': 'nm',
              'factor': 1e9, 'min': -0.1, 'max': 0.5}
gas = {'names': ['ng'], 'desc': 'gas content', 'label': 'gas', 'unit': '1e-22 mol',
       'factor': 1e22, 'min': 1.5, 'max': 2.0}

iNa_gates = {'names': ['m', 'h'], 'desc': 'iNa gates opening', 'label': 'iNa gates', 'unit': '-',
             'factor': 1, 'min': -0.1, 'max': 1.1}
iK_gate = {'names': ['n'], 'desc': 'iK gate opening', 'label': 'iK gate', 'unit': '-',
           'factor': 1, 'min': -0.1, 'max': 1.1}
iM_gate = {'names': ['p'], 'desc': 'iM gate opening', 'label': 'iM gate', 'unit': '-',
           'factor': 1, 'min': -0.1, 'max': 1.1}
iCa_gates = {'names': ['s', 'u'], 'desc': 'iCa gates opening', 'label': 'iCa gates', 'unit': '-',
             'factor': 1, 'min': -0.1, 'max': 1.1}
iH_gates = {'names': ['O', 'OL'], 'desc': 'iH gates opening', 'label': 'iH gates', 'unit': '-',
            'factor': 1, 'min': -0.1, 'max': 1.1}
iH_reg_factor = {'names': ['P0'], 'desc': 'iH regulating factor activation',
                 'label': 'iH reg.', 'unit': '-', 'factor': 1, 'min': -0.1, 'max': 1.1}
Ca_conc = {'names': ['C_Ca'], 'desc': 'sumbmembrane Ca2+ concentration', 'label': '[Ca2+]',
           'unit': 'uM', 'factor': 1e6, 'min': 0, 'max': 150.0}


# Define neurons with specific output variables
neurons = {
    'RS': {
        'desc': 'Cortical regular-spiking neuron',
        'vars_US': [charge, deflection, gas, iNa_gates, iK_gate, iM_gate],
        'vars_elec': [potential, iNa_gates, iK_gate, iM_gate]
    },
    'FS': {
        'desc': 'Cortical fast-spiking neuron',
        'vars_US': [charge, deflection, gas, iNa_gates, iK_gate, iM_gate],
        'vars_elec': [potential, iNa_gates, iK_gate, iM_gate]
    },
    'LTS': {
        'desc': 'Cortical, low-threshold spiking neuron',
        'vars_US': [charge, deflection, gas, iNa_gates, iK_gate, iM_gate, iCa_gates],
        'vars_elec': [potential, iNa_gates, iK_gate, iM_gate, iCa_gates]
    },
    'RE': {
        'desc': 'Thalamic reticular neuron',
        'vars_US': [charge, deflection, gas, iNa_gates, iK_gate, iCa_gates],
        'vars_elec': [potential, iNa_gates, iK_gate, iCa_gates]
    },
    'TC': {
        'desc': 'Thalamo-cortical neuron',
        'vars_US': [charge, deflection, gas, iNa_gates, iK_gate, iCa_gates, iH_reg_factor, Ca_conc],
        'vars_elec': [potential, iNa_gates, iK_gate, iCa_gates, iH_reg_factor, Ca_conc]
    }
}


# Set plotting parameters
ngraphs = 3
colorset = cl.scales[str(2 * ngraphs + 1)]['qual']['Set1']
del colorset[5]
tmin_plot = -5  # ms
tmax_plot = 350  # ms


# -------------------------------- INPUT RANGES & DEFAULT VARIABLES --------------------------------


# Define parameter ranges for input sliders
diams = [16.0, 32.0, 64.0]  # nm
US_freqs = [200, 400, 600, 800, 1000]  # kHz
US_amps = [10, 20, 40, 80, 150, 300, 600]  # kPa
elec_amps = [-30, -20, -15, -10, -5, -2, 2, 5, 10, 15, 20, 30]  # mA/m2
durs = [20, 40, 60, 80, 100, 150, 200, 250, 300]  # ms
PRFs = [0.1, 0.2, 0.5, 1, 2, 5, 10]  # kHz
DFs = [0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 0.75, 1]

# Define default and initial parameters
default_cell = {'neuron': 'RS', 'diameter': 1}
default_US = {'freq': 1, 'amp': 4, 'dur': 2, 'PRF': 3, 'DF': 7}
default_elec = {'amp': 10, 'dur': 2, 'PRF': 3, 'DF': 7}
modalities = {'US': 1, 'elec': 2}
current_cell = default_cell
current_modality = modalities['US']
current_stim = default_US if current_modality == modalities['US'] else default_elec
default_vars = 'vars_US' if current_modality == modalities['US'] else 'vars_elec'
current_stim = None

# Initialize global variables
localfilepath = None
df = None
previous_n_US_submits = 0
previous_n_elec_submits = 0
is_submit = False
iprop = 0
is_submit_prop = False
localdir = '{}/tmp'.format(os.getcwd()).replace('\\', '/')

mechanisms = getNeuronsDict()
solver_elec = SolverElec()

n_updates_submit = 0

# -------------------------------- APPLICATION --------------------------------

# Create Dash app
app = dash.Dash(
    name='viewer',
    server=server,
    url_base_pathname='/viewer',
    csrf_protect=True
)
app.title = 'TNEWebNICE viewer'


# Protect app with login
auth = dash_auth.BasicAuth(app, VALID_USERNAME_PASSWORD_PAIRS)


# -------------------------------- LAYOUT --------------------------------


# Load internal style sheets
for stylesheet in stylesheets:
    app.css.append_css({"external_url": "/css/{}".format(stylesheet)})


app.layout = html.Div([

    # Favicon
    # html.Link(rel='shortcut icon', href=static_route + 'icon')

    # Header
    html.Div([
        html.Div(
            [html.A(
                html.Img(src='{}EPFL.png'.format(static_route), className='logo'),
                href='https://www.epfl.ch')
             # html.A(
             #    html.Img(src='{}TNE.png'.format(static_route), className='logo'),
             #    href='https://tne.epfl.ch')
            ],
            className='header-side', id='header-left'
        ),
        html.Div([
            html.H1('Ultrasound vs. Electrical stimulation', className='header-txt'),
            html.H3(['Exploring predictions of the ',
                     html.I('NICE'),
                     ' and ',
                     html.I('Hodgkin-Huxley'),
                     ' models'], className='header-txt'),
            # html.Img(src='{}nbls.svg'.format(static_route), id='main-logo')
        ], id='header-middle'),
        html.Div(
            [html.A(
                html.Img(src='{}ITIS.svg'.format(static_route), className='logo'),
                href='https://www.itis.ethz.ch')],
            className='header-side', id='header-right'
        )
    ], id='header'),

    html.Hr(className='separator'),

    # Main div
    html.Div([

        # Left side
        html.Div([

            # Cell parameters panel (collapsable)
            html.Details([
                html.Summary('Cell parameters', className='panel-title'),
                html.Table([
                    html.Tr([
                        html.Td('Cell type', style={'width': '35%'}),
                        html.Td(
                            dcc.Dropdown(
                                id='mechanism-type',
                                options=[{'label': v['desc'], 'value': k}
                                         for k, v in neurons.items()],
                                value=default_cell['neuron']
                            ), style={'width': '65%'}
                        )
                    ]),
                    html.Tr([
                        html.Td('Membrane mechanism'),
                        html.Td(html.Img(id='neuron-mechanism', style={'width': '100%'}))
                    ]),
                    html.Tr([
                        html.Td('Sonophore diameter'),
                        html.Td(
                            dcc.Slider(
                                id='diam-slider', min=0, max=len(diams) - 1, step=1,
                                value=default_cell['diameter'],
                                marks={i: '{:.0f} nm'.format(diams[i]) if i == default_cell['diameter']
                                          else '' for i in range(len(diams))},
                                disabled=True,
                            )
                        )
                    ], className='slider-row')
                ], className='table'),
            ], open=1, className='panel'),


            # Stim parameters panel (collapsable)
            html.Details([
                html.Summary('Stimulation parameters', className='panel-title'),

                dcc.Tabs(
                    tabs=[{'label': 'Ultrasound', 'value': modalities['US']},
                          {'label': 'Electricity', 'value': modalities['elec']}],
                    value=current_modality,
                    id='modality-tabs'
                ),

                dcc.Checklist(
                    options=[{'label': 'custom', 'value': 'custom'}],
                    values=[],
                    id='custom-params-check'
                ),

                html.Table([
                    html.Tr([
                        html.Td('Amplitude', style={'width': '30%'}),
                        html.Td(
                            dcc.Slider(
                                id='elec-amp-slider',
                                min=0, max=len(elec_amps) - 1, step=1, value=default_elec['amp']
                            ), style={'width': '70%'}
                        )
                    ], className='slider-row'),
                    html.Tr([
                        html.Td('Duration'),
                        html.Td(
                            dcc.Slider(
                                id='elec-dur-slider',
                                min=0, max=len(durs) - 1, step=1, value=default_elec['dur']
                            )
                        )
                    ], className='slider-row'),
                    html.Tr([
                        html.Td('PRF'),
                        html.Td(
                            dcc.Slider(
                                id='elec-PRF-slider',
                                min=0, max=len(PRFs) - 1, step=1, value=default_elec['PRF']
                            )
                        )
                    ], className='slider-row'),
                    html.Tr([
                        html.Td('Duty cycle'),
                        html.Td(
                            dcc.Slider(
                                id='elec-DF-slider',
                                min=0, max=len(DFs) - 1, step=1, value=default_elec['DF']
                            )
                        )
                    ], className='slider-row')
                ], id='elec-slider-table', className='table', hidden=0),


                html.Div([
                    html.Table([
                        html.Tr([
                            html.Td('Amplitude (mA/m2)', style={'width': '30%'}),
                            html.Td(dcc.Input(id='elec-amp-input', className='input-box',
                                              type='number', min=-100.0, max=100.0, value=10.0),
                                    style={'width': '70%'})
                        ], className='input-row'),
                        html.Tr([
                            html.Td('Duration (ms)'),
                            html.Td(dcc.Input(id='elec-dur-input', className='input-box',
                                              type='number', min=0.0, max=350.0, value=50.0))
                        ], className='input-row'),
                        html.Tr([
                            html.Td('PRF (kHz)'),
                            html.Td(dcc.Input(id='elec-PRF-input', className='input-box',
                                              type='number', min=0.001, max=10.0, value=0.1))
                        ], className='input-row'),
                        html.Tr([
                            html.Td('Duty cycle (%)'),
                            html.Td(dcc.Input(id='elec-DF-input', className='input-box',
                                              type='number', min=0.0, max=100.0, value=100.0))
                        ], className='input-row')
                    ], className='table'),

                    html.Button('Submit', id='elec-input-submit', className='submit-button')

                ], id='elec-input-table', className='input-div', hidden=0),



                html.Table([
                    html.Tr([
                        html.Td('Frequency', style={'width': '30%'}),
                        html.Td(
                            dcc.Slider(
                                id='US-freq-slider',
                                min=0, max=len(US_freqs) - 1, step=1, value=default_US['freq']
                            ), style={'width': '70%'}
                        ),
                    ], className='slider-row'),
                    html.Tr([
                        html.Td('Amplitude'),
                        html.Td(
                            dcc.Slider(
                                id='US-amp-slider',
                                min=0, max=len(US_amps) - 1, step=1, value=default_US['amp']
                            )
                        )
                    ], className='slider-row'),
                    html.Tr([
                        html.Td('Duration'),
                        html.Td(
                            dcc.Slider(
                                id='US-dur-slider',
                                min=0, max=len(durs) - 1, step=1, value=default_US['dur']
                            )
                        )
                    ], className='slider-row'),
                    html.Tr([
                        html.Td('PRF'),
                        html.Td(
                            dcc.Slider(
                                id='US-PRF-slider',
                                min=0, max=len(PRFs) - 1, step=1, value=default_US['PRF']
                            )
                        )
                    ], className='slider-row'),
                    html.Tr([
                        html.Td('Duty cycle'),
                        html.Td(
                            dcc.Slider(
                                id='US-DF-slider',
                                min=0, max=len(DFs) - 1, step=1, value=default_US['DF']
                            )
                        )
                    ], className='slider-row')
                ], id='US-slider-table', className='table', hidden=0),


                html.Div([
                    html.Table([
                        html.Tr([
                            html.Td('Frequency (kHz)', style={'width': '30%'}),
                            html.Td(dcc.Input(id='US-freq-input', className='input-box',
                                              type='number', min=100.0, max=1000.0, value=100.0),
                                    style={'width': '70%'})
                        ], className='input-row'),
                        html.Tr([
                            html.Td('Amplitude (kPa)'),
                            html.Td(dcc.Input(id='US-amp-input', className='input-box',
                                              type='number', min=0.0, max=650.0, value=100.0))
                        ], className='input-row'),
                        html.Tr([
                            html.Td('Duration (ms)'),
                            html.Td(dcc.Input(id='US-dur-input', className='input-box',
                                              type='number', min=0.0, max=350.0, value=50.0))
                        ], className='input-row'),
                        html.Tr([
                            html.Td('PRF (kHz)'),
                            html.Td(dcc.Input(id='US-PRF-input', className='input-box',
                                              type='number', min=0.001, max=10.0, value=0.1))
                        ], className='input-row'),
                        html.Tr([
                            html.Td('Duty cycle (%)'),
                            html.Td(dcc.Input(id='US-DF-input', className='input-box',
                                              type='number', min=0.0, max=100.0, value=100.0))
                        ], className='input-row'),
                    ], className='table'),

                    html.Button('Submit', id='US-input-submit', className='submit-button')

                ], id='US-input-table', className='input-div', hidden=0),

            ], open=1, className='panel'),


            # Output metrics panel (collapsable)
            html.Details([
                html.Summary('Output metrics', className='panel-title'),
                html.Table(id='info-table', className='table')
            ], open=1, className='panel'),

        ], id='left-div', className='grid-div'),

        # Right side
        html.Div([

            # Graphs panel
            html.Div(
                [
                    html.H5('Neural response', className='panel-title', id='output-panel-title'),

                    *[html.Div(
                        [

                            # html.Summary('graph {}'.format(i + 1), className='add-graph'),

                            html.Hr(className='graph-separator') if i > 0 else None,

                            # Dropdown list
                            dcc.Dropdown(
                                id='output-dropdown-{}'.format(i + 1),

                                options=[
                                    {'label': v['desc'],
                                     'value': v['label']}
                                    for v in neurons[default_cell['neuron']][default_vars]
                                ],
                                value=neurons[default_cell['neuron']][default_vars][i]['label'],
                            ),

                            # Graph
                            dcc.Graph(
                                id='output-curve-{}'.format(i + 1),
                                style={'height': '15em'},
                                animate=False,
                                config={
                                    'editable': True,
                                    'modeBarButtonsToRemove': ['sendDataToCloud', 'displaylogo']
                                }
                            ),
                        ],
                        # open=1 if i < 1 else 0,
                        id='output-{}'.format(i),
                        className='graph-div')
                      for i in range(ngraphs)],

                    html.Div([
                        html.A('Download Data',
                               id='download-link',
                               download="",
                               href="",
                               target="_blank")],
                        id='download-wrapper')
                ],
                className='panel'
            )

        ], id='right-div', className='grid-div')

    ], id='container'),

    html.Br(),
    html.Hr(className='separator'),

    # Footer
    html.Div([
        'Translational Neural Engineering Lab, EPFL - 2017',
        html.Br(),
        'contact: ', html.A('theo.lemaire@epfl.ch', href='mailto:theo.lemaire@epfl.ch')
    ], id='footer')


])


# -------------------------------- NEURON MECHANISM CALLBACK --------------------------------

@app.callback(Output('neuron-mechanism', 'src'), [Input('mechanism-type', 'value')])
def updateImgSrc(value):
    return '{}{}_mech.png'.format(static_route, value)


# -------------------------------- TABLES CALLBACKS --------------------------------

@app.callback(
    Output('US-slider-table', 'hidden'),
    [Input('modality-tabs', 'value'), Input('custom-params-check', 'values')])
def toggleSliderTableUS(mod_value, is_custom):
    if mod_value == 1 and is_custom == []:
        hide = 0
    else:
        hide = 1
    return hide


@app.callback(
    Output('US-input-table', 'hidden'),
    [Input('modality-tabs', 'value'), Input('custom-params-check', 'values')])
def toggleInputTableUS(mod_value, is_custom):
    if mod_value == 1 and is_custom == ['custom']:
        hide = 0
    else:
        hide = 1
    return hide


@app.callback(
    Output('elec-slider-table', 'hidden'),
    [Input('modality-tabs', 'value'), Input('custom-params-check', 'values')])
def toggleSliderTableElec(mod_value, is_custom):
    if mod_value == 2 and is_custom == []:
        hide = 0
    else:
        hide = 1
    return hide


@app.callback(
    Output('elec-input-table', 'hidden'),
    [Input('modality-tabs', 'value'), Input('custom-params-check', 'values')])
def toggleInputTableElec(mod_value, is_custom):
    if mod_value == 2 and is_custom == ['custom']:
        hide = 0
    else:
        hide = 1
    return hide


@app.callback(Output('custom-params-check', 'values'),
              [Input('modality-tabs', 'value'), Input('mechanism-type', 'value')])
def uncheckCustomParams(*_):
    return []


# -------------------------------- US SLIDERS CALLBACKS --------------------------------

def updateSlider(values, curr, factor=1, precision=0, suffix=''):
    marks = {i: '{:.{}f}{}'.format(values[i] * factor, precision, suffix) if i == curr else ''
             for i in range(len(values))}
    return marks


@app.callback(Output('US-freq-slider', 'marks'), [Input('US-freq-slider', 'value')])
def updateUSFreqSlider(value):
    return updateSlider(US_freqs, value, suffix='kHz')


@app.callback(Output('US-amp-slider', 'marks'), [Input('US-amp-slider', 'value')])
def updateUSAmpSlider(value):
    return updateSlider(US_amps, value, suffix='kPa')


@app.callback(Output('US-dur-slider', 'marks'), [Input('US-dur-slider', 'value')])
def updateUSDurSlider(value):
    return updateSlider(durs, value, suffix='ms')


@app.callback(Output('US-PRF-slider', 'marks'), [Input('US-PRF-slider', 'value')])
def updateUSPRFSlider(value):
    return updateSlider(PRFs, value, precision=1, suffix='kHz')


@app.callback(Output('US-DF-slider', 'marks'), [Input('US-DF-slider', 'value')])
def updateUSDutySlider(value):
    return updateSlider(DFs, value, factor=100, precision=0, suffix='%')


@app.callback(Output('US-PRF-slider', 'disabled'), [Input('US-DF-slider', 'value')])
def toggleUSPRFSlider(value):
    return value == len(DFs) - 1


# -------------------------------- ELEC SLIDERS CALLBACKS --------------------------------

@app.callback(Output('elec-amp-slider', 'marks'), [Input('elec-amp-slider', 'value')])
def updateElecAmpSlider(value):
    return updateSlider(elec_amps, value, suffix='mA/m2')


@app.callback(Output('elec-dur-slider', 'marks'), [Input('elec-dur-slider', 'value')])
def updateElecDurSlider(value):
    return updateSlider(durs, value, suffix='ms')


@app.callback(Output('elec-PRF-slider', 'marks'), [Input('elec-PRF-slider', 'value')])
def updateElecPRFSlider(value):
    return updateSlider(PRFs, value, precision=1, suffix='kHz')


@app.callback(Output('elec-DF-slider', 'marks'), [Input('elec-DF-slider', 'value')])
def updateElecDutySlider(value):
    return updateSlider(DFs, value, factor=100, precision=0, suffix='%')


@app.callback(Output('elec-PRF-slider', 'disabled'), [Input('elec-DF-slider', 'value')])
def toggleElecPRFSlider(value):
    return value == len(DFs) - 1


# -------------------------------- OUTPUT DROPDOWNS CALLBACKS --------------------------------

def updateOutputDropdowns(mech_type, stim_type):
    if stim_type == 1:
        varlist = neurons[mech_type]['vars_US']
    else:
        varlist = neurons[mech_type]['vars_elec']
    return [{'label': v['desc'], 'value': v['label']} for v in varlist]


for i in range(ngraphs):
    app.callback(
        Output('output-dropdown-{}'.format(i + 1), 'options'),
        [Input('mechanism-type', 'value'),
         Input('modality-tabs', 'value')])(updateOutputDropdowns)


def updateOutputDropdownsValue(mech_type, stim_type, varname):
    if stim_type == 1:
        varlist = neurons[mech_type]['vars_US']
    else:
        varlist = neurons[mech_type]['vars_elec']
    vargroups = [v['label'] for v in varlist]
    if varname not in vargroups:
        varname = vargroups[0]
    return varname


for i in range(ngraphs):
    app.callback(
        Output('output-dropdown-{}'.format(i + 1), 'value'),
        [Input('mechanism-type', 'value'), Input('modality-tabs', 'value')],
        state=[State('output-dropdown-{}'.format(i + 1), 'value')])(updateOutputDropdownsValue)


# -------------------------------- OUTPUT GRAPHS CALLBACKS --------------------------------

def propagateInputs(mech_type, i_diam, i_modality, is_custom,
                    i_US_freq, i_US_amp, i_US_dur, i_US_PRF, i_US_DF,
                    i_elec_amp, i_elec_dur, i_elec_PRF, i_elec_DF,
                    n_US_submits, n_elec_submits,
                    varname,
                    US_freq_input, US_amp_input, US_dur_input, US_PRF_input, US_DF_input,
                    elec_amp_input, elec_dur_input, elec_PRF_input, elec_DF_input,
                    dd_str):

    ''' Translate inputs components values into input parameters
        and propagate callback to updateCurve.
    '''

    global colorset
    global previous_n_US_submits
    global previous_n_elec_submits
    global is_submit
    global iprop

    print('call to propagateInputs')

    idx = int(dd_str[-1])
    colors = colorset[2 * idx - 2: 2 * idx]

    # US case
    if i_modality == modalities['US']:

        # Determine whether or not the callback comes from a submit event
        if isinstance(n_US_submits, int) and n_US_submits == previous_n_US_submits + 1:
            is_submit = True
            previous_n_US_submits += 1
            iprop = 1
        elif is_submit and iprop < ngraphs:
            is_submit = True
            iprop += 1
        else:
            is_submit = False
            iprop = 0

        # Callback comes from a submit event
        if is_submit:
            US_inputs = (US_freq_input, US_amp_input, US_dur_input, US_PRF_input, US_DF_input)
            US_values = [float(x) for x in US_inputs]
            US_values[-1] /= 1e2  # correcting DF unit
            return updateCurve(mech_type, diams[i_diam], *US_values, varname, colors)

        # Callback comes from input slider or output dropdown change
        else:
            return updateCurve(mech_type, diams[i_diam], US_freqs[i_US_freq], US_amps[i_US_amp],
                               durs[i_US_dur], PRFs[i_US_PRF], DFs[i_US_DF], varname, colors)

    # Elec case
    else:

        # Determine whether or not the callback comes from a submit event
        if isinstance(n_elec_submits, int) and n_elec_submits == previous_n_elec_submits + 1:
            is_submit = True
            previous_n_elec_submits += 1
            iprop = 1
        elif is_submit and iprop < ngraphs:
            is_submit = True
            iprop += 1
        else:
            is_submit = False
            iprop = 0

        # Callback comes from a submit event
        if is_submit:
            elec_inputs = (elec_amp_input, elec_dur_input, elec_PRF_input, elec_DF_input)
            elec_values = [float(x) for x in elec_inputs]
            elec_values[-1] /= 1e2  # correcting DF unit
            return updateCurve(mech_type, diams[i_diam], None, *elec_values, varname, colors)

        # Callback comes from input slider or output dropdown change
        else:
            return updateCurve(mech_type, diams[i_diam], None, elec_amps[i_elec_amp],
                               durs[i_elec_dur], PRFs[i_elec_PRF], DFs[i_elec_DF], varname, colors)


def updateCurve(mech_type, diameter, Fdrive, Astim, tstim, PRF, DF, varname, colors):
    ''' Update curve based on new parameters.

        :param mech_type: type of channel mechanism (cell-type specific).
        :param diameter: diameter of the typical BLS structure (nm).
        :param Fdrive: driving frequency for acoustic stimuli (kHz), None for Elec stimuli.
        :param Astim: stimulus amplitude (kPa for acoustic, mA/m2 for Elec).
        :param tstim: stimulus duration (ms).
        :param PRF: Pulse-repetition frequency (kHz)
        :param DF: stimulus duty factor.
        :param varname: name of the output variable to display.
        :param colors: RGB colors for the variables to display.
        :return: variable curve, stimulus patches and graph and layout objects
    '''

    global data
    global current_cell
    global current_stim

    # Define new parameters
    cell_new = {
        'neuron': mech_type,
        'diameter': diameter
    }
    stim_new = {
        'freq': Fdrive,
        'amp': Astim,
        'dur': tstim,
        'PRF': PRF,
        'DF': DF
    }

    print(current_stim)
    print(stim_new)

    # Load new data if parameters have changed
    if cell_new != current_cell or stim_new != current_stim:
        data = updateData(cell_new, stim_new)
        current_cell = cell_new
        current_stim = stim_new

    # Get info about variables to plot
    if stim_new['freq'] is None:
        varlist = neurons[mech_type]['vars_elec']
    else:
        varlist = neurons[mech_type]['vars_US']

    vargroups = [v['label'] for v in varlist]
    if varname not in vargroups:
        varname = vargroups[0]
    for v in varlist:
        if v['label'] == varname:
            pltvar = v
            break

    if data is not None:

        # Get time, states and output variable vectors
        t = data['t']
        varlist = [data[v] for v in pltvar['names']]
        states = data['states']

        # Determine patches location
        npatches, tpatch_on, tpatch_off = getPatchesLoc(t, states)

        # Add onset
        t = np.insert(t, 0, tmin_plot * 1e-3)
        varlist = [np.insert(var, 0, var[0]) for var in varlist]

        # Define curves
        curves = [
            {
                'name': pltvar['names'][i],
                'x': t * 1e3,
                'y': varlist[i] * pltvar['factor'],
                'mode': 'lines',
                'line': {'color': colors[i]},
                'showlegend': True
            } for i in range(len(varlist))
        ]

        # Define stimulus patches
        patches = [
            {
                'x': np.array([tpatch_on[i], tpatch_off[i], tpatch_off[i], tpatch_on[i]]) * 1e3,
                'y': np.array([pltvar['min'], pltvar['min'], pltvar['max'], pltvar['max']]),
                'mode': 'none',
                'fill': 'toself',
                'fillcolor': 'grey',
                'opacity': 0.2,
                'showlegend': False
            } for i in range(npatches)
        ]

    # If file does not exist, define empty curve and patches
    else:
        curves = [{
            'name': pltvar['label'],
            'mode': 'none',
            'showlegend': False
        }]
        patches = []


    # Set axes layout
    layout = go.Layout(
        xaxis={
            'type': 'linear',
            'title': 'time (ms)',
            'range': [tmin_plot, tmax_plot],
            'zeroline': False
        },
        yaxis={
            'type': 'linear',
            'title': '{} ({})'.format(pltvar['label'], pltvar['unit']),
            'range': [pltvar['min'], pltvar['max']],
            'zeroline': False
        },
        margin={'l': 60, 'b': 40, 't': 10, 'r': 10},
        title=''
    )

    # Return curve, patches and layout objects
    return {'data': [*curves, *patches], 'layout': layout}


def updateData(cell_params, stim_params):
    ''' Update data either by loading a pre-computed simulation file from the remote server
        or by running a custom simulation locally.

        :param cell_params: dictionary of cell type and BLS diameter.
        :param params: dictionary of stimulation parameters.
        :return: the simulation data for that specific cell and stimulation parameters.
    '''

    print('call to updateData')

    global df
    global localfilepath

    # Split parameters explicitly
    mech_type = cell_params['neuron']
    a = cell_params['diameter']

    # Define simulation file name and remote/local path (ESTIM or ASTIM)
    if stim_params['freq'] is None:
        vardict = neurons[mech_type]['vars_elec']
        remotedir = '{}/{}/Elec/{:.0f}mAm2'.format(remoteroot, mech_type, stim_params['amp'])
        if stim_params['DF'] == 1.0:
            filecode = 'ESTIM_{}_CW_{:.1f}mA_per_m2_{:.0f}ms'.format(
                mech_type, stim_params['amp'], stim_params['dur'])
        else:
            filecode = 'ESTIM_{}_PW_{:.1f}mA_per_m2_{:.0f}ms_PRF{:.2f}kHz_DF{:.2f}'.format(
                mech_type, stim_params['amp'], stim_params['dur'], stim_params['PRF'],
                stim_params['DF'])
    else:
        vardict = neurons[mech_type]['vars_US']
        Fdrive = stim_params['freq']
        remotedir = '{}/{}/US/{:.0f}nm/{:.0f}kHz'.format(remoteroot, mech_type, a, Fdrive)
        if stim_params['DF'] == 1.0:
            filecode = 'ASTIM_{}_CW_{:.0f}nm_{:.0f}kHz_{:.0f}kPa_{:.0f}ms_effective'.format(
                mech_type, a, Fdrive, stim_params['amp'], stim_params['dur'])
        else:
            filecode = 'ASTIM_{}_PW_{:.0f}nm_{:.0f}kHz_{:.0f}kPa_{:.0f}ms_PRF{:.2f}kHz_DF{:.2f}_effective'.format(
                mech_type, a, Fdrive, stim_params['amp'], stim_params['dur'], stim_params['PRF'],
                stim_params['DF'])

    localfilepath = '{}/{}.pkl'.format(localdir, filecode)

    # Custom parameters -> run simulation
    if is_submit:
        mech = mechanisms[mech_type]()
        if stim_params['freq'] is None:  # ESTIM
            print('running ESTIM simulation')
            logfilepath = '{}/log_ESTIM.xlsx'.format(localdir)
            t0 = time.time()
            outfilepath = runEStim(localdir, logfilepath, solver_elec, mech, stim_params['amp'],
                                   stim_params['dur'] * 1e-3, (350 - stim_params['dur']) * 1e-3,
                                   stim_params['PRF'] * 1e3, stim_params['DF'])
        else:  # ASTIM
            print('running ASTIM simulation')
            logfilepath = '{}/log_ASTIM.xlsx'.format(localdir)
            Fdrive = stim_params['freq'] * 1e3
            t0 = time.time()
            solver = SolverUS(a * 1e-9, mech, Fdrive)
            outfilepath = runAStim(localdir, logfilepath, solver, mech, Fdrive,
                                   stim_params['amp'] * 1e3, stim_params['dur'] * 1e-3,
                                   (350 - stim_params['dur']) * 1e-3, stim_params['PRF'] * 1e3,
                                   stim_params['DF'])
        print(outfilepath)
        print(localfilepath)
        assert outfilepath == localfilepath, 'Local filepath not matching'

    # Standard parameters -> retrieve file from server
    else:
        remotefilepath = '{}/{}.pkl'.format(remotedir, filecode)
        if channel.isfile(remotefilepath):
            print('downloading "{}.pkl" file from server...'.format(filecode))
            t0 = time.time()
            channel.get(remotefilepath, localpath=localfilepath)
        else:
            print('"{}" file not found on server'.format(remotefilepath))
            return None

    # Load data from downloaded/generated local file
    with open(localfilepath, 'rb') as pkl_file:
        file_data = pickle.load(pkl_file)
    if os.path.isfile(localfilepath):
        os.remove(localfilepath)
        print('file data loaded in {:.0f} ms'.format((time.time() - t0) * 1e3))

    # Create pandas dataframe from file data (for further download purposes)
    varlist = ['t']
    unitlist = ['ms']
    factorlist = [1e3]
    for x in vardict:
        names = x['names']
        varlist += names
        unitlist += [x['unit']] * len(names)
        factorlist += [x['factor']] * len(names)
    df = pd.DataFrame(data={'{} ({})'.format(key, unitlist[i]): file_data[key] * factorlist[i]
                            for i, key in enumerate(varlist)})

    # Return raw file data
    return file_data


for i in range(ngraphs):
    app.callback(
        Output('output-curve-{}'.format(i + 1), 'figure'),
        [Input('mechanism-type', 'value'),
         Input('diam-slider', 'value'),
         Input('modality-tabs', 'value'),
         Input('custom-params-check', 'values'),
         Input('US-freq-slider', 'value'),
         Input('US-amp-slider', 'value'),
         Input('US-dur-slider', 'value'),
         Input('US-PRF-slider', 'value'),
         Input('US-DF-slider', 'value'),
         Input('elec-amp-slider', 'value'),
         Input('elec-dur-slider', 'value'),
         Input('elec-PRF-slider', 'value'),
         Input('elec-DF-slider', 'value'),
         Input('US-input-submit', 'n_clicks'),
         Input('elec-input-submit', 'n_clicks'),
         Input('output-dropdown-{}'.format(i + 1), 'value')],
        [State('US-freq-input', 'value'),
         State('US-amp-input', 'value'),
         State('US-dur-input', 'value'),
         State('US-PRF-input', 'value'),
         State('US-DF-input', 'value'),
         State('elec-amp-input', 'value'),
         State('elec-dur-input', 'value'),
         State('elec-PRF-input', 'value'),
         State('elec-DF-input', 'value'),
         State('output-dropdown-{}'.format(i + 1), 'id')])(propagateInputs)


# -------------------------------- OUTPUT METRICS CALLBACKS --------------------------------

@app.callback(Output('info-table', 'children'), [Input('output-curve-1', 'figure')])
def updateInfoTable(_):

    # Spike detection
    global data
    if data:
        if 'Qm' in data:
            n_spikes, lat, sr = detectSpikes(data['t'], data['Qm'], SPIKE_MIN_QAMP, SPIKE_MIN_DT)
        elif 'Vm' in data:
            n_spikes, lat, sr = detectSpikes(data['t'], data['Vm'], SPIKE_MIN_VAMP, SPIKE_MIN_DT)
        else:
            n_spikes, lat, sr = (0, None, None)
    else:
        n_spikes = 0
        lat = None
        sr = None

    rows = [
        html.Tr([
            html.Td('# spikes', style={'width': '30%'}),
            html.Td('{}'.format(n_spikes), style={'width': '70%'})
        ])
    ]
    if n_spikes > 0:
        rows.append(
            html.Tr([
                html.Td('Latency'),
                html.Td('{:.2f} ms'.format(lat * 1e3) if isinstance(lat, float) else '---')
            ])
        )
    if n_spikes > 1:
        rows.append(
            html.Tr([
                html.Td('Firing rate'),
                html.Td('{:.2f} kHz'.format(sr * 1e-3) if isinstance(sr, float) else '---')
            ])
        )

    return rows


# -------------------------------- DOWNLOAD LINK CALLBACK --------------------------------

@app.callback(Output('download-link', 'href'), [Input('output-curve-1', 'figure')])
def update_download_content(_):
    csv_string = df.to_csv(index=False, encoding='utf-8')
    csv_string = "data:text/csv;charset=utf-8," + urllib.parse.quote(csv_string)
    return csv_string


@app.callback(Output('download-link', 'download'), [Input('output-curve-1', 'figure')])
def update_download_name(_):
    filecode = os.path.splitext(os.path.basename(localfilepath))[0]
    return '{}.csv'.format(filecode)
