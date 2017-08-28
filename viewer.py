#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2017-06-22 16:57:14
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2017-08-28 15:52:47

''' Layout and callbacks of the web app. '''

import os
import pickle
import base64
import numpy as np
import dash
from dash.dependencies import Input, Output
import dash_auth
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import colorlover as cl

from server import server, static_route, stylesheets
from root import data_root, image_directory
from login import VALID_USERNAME_PASSWORD_PAIRS
from PointNICE.solvers import detectSpikes
from PointNICE.plt import getPatchesLoc
from PointNICE.constants import SPIKE_MIN_DT, SPIKE_MIN_QAMP



# -------------------------------- PARAMETERS --------------------------------

# Define output variables
charge = {'names': ['Qm'], 'desc': 'charge density', 'label': 'charge', 'unit': 'nC/cm2',
          'factor': 1e5, 'min': -90, 'max': 50}
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
        'vars': [charge, deflection, gas, iNa_gates, iK_gate, iM_gate]
    },
    'FS': {
        'desc': 'Cortical fast-spiking neuron',
        'vars': [charge, deflection, gas, iNa_gates, iK_gate, iM_gate]
    },
    'LTS': {
        'desc': 'Cortical, low-threshold spiking neuron',
        'vars': [charge, deflection, gas, iNa_gates, iK_gate, iM_gate, iCa_gates]
    },
    'RE': {
        'desc': 'Thalamic reticular neuron',
        'vars': [charge, deflection, gas, iNa_gates, iK_gate, iCa_gates]
    },
    'TC': {
        'desc': 'Thalamo-cortical neuron',
        'vars': [charge, deflection, gas, iNa_gates, iK_gate, iCa_gates, iH_reg_factor, Ca_conc]
    }
}


# Define parameter ranges for input sliders
diams = [16.0, 32.0, 64.0]
freqs = [200, 400, 600, 800, 1000]
amps = [10, 20, 40, 80, 150, 300, 600]
durs = [20, 40, 60, 80, 100, 150, 200, 250, 300]
PRFs = [0.1, 0.2, 0.5, 1, 2, 5, 10]
DFs = [0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 0.75, 1]

# Define default parameters and set current parameters to default
default = {'neuron': 'RS', 'diameter': 1, 'freq': 1, 'amp': 4, 'dur': 2, 'PRF': 3, 'DF': 7}
current = default

# Set plotting parameters
ngraphs = 3
colorset = cl.scales[str(2 * ngraphs + 1)]['qual']['Set1']
del colorset[5]
tmin_plot = -5  # ms
tmax_plot = 350  # ms


# -------------------------------- APPLICATION --------------------------------

# Create Dash app
app = dash.Dash(
    name='viewer',
    server=server,
    url_base_pathname='/viewer',
    csrf_protect=True
)
app.title = 'NICE: model predictions viewer'

# Protect app with login
auth = dash_auth.BasicAuth(app, VALID_USERNAME_PASSWORD_PAIRS)


# -------------------------------- LAYOUT --------------------------------

# Load external style sheet
app.css.append_css({"external_url": "https://codepen.io/chriddyp/pen/bWLwgP.css"})

# Load internal style sheet
for stylesheet in stylesheets:
    app.css.append_css({"external_url": "/css/{}".format(stylesheet)})



# Load static image files into base64 strings
epfl_logo = base64.b64encode(open(image_directory + 'EPFL.png', 'rb').read()).decode()
tne_logo = base64.b64encode(open(image_directory + 'TNE.png', 'rb').read()).decode()

app.layout = html.Div([

    # Header
    html.Div([
        html.Div(
            [html.Img(src='data:image/png;base64,{}'.format(epfl_logo), className='logo')],
            className='header-side', id='header-left'
        ),
        html.Div([
            html.H2('Neuronal Intramembrane Cavitation Excitation:', className='header-txt'),
            html.H3('model predictions for various neuron types', className='header-txt')
        ], id='header-middle'),
        html.Div(
            [html.Img(src='data:image/png;base64,{}'.format(tne_logo), className='logo')],
            className='header-side', id='header-right'
        )
    ], id='header'),

    html.Hr(className='separator'),

    # Main div
    html.Div([

        # Left side
        html.Div([

            # Cell parameters panel
            html.Div([
                html.H5('Cell parameters', className='panel-title'),

                html.Table([
                    html.Tr([
                        html.Td('Cell type', style={'width': '35%'}),
                        html.Td(
                            dcc.Dropdown(
                                id='mechanism-type',
                                options=[{'label': v['desc'], 'value': k}
                                         for k, v in neurons.items()],
                                value=default['neuron']
                            ), style={'width': '65%'}
                        )
                    ]),
                    html.Tr([
                        html.Td('E-STIM response'),
                        html.Td(html.Img(id='neuron-anim', style={'width': '100%'}))
                    ]),
                    html.Tr([
                        html.Td('Sonophore diameter'),
                        html.Td(
                            dcc.Slider(
                                id='diam-slider', min=0, max=len(diams) - 1, step=1,
                                value=default['diameter'],
                                marks={i: '{:.0f} nm'.format(diams[i]) if i == default['diameter']
                                          else '' for i in range(len(diams))},
                                disabled=True,
                            )
                        )
                    ], className='slider-row')
                ], className='table'),

            ], className='panel'),


            # Stim parameters panel
            html.Div([
                html.H5('US stimulation parameters', className='panel-title'),

                html.Table([

                    html.Tr([
                        html.Td('Frequency', style={'width': '30%'}),
                        html.Td(
                            dcc.Slider(
                                id='freq-slider',
                                min=0, max=len(freqs) - 1, step=1, value=default['freq']
                            ), style={'width': '70%'}
                        ),
                    ], className='slider-row'),

                    html.Tr([
                        html.Td('Amplitude'),
                        html.Td(
                            dcc.Slider(
                                id='amp-slider',
                                min=0, max=len(amps) - 1, step=1, value=default['amp']
                            )
                        )
                    ], className='slider-row'),

                    html.Tr([
                        html.Td('Duration'),
                        html.Td(
                            dcc.Slider(
                                id='dur-slider',
                                min=0, max=len(durs) - 1, step=1, value=default['dur']
                            )
                        )
                    ], className='slider-row'),

                    html.Tr([
                        html.Td('PRF'),
                        html.Td(
                            dcc.Slider(
                                id='PRF-slider',
                                min=0, max=len(PRFs) - 1, step=1, value=default['PRF']
                            )
                        )
                    ], className='slider-row'),

                    html.Tr([
                        html.Td('Duty cycle'),
                        html.Td(
                            dcc.Slider(
                                id='DF-slider',
                                min=0, max=len(DFs) - 1, step=1, value=default['DF']
                            )
                        )
                    ], className='slider-row')
                ], className='table'),
            ], className='panel'),


            # Output metrics panel
            html.Div([
                html.H5('Output metrics', className='panel-title'),
                html.Table(id='info-table', className='table')
            ], className='panel'),

        ], id='left-div', className='grid-div'),

        # Right side
        html.Div([

            # Graphs panel
            html.Div(
                [
                    html.H5('Neural response', className='panel-title'),

                    *[html.Div(
                        [
                            # Dropdown list
                            dcc.Dropdown(
                                id='output-dropdown-{}'.format(i + 1),

                                options=[
                                    {'label': v['desc'],
                                     'value': v['label']}
                                    for v in neurons[default['neuron']]['vars']
                                ],
                                value=neurons[default['neuron']]['vars'][i]['label']
                            ),

                            # Graph
                            dcc.Graph(
                                id='output-curve-{}'.format(i + 1),
                                style={'height': '15em'},
                                animate=False
                            ),

                            # Horizontal line
                            html.Hr(className='graph-separator') if i < ngraphs - 1 else None,
                        ],
                        id='output-{}'.format(i),
                        className='graph-div')
                      for i in range(ngraphs)],

                ],
                className='panel'
            )

        ], id='right-div', className='grid-div')

    ], id='container'),

    html.Hr(className='separator'),

    # Footer
    html.Div([
        'Translational Neural Engineering Lab, EPFL - 2017',
        html.Br(),
        'contact: ', html.A('theo.lemaire@epfl.ch', href='mailto:theo.lemaire@epfl.ch')
    ], id='footer')


])


# -------------------------------- INPUT ANIMATION CALLBACK --------------------------------

@app.callback(Output('neuron-anim', 'src'), [Input('mechanism-type', 'value')])
def update_image_src(value):
    return static_route + value


# -------------------------------- SLIDERS CALLBACKS --------------------------------

def updateSlider(values, curr, factor=1, precision=0, suffix=''):
    marks = {i: '{:.{}f}{}'.format(values[i] * factor, precision, suffix) if i == curr else ''
             for i in range(len(values))}
    return marks


@app.callback(Output('freq-slider', 'marks'), [Input('freq-slider', 'value')])
def updateFreqSlider(value):
    return updateSlider(freqs, value, suffix='kHz')


@app.callback(Output('amp-slider', 'marks'), [Input('amp-slider', 'value')])
def updateAmpSlider(value):
    return updateSlider(amps, value, suffix='kPa')


@app.callback(Output('dur-slider', 'marks'), [Input('dur-slider', 'value')])
def updateDurSlider(value):
    return updateSlider(durs, value, suffix='ms')


@app.callback(Output('PRF-slider', 'marks'), [Input('PRF-slider', 'value')])
def updatePRFSlider(value):
    return updateSlider(PRFs, value, precision=1, suffix='kHz')


@app.callback(Output('DF-slider', 'marks'), [Input('DF-slider', 'value')])
def updateDutySlider(value):
    return updateSlider(DFs, value, factor=100, precision=0, suffix='%')


@app.callback(Output('PRF-slider', 'disabled'), [Input('DF-slider', 'value')])
def togglePRFSlider(value):
    return value == len(DFs) - 1


# -------------------------------- OUTPUT DROPDOWNS CALLBACKS --------------------------------

def updateOutputDropdowns(mech_type):
    return [{'label': v['desc'], 'value': v['label']} for v in neurons[mech_type]['vars']]


for i in range(ngraphs):
    app.callback(
        Output('output-dropdown-{}'.format(i + 1), 'options'),
        [Input('mechanism-type', 'value')])(updateOutputDropdowns)


# -------------------------------- OUTPUT GRAPHS CALLBACKS --------------------------------

def updateData(mech_type, i_diam, i_freq, i_amp, i_dur, i_PRF, i_DF, varname, dd_str):
    global colorset
    idx = int(dd_str[-1])
    colors = colorset[2 * idx - 2: 2 * idx]
    return updateCurve(mech_type, diams[i_diam], freqs[i_freq], amps[i_amp], durs[i_dur],
                       PRFs[i_PRF], DFs[i_DF], varname, colors)


def updateCurve(mech_type, diameter, Fdrive, Adrive, tstim, PRF, DF, varname, colors):
    ''' Update curve based on new parameters.

        :param mech_type: type of channel mechanism (cell-type specific).
        :param diameter: diameter of the typical BLS structure (nm).
        :param Fdrive: acoustic stimulus driving frequency (kHz).
        :param Adrive: acoustic stimulus pressure amplitude (kPa).
        :param tstim: stimulus duration (ms).
        :param PRF: Pulse-repetition frequency (kHz)
        :param DF: stimulus duty factor.
        :param varname: name of the output variable to display.
        :param colors: RGB colors for the variables to display.
        :return: variable curve, stimulus patches and graph and layout objects
    '''

    global data
    global current

    # Define new parameters
    new = {
        'neuron': mech_type,
        'diameter': diameter,
        'freq': Fdrive,
        'amp': Adrive,
        'dur': tstim,
        'PRF': PRF,
        'DF': DF
    }

    # Load new data if parameters have changed
    if new != current:
        data = getData(new, data_root)
        current = new

    # Get info about variables to plot
    vargroups = [v['label'] for v in neurons[mech_type]['vars']]
    if varname not in vargroups:
        varname = vargroups[0]
    for v in neurons[mech_type]['vars']:
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
    )

    # Return curve, patches and layout objects
    return {'data': [*curves, *patches], 'layout': layout}


def getData(params, data_root):
    ''' Load the appropriate data file and return data structure.

        :param params: dictionary of cell type and stimulation parameters.
        :param data_root: the absolute path to the root of data directories
        :return: the simulation data for that specific cell and stimulation parameters.
    '''

    # Split parameters explicitly
    mech_type = params['neuron']
    a = params['diameter']
    Fdrive = params['freq']

    # Define path to input file
    filedir = '{}/{}/{:.0f}nm/{:.0f}kHz'.format(data_root, mech_type, a, Fdrive)
    if params['DF'] == 1.0:
        filecode = 'sim_{}_CW_{:.0f}nm_{:.0f}kHz_{:.0f}kPa_{:.0f}ms'.format(
            mech_type, a, Fdrive, params['amp'], params['dur'])
    else:
        filecode = 'sim_{}_PW_{:.0f}nm_{:.0f}kHz_{:.0f}kPa_{:.0f}ms_PRF{:.2f}kHz_DF{:.2f}'.format(
            mech_type, a, Fdrive, params['amp'], params['dur'], params['PRF'], params['DF'])
    pkl_filepath = '{}/{}_effective.pkl'.format(filedir, filecode)

    # Check input file existence
    if os.path.isfile(pkl_filepath):
        # Load and return input data if present
        with open(pkl_filepath, 'rb') as pkl_file:
            file_data = pickle.load(pkl_file)
    else:
        # Return None if absent
        print('Data file "{}" does not exist'.format(pkl_filepath))
        file_data = None
    return file_data


for i in range(ngraphs):
    app.callback(
        Output('output-curve-{}'.format(i + 1), 'figure'),
        [Input('mechanism-type', 'value'),
         Input('diam-slider', 'value'),
         Input('freq-slider', 'value'),
         Input('amp-slider', 'value'),
         Input('dur-slider', 'value'),
         Input('PRF-slider', 'value'),
         Input('DF-slider', 'value'),
         Input('output-dropdown-{}'.format(i + 1), 'value'),
         Input('output-dropdown-{}'.format(i + 1), 'id')])(updateData)


# -------------------------------- OUTPUT METRICS CALLBACKS --------------------------------

@app.callback(Output('info-table', 'children'), [Input('output-curve-1', 'figure')])
def updateInfoTable(_):

    # Spike detection
    global data
    if data:
        n_spikes, lat, sr = detectSpikes(data['t'], data['Qm'], SPIKE_MIN_QAMP, SPIKE_MIN_DT)
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
                html.Td('{:.1f} ms'.format(lat * 1e3) if isinstance(lat, float) else '---')
            ])
        )
    if n_spikes > 1:
        rows.append(
            html.Tr([
                html.Td('Firing rate'),
                html.Td('{:.1f} kHz'.format(sr * 1e-3) if isinstance(sr, float) else '---')
            ])
        )

    return rows
