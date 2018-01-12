#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2017-06-22 16:57:14
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2018-01-12 09:17:26

''' Layout and callbacks of the web app. '''

import os
import time
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
from sftp import channel, data_root
from login import VALID_USERNAME_PASSWORD_PAIRS
from PointNICE.solvers import detectSpikes
from PointNICE.plt import getPatchesLoc
from PointNICE.constants import SPIKE_MIN_DT, SPIKE_MIN_QAMP, SPIKE_MIN_VAMP



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
diams = [16.0, 32.0, 64.0]  # nm
US_freqs = [200, 400, 600, 800, 1000]  # kHz
US_amps = [10, 20, 40, 80, 150, 300, 600]  # kPa
elec_amps = [-30, -20, -15, -10, -5, -2, 2, 5, 10, 15, 20, 30]  # mA/m2
durs = [20, 40, 60, 80, 100, 150, 200, 250, 300]  # ms
PRFs = [0.1, 0.2, 0.5, 1, 2, 5, 10]  # kHz
DFs = [0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 0.75, 1]

# Define default cell, Us and electricity parameters
# default = {'neuron': 'RS', 'diameter': 1, 'freq': 1, 'amp': 4, 'dur': 2, 'PRF': 3, 'DF': 7}
cell_default = {'neuron': 'RS', 'diameter': 1}
US_default = {'freq': 1, 'amp': 4, 'dur': 2, 'PRF': 3, 'DF': 7}
elec_default = {'amp': 8, 'dur': 2, 'PRF': 3, 'DF': 7}

# Set current parameters to default
# current = default
cell_current = cell_default
stim_current = US_default


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
app.title = 'TNEWebNICE viewer'

# Protect app with login
auth = dash_auth.BasicAuth(app, VALID_USERNAME_PASSWORD_PAIRS)


# -------------------------------- LAYOUT --------------------------------


# Load internal style sheets
for stylesheet in stylesheets:
    app.css.append_css({"external_url": "/css/{}".format(stylesheet)})



# Load static image files into base64 strings
epfl_logo = base64.b64encode(open('img/EPFL.png', 'rb').read()).decode()
tne_logo = base64.b64encode(open('img/TNE.png', 'rb').read()).decode()

app.layout = html.Div([

    # Header
    html.Div([
        html.Div(
            [html.A(
                html.Img(src='data:image/png;base64,{}'.format(epfl_logo), className='logo'),
                href='https://www.epfl.ch')],
            className='header-side', id='header-left'
        ),
        html.Div([
            html.H1('Ultrasound Neuromodulation', className='header-txt'),
            html.H3(['Exploring predictions of the ',
                     html.I('NICE'),
                     ' model'], className='header-txt')
        ], id='header-middle'),
        html.Div(
            [html.A(
                html.Img(src='data:image/png;base64,{}'.format(tne_logo), className='logo'),
                href='https://tne.epfl.ch')],
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
                                value=cell_default['neuron']
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
                                value=cell_default['diameter'],
                                marks={i: '{:.0f} nm'.format(diams[i]) if i == cell_default['diameter']
                                          else '' for i in range(len(diams))},
                                disabled=True,
                            )
                        )
                    ], className='slider-row')
                ], className='table'),

            ], className='panel'),


            # Stim parameters panel
            html.Div([
                html.H5('Stimulation parameters', className='panel-title'),

                dcc.Tabs(
                    tabs=[{'label': 'Ultrasound', 'value': 1},
                          {'label': 'Electricity', 'value': 2}],
                    value=1,
                    id='tabs'
                ),

                html.Table([

                    html.Tr([
                        html.Td('Frequency', style={'width': '30%'}),
                        html.Td(
                            dcc.Slider(
                                id='US-freq-slider',
                                min=0, max=len(US_freqs) - 1, step=1, value=US_default['freq']
                            ), style={'width': '70%'}
                        ),
                    ], className='slider-row'),

                    html.Tr([
                        html.Td('Amplitude'),
                        html.Td(
                            dcc.Slider(
                                id='US-amp-slider',
                                min=0, max=len(US_amps) - 1, step=1, value=US_default['amp']
                            )
                        )
                    ], className='slider-row'),

                    html.Tr([
                        html.Td('Duration'),
                        html.Td(
                            dcc.Slider(
                                id='US-dur-slider',
                                min=0, max=len(durs) - 1, step=1, value=US_default['dur']
                            )
                        )
                    ], className='slider-row'),

                    html.Tr([
                        html.Td('PRF'),
                        html.Td(
                            dcc.Slider(
                                id='US-PRF-slider',
                                min=0, max=len(PRFs) - 1, step=1, value=US_default['PRF']
                            )
                        )
                    ], className='slider-row'),

                    html.Tr([
                        html.Td('Duty cycle'),
                        html.Td(
                            dcc.Slider(
                                id='US-DF-slider',
                                min=0, max=len(DFs) - 1, step=1, value=US_default['DF']
                            )
                        )
                    ], className='slider-row')
                ], id='US-table', className='table', hidden=0),

                html.Table([

                    html.Tr([
                        html.Td('Amplitude'),
                        html.Td(
                            dcc.Slider(
                                id='elec-amp-slider',
                                min=0, max=len(elec_amps) - 1, step=1, value=elec_default['amp']
                            )
                        )
                    ], className='slider-row'),

                    html.Tr([
                        html.Td('Duration'),
                        html.Td(
                            dcc.Slider(
                                id='elec-dur-slider',
                                min=0, max=len(durs) - 1, step=1, value=elec_default['dur']
                            )
                        )
                    ], className='slider-row'),

                    html.Tr([
                        html.Td('PRF'),
                        html.Td(
                            dcc.Slider(
                                id='elec-PRF-slider',
                                min=0, max=len(PRFs) - 1, step=1, value=elec_default['PRF']
                            )
                        )
                    ], className='slider-row'),

                    html.Tr([
                        html.Td('Duty cycle'),
                        html.Td(
                            dcc.Slider(
                                id='elec-DF-slider',
                                min=0, max=len(DFs) - 1, step=1, value=elec_default['DF']
                            )
                        )
                    ], className='slider-row')
                ], id='elec-table', className='table', hidden=1),
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
                                    for v in neurons[cell_default['neuron']]['vars']
                                ],
                                value=neurons[cell_default['neuron']]['vars'][i]['label']
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


@app.callback(Output('US-freq-slider', 'marks'), [Input('US-freq-slider', 'value')])
def updateFreqSlider(value):
    return updateSlider(US_freqs, value, suffix='kHz')


@app.callback(Output('US-amp-slider', 'marks'), [Input('US-amp-slider', 'value')])
def updateAmpSlider(value):
    return updateSlider(US_amps, value, suffix='kPa')


@app.callback(Output('US-dur-slider', 'marks'), [Input('US-dur-slider', 'value')])
def updateDurSlider(value):
    return updateSlider(durs, value, suffix='ms')


@app.callback(Output('US-PRF-slider', 'marks'), [Input('US-PRF-slider', 'value')])
def updatePRFSlider(value):
    return updateSlider(PRFs, value, precision=1, suffix='kHz')


@app.callback(Output('US-DF-slider', 'marks'), [Input('US-DF-slider', 'value')])
def updateDutySlider(value):
    return updateSlider(DFs, value, factor=100, precision=0, suffix='%')


@app.callback(Output('US-PRF-slider', 'disabled'), [Input('US-DF-slider', 'value')])
def togglePRFSlider(value):
    return value == len(DFs) - 1



@app.callback(Output('elec-amp-slider', 'marks'), [Input('elec-amp-slider', 'value')])
def updateElecAmpSlider(value):
    return updateSlider(elec_amps, value, suffix='kPa')


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

def updateOutputDropdowns(mech_type):
    return [{'label': v['desc'], 'value': v['label']} for v in neurons[mech_type]['vars']]


for i in range(ngraphs):
    app.callback(
        Output('output-dropdown-{}'.format(i + 1), 'options'),
        [Input('mechanism-type', 'value')])(updateOutputDropdowns)


# -------------------------------- OUTPUT GRAPHS CALLBACKS --------------------------------

def updateData(mech_type, i_diam, i_modality,
               i_US_freq, i_US_amp, i_US_dur, i_US_PRF, i_US_DF,
               i_elec_amp, i_elec_dur, i_elec_PRF, i_elec_DF,
               varname, dd_str):
    global colorset
    idx = int(dd_str[-1])
    colors = colorset[2 * idx - 2: 2 * idx]

    if i_modality == 1:  # US
        return updateCurve(mech_type, diams[i_diam], US_freqs[i_US_freq], US_amps[i_US_amp],
                           durs[i_US_dur], PRFs[i_US_PRF], DFs[i_US_DF], varname, colors)
    else:  # Elec
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
    global cell_current
    global stim_current

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

    # Load new data if parameters have changed
    if cell_new != cell_current or stim_new != stim_current:
        data = getData(cell_new, stim_new, data_root)
        cell_current = cell_new
        stim_current = stim_new

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
        title=''
    )

    # Return curve, patches and layout objects
    return {'data': [*curves, *patches], 'layout': layout}


def getData(cell_params, stim_params, data_root):
    ''' Load the appropriate data file and return data structure.

        :param cell_params: dictionary of cell type and BLS diameter.
        :param params: dictionary of stimulation parameters.
        :param data_root: the absolute path to the root of data directories
        :return: the simulation data for that specific cell and stimulation parameters.
    '''

    # Split parameters explicitly
    mech_type = cell_params['neuron']
    a = cell_params['diameter']

    # Define path to input file (ESTIM or ASTIM)
    if stim_params['freq'] is None:
        filedir = '{}/{}/Elec'.format(data_root, mech_type)
        if stim_params['DF'] == 1.0:
            filecode = 'ESTIM_{}_CW_{:.0f}mA_per_m2_{:.0f}ms'.format(
                mech_type, stim_params['amp'], stim_params['dur'])
        else:
            filecode = 'ESTIM_{}_PW_{:.0f}mA_per_m2_{:.0f}ms_PRF{:.2f}kHz_DF{:.2f}'.format(
                mech_type, stim_params['amp'], stim_params['dur'], stim_params['PRF'],
                stim_params['DF'])
    else:
        Fdrive = stim_params['freq']
        filedir = '{}/{}/US/{:.0f}nm/{:.0f}kHz'.format(data_root, mech_type, a, Fdrive)
        if stim_params['DF'] == 1.0:
            filecode = 'ASTIM_{}_CW_{:.0f}nm_{:.0f}kHz_{:.0f}kPa_{:.0f}ms_effective'.format(
                mech_type, a, Fdrive, stim_params['amp'], stim_params['dur'])
        else:
            filecode = 'ASTIM_{}_PW_{:.0f}nm_{:.0f}kHz_{:.0f}kPa_{:.0f}ms_PRF{:.2f}kHz_DF{:.2f}_effective'.format(
                mech_type, a, Fdrive, stim_params['amp'], stim_params['dur'], stim_params['PRF'],
                stim_params['DF'])


    pkl_filepath = '{}/{}.pkl'.format(filedir, filecode)

    if channel.isfile(pkl_filepath):
        t0 = time.time()
        tmpfile = 'tmp/{}.pkl'.format(filecode)
        channel.get(pkl_filepath, localpath=tmpfile)
        with open(tmpfile, 'rb') as pkl_file:
            file_data = pickle.load(pkl_file)
        os.remove(tmpfile)
        print('file loaded in {:.0f} ms'.format((time.time() - t0) * 1e3))
    else:
        print('Data file "{}" not found on server'.format(pkl_filepath))
        file_data = None

    return file_data


for i in range(ngraphs):
    app.callback(
        Output('output-curve-{}'.format(i + 1), 'figure'),
        [Input('mechanism-type', 'value'),
         Input('diam-slider', 'value'),
         Input('tabs', 'value'),
         Input('US-freq-slider', 'value'),
         Input('US-amp-slider', 'value'),
         Input('US-dur-slider', 'value'),
         Input('US-PRF-slider', 'value'),
         Input('US-DF-slider', 'value'),
         Input('elec-amp-slider', 'value'),
         Input('elec-dur-slider', 'value'),
         Input('elec-PRF-slider', 'value'),
         Input('elec-DF-slider', 'value'),
         Input('output-dropdown-{}'.format(i + 1), 'value'),
         Input('output-dropdown-{}'.format(i + 1), 'id')])(updateData)


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


@app.callback(Output('US-table', 'hidden'), [Input('tabs', 'value')])
def toggle_US_table(value):
    if value == 1:
        hide = 0
    else:
        hide = 1
    return hide


@app.callback(Output('elec-table', 'hidden'), [Input('tabs', 'value')])
def toggle_elec_table(value):
    if value == 1:
        hide = 1
    else:
        hide = 0
    return hide
