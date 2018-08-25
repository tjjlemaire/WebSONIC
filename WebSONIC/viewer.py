#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2017-06-22 16:57:14
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2018-08-25 02:24:32

''' Definition of the SONICViewer class. '''

import os
import time
import pickle
import urllib
import numpy as np

import dash
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import dash_auth

from PySONIC.plt import getPatchesLoc
from PySONIC.solvers import SolverElec, SolverUS, EStimWorker, AStimWorker, findPeaks
from PySONIC.constants import *
from PySONIC.utils import getNeuronsDict, si_prefixes, checkNumBounds

from .components import *
from .pltvars import neuronvars


class SONICViewer(dash.Dash):
    ''' SONIC viewer application inheriting from dash.Dash. '''

    def __init__(self, server, tmpdir, remoteroot, ssh_channel, inputs, pltparams,
                 name='viewer', title='SONIC viewer', ngraphs=1, credentials=None):

        self.tmpdir = tmpdir
        self.remoteroot = remoteroot
        self.ssh_channel = ssh_channel

        # Initialize Dash app
        super(SONICViewer, self).__init__(
            name=name,
            server=server,
            url_base_pathname='/{}/'.format(name),
            csrf_protect=True
        )

        self.prefixes = {v: k for k, v in si_prefixes.items()}

        # Protecting app with authentifier if credentials provided
        if credentials is not None:
            self.authentifier = dash_auth.BasicAuth(self, credentials)
        else:
            self.authentifier = None

        self.title = title
        self.ngraphs = ngraphs
        self.colorset = pltparams['colorset']
        self.tbounds = pltparams['tbounds']  # ms

        self.neurons = {key: getNeuronsDict()[key]() for key in neuronvars.keys()}

        self.cell_params = {
            'mech': dict(label='Cell Type', values=list(neuronvars.keys()), idef=0),
            'diam': dict(label='Sonophore diameter', values=inputs['diams'], idef=1),
        }
        self.default_cell = self.cell_params['mech']['values'][self.cell_params['mech']['idef']]

        self.stimmod = 'US'

        self.stim_params = {
            'US': {
                'freq': dict(label='Frequency', values=inputs['US_freqs'],
                             unit='Hz', factor=1e-3, idef=2),
                'amp': dict(label='Amplitude', values=inputs['US_amps'],
                            unit='Pa', factor=1e-3, idef=3),
                'PRF': dict(label='PRF', values=inputs['PRFs'], unit='Hz', idef=1),
                'DC': dict(label='Duty Cycle', values=inputs['DCs'], unit='%', idef=6)
            },
            'elec': {
                'amp': dict(label='Amplitude', values=inputs['elec_amps'], unit='mA/m2', idef=6),
                'PRF': dict(label='PRF', values=inputs['PRFs'], unit='Hz', idef=1),
                'DC': dict(label='Duty Cycle', values=inputs['DCs'], unit='%', idef=6)
            }
        }
        self.tstim = inputs['tstim']

        self.prev_nsubmits = 0
        self.current_params = None
        self.data = None
        self.localfilepath = ''

        self.setLayout()
        self.registerCallbacks()


    def __str__(self):
        return '{} app ({}) with {} graphs'.format(
            self.title,
            'password protected' if self.authentifier is not None else 'unprotected',
            self.ngraphs
        )

    # ------------------------------------------ LAYOUT ------------------------------------------

    def setLayout(self):
        ''' Set app layout. '''
        self.layout = html.Div(id='body', children=[
            # # Favicon
            # html.Link(rel='shortcut icon', href='/favicon.ico'),

            # Header
            self.header(),
            separator(),

            # Content
            html.Div(id='content', children=[
                # Left side
                html.Div(id='left-col', className='content-column', children=[
                    self.cellPanel(),
                    self.stimPanel(),
                    self.metricsPanel(),
                ]),

                # Right side
                html.Div(id='right-col', className='content-column', children=[
                    self.outputPanel()
                ])
            ]),

            # Footer
            html.Br(),
            separator(),
            self.footer()
        ])

    def header(self):
        ''' Set app header. '''
        return html.Div(id='header', children=[

            html.Div(className='header-side', id='header-left', children=[
                html.A(html.Img(src='/assets/EPFL.png', className='logo'),
                       href='https://www.epfl.ch')]),

            html.Div(id='header-middle', children=[
                html.H1('Ultrasound vs. Electrical stimulation', className='header-txt'),
                html.H3(['Exploring predictions of the ', html.I('NICE'), ' and ',
                         html.I('Hodgkin-Huxley'), ' models'], className='header-txt')]),

            html.Div(className='header-side', id='header-right', children=[
                html.A(html.Img(src='/assets/ITIS.svg', className='logo'),
                       href='https://www.itis.ethz.ch')])
        ])

    def footer(self):
        ''' Set app footer. '''
        return html.Div(id='footer', children=[
            'Developed with the ', html.A('Dash', href='https://dash.plot.ly/'), ' framework.',
            html.Br(),
            'Translational Neural Engineering Lab, EPFL - 2018',
            html.Br(),
            'contact: ', html.A('theo.lemaire@epfl.ch', href='mailto:theo.lemaire@epfl.ch')
        ])

    def cellPanel(self):
        ''' Construct cell parameters panel. '''
        return collapsablePanel('Cell parameters', children=[
            html.Table(className='table', children=[
                html.Tr([
                    html.Td(self.cell_params['mech']['label'], style={'width': '35%'}),
                    html.Td(style={'width': '65%'}, children=[
                        dcc.Dropdown(
                            id='mechanism-type',
                            options=[{'label': v['desc'], 'value': k} for k, v in neuronvars.items()],
                            value=self.default_cell)
                    ])]),
                html.Tr([
                    html.Td('Membrane mechanism'),
                    html.Td(html.Img(id='neuron-mechanism', style={'width': '100%'}))]),

                labeledSliderRow(self.cell_params['diam']['label'], 'diam-slider',
                                 len(self.cell_params['diam']['values']),
                                 value=self.cell_params['diam']['idef'],
                                 disabled=True)
            ])
        ])

    def stimPanel(self):
        ''' Construct stimulation parameters panel. '''

        return collapsablePanel('Stimulation parameters', children=[

            dcc.Tabs(id='modality-tabs', value=self.stimmod, children=[
                dcc.Tab(label='Ultrasound', value='US'),
                dcc.Tab(label='Electricity', value='elec')]),

            html.Br(),

            dcc.RadioItems(id='toggle-custom', value=True, labelStyle={'display': 'inline-block'},
                           options=[{'label': 'Standard', 'value': True},
                                    {'label': 'Custom', 'value': False}]),

            *[labeledSlidersTable(
                '{}-slider-table'.format(mod_type),
                labels=[p['label'] for p in self.stim_params[mod_type].values()],
                ids=['{}-{}-slider'.format(mod_type, p) for p in self.stim_params[mod_type].keys()],
                sizes=[len(p['values']) for p in self.stim_params[mod_type].values()],
                values=[p['idef'] for p in self.stim_params[mod_type].values()])

                for mod_type in self.stim_params.keys()],


            html.Div(id='inputs-form', className='input-div', hidden=0, children=[

                *[labeledInputsTable(
                    '{}-input-table'.format(mod_type),
                    labels=['{} ({}{})'.format(p['label'], self.prefixes[1 / p.get('factor', 1)],
                                               p['unit'])
                            for p in self.stim_params[mod_type].values()],
                    ids=['{}-{}-input'.format(mod_type, p)
                         for p in self.stim_params[mod_type].keys()],
                    mins=[min(p['values']) * p.get('factor', 1)
                          for p in self.stim_params[mod_type].values()],
                    maxs=[max(p['values']) * p.get('factor', 1)
                          for p in self.stim_params[mod_type].values()],
                    values=[p['values'][p['idef']] * p.get('factor', 1)
                            for p in self.stim_params[mod_type].values()])

                    for mod_type in self.stim_params.keys()],

                html.Div(id='inputs-submit-div', hidden=True, children=[
                    html.Button('Run', id='inputs-submit', className='submit-button')
                ])
            ])
        ])

    def metricsPanel(self):
        return collapsablePanel('Output metrics', children=[
            html.Table(id='info-table', className='table')])

    def outputPanel(self):
        ddgraphpanels = []
        for i in range(self.ngraphs):
            graphvars = neuronvars[self.default_cell]['vars_{}'.format(self.stimmod)]
            ddgraphpanels.append(collapsablePanel(title=None, children=[ddGraph(
                id='out{}'.format(i),
                labels=[v['desc'] for v in graphvars],
                values=[v['label'] for v in graphvars],
                default=graphvars[i]['label'],
                sep=False)]))

        return html.Div(children=[
            *ddgraphpanels,
            html.Div(id='download-wrapper', children=[
                html.A('Download Data', id='download-link', download="", href="", target="_blank")])
        ])

    # ------------------------------------------ CALLBACKS ------------------------------------------

    def registerCallbacks(self):

        # Cell panel: mechanism type
        self.callback(
            Output('neuron-mechanism', 'src'),
            [Input('mechanism-type', 'value')])(self.updateImgSrc)

        # Stimulation panel: tables visibility
        for table_mod in ['US', 'elec']:
            for table_type in ['slider', 'input']:
                key = '{}-{}' .format(table_mod, table_type)
                is_standard = table_type == 'slider'
                self.callback(
                    Output('{}-table'.format(key), 'hidden'),
                    [Input('modality-tabs', 'value'),
                     Input('toggle-custom', 'value')])(self.showTable(table_mod, is_standard))

        self.callback(
            Output('inputs-submit-div', 'hidden'),
            [Input('toggle-custom', 'value')])(self.hideSubmitButton)

        # Stimulation panel: sliders
        for mod_type, refparams in self.stim_params.items():
            for key, p in refparams.items():
                id = '{}-{}-slider'.format(mod_type, key)
                self.callback(Output(id, 'marks'), [Input(id, 'value')])(self.updateSlider(p))

        # Output metrics table
        self.callback(
            Output('info-table', 'children'),
            [Input('out0-graph', 'figure')])(self.updateInfoTable)

        # Output panels
        for i in range(self.ngraphs):

            # drop-down list
            self.callback(
                Output('out{}-dropdown'.format(i), 'options'),
                [Input('mechanism-type', 'value'),
                 Input('modality-tabs', 'value')])(self.updateOutputOptions)
            self.callback(
                Output('out{}-dropdown'.format(i), 'value'),
                [Input('mechanism-type', 'value'), Input('modality-tabs', 'value')],
                state=[State('out{}-dropdown'.format(i), 'value')])(self.updateOutputVar)

        # 1st graph
        self.callback(
            Output('out0-graph', 'figure'),
            [Input('mechanism-type', 'value'),
             Input('diam-slider', 'value'),
             Input('modality-tabs', 'value'),
             Input('toggle-custom', 'value'),
             Input('US-freq-slider', 'value'),
             Input('US-amp-slider', 'value'),
             Input('US-PRF-slider', 'value'),
             Input('US-DC-slider', 'value'),
             Input('elec-amp-slider', 'value'),
             Input('elec-PRF-slider', 'value'),
             Input('elec-DC-slider', 'value'),
             Input('inputs-submit', 'n_clicks'),
             Input('out0-dropdown', 'value')],
            [State('US-freq-input', 'value'),
             State('US-amp-input', 'value'),
             State('US-PRF-input', 'value'),
             State('US-DC-input', 'value'),
             State('elec-amp-input', 'value'),
             State('elec-PRF-input', 'value'),
             State('elec-DC-input', 'value')])(self.propagateInputs)

        # from 2nd graph on
        for i in range(1, self.ngraphs):
            self.callback(
                Output('out{}-graph'.format(i), 'figure'),
                [Input('out0-graph', 'figure'),
                 Input('out{}-dropdown'.format(i), 'value')],
                [State('mechanism-type', 'value'),
                 State('modality-tabs', 'value'),
                 State('out{}-graph'.format(i), 'id')])(self.updateGraph)

        # Downbload link
        self.callback(
            Output('download-link', 'href'),
            [Input('out0-graph', 'figure')])(self.updateDownloadContent)
        self.callback(
            Output('download-link', 'download'),
            [Input('out0-graph', 'figure')])(self.updateDownloadName)

    def updateImgSrc(self, value):
        ''' Update the image of neuron mechanism on neuron switch. '''
        return '/assets/{}_mech.png'.format(value)

    def showTableGeneric(self, stim_mod, is_standard, table_mod, is_standard_table):
        return not (stim_mod == table_mod and is_standard == is_standard_table)

    def showTable(self, table_mod, is_standard_table):
        ''' For correct assignment of updateSlider functions with lambda expressions. '''
        return lambda x, y: self.showTableGeneric(x, y, table_mod, is_standard_table)

    def hideSubmitButton(self, is_standard):
        ''' Show submit button only when stimulation panel is in input mode. '''
        return is_standard

    def updateSliderGeneric(self, values, curr, factor=1, precision=0, suffix=''):
        ''' Generic function to update a slider value. '''
        return {i: '{}{}'.format(si_format(values[i], precision, space=' '), suffix)
                   if i == curr else '' for i in range(len(values))}

    def updateSlider(self, p):
            ''' For correct assignment of updateSlider functions with lambda expressions. '''
            return lambda x: self.updateSliderGeneric(p['values'], x, suffix=p['unit'])

    def getVars(self, cell_type, mod_type):
        return neuronvars[cell_type]['vars_{}'.format(mod_type)]

    def updateOutputOptions(self, cell_type, mod_type):
        ''' Update the list of available variables in a graph dropdown menu on neuron switch. '''
        return [{'label': v['desc'], 'value': v['label']} for v in self.getVars(cell_type, mod_type)]

    def updateOutputVar(self, cell_type, mod_type, varname):
        ''' Update the selected variable in a graph dropdown menu on neuron switch. '''
        vargroups = [v['label'] for v in self.getVars(cell_type, mod_type)]
        if varname not in vargroups:
            varname = vargroups[0]
        return varname

    def validateInputs(self, inputs, refparams):
        ''' Convert inputs to float and check validity. '''

        # converting to float and optional rescaling
        values = [float(x) / p.get('factor', 1)
                  for x, p in zip(inputs, refparams.values())]

        mins = [min(p['values']) for p in refparams.values()],
        maxs = [max(p['values']) for p in refparams.values()],
        mins = mins[0]
        maxs = maxs[0]
        checkNumBounds(values, list(zip(mins, maxs)))
        return values

    def getSlidersValues(self, inputs, refparams):
        ''' Get the parameters values corresponding to the sliders positions. '''
        return [p['values'][inputs[i]] for i, p in enumerate(refparams.values())]

    def isSubmitButtonTriggered(self, nsubmits):
        ''' Determine whether or not the callback comes from a submit event. '''
        if isinstance(nsubmits, int) and nsubmits == self.prev_nsubmits + 1:
            self.prev_nsubmits += 1
            return True
        else:
            return False

    def propagateInputs(self, mech_type, i_diam, mod_type, is_standard, i_US_freq, i_US_amp,
                        i_US_PRF, i_US_DC, i_elec_amp, i_elec_PRF, i_elec_DC, nsubmits, varname,
                        US_freq_input, US_amp_input, US_PRF_input, US_DC_input, elec_amp_input,
                        elec_PRF_input, elec_DC_input):
        ''' Translate inputs into parameters and propagate callback to updateCurve. '''

        is_submit = self.isSubmitButtonTriggered(nsubmits)
        refparams = self.stim_params[mod_type]

        # Determine parameters
        a = self.cell_params['diam']['values'][i_diam]
        try:
            if mod_type == 'US':
                if is_submit:
                    Fdrive, A, PRF, DC = self.validateInputs(
                        (US_freq_input, US_amp_input, US_PRF_input, US_DC_input), refparams)
                else:
                    Fdrive, A, PRF, DC = self.getSlidersValues(
                        (i_US_freq, i_US_amp, i_US_PRF, i_US_DC),
                        refparams)
            else:
                Fdrive = None
                if is_submit:
                    A, PRF, DC = self.validateInputs(
                        (elec_amp_input, elec_PRF_input, elec_DC_input), refparams)
                else:
                    A, PRF, DC = self.getSlidersValues((i_elec_amp, i_elec_PRF, i_elec_DC), refparams)
        except ValueError:
            print('Error in custom inputs')
            Fdrive = A = PRF = DC = None
        new_params = [mech_type, a, mod_type, Fdrive, A, self.tstim, PRF, DC * 1e-2]

        # print('current params: {}, new params: {}'.format(self.current_params, new_params))

        # Handle incorrect submissions
        if A is None:
            self.data = None

        # Load new data if parameters have changed
        elif new_params != self.current_params:
            print('getting data for new set of parameters')
            self.current_params = new_params
            self.getData(*self.current_params)

        # Update graph accordingly
        return self.updateGraph(None, varname, mech_type, mod_type, 'out0-graph')

    def getData(self, mech_type, a, mod_type, Fdrive, A, tstim, PRF, DC):
        ''' Update data either by loading a pre-computed simulation file from the remote server
            or by running a custom simulation locally. '''

        # Get file name from parameters
        filename = self.getFileName(mech_type, a, mod_type, Fdrive, A, tstim, PRF, DC)
        self.localfilepath = '{}/{}'.format(self.tmpdir, filename)

        # Check file existence on server
        remotedir = self.getRemoteDir(mech_type, a, mod_type)
        remotefilepath = '{}/{}'.format(remotedir, filename)
        remotefileexists = self.ssh_channel.isfile(remotefilepath)

        # Download if available
        if remotefileexists:
            print('downloading "{}.pkl" file from server...'.format(filename))
            t0 = time.time()
            self.ssh_channel.get(remotefilepath, localpath=self.localfilepath)

        # Otherwise run simulation
        else:
            print('"{}" file not found on server'.format(remotefilepath))
            neuron = self.neurons[mech_type]
            tstop = self.tbounds[1]
            if mod_type == 'elec':
                logfilepath = '{}/log_ESTIM.xlsx'.format(self.tmpdir)
                worker = EStimWorker(1, self.tmpdir, logfilepath, SolverElec(), neuron,
                                     A, tstim * 1e-3, (tstop - tstim) * 1e-3, PRF, DC, 1)
            else:
                logfilepath = '{}/log_ASTIM.xlsx'.format(self.tmpdir)
                worker = AStimWorker(1, self.tmpdir, logfilepath, SolverUS(a, neuron, Fdrive), neuron,
                                     Fdrive, A, tstim * 1e-3, (tstop - tstim) * 1e-3, PRF, DC,
                                     'sonic', 1)
            t0 = time.time()
            print(worker)
            outfilepath = worker.__call__()

            assert outfilepath == self.localfilepath, 'Local filepath not matching'

        # Load data from downloaded/generated local file and delete it afterwards
        with open(self.localfilepath, 'rb') as pkl_file:
            frame = pickle.load(pkl_file)
            self.data = frame['data']
        if os.path.isfile(self.localfilepath):
            os.remove(self.localfilepath)
            print('file data loaded in {}s'.format(si_format((time.time() - t0), space=' ')))


    def getFileName(self, mech_type, a, mod_type, Fdrive, A, tstim, PRF, DC):
        ''' Get simulation filename for the given parameters.

            :param mech_type: type of ssh_channel mechanism (cell-type specific)
            :param a: Sonophore diameter (m)
            :param mod_type: stimulation modality ('US' or 'elec')
            :param Fdrive: Ultrasound frequency (Hz) for A-STIM / None for E-STIM
            :param A: Acoustic amplitude (Pa) for A-STIM / electrical amplitude (mA/m2) for E-STIM
            :param tstim: Stimulus duration (s)
            :param PRF: Pulse-repetition frequency (Hz)
            :param DC: duty cycle (-)
            :return: filename
        '''
        if mod_type == 'elec':
            if DC == 1.0:
                filecode = 'ESTIM_{}_CW_{:.1f}mA_per_m2_{:.0f}ms'.format(mech_type, A, tstim)
            else:
                filecode = 'ESTIM_{}_PW_{:.1f}mA_per_m2_{:.0f}ms_PRF{:.2f}Hz_DC{:.2f}%'.format(
                    mech_type, A, tstim, PRF, DC * 1e2)
        else:
            if DC == 1.0:
                filecode = 'ASTIM_{}_CW_{:.0f}nm_{:.0f}kHz_{:.1f}kPa_{:.0f}ms_effective'.format(
                    mech_type, a * 1e9, Fdrive * 1e-3, A * 1e-3, tstim)
            else:
                filecode = ('ASTIM_{}_PW_{:.0f}nm_{:.0f}kHz_{:.1f}kPa_{:.0f}ms' +
                            '_PRF{:.2f}Hz_DC{:.2f}%_effective').format(mech_type, a * 1e9,
                                                                       Fdrive * 1e-3, A * 1e-3, tstim,
                                                                       PRF, DC * 1e2)
        return '{}.pkl'.format(filecode)


    def getRemoteDir(self, mech_type, a, mod_type):
        ''' Get remote directory that potentially holds the simulation file corresponding to
            the given parameters.

            :param mech_type: type of ssh_channel mechanism (cell-type specific)
            :param a: Sonophore diameter (m)
            :param mod_type: stimulation modality ('US' or 'elec')
        '''
        if mod_type == 'elec':
            return '{}/EL/{}'.format(self.remoteroot, mech_type)
        else:
            return '{}/US/{:.0f}nm/{}'.format(self.remoteroot, a * 1e9, mech_type)


    def updateGraph(self, _, varname, mech_type, mod_type, id):
        ''' Update graph with new data.

            :param _: 1st vgraph figure content (used to trigger callback for subsequent graphs)
            :param varname: name of the output variable to display
            :param mech_type: type of ssh_channel mechanism (cell-type specific)
            :param mod_type: type of stimulation modality (US or elec)
            :param id: id of the graph to update
            :return: graph content
        '''

        # Get graph-specific colorset
        igraph = int(id[3])
        colors = self.colorset[2 * igraph: 2 * (igraph + 1)]

        # Get info about variables to plot
        varlist = neuronvars[mech_type]['vars_{}'.format(mod_type)]

        vargroups = [v['label'] for v in varlist]
        if varname not in vargroups:
            varname = vargroups[0]
        for v in varlist:
            if v['label'] == varname:
                pltvar = v
                break

        if self.data is not None:

            # Get time, states and output variable vectors
            t = self.data['t'].values
            varlist = [self.data[v].values for v in pltvar['names']]
            states = self.data['states'].values

            # Determine patches location
            npatches, tpatch_on, tpatch_off = getPatchesLoc(t, states)

            # Add onset
            dt = t[1] - t[0]
            tplot = np.hstack((np.array([self.tbounds[0] * 1e-3, -dt]), t))
            varlistplot = []
            for name, var in zip(pltvar['names'], varlist):
                if name is 'Vm':
                    var0 = neuronvars[mech_type]['Vm0']
                else:
                    var0 = var[0]
                varlistplot.append(np.hstack((np.array([var0] * 2), var)))

            # Define curves
            curves = [
                {
                    'name': pltvar['names'][i],
                    'x': tplot * 1e3,
                    'y': varlistplot[i] * pltvar['factor'],
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
                'range': self.tbounds,
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


    # def crossFilterTime(self, input_relayout, self_relayout):

    #     # If no input relayout -> no changes
    #     if input_relayout is None:
    #         output_layout = self_layout
    #     else:
    #         print(input_layout)

    #         # if input layout on autosize or autorange -> return autosize
    #         if input_layout in [
    #             {'autosize': True},
    #             {'xaxis.autorange': True, 'yaxis.autorange': True}
    #         ]:
    #             print('autosized input -> autosize output')
    #             return {'autosize': True}

    #         # if xaxis range is specified in input layout -> return this xaxis range (but not yaxis)
    #         elif 'xaxis.range[0]' in input_layout:
    #             print('x-ranged input -> x-ranged output')
    #             tmin = input_layout['xaxis.range[0]']
    #             tmax = input_layout['xaxis.range[1]']
    #             return{'xaxis.range[0]': tmin, 'xaxis.range[1]': tmax}
    #         #     print('time range: {:.0f} - {:.0f} ms'.format(tmin, tmax))

    #         # if not handled -> return self_layout
    #         else:
    #             print('not handled !!!')
    #             output_layout = self_layout

    #     return self_figure


    def updateInfoTable(self, _):
        ''' Update the content of the output metrics table on neuron/modality/stimulation change. '''

        # Spike detection
        if self.data is not None:
            t = self.data['t']
            dt = t[1] - t[0]
            mpd = int(np.ceil(SPIKE_MIN_DT / dt))
            ipeaks, *_ = findPeaks(self.data['Qm'].values, SPIKE_MIN_QAMP, mpd, SPIKE_MIN_QPROM)
            nspikes = ipeaks.size
            lat = t[ipeaks[0]] if nspikes > 0 else None
            sr = np.mean(1 / np.diff(t[ipeaks])) if nspikes > 1 else None
        else:
            nspikes = 0
            lat = None
            sr = None

        return dataRows(labels=['# spikes', 'Latency', 'Firing rate'],
                        values=[nspikes, lat, sr],
                        units=['', 's', 'Hz'])

    def updateDownloadContent(self, _):
        ''' Update the content of the downloadable pandas dataframe. '''
        csv_string = self.data.to_csv(index=False, encoding='utf-8')
        csv_string = "data:text/csv;charset=utf-8," + urllib.parse.quote(csv_string)
        return csv_string

    def updateDownloadName(self, _):
        ''' Update the name of the downloadable pandas dataframe. '''
        filecode = os.path.splitext(os.path.basename(self.localfilepath))[0]
        return '{}.csv'.format(filecode)
