#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2017-06-22 16:57:14
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-03-04 16:26:54

''' Definition of the SONICViewer class. '''

import time
import urllib
import numpy as np
import pandas as pd

import dash
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go

from PySONIC.postpro import findPeaks
from PySONIC.constants import *
from PySONIC.neurons import getNeuronsDict
from PySONIC.utils import si_prefixes, getStimPulses, isWithin
from ExSONIC._0D import Sonic0D

from .components import *


def getDefaultIndexes(params, defaults):
    ''' Return the indexes of default values found in lists of parameters.

        :param params: dictionary of parameter arrays
        :param defaults: dictionary of default values
        :return: dictionary of resolved default indexes
    '''
    idefs = {}
    for key, default in defaults.items():
        imatches = np.where(np.isclose(params[key], default, rtol=1e-9, atol=1e-16))[0]
        if len(imatches) == 0:
            raise ValueError('default {} ({}) not found in parameter values'.format(key, default))
        else:
            idefs[key] = imatches[0]
    return idefs



class SONICViewer(dash.Dash):
    ''' SONIC viewer application inheriting from dash.Dash. '''

    def __init__(self, inputparams, inputdefaults, pltparams, celltypes, ngraphs=1):

        # Initialize Dash app
        super(SONICViewer, self).__init__(
            name='viewer',
            url_base_pathname='/viewer/',
            csrf_protect=True
        )
        self.title = 'SONIC viewer'

        # Initialize constant parameters
        self.prefixes = {v: k for k, v in si_prefixes.items()}
        self.ngraphs = ngraphs
        self.colorset = pltparams['colorset']
        self.tbounds = pltparams['tbounds']  # ms
        self.celltypes = celltypes

        # Initialize parameters that will change upon requests
        self.prev_nsubmits = 0
        self.current_params = None
        self.data = None

        self.spatialdist = False

        # Initialize cell and stimulation parameters
        idefs = getDefaultIndexes(inputparams, inputdefaults)
        self.neurons = {key: getNeuronsDict()[key]() for key in self.celltypes.keys()}
        self.cell_params = {
            'mech': dict(label='Cell Type', values=list(self.celltypes.keys()), idef=0),
            'diam': dict(label='Sonophore diameter', values=inputparams['diams'],
                         idef=idefs['diams'], unit='m')
        }
        self.stim_params = {
            'US': {
                'freq': dict(label='Frequency', values=inputparams['US_freqs'],
                             unit='Hz', factor=1e-3, idef=idefs['US_freqs']),
                'amp': dict(label='Amplitude', values=inputparams['US_amps'],
                            unit='Pa', factor=1e-3, idef=idefs['US_amps']),
                'PRF': dict(label='PRF', values=inputparams['PRFs'], unit='Hz', idef=idefs['PRFs']),
                'DC': dict(label='Duty Cycle', values=inputparams['DCs'], unit='%',
                           idef=idefs['DCs'])},
            'elec': {
                'amp': dict(label='Amplitude', values=inputparams['elec_amps'], unit='mA/m2',
                            idef=idefs['elec_amps']),
                'PRF': dict(label='PRF', values=inputparams['PRFs'], unit='Hz', idef=idefs['PRFs']),
                'DC': dict(label='Duty Cycle', values=inputparams['DCs'], unit='%',
                           idef=idefs['DCs'])}
        }
        self.tstim = inputparams['tstim']

        # Initialize UI layout components
        default_cell = self.cell_params['mech']['values'][self.cell_params['mech']['idef']]
        default_mod = 'US'
        self.setLayout(default_cell, default_mod)

        # Link UI components callbacks to appropriate functions
        self.registerCallbacks()


    def __str__(self):
        return '{} app with {} graphs'.format(self.title, self.ngraphs)

    # ------------------------------------------ LAYOUT ------------------------------------------

    def setLayout(self, default_cell, default_mod):
        ''' Set app layout. '''
        self.layout = html.Div(id='body', children=[

            # Header
            self.header(),
            separator(),

            # Content
            html.Div(id='content', children=[
                # Left side
                html.Div(id='left-col', className='content-column', children=[
                    self.cellPanel(default_cell),
                    self.stimPanel(default_mod),
                    self.metricsPanel(),
                ]),

                # Right side
                html.Div(id='right-col', className='content-column', children=[
                    self.outputPanel(default_cell, default_mod)
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
            'Developed with ', html.A('Dash', href='https://dash.plot.ly/'), '. ',
            'Powered by ', html.A('NEURON', href='https://www.neuron.yale.edu/neuron/'), '. ',
            html.Br(),
            'Translational Neural Engineering Lab, EPFL - 2018',
            html.Br(),
            'contact: ', html.A('theo.lemaire@epfl.ch', href='mailto:theo.lemaire@epfl.ch')
        ])

    def cellPanel(self, default_cell):
        ''' Construct cell parameters panel. '''
        return collapsablePanel('Cell parameters', children=[
            html.Table(className='table', children=[
                html.Tr([
                    html.Td(self.cell_params['mech']['label'], style={'width': '35%'}),
                    html.Td(style={'width': '65%'}, children=[
                        dcc.Dropdown(
                            id='mechanism-type',
                            options=[{'label': v.desc, 'value': k} for k, v in self.celltypes.items()],
                            value=default_cell),
                        html.Br(),
                        html.Div(id='membrane-currents'),
                    ])]),

                labeledSliderRow(self.cell_params['diam']['label'], 'diam-slider',
                                 len(self.cell_params['diam']['values']),
                                 value=self.cell_params['diam']['idef'])
            ])
        ])

    def stimPanel(self, default_mod):
        ''' Construct stimulation parameters panel. '''

        return collapsablePanel('Stimulation parameters', children=[

            dcc.Tabs(id='modality-tabs', className='tabs', value=default_mod, children=[
                dcc.Tab(label='Ultrasound', value='US'),
                dcc.Tab(label='Electricity', value='elec')]),

            labeledToggleSwitch(
                'toggle-stim-inputs',
                labelLeft='Standard',
                labelRight='Custom',
                value=False,
                boldLabels=True
            ),

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

    def outputPanel(self, default_cell, default_mod):
        ddgraphpanels = []
        pltvars = self.celltypes[default_cell].pltvars
        for i in range(self.ngraphs):
            ddgraphpanels.append(collapsablePanel(title=None, children=[ddGraph(
                id='out{}'.format(i),
                labels=[v.desc for v in pltvars],
                values=[v.label for v in pltvars],
                default=pltvars[i].label,
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
            Output('membrane-currents', 'children'),
            [Input('mechanism-type', 'value')])(self.updateMembraneCurrents)

        # Cell panel: sliders
        for p in ['diam']:
            id = '{}-slider'.format(p)
            self.callback(Output(id, 'marks'), [Input(id, 'value')])(self.updateSlider(
                self.cell_params[p]))

        # Stimulation panel: tables visibility
        for table_mod in ['US', 'elec']:
            for table_type in ['slider', 'input']:
                key = '{}-{}' .format(table_mod, table_type)
                is_standard_table = table_type == 'slider'
                self.callback(
                    Output('{}-table'.format(key), 'hidden'),
                    [Input('modality-tabs', 'value'),
                     Input('toggle-stim-inputs', 'value')])(self.showTable(table_mod, is_standard_table))

        self.callback(
            Output('inputs-submit-div', 'hidden'),
            [Input('toggle-stim-inputs', 'value')])(self.hideSubmitButton)

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
                [Input('mechanism-type', 'value')])(self.updateOutputOptions)
            self.callback(
                Output('out{}-dropdown'.format(i), 'value'),
                [Input('mechanism-type', 'value')],
                state=[State('out{}-dropdown'.format(i), 'value')])(self.updateOutputVar)

        # 1st graph
        self.callback(
            Output('out0-graph', 'figure'),
            [Input('mechanism-type', 'value'),
             Input('diam-slider', 'value'),
             Input('modality-tabs', 'value'),
             Input('toggle-stim-inputs', 'value'),
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
                 Input('out{}-graph'.format(0), 'relayoutData'),
                 Input('out{}-dropdown'.format(i), 'value')],
                [State('mechanism-type', 'value'),
                 State('out{}-graph'.format(i), 'id')])(self.updateGraph)

        # Download link
        self.callback(
            Output('download-link', 'href'),
            [Input('out0-graph', 'figure')])(self.updateDownloadContent)
        self.callback(
            Output('download-link', 'download'),
            [Input('out0-graph', 'figure')])(self.updateDownloadName)

    def updateMembraneCurrents(self, cell_type):
        ''' Update the list of membrane currents on neuron switch. '''
        currents = self.celltypes[cell_type].currents
        return unorderedList(['{} ({})'.format(c.desc, c.name) for c in currents])

    def showTableGeneric(self, stim_mod, is_custom, table_mod, is_standard_table):
        return not (stim_mod == table_mod and is_custom != is_standard_table)

    def showTable(self, table_mod, is_standard_table):
        ''' For correct assignment of updateSlider functions with lambda expressions. '''
        return lambda x, y: self.showTableGeneric(x, y, table_mod, is_standard_table)

    def hideSubmitButton(self, is_custom):
        ''' Show submit button only when stimulation panel is in input mode. '''
        return not is_custom

    def updateSliderGeneric(self, values, curr, factor=1, precision=0, suffix=''):
        ''' Generic function to update a slider value. '''
        return {i: '{}{}'.format(si_format(values[i], precision, space=' '), suffix)
                   if i == curr else '' for i in range(len(values))}

    def updateSlider(self, p):
            ''' For correct assignment of updateSlider functions with lambda expressions. '''
            return lambda x: self.updateSliderGeneric(p['values'], x, suffix=p['unit'])

    def updateOutputOptions(self, cell_type):
        ''' Update the list of available variables in a graph dropdown menu on neuron switch. '''
        return [{'label': v.desc, 'value': v.label} for v in self.celltypes[cell_type].pltvars]

    def updateOutputVar(self, cell_type, varname):
        ''' Update the selected variable in a graph dropdown menu on neuron switch. '''
        varlabels = [v.label for v in self.celltypes[cell_type].pltvars]
        if varname not in varlabels:
            varname = varlabels[0]
        return varname

    def validateInputs(self, inputs, refparams):
        ''' Convert inputs to float and check validity. '''

        # Convert to float and optional rescaling
        values = [float(x) / p.get('factor', 1)
                  for x, p in zip(inputs, refparams.values())]
        mins = [min(p['values']) for p in refparams.values()]
        maxs = [max(p['values']) for p in refparams.values()]

        # Check parameters against reference bounds
        for i in range(len(values)):
            values[i] = isWithin('', values[i], (mins[i], maxs[i]))

        # Return values
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

    def propagateInputs(self, mech_type, i_diam, mod_type, is_custom, i_US_freq, i_US_amp,
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

        # Handle incorrect submissions
        if A is None:
            self.data = None

        # Load new data if parameters have changed
        elif new_params != self.current_params:
            print('getting data for new set of parameters')
            self.current_params = new_params
            self.getData(*self.current_params)

        # Update graph accordingly
        return self.updateGraph(None, None, varname, mech_type, 'out0-graph')

    def getData(self, mech_type, a, mod_type, Fdrive, A, tstim, PRF, DC):
        ''' Run NEURON simulaiton to update data.

            :param mech_type: type of mechanism (cell-type specific)
            :param a: Sonophore diameter (m)
            :param mod_type: stimulation modality ('US' or 'elec')
            :param Fdrive: Ultrasound frequency (Hz) for A-STIM / None for E-STIM
            :param A: Acoustic amplitude (Pa) for A-STIM / electrical amplitude (mA/m2) for E-STIM
            :param tstim: Stimulus duration (s)
            :param PRF: Pulse-repetition frequency (Hz)
            :param DC: duty cycle (-)
        '''
        tstart = time.time()

        # Initialize 0D NEURON model
        neuron = self.neurons[mech_type]
        tstop = self.tbounds[1]
        if mod_type == 'elec':
            model = Sonic0D(neuron)
            model.setIinj(A)
        else:
            model = Sonic0D(neuron, a=a * 1e9, Fdrive=Fdrive * 1e-3)
            model.setUSdrive(A * 1e-3)

        # Run simulation
        (t, y, stimon) = model.simulate(tstim * 1e-3, (tstop - tstim) * 1e-3, PRF, DC)
        Qm, Vm, *states = y

        # Store output in dataframe
        self.data = pd.DataFrame({'t': t, 'states': stimon, 'Qm': Qm, 'Vm': Vm})
        for j in range(len(neuron.states_names)):
            self.data[neuron.states_names[j]] = states[j]

        tcomp = time.time() - tstart
        print('data loaded in {}s'.format(si_format(tcomp, space=' ')))


    def getFileCode(self, mech_type, a, mod_type, Fdrive, A, tstim, PRF, DC):
        ''' Get simulation filecode for the given parameters.

            :param mech_type: type of mechanism (cell-type specific)
            :param a: Sonophore diameter (m)
            :param mod_type: stimulation modality ('US' or 'elec')
            :param Fdrive: Ultrasound frequency (Hz) for A-STIM / None for E-STIM
            :param A: Acoustic amplitude (Pa) for A-STIM / electrical amplitude (mA/m2) for E-STIM
            :param tstim: Stimulus duration (s)
            :param PRF: Pulse-repetition frequency (Hz)
            :param DC: duty cycle (-)
            :return: filecode
        '''
        PW_str = '_PRF{:.2f}Hz_DC{:.2f}%'.format(PRF, DC * 1e2) if DC < 1.0 else ''
        W_str = 'PW' if DC < 1.0 else 'CW'
        if mod_type == 'elec':
            filecode = 'ESTIM_{}_{}_{:.1f}mA_per_m2_{:.0f}ms{}'.format(
                mech_type, W_str, A, tstim, PW_str)
        else:
            filecode = 'ASTIM_{}_{}_{:.0f}nm_{:.0f}kHz_{:.1f}kPa_{:.0f}ms{}'.format(
                mech_type, W_str, a * 1e9, Fdrive * 1e-3, A * 1e-3, tstim, PW_str)
        return filecode

    def updateGraph(self, _, relayout_data, varname, mech_type, id):
        ''' Update graph with new data.

            :param _: input graph figure content (used to trigger callback for subsequent graphs)
            :param relayout_data: input graph relayout data
            :param varname: name of the output variable to display
            :param mech_type: type of mechanism (cell-type specific)
            :param id: id of the graph to update
            :return: graph content
        '''

        # Get the x-range of the zoomed in data
        startx = 'xaxis.range[0]' in relayout_data if relayout_data else None
        endx = 'xaxis.range[1]' in relayout_data if relayout_data else None
        sliderange = 'xaxis.range' in relayout_data if relayout_data else None
        if startx and endx:
            xrange = [relayout_data['xaxis.range[0]'], relayout_data['xaxis.range[1]']]
        elif startx and not endx:
            xrange = [relayout_data['xaxis.range[0]'], thedates.max()]
        elif not startx and endx:
            xrange = [thedates.min(), relayout_data['xaxis.range[1]']]
        elif sliderange:
            xrange = relayout_data['xaxis.range']
        else:
            xrange = None

        # Get graph-specific colorset
        igraph = int(id[3])
        colors = self.colorset[2 * igraph: 2 * (igraph + 1)]

        # Get info about variables to plot
        varlist = self.celltypes[mech_type].pltvars
        varlabels = [v.label for v in varlist]
        if varname not in varlabels:
            varname = varlabels[0]
        for v in varlist:
            if v.label == varname:
                pltvar = v
                break

        if self.data is not None:

            # Get time vector and add onset
            t = self.data['t'].values
            dt = t[1] - t[0]
            tonset = np.array([self.tbounds[0] * 1e-3, -dt])
            tplot = np.hstack((tonset, t))

            # Get states vector and Determind patches location
            states = self.data['states'].values
            npatches, tpatch_on, tpatch_off = getStimPulses(t, states)

            # Get vector(s) of variable(s) to plot, rescaled and with appropriate onset
            yplot = pltvar.getData(self.data, nonset=len(tonset))

            # Define curve objects
            curves = [
                go.Scatter(
                    x=tplot * 1e3,
                    y=yplot[i],
                    mode='lines',
                    name=pltvar.names[i],
                    line={'color': colors[i]},
                    showlegend=True
                ) for i in range(len(yplot))
            ]

            # Define stimulus patches
            patches = [
                go.Scatter(
                    x=np.array([tpatch_on[i], tpatch_off[i], tpatch_off[i], tpatch_on[i]]) * 1e3,
                    y=np.array([pltvar.bounds[0]] * 2 + [pltvar.bounds[1]] * 2),
                    mode='none',
                    fill='toself',
                    fillcolor='grey',
                    opacity=0.2,
                    showlegend=False
                ) for i in range(npatches)
            ]

        # If file does not exist, define empty curve and patches
        else:
            curves = []
            patches = []

        # Set axes layout
        layout = go.Layout(
            xaxis={
                'type': 'linear',
                'title': 'time (ms)',
                'range': self.tbounds if xrange is None else xrange,
                'zeroline': False
            },
            yaxis={
                'type': 'linear',
                'title': '{} ({})'.format(pltvar.label, pltvar.unit),
                'range': pltvar.bounds,
                'zeroline': False
            },
            margin={'l': 60, 'b': 40, 't': 10, 'r': 10},
            title=''
        )

        # Return curve, patches and layout objects
        return {'data': [*curves, *patches], 'layout': layout}

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
        return '{}.csv'.format(self.getFileCode(*self.current_params))
