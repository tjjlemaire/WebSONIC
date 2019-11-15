# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2017-06-22 16:57:14
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-11-15 01:21:31

''' Definition of the SONICViewer class. '''

import time
import urllib
import numpy as np
import pandas as pd
import dash
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go

from PySONIC.postpro import detectSpikes
from PySONIC.constants import *
from PySONIC.core import PulsedProtocol
from PySONIC.neurons import getNeuronsDict
from PySONIC.utils import si_prefixes, isWithin, getIndex
from PySONIC.plt import GroupedTimeSeries, extractPltVar
from ExSONIC.core import IintraNode, SonicNode

from .components import *


class SONICViewer(dash.Dash):
    ''' SONIC viewer application inheriting from dash.Dash. '''

    tscale = 1e3  # time scaling factor

    def __init__(self, input_params, plt_params, ngraphs, no_run=False, verbose=False):

        # Initialize Dash app
        super(SONICViewer, self).__init__(
            name='viewer',
            url_base_pathname='/viewer/',
            external_stylesheets=[dbc.themes.BOOTSTRAP]
        )
        self.title = 'SONIC viewer'

        # Initialize constant parameters
        self.prefixes = {v: k for k, v in si_prefixes.items()}
        self.ngraphs = ngraphs
        self.colors = plt_params['colors']
        self.no_run = no_run
        self.verbose = verbose

        # Initialize parameters that will change upon requests
        self.prev_nsubmits = 0
        self.current_params = None
        self.data = None

        # Initialize neuron objects
        self.pneurons = {
            key: getNeuronsDict()[key]()
            for key in input_params['cell_type']['values']}

        # Initialize cell and stimulation parameters
        self.cell_params = {
            x: self.parseParam(input_params[x])
            for x in ['cell_type', 'sonophore_radius', 'sonophore_coverage_fraction']
        }
        self.stim_params = {
            'US': {
                x if '_US' in x else '{}_US'.format(x): self.parseParam(input_params[x])
                for x in ['f_US', 'A_US', 'tstim', 'PRF', 'DC']
            },
            'elec': {
                x if '_elec' in x else '{}_elec'.format(x): self.parseParam(input_params[x])
                for x in ['A_elec', 'tstim', 'PRF', 'DC']
            }
        }

        # Initialize plot variables and plot scheme
        default_cell = input_params['cell_type']['default']
        self.pltvars = self.pneurons[default_cell].getPltVars()
        self.pltscheme = self.pneurons[default_cell].getPltScheme()

        # Initialize UI layout components
        self.setLayout(default_cell, 'US')

        # Link UI components callbacks to appropriate functions
        self.registerCallbacks()


    @staticmethod
    def parseParam(p):
        parsed_p = {
            'label': p['label'],
            'values': p['values'],
            'idef': getIndex(p['values'], p['default']),
            'unit': p.get('unit', None),
            'disabled': p.get('disabled', False),
        }
        if 'factor' in p:
            parsed_p['factor'] = p['factor']
        if 'allow_titrate' in p:
            parsed_p['allow_titrate'] = p['allow_titrate']
        return parsed_p


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
            separator(),
            self.footer()
        ])

    @staticmethod
    def header():
        ''' Set app header. '''
        return html.Div(id='header', children=[

            html.Div(className='header-side', id='header-left', children=[
                html.A(html.Img(src='assets/EPFL.svg', className='logo'),
                       href='https://www.epfl.ch')]),

            html.Div(className='header-side', id='header-right', children=[
                html.A(html.Img(src='assets/ITIS.svg', className='logo'),
                       href='https://www.itis.ethz.ch')]),

            html.Div(id='header-middle', children=[
                html.H1('Ultrasound Neuromodulation: exploring predictions of the SONIC model',
                        className='header-txt'),
                html.Br()
            ])
        ])

    @classmethod
    def footer(cls):
        ''' Set app footer. '''
        return html.Div(id='footer', children=[
            html.Span([
                'Ref: Lemaire, T., Neufeld, E., Kuster, N., and Micera, S. (2019). ',
                html.A(html.I('Understanding ultrasound neuromodulation using a computationally\
                              efficient and interpretable model of intramembrane cavitation. '),
                       href='https://iopscience.iop.org/article/10.1088/1741-2552/ab1685'),
                'J. Neural Eng. '], id='ref'),
            html.Br(),
            'Developed with ', html.A('Dash', href='https://dash.plot.ly/'), '. ',
            'Powered by ', html.A('NEURON', href='https://www.neuron.yale.edu/neuron/'), '.',
            html.Br(),
            'Translational Neural Engineering Lab, EPFL - 2019',
            html.Br(),
            'contact: ', html.A('theo.lemaire@epfl.ch', href='mailto:theo.lemaire@epfl.ch'),
            html.Br(),
            '>>> ', html.A('About', id='about-link'), ' <<<',
            dbc.Modal(
                id='about-modal',
                size='lg',
                scrollable=True,
                centered=True,
                children=[
                    dbc.ModalHeader('About'),
                    dbc.ModalBody(dcc.Markdown(f'''{cls.about()}''')),
                    dbc.ModalFooter(dbc.Button('Close', id='close-about', className='ml-auto')),
                ]
            )
        ])

    @staticmethod
    def about():
        with open('about.md', encoding="utf8") as f:
            return f.read()

    def cellPanel(self, default_cell):
        ''' Construct cell parameters panel. '''
        return collapsablePanel('Cell parameters', children=[
            html.Table(className='table', children=[
                html.Tr([
                    html.Td(self.cell_params['cell_type']['label'], className='row-label'),
                    html.Td(className='row-data', children=[
                        dcc.Dropdown(
                            className='ddlist',
                            id='cell_type-dropdown',
                            options=[{
                                'label': '{} ({})'.format(self.pneurons[name].description(), name),
                                'value': name
                            } for name in self.pneurons.keys()],
                            value=default_cell),
                        html.Div(id='membrane-currents'),
                    ])]),

                *[labeledSliderRow(
                    self.cell_params[k]['label'], f'{k}-slider',
                    len(self.cell_params[k]['values']),
                    value=self.cell_params[k]['idef'],
                    disabled=self.cell_params[k]['disabled'])
                  for k in ['sonophore_radius', 'sonophore_coverage_fraction']]
            ])
        ])

    def stimPanel(self, default_mod):
        ''' Construct stimulation parameters panel. '''

        return collapsablePanel('Stimulation parameters', children=[

            labeledToggleSwitch(
                'toggle-stim-inputs',
                labelLeft='Sliders',
                labelRight='Inputs',
                value=True,
                boldLabels=True
            ),

            dcc.Tabs(id='modality-tabs', className='tabs', value=default_mod, children=[
                dcc.Tab(label='LIFUS', value='US'),
                dcc.Tab(label='Intracellular current', value='elec')]),


            *[labeledSlidersTable(
                '{}-slider-table'.format(mod_type),
                labels=[p['label'] for p in self.stim_params[mod_type].values()],
                ids=['{}-slider'.format(p) for p in self.stim_params[mod_type].keys()],
                sizes=[len(p['values']) for p in self.stim_params[mod_type].values()],
                values=[p['idef'] for p in self.stim_params[mod_type].values()])

                for mod_type in self.stim_params.keys()],


            html.Div(id='inputs-form', className='input-div', hidden=False, children=[

                *[labeledInputsTable(
                    '{}-input-table'.format(mod_type),
                    labels=['{} ({}{})'.format(p['label'], self.prefixes[1 / p.get('factor', 1)],
                                               p['unit'])
                            for p in self.stim_params[mod_type].values()],
                    ids=['{}-input'.format(p)
                         for p in self.stim_params[mod_type].keys()],
                    mins=[min(p['values']) * p.get('factor', 1)
                          for p in self.stim_params[mod_type].values()],
                    maxs=[max(p['values']) * p.get('factor', 1)
                          for p in self.stim_params[mod_type].values()],
                    values=[p['values'][p['idef']] * p.get('factor', 1)
                            for p in self.stim_params[mod_type].values()],
                    allow_titrate=[p.get('allow_titrate', False)
                                   for p in self.stim_params[mod_type].values()])

                    for mod_type in self.stim_params.keys()],

                dbc.Alert(id='status-bar', color='success', is_open=True, children=['']),

                html.Div(id='inputs-submit-div', hidden=True, children=[
                    html.Button('Run', id='run-button', className='submit-button')
                ])
            ])
        ])

    @staticmethod
    def metricsPanel():
        return collapsablePanel('Output metrics', children=[
            html.Table(id='info-table', className='table')])

    def outputPanel(self, default_cell, default_mod):

        # Get options values and generate options labels
        values = list(self.pltscheme.keys())
        labels = self.getOutputDropDownLabels()

        ddgraphpanels = [
            panel(children=[ddGraph(
                id=str(i + 1),
                values=values,
                labels=labels,
                default=values[i])])
            for i in range(self.ngraphs)]

        return html.Div(children=[
            *ddgraphpanels,
            html.Div(id='download-wrapper', children=[
                html.A('Download Data', id='download-link', download="", href="", target="_blank")])
        ])

    # ------------------------------------------ CALLBACKS ------------------------------------------

    def registerCallbacks(self):

        # Cell panel: cell type
        self.callback(
            Output('membrane-currents', 'children'),
            [Input('cell_type-dropdown', 'value')])(self.updateMembraneCurrents)

        # Cell panel: radius slider
        for p in ['sonophore_radius', 'sonophore_coverage_fraction']:
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
        for refparams in self.stim_params.values():
            for key, p in refparams.items():
                id = '{}-slider'.format(key)
                self.callback(Output(id, 'marks'), [Input(id, 'value')])(self.updateSlider(p))

        # Output metrics table
        self.callback(
            Output('info-table', 'children'),
            [Input('graph1', 'figure')])(self.updateInfoTable)

        # Output panels
        for i in range(self.ngraphs):

            # drop-down list
            self.callback(
                Output('graph{}-dropdown'.format(i + 1), 'options'),
                [Input('cell_type-dropdown', 'value')])(self.updateOutputOptions)
            self.callback(
                Output('graph{}-dropdown'.format(i + 1), 'value'),
                [Input('cell_type-dropdown', 'value')],
                state=[State('graph{}-dropdown'.format(i + 1), 'value')])(self.updateOutputVar)

        # Update status bar
        self.callback(
            [Output('A_US-input', 'value'),
             Output('A_elec-input', 'value'),
             Output('status-bar', 'children'),
             Output('status-bar', 'color')],
            [Input('cell_type-dropdown', 'value'),
             Input('sonophore_radius-slider', 'value'),
             Input('sonophore_coverage_fraction-slider', 'value'),
             Input('modality-tabs', 'value'),
             Input('toggle-stim-inputs', 'value'),
             Input('f_US-slider', 'value'),
             Input('A_US-slider', 'value'),
             Input('tstim_US-slider', 'value'),
             Input('PRF_US-slider', 'value'),
             Input('DC_US-slider', 'value'),
             Input('A_elec-slider', 'value'),
             Input('tstim_elec-slider', 'value'),
             Input('PRF_elec-slider', 'value'),
             Input('DC_elec-slider', 'value'),
             Input('run-button', 'n_clicks')],
            [State('f_US-input', 'value'),
             State('A_US-input', 'value'),
             State('tstim_US-input', 'value'),
             State('PRF_US-input', 'value'),
             State('DC_US-input', 'value'),
             State('A_elec-input', 'value'),
             State('tstim_elec-input', 'value'),
             State('PRF_elec-input', 'value'),
             State('DC_elec-input', 'value')])(self.propagateInputs)

        # Update graphs whenever status bar or dropdown value changes,
        # or when 1st graph layout is changed
        for i in range(self.ngraphs):
            self.callback(
                Output(f'graph{i + 1}', 'figure'),
                [Input('status-bar', 'children'),
                 Input(f'graph{i + 1}-dropdown', 'value'),
                 Input('graph1', 'relayoutData')],
                [State('cell_type-dropdown', 'value'),
                 State('graph{}'.format(i + 1), 'id')])(self.updateGraph)

        # Download link
        self.callback(
            Output('download-link', 'href'),
            [Input('status-bar', 'children')])(self.updateDownloadContent)
        self.callback(
            Output('download-link', 'download'),
            [Input('status-bar', 'children')])(self.updateDownloadName)

        self.callback(
            Output('about-modal', 'is_open'),
            [Input('about-link', 'n_clicks'), Input('close-about', 'n_clicks')],
            [State('about-modal', 'is_open')]
        )(self.toggleAbout)

    @staticmethod
    def toggleAbout(n1, n2, is_open):
        if n1 or n2:
            return not is_open
        return is_open

    def updateMembraneCurrents(self, cell_type):
        ''' Update the list of membrane currents on neuron switch. '''
        currents = self.pneurons[cell_type].getCurrentsNames()
        return unorderedList(['{} ({})'.format(self.pltvars[c]['desc'], c) for c in currents])

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

    def getOutputDropDownLabels(self):
        ''' Generate output drop-down labels from pltscheme elements. '''
        labels = []
        for i, v in enumerate(list(self.pltscheme.keys())):
            ax_varnames = self.pltscheme[v]
            if len(ax_varnames) == 1:
                labels.append(self.pltvars[ax_varnames[0]]['desc'])
            elif v == 'I':
                labels.append('membrane currents')
            else:
                label = v
                for c in ['{', '}', '\\', '_', '^']:
                    label = label.replace(c, '')
                label = label.replace('kin.', 'kinetics')
                labels.append(label)
        return labels

    def updateOutputOptions(self, cell_type):
        ''' Update the list of available variables in a graph dropdown menu on neuron switch. '''

        # Update pltvars and pltscheme according to new cell type
        self.pltvars = self.pneurons[cell_type].getPltVars()
        self.pltscheme = self.pneurons[cell_type].getPltScheme()

        # Get options values and generate options labels
        values = list(self.pltscheme.keys())
        labels = self.getOutputDropDownLabels()

        # Return dictionary
        return [{'label': lbl, 'value': val} for lbl, val in zip(labels, values)]


    def updateOutputVar(self, cell_type, varname):
        ''' Update the selected variable in a graph dropdown menu on neuron switch. '''

        # Update pltvars and pltscheme according to new cell type
        self.pltvars = self.pneurons[cell_type].getPltVars()
        self.pltscheme = self.pneurons[cell_type].getPltScheme()

        # Get options values and generate options labels
        values = list(self.pltscheme.keys())
        if varname not in values:
            varname = values[0]

        return varname

    def validateInputs(self, inputs, refparams):
        ''' Convert inputs to float and check validity. '''

        mins = [min(p['values']) for p in refparams.values()]
        maxs = [max(p['values']) for p in refparams.values()]

        # Check parameters against reference bounds
        values = []
        for x, p in zip(inputs, refparams.values()):
            if x is None or x == '?' or x == '':
                values.append(None)
            else:
                # Convert to float and optional rescaling
                values.append(float(x) / p.get('factor', 1))

        for i in range(len(values)):
            if values[i] is not None:
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

    def propagateInputs(self, cell_type, i_radius, i_cov, mod_type, is_input, i_US_freq, i_US_amp,
                        i_US_tstim, i_US_PRF, i_US_DC, i_elec_amp, i_elec_tstim, i_elec_PRF,
                        i_elec_DC, nsubmits, US_freq_input, US_amp_input, US_tstim_input,
                        US_PRF_input, US_DC_input, elec_amp_input, elec_tstim_input,
                        elec_PRF_input, elec_DC_input):
        ''' Translate inputs into parameters and propagate callback to updateCurve. '''

        refparams = self.stim_params[mod_type]

        # Determine parameters
        a = self.cell_params['sonophore_radius']['values'][i_radius]
        fs = self.cell_params['sonophore_coverage_fraction']['values'][i_cov] * 1e-2

        try:
            if mod_type == 'US':
                if is_input:
                    Fdrive, A, tstim, PRF, DC = self.validateInputs(
                        (US_freq_input, US_amp_input, US_tstim_input, US_PRF_input, US_DC_input),
                        refparams)
                else:
                    Fdrive, A, tstim, PRF, DC = self.getSlidersValues(
                        (i_US_freq, i_US_amp, i_US_tstim, i_US_PRF, i_US_DC),
                        refparams)
            else:
                Fdrive = None
                if is_input:
                    A, tstim, PRF, DC = self.validateInputs(
                        (elec_amp_input, elec_tstim_input, elec_PRF_input, elec_DC_input),
                        refparams)
                else:
                    A, tstim, PRF, DC = self.getSlidersValues(
                        (i_elec_amp, i_elec_tstim, i_elec_PRF, i_elec_DC), refparams)
        except ValueError:
            print('Error in inputs')
            Fdrive = A = tstim = PRF = DC = None
        new_params = [cell_type, a, fs, mod_type, Fdrive, A, tstim, PRF, DC * 1e-2]
        iA = 5

        # Update plot variables if different cell type
        if self.current_params is None or cell_type != self.current_params[0]:
            self.pltvars = self.pneurons[cell_type].getPltVars()
            self.pltscheme = self.pneurons[cell_type].getPltScheme()

        # Load new data if parameters have changed
        is_successful = True
        if new_params != self.current_params:
            msg, A, is_successful = self.runSim(*new_params)
            if None in new_params:
                new_params[iA] = A
            self.current_params = new_params
        else:
            msg = None

        if msg is not None and is_successful:
            if mod_type == 'US':
                US_amp_input = A * 1e-3
            else:
                elec_amp_input = A

        color = 'success' if is_successful else 'danger'

        return US_amp_input, elec_amp_input, msg, color

    def getFakeData(self, pneuron, tstim, toffset):
        data = pd.DataFrame({
            't': np.array([0., tstim, tstim, tstim + toffset]),
            'stimstate': np.hstack((np.ones(2), np.zeros(2))),
            'Qm': pneuron.Qm0 * np.ones(4),
            'Vm': pneuron.Vm0 * np.ones(4)
        })
        for k in pneuron.states.keys():
            data[k] = 0.5 * np.ones(4)
        return data

    def runSim(self, cell_type, a, fs, mod_type, Fdrive, A, tstim, PRF, DC):
        ''' Run NEURON simulation to update data.

            :param cell_type: cell type
            :param a: Sonophore radius (m)
            :param fs: sonophore membrane coverage fraction (-)
            :param mod_type: stimulation modality ('US' or 'elec')
            :param Fdrive: Ultrasound frequency (Hz) for A-STIM / None for E-STIM
            :param A: Acoustic amplitude (Pa) for A-STIM / electrical amplitude (mA/m2) for E-STIM
            :param tstim: Stimulus duration (s)
            :param PRF: Pulse-repetition frequency (Hz)
            :param DC: duty cycle (-)
        '''

        # Initialize 0D NEURON model
        pneuron = self.pneurons[cell_type]
        toffset = 0.5 * tstim
        pp = PulsedProtocol(tstim, toffset, PRF, DC)

        # Create message
        if A is not None:
            keyword = 'simulation'
            Astr = 'A = {:.2f} {}, '.format(
                A * {'US': 1e-3, 'elec': 1.}[mod_type],
                {'US': 'kPa', 'elec': 'mA/m2'}[mod_type])
        else:
            keyword = 'titration'
            Astr = ''
        if mod_type == 'US':
            prefix = f'a = {(a * 1e9):.2f} nm, fs = {(fs * 1e2):.0f}%, f = {(Fdrive * 1e-3):.0f} kHz, '
        else:
            prefix = ''
        msg = f'running {mod_type} {keyword} on {pneuron.name} neuron ({prefix}{Astr}tstim = {tstim * 1e3} ms, PRF = {PRF} Hz, DC = {DC})'

        if self.verbose:
            print(msg)

        if self.no_run:
            self.data = self.getFakeData(pneuron, tstim, toffset)
        else:
            if mod_type == 'elec':
                model = IintraNode(pneuron)
            else:
                model = SonicNode(pneuron, a=a, Fdrive=Fdrive, fs=fs)
            try:
                if A is None:
                    A = model.titrate(pp)
                    if np.isnan(A):
                        raise ValueError('Impossible titration')
                self.data, _ = model.simulate(A, pp)
                is_successful = True
            except ValueError as err:
                print(err)
                msg = str(err)
                A = np.nan
                is_successful = False

        return [msg], A, is_successful

    def getFileCode(self, cell_type, a, fs, mod_type, Fdrive, A, tstim, PRF, DC):
        ''' Get simulation filecode for the given parameters.

            :param cell_type: cell type
            :param a: Sonophore radius (m)
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
                cell_type, W_str, A, tstim * 1e3, PW_str)
        else:
            fs_str = '_fs{:.0f}%'.format(fs * 1e2) if fs < 1 else ''
            filecode = 'ASTIM_{}_{}_{:.0f}nm{}_{:.0f}kHz_{:.1f}kPa_{:.0f}ms{}'.format(
                cell_type, W_str, a * 1e9, fs_str, Fdrive * 1e-3, A * 1e-3, tstim * 1e3, PW_str)
        return filecode

    def updateGraph(self, _, group_name, relayout_data, cell_type, id):
        ''' Update graph with new data.

            :param _: input graph figure content (used to trigger callback for subsequent graphs)
            :param group_name: name of the group of output variables to display
            :param relayout_data: input graph relayout data
            :param cell_type: cell type
            :param id: id of the graph to update
            :return: graph content
        '''

        # Determine plot variables
        ax_varnames = self.pltscheme[group_name]
        ax_pltvars = [self.pltvars[k] for k in ax_varnames]
        if self.verbose:
            print('{}: plotting {} set: {}'.format(id, group_name, ax_varnames))

        # Determine y-axis bounds and unit if needed
        if 'bounds' in ax_pltvars[0]:
            ax_min = min([ap['bounds'][0] for ap in ax_pltvars])
            ax_max = max([ap['bounds'][1] for ap in ax_pltvars])
            ybounds = (ax_min, ax_max)
        else:
            ybounds = None
        yunit = ax_pltvars[0].get('unit', '')

        # Process y-axis label
        ylabel = '{} ({})'.format(group_name, yunit)
        for c in ['{', '}', '\\', '_', '^']:
            ylabel = ylabel.replace(c, '')

        # Adjust color to black if only 1 variable to plot
        if len(ax_varnames) == 1:
            ax_pltvars[0]['color'] = 'black'

        # Process and plot data if any
        if self.data is not None:

            # Get time and states vector
            t = self.data['t'].values
            states = self.data['stimstate'].values

            # Determine stimulus patch(es) from states
            tpatch_on, tpatch_off = GroupedTimeSeries.getStimPulses(t, states)

            # Preset and rescale time vector
            tonset = np.array([-0.05 * np.ptp(t), 0.0])
            t = np.hstack((tonset, t))
            t *= self.tscale

            # Plot time series
            timeseries = []
            icolor = 0
            for name, pltvar in zip(ax_varnames, ax_pltvars):
                try:
                    var = extractPltVar(self.pneurons[cell_type], pltvar, self.data, None, t.size, name)
                except KeyError:
                    pass
                timeseries.append(go.Scatter(
                    x=t,
                    y=var,
                    mode='lines',
                    name=name,
                    line={'color': pltvar.get('color', self.colors[icolor])}
                ))
                if 'color' not in pltvar:
                    icolor += 1

            # Define stimulus patches as rectangles with y-reference to the plot
            patches = [{
                'type': 'rect',
                'xref': 'x',
                'yref': 'paper',
                'x0': tpatch_on[i] * self.tscale,
                'x1': tpatch_off[i] * self.tscale,
                'y0': 0,
                'y1': 1,
                'fillcolor': 'grey',
                'line': {'color': 'grey'},
                'opacity': 0.2
            } for i in range(tpatch_on.size)]

        # If data does not exist, define empty timeseries and patches
        else:
            timeseries = []
            patches = []

        # Set axes layout
        layout = go.Layout(
            xaxis={
                'type': 'linear',
                'title': 'time (ms)',
                'range': (t.min(), t.max()),
                'zeroline': False
            },
            yaxis={
                'type': 'linear',
                'title': ylabel,
                'range': ybounds,
                'zeroline': False
            },
            shapes=patches,
            margin={'l': 60, 'b': 40, 't': 10, 'r': 10},
            title='',
            showlegend=True
        )

        # Return curve, patches and layout objects
        fig = {'data': timeseries, 'layout': layout}
        fig['layout']['xaxis']['range'] = self.getXrange(relayout_data)
        return fig

    @staticmethod
    def getXrange(relayout_data):
        ''' Get the x-range of the zoomed in data '''
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
        return xrange

    def updateInfoTable(self, _):
        ''' Update the content of the output metrics table on neuron/modality/stimulation change. '''

        # Spike detection
        if self.data is not None:
            t = self.data['t']
            ispikes, _ = detectSpikes(self.data)
            nspikes = ispikes.size
            lat = t[ispikes[0]] if nspikes > 0 else None
            sr = np.mean(1 / np.diff(t[ispikes])) if nspikes > 1 else None
        else:
            nspikes = 0
            lat = None
            sr = None

        return dataRows(labels=['# spikes', 'Latency', 'Firing rate'],
                        values=[nspikes, lat, sr],
                        units=['', 's', 'Hz'])

    def updateDownloadContent(self, _):
        ''' Update the content of the downloadable pandas dataframe. '''
        try:
            csv_string = self.data.to_csv(index=False, encoding='utf-8')
            csv_string = "data:text/csv;charset=utf-8," + urllib.parse.quote(csv_string)
            return csv_string
        except AttributeError:
            pass

    def updateDownloadName(self, _):
        ''' Update the name of the downloadable pandas dataframe. '''
        # if self.data is None:
        #     return ''
        try:
            return '{}.csv'.format(self.getFileCode(*self.current_params))
        except TypeError:
            pass
