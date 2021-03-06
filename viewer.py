# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2017-06-22 16:57:14
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2021-03-06 02:33:06

import urllib
import numpy as np
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
from plotly.subplots import make_subplots

from PySONIC.utils import isIterable, bounds
from PySONIC.postpro import detectSpikes
from PySONIC.core import PulsedProtocol, ElectricDrive, AcousticDrive
from PySONIC.neurons import getNeuronsDict
from PySONIC.plt import GroupedTimeSeries, extractPltVar
from ExSONIC.core import Node
from ExSONIC.constants import S_TO_MS

from template import AppTemplate
from params import QualitativeParameter, RangeParameter


class SONICViewer(AppTemplate):
    ''' SONIC viewer application. '''

    # Properties
    name = 'viewer'
    title = 'SONIC viewer'
    author = 'Théo Lemaire'
    email = 'theo.lemaire@epfl.ch'
    copyright = 'Translational Neural Engineering Lab, EPFL - 2019'

    # Cell, drive and pulsing parameters
    params = {
        'cell': QualitativeParameter(
            'Cell type', ['RS', 'FS', 'LTS', 'IB', 'RE', 'TC', 'STN'], default='RS'),
        'sonophore': {
            'radius': RangeParameter(
                'Sonophore radius', (16e-9, 64e-9), 'm', default=32e-9, scale='log', n=5),
            'coverage_fraction': RangeParameter(
                'Coverage fraction', (1., 100.), '%', default=100., scale='lin', n=100)},
        'drive': {
            'US': {
                'f': RangeParameter(
                    'Frequency', (20e3, 4e6), 'Hz', default=500e3, scale='friendly-log'),
                'A': RangeParameter(
                    'Amplitude', (10e3, 600e3), 'Pa', default=80e3, scale='log', n=100,
                    round_factor=1e-3)},
            'EL': {
                'A': RangeParameter(
                    'Amplitude', (-25e-3, 25e-3), 'A/m2', default=10e-3, n=51)}},
        'pp': {
            'tstim': RangeParameter(
                'Duration', (20e-3, 1.0), 's', default=200e-3, scale='friendly-log'),
            'PRF': RangeParameter(
                'PRF', (1e1, 1e3), 'Hz', default=100., scale='friendly-log'),
            'DC': RangeParameter(
                'Duty cycle', (1., 100.), '%', default=50., scale='lin', n=100)}
    }

    def __init__(self, verbose=False):
        ''' App constructor.

            :param verbose: boolean stating whether or not to print app information in terminal
        '''
        # Initialize constant parameters
        self.verbose = verbose

        # Initialize defaults
        self.defaults = self.getDefaults(self.params)
        self.defaults['mod'] = 'US'
        self.defaults['pltvars'] = ['Q_m', 'V_m', 'I']

        # Initialize point-neuron objects
        self.pneurons = {k: getNeuronsDict()[k]() for k in self.params['cell'].values}

        # Initialize parameters that will change upon requests
        self.simcount = 0
        self.current_params = None
        self.model = None
        self.data = None

        # Create app
        super().__init__()

    # ------------------------------------------ LAYOUT ------------------------------------------

    def header(self):
        return [html.H2('Ultrasound Neuromodulation: exploring predictions of the SONIC model')]

    def content(self):
        return [
            # Left side
            html.Div(id='left-col', className='content-column', children=[
                self.cellPanel(),
                self.stimPanel(),
                self.metricsPanel()
            ]),

            # Right side
            html.Div(id='right-col', className='content-column', children=[
                self.outputPanel()
            ])
        ]

    def footer(self):
        return [self.reachout(), *super().footer()]

    def reachout(self):
        return dbc.Alert(id='reachout', color='info', is_open=True, children=[
            html.H5([html.I('Interested in using the SONIC model?')],
                    className='alert-heading'),
            ' Download the code from our ', html.A(
                'online repository', href='https://c4science.ch/diffusion/4670/',
                className='alert-link'), ' and check-out the ', html.A(
                'related paper',
                href='https://iopscience.iop.org/article/10.1088/1741-2552/ab1685',
                className='alert-link'),
            ' for more details.'
        ])

    def credentials(self):
        return html.Span([
            *super().credentials(),
            ' Designed with ', html.A('Dash', href='https://dash.plot.ly/'), '.',
            ' Powered by ', html.A('NEURON', href='https://www.neuron.yale.edu/neuron/'), '.',
        ])

    @property
    def about(self):
        with open('about.md', encoding="utf8") as f:
            return f.read()

    def footerImgs(self):
        return self.centered(id='footer-imgs', children=[
            html.Div(className='footer-img', children=[html.A(html.Img(
                src='assets/EPFL.svg', className='logo'), href='https://www.epfl.ch')]),
            html.Div(className='footer-img', children=[html.A(html.Img(
                src='assets/ITIS.svg', className='logo'), href='https://www.itis.ethz.ch')])
        ])

    def cellPanel(self):
        ''' Construct cell parameters panel. '''
        return self.collapsablePanel('Cell parameters', children=[
            html.Table(className='table', children=[
                html.Tr([
                    html.Td(self.params['cell'].label, className='row-label'),
                    html.Td(className='row-data', children=[
                        dcc.Dropdown(
                            className='ddlist',
                            id='cell_type-dropdown',
                            options=[{'label': f'{v.description()} ({k})', 'value': k}
                                     for k, v in self.pneurons.items()],
                            value=self.defaults['cell']),
                        html.Div(id='membrane-currents'),
                    ])])
            ]),
            self.paramSlidersTable('sonophore', self.params['sonophore'])
        ])

    def stimPanel(self):
        ''' Construct stimulation parameters panel. '''
        return self.collapsablePanel('Stimulation parameters', children=[
            self.tabs(
                'modality', ['Ultrasound', 'Injected current'], ['US', 'EL'], self.defaults['mod']),
            *[self.paramSlidersTable(k, v, id_prefix=k) for k, v in self.params['drive'].items()],
            self.paramSlidersTable('pp', self.params['pp'])
        ])

    def metricsPanel(self):
        ''' Construct metrics panel. '''
        return self.collapsablePanel('Output metrics', children=[
            html.Table(id='info-table', className='table')])

    def outputPanel(self):
        ''' Set output (graphs) panel layout. '''
        return self.panel(children=[
            html.Div('Neural response', className='panel-title'),
            dcc.Dropdown(className='ddlist', id=f'timeseries-dropdown', multi=True),
            self.graph('timeseries-graph'),
            dbc.Alert(id='simlog-box', className='logbox', color='light', children=[]),
            self.centered(children=[html.A(
                'Download Data', id='download-link', download='', href='', target='_blank')
            ])
        ])

    # ------------------------------------------ CALLBACKS -----------------------------------------

    def registerCallbacks(self):
        super().registerCallbacks()

        # Cell panel: cell type
        self.callback(
            Output('membrane-currents', 'children'),
            [Input('cell_type-dropdown', 'value')])(self.updateMembraneCurrents)

        # Cell panel: sliders
        for k, p in self.params['sonophore'].items():
            self.linkSliderValue(k, p)

        # Coverage slider
        self.callback(
            [Output('coverage_fraction-label', 'children'),
             Output('coverage_fraction-slider', 'value'),
             Output('coverage_fraction-slider', 'disabled')],
            [Input('cell_type-dropdown', 'value'),
             Input('radius-slider', 'value'),
             Input('modality-tabs', 'value'),
             Input('US-f-slider', 'value')],
            [State('coverage_fraction-slider', 'value')])(self.updateCoverageSlider)

        # Stimulation panel: US/EL drive parameters visibility
        for key in self.params['drive'].keys():
            self.callback(
                Output(f'{key}-slider-table', 'hidden'),
                [Input('modality-tabs', 'value')])(self.valueDependentVisibility(key))

        # Stimulation panel: sliders
        for key, val in self.params['drive'].items():
            for k, p in val.items():
                self.linkSliderValue(f'{key}-{k}', p)
        for k, p in self.params['pp'].items():
            self.linkSliderValue(k, p)

        # Inputs changes that trigger simulations
        self.callback(
            [Output('info-table', 'children'),
             Output('simlog-box', 'children'),
             Output('download-link', 'href'),
             Output('download-link', 'download')],
            [Input('modality-tabs', 'value'),
             Input('cell_type-dropdown', 'value'),
             Input('radius-slider', 'value'),
             Input('coverage_fraction-slider', 'value'),
             Input('US-f-slider', 'value'),
             Input('US-A-slider', 'value'),
             Input('EL-A-slider', 'value'),
             Input('tstim-slider', 'value'),
             Input('PRF-slider', 'value'),
             Input('DC-slider', 'value')])(self.onInputsChange)

        # Output dropdown
        self.callback(
            [Output('timeseries-dropdown', 'value'),
             Output('timeseries-dropdown', 'options')],
            [Input('cell_type-dropdown', 'value')],
            [State(f'timeseries-dropdown', 'value')])(self.updateOutputDropdown)

        # Update graph & title whenever simlog box or dropdown values change
        self.callback(
            Output('timeseries-graph', 'figure'),
            [Input('simlog-box', 'children'),
             Input('timeseries-dropdown', 'value')],
            [State('cell_type-dropdown', 'value')])(self.updateTimeseries)

    def updateMembraneCurrents(self, cell_type):
        ''' Update the list of membrane currents on neuron switch.

            :param cell_type: cell type
            :return: HTML list of cell-type-specific membrane currents
        '''
        currents = self.pneurons[cell_type].getCurrentsNames()
        return self.unorderedList([f'{self.pltvars[c]["desc"]} ({c})' for c in currents])

    def has_fs_lookup(self, cell_type, a, f):
        ''' Determine if an fs-dependent lookup exists for a specific parameter combination.
            :param cell_type: cell type
            :param a: sonophore radius (m)
            :param f: US frequency (Hz)
            :return: boolean stating whether a lookup file should exist.
        '''
        is_default_cell = cell_type == self.defaults['cell']
        is_default_radius = np.isclose(a, self.defaults['sonophore']['radius'],
                                       rtol=1e-9, atol=1e-16)
        is_default_freq = np.isclose(f, self.defaults['drive']['US']['f'],
                                     rtol=1e-9, atol=1e-16)
        return is_default_cell and is_default_radius and is_default_freq

    def updateCoverageSlider(self, cell_type, a_slider, mod_type, f_US_slider, fs_slider):
        ''' Update the value and state of the sonophore coverage fraction slider based on other
            input parameters.

            :param cell_type: cell type
            :param a_slider: value of the sonophore radius slider
            :param mod_type: selected modality tab
            :param: f_US_slider: value of the US frequency slider
            :param fs_slider: value of the sonophore coverage faction slider
            :return: (value, disabled) tuple to update the slider's state
        '''
        p = self.params['sonophore']['coverage_fraction']
        disabled_output = ([
            self.tooltip(
                'coverage-tooltip', p.label,
                '''Coverage-fraction-dependent lookup tables are unavailable
                for this parameters combination'''
            )],
            p.idefault, True)
        enabled_output = ([p.label], fs_slider, False)

        if mod_type != 'US':
            return disabled_output
        a = self.params['sonophore']['radius'].values[a_slider]
        f_US = self.params['drive']['US']['f'].values[f_US_slider]
        if not self.has_fs_lookup(cell_type, a, f_US):
            return disabled_output
        else:
            return enabled_output

    def getOutputDropDownLabels(self):
        ''' Generate output drop-down labels from pltscheme elements.

            :return list of output labels
        '''
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

    def updateOutputDropdown(self, cell_type, values):
        ''' Update the timeseries dropdown options and selected values on neuron switch.

            :param cell_type: cell type
            :param values: currently selected value(s)
            :return: new dropdown values and options
        '''
        # Update pltvars and pltscheme according to new cell type
        self.pltvars = self.pneurons[cell_type].getPltVars()
        self.pltscheme = self.pneurons[cell_type].pltScheme

        # Construct dropdown options list
        options = [{'label': lbl, 'value': val} for lbl, val in zip(
            self.getOutputDropDownLabels(), self.pltscheme.keys())]

        # Filter current values based on new options
        if not isIterable(values):
            values = [values]
        values = list(filter(lambda x: x in self.pltscheme.keys(), values))
        if len(values) == 0:
            values = self.defaults['pltvars']

        # Return new values and options
        return values, options

    def convertSliderInputs(self, values, refparams):
        ''' Convert sliders values into corresponding parameters values.

            :param values: sliders values
            :param refparams: dictionary of reference parameters
            :return: list of converted parameters values
        '''
        return [p.values[x] for x, p in zip(values, refparams.values())]

    def onInputsChange(self, mod_type, cell_type, a_slider, fs_slider, f_US_slider, A_US_slider,
                       I_EL_slider, tstim_slider, PRF_slider, DC_slider):
        ''' Translate inputs into parameter values and run model simulation. '''
        # Determine new parameters
        a, fs = self.convertSliderInputs(
            [a_slider, fs_slider], self.params['sonophore'])
        US_params = self.convertSliderInputs(
            [f_US_slider, A_US_slider], self.params['drive']['US'])
        EL_params = self.convertSliderInputs(
            [I_EL_slider], self.params['drive']['EL'])
        tstim, PRF, DC = self.convertSliderInputs(
            [tstim_slider, PRF_slider, DC_slider], self.params['pp'])

        # Construct drive and pulsed protocol accordingly
        if mod_type == 'US':
            drive = AcousticDrive(*US_params)
        else:
            drive = ElectricDrive(EL_params[0] * 1e3)
        pp = PulsedProtocol(tstim, 0.5 * tstim, PRF=max(PRF, 1 / tstim), DC=DC * 1e-2)

        # Update plot variables if different cell type
        new_params = [cell_type, a, fs * 1e-2, drive, pp]
        if self.current_params is None or cell_type != self.current_params[0]:
            self.pltvars = self.pneurons[cell_type].getPltVars()
            self.pltscheme = self.pneurons[cell_type].pltScheme

        # Run simulation if parameters have changed
        if new_params != self.current_params:
            self.runSim(*new_params)
            self.current_params = new_params

        # Return new info-table, sim log and download link-content
        return [self.infoTable(), self.simlog, *self.download()]

    def infoTable(self):
        ''' Return an output metrics table on the current data. '''
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

        return self.dataRows(
            labels=['# spikes', 'Latency', 'Firing rate'],
            values=[nspikes, lat, sr],
            units=['', 's', 'Hz'])

    def download(self):
        ''' Return a content-name download link according to the current data. '''
        if self.data is None:
            csv_string = ''
            code = 'none'
        else:
            csv_string = self.data.to_csv(index=False, encoding='utf-8')
            code = self.model.filecode(*self.current_params[-2:])
        content = "data:text/csv;charset=utf-8," + urllib.parse.quote(csv_string)
        name = f'{code}.csv'
        return content, name

    def runSim(self, cell_type, a, fs, drive, pp):
        ''' Run NEURON simulation to update data.

            :param cell_type: cell type
            :param a: Sonophore radius (m)
            :param fs: sonophore membrane coverage fraction (-)
            :param drive: drive object
            :param pp: pulsed protocol object
        '''
        # Construct model and run simulation
        self.model = Node(self.pneurons[cell_type], a=a, fs=fs)
        self.data, meta = self.model.simulate(drive, pp)

        # Update simulation count and log
        self.simcount += 1
        self.simlog = self.model.desc(meta)
        if self.verbose:
            print(self.simlog)

    def updateTimeseries(self, _, group_names, cell_type):
        ''' Update timeseries graph with new data.

            :param group_names: names of the groups of output variables to display
            :param cell_type: cell type
            :return: graph content
        '''
        # If data exists
        if self.data is not None:
            # Get time and states vector
            t = self.data['t'].values
            states = self.data['stimstate'].values

            # Determine stimulus pulses and their colors from states
            pulses = GroupedTimeSeries.getStimPulses(t, states)
            pcolors = GroupedTimeSeries.getPatchesColors([p[2] for p in pulses])

            # Preset and rescale time vector
            tonset = t.min() - 0.05 * np.ptp(t)
            t = np.insert(t, 0, tonset)
            t *= S_TO_MS
            trange = bounds(t)
            nsamples = t.size

            # Define stimulus patches as rectangles with y-reference to the plot
            patches = [{
                'type': 'rect',
                'xref': 'x',
                'yref': 'paper',
                'x0': pulse[0] * S_TO_MS,
                'x1': pulse[1] * S_TO_MS,
                'y0': 0,
                'y1': 1,
                'fillcolor': self.rgb2hex(pcolor),
                'line': {'color': self.rgb2hex(pcolor)},
                'opacity': 0.2
            } for pulse, pcolor in zip(pulses, pcolors)]
        else:
            patches = []
            trange = (0, 100)

        # Determine plot variables
        if not isIterable(group_names):
            group_names = [group_names]

        # Create figure with shared x-axes
        nrows = len(group_names)
        default_row_height = 200
        vertical_spacing = 0.02  # pt
        max_height = 700
        total_height = min(max_height, default_row_height * nrows)
        row_height = total_height / nrows
        fig = make_subplots(
            rows=nrows, cols=1, shared_xaxes=True,
            vertical_spacing=vertical_spacing,
            row_heights=[row_height] * nrows)
        fig.update_xaxes(title_text='time (ms)', range=trange, row=nrows, col=1)

        # For each axis-group pair
        icolor = 0
        for j, group_name in enumerate(group_names):
            # Get axis variables
            ax_varnames = self.pltscheme[group_name]
            ax_pltvars = [self.pltvars[k] for k in ax_varnames]
            if self.verbose:
                print(f'{id}: plotting {group_name} set: {ax_varnames}')

            # Determine y-axis bounds and unit if needed
            if 'bounds' in ax_pltvars[0]:
                ax_min = min([ap['bounds'][0] for ap in ax_pltvars])
                ax_max = max([ap['bounds'][1] for ap in ax_pltvars])
                ybounds = (ax_min, ax_max)
            else:
                ybounds = None
            yunit = ax_pltvars[0].get('unit', '')

            # Determine y-axis label
            ylabel = f'{group_name} ({yunit})'
            for c in ['{', '}', '\\', '_', '^']:
                ylabel = ylabel.replace(c, '')

            # Set y-axis properties
            irow = j + 1
            fig.update_yaxes(title_text=ylabel, range=ybounds, row=irow, col=1)

            # Extract and plot variables timeseries if data exists
            if self.data is not None:
                for name, pltvar in zip(ax_varnames, ax_pltvars):
                    try:
                        var = extractPltVar(
                            self.pneurons[cell_type], pltvar, self.data, None, nsamples, name)
                    except (KeyError, UnboundLocalError):
                        pass
                    fig.add_trace(
                        go.Scatter(
                            x=t,
                            y=var,
                            mode='lines',
                            name=name,
                            line={'color': self.colors[icolor]}
                        ), row=irow, col=1)
                    icolor += 1

        # Update figure layout
        fig.update_layout(
            height=total_height,
            shapes=patches,
            template='plotly_white',
            margin={'l': 60, 'b': 40, 't': 30, 'r': 10},
        )

        # Return figure object
        return fig
