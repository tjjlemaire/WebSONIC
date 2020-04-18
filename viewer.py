# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2017-06-22 16:57:14
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2020-04-18 16:47:44

''' Definition of the SONICViewer class. '''

import urllib
import numpy as np
import pandas as pd
from matplotlib.colors import rgb2hex
import dash
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
from plotly.subplots import make_subplots

from PySONIC.utils import isIterable, bounds, getMeta
from PySONIC.postpro import detectSpikes
from PySONIC.constants import *
from PySONIC.core import PulsedProtocol, ElectricDrive, AcousticDrive
from PySONIC.neurons import getNeuronsDict
from PySONIC.plt import GroupedTimeSeries, extractPltVar
from ExSONIC.core import Node

from components import *


class SONICViewer(dash.Dash):
    ''' SONIC viewer application inheriting from dash.Dash. '''

    tscale = 1e3  # time scaling factor

    def __init__(self, ctrl_params, plt_params, no_run=False, verbose=False):
        ''' App constructor.

            :param ctrl_params: dictionary of input parameters that determine input controls
            :param plt_params: dictionary of parameters that determine plot outputs
            :param no_run: boolean stating whether to test the app UI without running simulations
            :param verbose: boolean stating whether or not to print app information in terminal
        '''
        # Initialize Dash app
        super(SONICViewer, self).__init__(
            name='viewer',
            url_base_pathname='/viewer/',
            external_stylesheets=[dbc.themes.BOOTSTRAP]
        )
        self.title = 'SONIC viewer'

        # Initialize constant parameters
        self.colors = plt_params['colors']
        self.default_vars = plt_params['default_vars']
        self.no_run = no_run
        self.verbose = verbose

        # Initialize parameters that will change upon requests
        self.current_params = None
        self.model = None
        self.data = None

        # Initialize point-neuron objects
        self.pneurons = {
            key: getNeuronsDict()[key]()
            for key in ctrl_params['cell_type'].values}

        # Initialize cell, drive and pulsing parameters
        self.cell_param = ctrl_params['cell_type']
        self.sonophore_params = {
            x: ctrl_params[x] for x in ['sonophore_radius', 'sonophore_coverage_fraction']}
        self.drive_params = {
            'US': {x: ctrl_params[x] for x in ['f_US', 'A_US']},
            'EL': {'A_EL': ctrl_params['A_EL']}}
        self.pp_params = {x: ctrl_params[x] for x in ['tstim', 'PRF', 'DC']}

        # Initialize plot variables and plot scheme
        default_cell = self.cell_param.default
        self.pltvars = self.pneurons[default_cell].getPltVars()
        self.pltscheme = self.pneurons[default_cell].pltScheme

        self.simcount = 0

        # Initialize UI layout components
        self.setLayout(default_cell, 'US')

        # Link UI components callbacks to appropriate functions
        self.registerCallbacks()

    def __str__(self):
        ''' Dsecriptive information about the app object. '''
        return f'{self.title} app'

    # ------------------------------------------ LAYOUT ------------------------------------------

    def setLayout(self, default_cell, default_mod):
        ''' Set app layout.

            :param default_cell: default cell type for the layout initialization
            :param default_mod: default modality for the layout initialization
        '''
        self.layout = html.Div(id='body', children=[
            # Header
            self.header(),

            # Content
            html.Div(id='content', children=[
                # Left side
                html.Div(id='left-col', className='content-column', children=[
                    self.cellPanel(default_cell),
                    self.stimPanel(default_mod),
                    self.metricsPanel(),
                    self.statusBar()
                ]),

                # Right side
                html.Div(id='right-col', className='content-column', children=[
                    self.outputPanel(default_cell, default_mod)])
            ]),

            # Footer
            separator(),
            html.Br(),
            self.footer()
        ])

    @staticmethod
    def header():
        ''' Set app header. '''
        return html.Div(id='header', children=[
            html.H2('Ultrasound Neuromodulation: exploring predictions of the SONIC model',
                    className='header-txt')
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
                    dbc.ModalBody(children=[
                        html.Img(src="assets/sonic_logo.svg", id='about-logo'),
                        dcc.Markdown(f'''{cls.about()}''')]),
                    dbc.ModalFooter(dbc.Button('Close', id='close-about', className='ml-auto')),
                ]
            ),
            html.Br(),
            html.Div(className='footer-img', children=[html.A(html.Img(
                src='assets/EPFL.svg', className='logo'), href='https://www.epfl.ch')]),
            html.Div(className='footer-img', children=[html.A(html.Img(
                src='assets/ITIS.svg', className='logo'), href='https://www.itis.ethz.ch')])
        ])

    @staticmethod
    def about():
        ''' Retrieve the "about" text from file.

            :return: text string
        '''
        with open('about.md', encoding="utf8") as f:
            return f.read()

    def slidersTable(self, label, params_dict):
        ''' Set a table of labeled slider controls based on a dictionary of parameters.

            :param label: table label
            :param params_dict: dictionary of parameter objects
        '''
        return labeledSlidersTable(
            f'{label}-slider-table',
            labels=[p.label for p in params_dict.values()],
            ids=[f'{p}-slider' for p in params_dict.keys()],
            bounds=[p.bounds for p in params_dict.values()],
            n=[p.n for p in params_dict.values()],
            values=[p.default for p in params_dict.values()],
            scales=[p.scale for p in params_dict.values()],
            disabled=[p.disabled for p in params_dict.values()]
        )

    def cellPanel(self, default_cell):
        ''' Construct cell parameters panel.

            :param default_cell: default cell type for the layout initialization
        '''
        return collapsablePanel('Cell parameters', children=[
            html.Table(className='table', children=[
                html.Tr([
                    html.Td(self.cell_param.label, className='row-label'),
                    html.Td(className='row-data', children=[
                        dcc.Dropdown(
                            className='ddlist',
                            id='cell_type-dropdown',
                            options=[{
                                'label': f'{self.pneurons[name].description()} ({name})',
                                'value': name
                            } for name in self.pneurons.keys()],
                            value=default_cell),
                        html.Div(id='membrane-currents'),
                    ])])
            ]),
            self.slidersTable('sonophore', self.sonophore_params)
        ])

    def stimPanel(self, default_mod):
        ''' Construct stimulation parameters panel.

            :param default_mod: default modality for the layout initialization
        '''
        return collapsablePanel('Stimulation parameters', children=[
            # US-EL tabs
            dcc.Tabs(
                id='modality-tabs',
                className='custom-tabs-container',
                parent_className='custom-tabs',
                value=default_mod, children=[
                    dcc.Tab(
                        label='Ultrasound',
                        value='US',
                        className='custom-tab',
                        selected_className='custom-tab--selected'
                    ),
                    dcc.Tab(
                        label='Injected current',
                        value='EL',
                        className='custom-tab',
                        selected_className='custom-tab--selected'
                    )
                ]),

            # Ctrl sliders
            *[self.slidersTable(k, v) for k, v in self.drive_params.items()],
            self.slidersTable('pp', self.pp_params)
        ])

    def statusBar(self):
        ''' Status bar object. '''
        return dbc.Alert(id='status-bar', color='light', is_open=True, children=[])

    @staticmethod
    def metricsPanel():
        ''' Metric panel. '''
        return collapsablePanel('Output metrics', children=[
            html.Table(id='info-table', className='table')])

    def outputPanel(self, default_cell, default_mod):
        ''' Set output (graphs) panel layout.

            :param default_cell: default cell type for the layout initialization
            :param default_mod: default modality for the layout initialization
        '''
        # Get options values and generate options labels
        values = list(self.pltscheme.keys())
        labels = self.getOutputDropDownLabels()

        return html.Div(children=[
            panel(children=[
                html.Div(className='graph-div', children=[
                    # Title
                    html.Div(id='graph-title', className='graph-title', children=[]),
                    # Multi-dropdown
                    dcc.Dropdown(
                        className='ddlist',
                        id=f'graph-dropdown',
                        options=[{'label': l, 'value': v} for l, v in zip(labels, values)],
                        value=self.default_vars,
                        multi=True
                    ),
                    # Graph
                    dcc.Loading(dcc.Graph(
                        id='graph',
                        className='graph',
                        animate=False,
                        config={
                            'editable': False,
                            'modeBarButtonsToRemove': [
                                'sendDataToCloud',
                                'displaylogo',
                                'toggleSpikelines']
                        },
                        figure={'data': [], 'layout': {}}
                    ))
                ])
            ]),
            html.Div(id='download-wrapper', children=[
                html.A('Download Data', id='download-link', download="", href="", target="_blank")])
        ])

    # ------------------------------------------ CALLBACKS ------------------------------------------

    def registerCallbacks(self):
        ''' Assign callbacks between inputs and outputs in order to make the app interactive. '''
        # Cell panel: cell type
        self.callback(
            Output('membrane-currents', 'children'),
            [Input('cell_type-dropdown', 'value')])(self.updateMembraneCurrents)

        # Cell panel: sliders
        for key, p in self.sonophore_params.items():
            id = f'{key}-slider'
            self.callback(
                Output(f'{id}-value', 'children'),
                [Input(id, 'value')])(self.updateSliderValue(p))

        # Stimulation panel: US/EL drive parameters visibility
        for key in self.drive_params.keys():
            self.callback(
                Output(f'{key}-slider-table', 'hidden'),
                [Input('modality-tabs', 'value')])(self.tabDependentVisibility(key))

        # Stimulation panel: sliders
        for refparams in [*self.drive_params.values(), self.pp_params]:
            for key, p in refparams.items():
                id = f'{key}-slider'
                self.callback(
                    Output(f'{id}-value', 'children'),
                    [Input(id, 'value')])(self.updateSliderValue(p))

        # Coverage slider
        self.callback(
            [Output('sonophore_coverage_fraction-slider', 'value'),
             Output('sonophore_coverage_fraction-slider', 'disabled')],
            [Input('cell_type-dropdown', 'value'),
             Input('sonophore_radius-slider', 'value'),
             Input('modality-tabs', 'value'),
             Input('f_US-slider', 'value')],
            [State('sonophore_coverage_fraction-slider', 'value')])(self.updateCoverageSlider)

        # Output metrics table
        self.callback(
            Output('info-table', 'children'),
            [Input('status-bar', 'children')])(self.updateInfoTable)

        # Inputs change that trigger simulations
        self.callback(
            Output('status-bar', 'children'),
            [Input('cell_type-dropdown', 'value'),
             Input('sonophore_radius-slider', 'value'),
             Input('sonophore_coverage_fraction-slider', 'value'),
             Input('modality-tabs', 'value'),
             Input('f_US-slider', 'value'),
             Input('A_US-slider', 'value'),
             Input('A_EL-slider', 'value'),
             Input('tstim-slider', 'value'),
             Input('PRF-slider', 'value'),
             Input('DC-slider', 'value')])(self.onInputsChange)

        # Output panel
        self.callback(
            Output('graph-dropdown', 'options'),
            [Input('cell_type-dropdown', 'value')])(self.updateOutputOptions)
        self.callback(
            Output('graph-dropdown', 'value'),
            [Input('cell_type-dropdown', 'value')],
            state=[State(f'graph-dropdown', 'value')])(self.updateOutputVars)

        # Update graph & title whenever status bar or dropdown values change
        self.callback(
            [Output('graph', 'figure'),
             Output('graph-title', 'children')],
            [Input('status-bar', 'children'),
             Input('graph-dropdown', 'value')],
            [State('cell_type-dropdown', 'value')])(self.updateGraph)

        # Download link
        self.callback(
            Output('download-link', 'href'),
            [Input('status-bar', 'children')])(self.updateDownloadContent)
        self.callback(
            Output('download-link', 'download'),
            [Input('status-bar', 'children')])(self.updateDownloadName)

        # About modal
        self.callback(
            Output('about-modal', 'is_open'),
            [Input('about-link', 'n_clicks'), Input('close-about', 'n_clicks')],
            [State('about-modal', 'is_open')])(self.toggleAbout)

    @staticmethod
    def toggleAbout(n1, n2, is_open):
        ''' Toggle the visibility of a modal HTML element.

            :param n1: number of clicks on the opening link
            :param n2: number of clicks on the close button
            :param is_open: current state of the modal element (open or closed)
        '''
        if n1 or n2:
            return not is_open
        return is_open

    def updateMembraneCurrents(self, cell_type):
        ''' Update the list of membrane currents on neuron switch.

            :param cell_type: cell type
            :return: HTML list of cell-type-specific membrane currents
        '''
        currents = self.pneurons[cell_type].getCurrentsNames()
        return unorderedList([f'{self.pltvars[c]["desc"]} ({c})' for c in currents])

    def tabDependentVisibility(self, ref_value):
        ''' Set the bisibility of an element if according to match with a tab state.

            :param ref_value: reference value that needs to match the tab state to show the element
            :return: lambda function handling the visibility toggle
        '''
        return lambda x: x != ref_value

    def updateSliderValue(self, p):
        ''' Update the value of a slider label when the slider is moved.

            :param p: parameter object with information about the label formatting
            :return: lambda function handling the label formatting upon slider change.
        '''
        return lambda x: f'{si_format(p.scaling_func(x), 1)}{p.unit}'

    def has_fs_lookup(self, cell_type, a, f):
        ''' Determine if an fs-dependent lookup exists for a specific parameter combination.
            :param cell_type: cell type
            :param a: sonophore radius (m)
            :param f: US frequency (Hz)
            :return: boolean stating whether a lookup file should exist.
        '''
        is_default_cell = cell_type == 'RS'
        is_default_radius = np.isclose(a, self.sonophore_params['sonophore_radius'].default,
                                       rtol=1e-9, atol=1e-16)
        is_default_freq = np.isclose(f, self.drive_params['US']['f_US'].default,
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
        disabled_output = (self.sonophore_params['sonophore_coverage_fraction'].default, True)
        enabled_output = (fs_slider, False)
        if mod_type != 'US':
            return disabled_output
        a = self.convertSliderInput(a_slider, self.sonophore_params['sonophore_radius'])
        f_US = self.convertSliderInput(f_US_slider, self.drive_params['US']['f_US'])
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

    def updateOutputOptions(self, cell_type):
        ''' Update the list of available variables in a graph dropdown menu on neuron switch.

            :param cell_type: cell type
            :return: label:value dictionary of update droppdown options
        '''
        # Update pltvars and pltscheme according to new cell type
        self.pltvars = self.pneurons[cell_type].getPltVars()
        self.pltscheme = self.pneurons[cell_type].pltScheme

        # Get options values and generate options labels
        values = list(self.pltscheme.keys())
        labels = self.getOutputDropDownLabels()

        # Return dictionary
        return [{'label': lbl, 'value': val} for lbl, val in zip(labels, values)]

    def updateOutputVars(self, cell_type, varnames):
        ''' Update the selected variable in a graph dropdown menu on neuron switch.

            :param cell_type: cell type
            :param varnames: name of currently selected variables
            :return: name of the selected variable, updated if needed
        '''
        # Update pltvars and pltscheme according to new cell type
        self.pltvars = self.pneurons[cell_type].getPltVars()
        self.pltscheme = self.pneurons[cell_type].pltScheme

        # Get options values and generate options labels
        if not isIterable(varnames):
            varnames = [varnames]
        varnames = list(filter(lambda x: x in self.pltscheme.keys(), varnames))
        if len(varnames) == 0:
            varnames = self.default_vars
        return varnames

    def convertSliderInput(self, value, refparam):
        ''' Convert slider value into corresponding parameters value.

            :param value: slider value
            :param refparam: reference parameter object
            :return: converted parameters value
        '''
        return refparam.scaling_func(value) * refparam.factor

    def convertSliderInputs(self, values, refparams):
        ''' Convert sliders values into corresponding parameters values.

            :param values: sliders values
            :param refparams: dictionary of reference parameters
            :return: list of converted parameters values
        '''
        return [self.convertSliderInput(x, p) for x, p in zip(values, refparams.values())]

    def onInputsChange(self, cell_type, a_slider, fs_slider, mod_type, f_US_slider, A_US_slider,
                       I_EL_slider, tstim_slider, PRF_slider, DC_slider):
        ''' Translate inputs into parameter values and run model simulation.

            :return: status message
        '''
        # Determine new parameters
        a, fs = self.convertSliderInputs([a_slider, fs_slider], self.sonophore_params)
        US_params = self.convertSliderInputs([f_US_slider, A_US_slider], self.drive_params['US'])
        EL_params = self.convertSliderInputs([I_EL_slider], self.drive_params['EL'])
        tstim, PRF, DC = self.convertSliderInputs([
            tstim_slider, PRF_slider, DC_slider], self.pp_params)

        # Assign them
        drive = AcousticDrive(*US_params) if mod_type == 'US' else ElectricDrive(*EL_params)
        pp = PulsedProtocol(tstim, 0.5 * tstim, PRF=PRF, DC=DC * 1e-2)

        # Update plot variables if different cell type
        new_params = [cell_type, a, fs * 1e-2, drive, pp]
        if self.current_params is None or cell_type != self.current_params[0]:
            self.pltvars = self.pneurons[cell_type].getPltVars()
            self.pltscheme = self.pneurons[cell_type].pltScheme

        # Run simulation if parameters have changed
        if new_params != self.current_params:
            self.runSim(*new_params)
            self.current_params = new_params

        # Return status message inside a list
        return [self.statusMessage()]

    def statusMessage(self):
        return f'Number of simulations: {self.simcount}'

    def getShamData(self, pneuron, pp):
        ''' Get Sham output data without running a simulation.data
            :param pneuron: point-neuron object
            :param pp: pulsed protocol object
            :return artificially generated dataframe matching input "dimensionality"
        '''
        data = pd.DataFrame({
            't': np.array([0., pp.tstim, pp.tstim, pp.tstim + pp.toffset]),
            'stimstate': np.hstack((np.ones(2), np.zeros(2))),
            'Qm': pneuron.Qm0 * np.ones(4),
            'Vm': pneuron.Vm0 * np.ones(4)
        })
        for k in pneuron.states.keys():
            data[k] = 0.5 * np.ones(4)
        return data

    def runSim(self, cell_type, a, fs, drive, pp):
        ''' Run NEURON simulation to update data.

            :param cell_type: cell type
            :param a: Sonophore radius (m)
            :param fs: sonophore membrane coverage fraction (-)
            :param drive: drive object
            :param pp: pulsed protocol object
        '''
        pneuron = self.pneurons[cell_type]
        self.model = Node(pneuron, a=a, fs=fs)
        if self.no_run:
            # If no-run mode, get Sham data
            self.data = self.getShamData(pneuron, pp)
            meta = getMeta(self.model, self.model.simulate, drive, pp)
        else:
            # Otherwise, run model simulation
            self.data, meta = self.model.simulate(drive, pp)

        # Update simulation count and log
        self.simcount += 1
        self.simlog = self.model.desc(meta)
        if self.verbose:
            print(self.simlog)

    def getFileCode(self, drive, pp):
        ''' Get simulation filecode for the given parameters.

            :param drive: drive object
            :param pp: pulsed protocol object
            :return: filecode
        '''
        return self.model.filecode(drive, pp)

    def updateGraph(self, _, group_names, cell_type):
        ''' Update graph with new data.

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
            t *= self.tscale
            trange = bounds(t)
            nsamples = t.size

            # Define stimulus patches as rectangles with y-reference to the plot
            patches = [{
                'type': 'rect',
                'xref': 'x',
                'yref': 'paper',
                'x0': pulse[0] * self.tscale,
                'x1': pulse[1] * self.tscale,
                'y0': 0,
                'y1': 1,
                'fillcolor': 'grey',
                'line': {'color': rgb2hex(pcolor)},
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

            # Process and add y-axis label
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

        # Return figure object and title
        return fig, self.simlog

    @staticmethod
    def getXrange(relayout_data):
        ''' Get the x-range of the zoomed in data

            :param relayout_data: graph relayout data structure
            :return x-axis range
        '''
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
        ''' Update the content of the output metrics table on neuron/modality/stimulation change.

            return: updated table data rows
        '''
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
        ''' Update the content of the downloadable pandas dataframe.

            :return: string-encoded CSV
        '''
        if self.data is None:
            csv_string = ''
        else:
            csv_string = self.data.to_csv(index=False, encoding='utf-8')
        return "data:text/csv;charset=utf-8," + urllib.parse.quote(csv_string)

    def updateDownloadName(self, _):
        ''' Update the name of the downloadable pandas dataframe.

            :return: download file name
        '''
        if self.current_params is None:
            code = 'none'
        else:
            code = self.getFileCode(*self.current_params[-2:])
        return f'{code}.csv'
