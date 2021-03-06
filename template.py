# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2017-06-22 16:57:14
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2021-03-06 12:47:55

''' Definition of the SONICViewer class. '''

from datetime import datetime
from matplotlib.pyplot import get_cmap
from matplotlib.colors import rgb2hex
import dash
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from gitinfo import get_git_info

from PySONIC.utils import customStrftime, si_format

from params import Parameter


class AppTemplate(dash.Dash):
    ''' Dash application template. '''

    def __init__(self):
        super().__init__(
            name=self.name,
            url_base_pathname=self.urlpath,
            external_stylesheets=[dbc.themes.BOOTSTRAP]
        )
        self.colors = self.getHexColors()
        self.setLayout()
        self.registerCallbacks()
        print(f'Initialized {self}')

    # ---------------------------------------- PROPERTIES -----------------------------------------

    def __repr__(self):
        return f'{self.title} app'

    @property
    def name(self):
        return 'app'

    @property
    def urlpath(self):
        return f'/{self.name.replace(" ", "-")}/'

    @property
    def title(self):
        return 'My App'

    @property
    def author(self):
        return 'Xxx Xxx'

    @property
    def email(self):
        raise NotImplementedError

    @property
    def copyright(self):
        return f'Some institution - {datetime.now().year}'

    @property
    def about(self):
        return 'Some information about the app'

    @property
    def params(self):
        return {}

    # ------------------------------------------ LAYOUT ------------------------------------------

    def getDefaults(self, x):
        ''' Get dictionary of defaults by searching recursively through parameters. '''
        if isinstance(x, Parameter):
            return x.default
        else:
            return {k: self.getDefaults(v) for k, v in x.items()}

    def setLayout(self):
        ''' Set app layout. '''
        self.layout = html.Div(id='body', children=[
            html.Div(className='centered-wrapper', children=self.header()),
            html.Div(id='content', children=self.content()),
            self.separator(),
            html.Div(className='centered-wrapper', children=self.footer())
        ])

    def header(self):
        return [html.H1('App title')]

    def content(self):
        return [dbc.Alert(color='info', is_open=True, style={'height': '40em'}, children=[
            html.H1('Content', className='alert-heading'),
            html.P('App content goes here.')
        ])]

    def footer(self):
        return [
            html.Div(self.credentials()),
            html.Div([self.lastUpdated()]),
            self.aboutModal(),
            self.footerImgs(),
            self.copyrightBanner()
        ]

    def credentials(self):
        return ['Developed by ', html.A(self.author, href=f'mailto:{self.email}'), '.']

    def getLastUpdateTime(self):
        ''' Get the last time the repository was updated. '''
        return datetime.strptime(get_git_info()['author_date'], '%Y-%m-%d %H:%M:%S')

    def lastUpdated(self):
        return f'Last updated on {customStrftime("%B {S}, %Y", self.getLastUpdateTime())}.'

    def aboutModal(self):
        return html.Div([
            '>>> ', html.A('About', id='about-link'), ' <<<',
            dbc.Modal(
                id='about-modal',
                size='lg',
                scrollable=True,
                centered=True,
                children=[
                    dbc.ModalHeader('About'),
                    dbc.ModalBody(children=[dcc.Markdown(f'''{self.about}''')]),
                    dbc.ModalFooter(dbc.Button('Close', id='close-about', className='ml-auto')),
                ]
            )
        ])

    def footerImgs(self):
        return html.Br()

    def copyrightBanner(self):
        return html.Div(id='copyright', children=['Copyright ', u'\u00A9', f' {self.copyright}'])

    # ---------------------------------------- UTILITARIES ----------------------------------------

    @staticmethod
    def rgb2hex(*args, **kwargs):
        return rgb2hex(*args, **kwargs)

    def getHexColors(self):
        ''' Generate a list of HEX colors for timeseries plots. '''
        colors = []
        for cmap in ['Set1', 'Set2']:
            colors += get_cmap(cmap).colors
        return [self.rgb2hex(c) for c in colors]

    # ----------------------------------------- CALLBACKS -----------------------------------------

    def registerCallbacks(self):
        ''' Assign callbacks between inputs and outputs in order to make the app interactive. '''
        self.callback(
            Output('about-modal', 'is_open'),
            [Input('about-link', 'n_clicks'), Input('close-about', 'n_clicks')],
            [State('about-modal', 'is_open')])(self.toggleAbout)

    def toggleAbout(self, n1, n2, is_open):
        ''' Toggle the visibility of a modal HTML element.

            :param n1: number of clicks on the opening link
            :param n2: number of clicks on the close button
            :param is_open: current state of the modal element (open or closed)
        '''
        if n1 or n2:
            return not is_open
        return is_open

    # ------------------------------------- GENERIC COMPONENTS -------------------------------------

    def separator(self):
        return html.Hr(className='separator')

    def unorderedList(self, items):
        return dcc.Markdown(children=['''* {}'''.format('\r\n* '.join(items))])

    def paramSlider(self, id, p):
        ''' Linearly-spaced slider for a range parameter. '''
        return dcc.Slider(
            id=id,
            className='slider',
            min=0,
            max=p.n - 1,
            step=1,
            value=p.idefault
        )

    def paramSliderRow(self, id, p):
        ''' Return a label:slider:value table row for a range parameter. '''
        return html.Tr(className='table-row', children=[
            html.Td(id=f'{id}-label', className='row-label', children=[p.label]),
            html.Td(className='row-slider', children=[self.paramSlider(f'{id}-slider', p)]),
            html.Td(id=f'{id}-value', className='row-value')
        ])

    def linkSliderValue(self, id, p):
        ''' Assign a callback to update the corresponding text value upon slider change. '''
        self.callback(
            Output(f'{id}-value', 'children'),
            [Input(f'{id}-slider', 'value')])(lambda x: f'{si_format(p.values[x], 1)}{p.unit}')

    def paramSlidersTable(self, id, pdict, id_prefix=''):
        ''' Return a table of labeled sliders. '''
        if len(id_prefix) > 0:
            id_prefix = f'{id_prefix}-'
        return html.Table(id=f'{id}-slider-table', className='table', children=[
            self.paramSliderRow(f'{id_prefix}{k}', p) for k, p in pdict.items()])

    def panel(self, children):
        ''' Return a panel with contents. '''
        return html.Div(className='panel', children=children)

    def collapsablePanel(self, title, children):
        ''' Return a collapsable panel with title and contents. '''
        if title is None:
            title = ''
        return html.Details(open=True, className='panel', children=[
            html.Summary(title, className='panel-title'), *children])

    def dataRows(self, labels, values, units):
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

    def tabs(self, id, labels, values, default):
        ''' Construct tabs. '''
        return dcc.Tabs(
            id=f'{id}-tabs',
            className='custom-tabs-container',
            parent_className='custom-tabs',
            value=values[0], children=[
                dcc.Tab(
                    label=lbl,
                    value=val,
                    className='custom-tab',
                    selected_className='custom-tab--selected'
                ) for lbl, val in zip(labels, values)]
        )

    def valueDependentVisibility(self, ref_value):
        ''' Set the visibility of an element according to a ref value.

            :param ref_value: reference value that needs to be matched to show the element
            :return: lambda function handling the visibility toggle
        '''
        return lambda x: x != ref_value

    def tooltip(self, id, text, desc):
        return html.Div([
            html.Span(text, id=id, style={'textDecoration': 'underline', 'cursor': 'pointer'}),
            dbc.Tooltip(desc, target=id)
        ])

    def graph(self, id):
        return dcc.Loading(dcc.Graph(
            id=id, className='graph', figure={'data': [], 'layout': {}},
            config={
                'editable': False,
                'modeBarButtonsToRemove': [
                    'sendDataToCloud',
                    'displaylogo',
                    'toggleSpikelines']
            }
        ))

    def centered(self, *args, **kwargs):
        kwargs.pop('className', None)
        return html.Div(*args, className='centered-wrapper', **kwargs)
