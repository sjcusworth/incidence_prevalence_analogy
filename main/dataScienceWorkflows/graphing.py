from typing import Any, Union, Optional

from collections.abc import Callable
from itertools import compress
from math import ceil, floor
from os import getcwd, listdir, mkdir, rename, scandir
from os.path import isdir
from re import search, sub
from tqdm import tqdm
from numpy import isnan as npisnan
from numpy import vectorize as npvectorize
from seaborn import husl_palette

try:
    from dash import html, dcc
except ImportError:
    html = None
    dcc = None

import pandas as pd
from plotly.graph_objects import Layout, Scatter, Table, Figure, Bar, Box, Histogram
from plotly.subplots import make_subplots
import plotly.io as pio

class LayoutOpts:
    """
    """

    def __init__(self, colours=None,):
        """
        """
        self.colours=colours

        self.dpi=226.98 #determining page size relative to pixels
        self.overideColours=False

        #font
        self.font_family="SF Pro Text"
        self.font_color="black"

        #title
        self.title_font_size=35
        self.title_xanchor="left"
        self.title_pad={'t':20, 'b':20, 'l':30, 'r':20,}

        #axes
        self.axes_title_font_size=25
        self.x_axis_showgrid=True
        self.y_axis_showgrid=False
        self.axes_showline=True
        self.axes_showspikes=True
        self.axes_gridcolor="rgba(156, 156, 156, 0.7)"
        self.axes_linecolor="grey"
        self.axes_gridwidth=0.5
        self.axes_showticklabels=True
        self.axes_tickfont_size=20
        self.x_axis_tickangle=45
        self.y_axis_tickangle=0
        self.x_axis_dtick="M12"
        self.y_axis_dtick=None
        self.x_axis_tick0=None
        self.x_axis_autorange=True
        self.y_axis_autorange=False
        self.y_axis_nticks=5
        self.x_axis_nticks=5
        self.x_axis_ticklabelstep=5
        self.y_axis_ticklabelstep=10
        self.axes_title_standoff=20
        self.plot_scattergap = 0.75
        self.axes_exponentformat="e"
        self.x_axis_rndFactor=1
        self.y_axis_rndFactor=1

        #background
        self.plot_bgcolor="white"
        self.background_fillcolor="#C4ECFF"
        self.background_opacity=0
        self.background_y0=-0.1
        self.background_x0=-0.1
        self.background_y1=1.012
        self.background_x1=1

        #plot lines and points
        self.dash_types=["solid"]#["solid", "dot", "dash", "longdash", "dashdot", "longdashdot"],
        self.line_width=1.5
        self.marker_size=7
        self.highlightMarker_size=15
        self.highlightMarker_colour="red"
        self.marker_opacity=0.8
        self.line_opacity=0.8
        self.error_y_type="data"
        self.error_y_symmetric=False
        self.error_y_thickness=1.3
        self.error_y_width=0.5

        #bars
        self.bar_opacity=0.8

        #box
        self.box_opacity=0.8

        #legend
        self.legend_font_size=18
        self.legend_yanchor="top"
        self.legend_xanchor="left"
        self.legend_y=0.99
        self.legend_x=1.01
        self.legend_title_text="Subgroups: "
        self.legend_valign="middle"

        #output format
        self.autosize=False
        self.height=3
        self.width=5
        self.title_xref="paper"
        self.title_x=0
        self.table_font_size=12
        self.table_font_color=None
        self.table_columnwidth=[1,3,3]
        self.table_header_height=20
        self.table_header_sizeMulti=1.2
        self.table_cells_align="left"
        self.table_header_align="left"
        self.table_cells_height=45
        self.table_header_bgcolor=None
        self.table_cells_bgcolor=None
        self.subplot_vertical_spacing=None
        self.margin={'t':100, 'b':20, 'l':50, 'r':20,}

    def scaleOpts(self, scaleFactor):
        """
        """
        self.title_font_size=self.title_font_size * scaleFactor
        self.axes_title_font_size=self.axes_title_font_size * scaleFactor
        self.axes_gridwidth=self.axes_gridwidth * scaleFactor
        self.axes_tickfont_size=self.axes_tickfont_size * scaleFactor
        self.axes_title_standoff=self.axes_title_standoff*scaleFactor
        self.line_width=self.line_width * scaleFactor
        self.marker_size=self.marker_size * scaleFactor
        self.highlightMarker_size=self.highlightMarker_size * scaleFactor
        self.error_y_thickness=self.error_y_thickness * scaleFactor
        self.error_y_width=self.error_y_width * scaleFactor
        self.legend_font_size=self.legend_font_size * scaleFactor
        self.height=self.height * scaleFactor
        self.width=self.width * scaleFactor
        self.table_font_size=self.table_font_size * scaleFactor
        self.table_header_height=self.table_header_height * scaleFactor
        self.table_cells_height=self.table_cells_height * scaleFactor

    def make_dict_font(self):
        return dict(
            family=self.font_family, color=self.font_color
        )


    def make_dict_title(self):
        return dict(
            font_size=self.title_font_size,
            xanchor=self.title_xanchor,
            pad=self.title_pad,
            font_family=self.font_family,
            xref=self.title_xref,
        )


    def make_dict_line(self, color: str, dash: str) -> dict[str, object]:
        """
        Make layout dictionary for

        Parameters:
            color (str): color
            dash (str): dash

        Returns:
            dict: dictionary of layout options

        Usage:
            make_dict_line()
        """
        opacity = self.line_opacity
        colour = color[1:]
        colour = [str(int(colour[0:2], 16)),
            str(int(colour[2:4], 16)),
            str(int(colour[4:6], 16)),
            str(opacity)]
        colour = ", ".join(colour)


        dict_line = dict(
                dash=dash,
                color=f"rgba({colour})",
                width=self.line_width)
        return dict_line


    def make_dict_marker(self, color: str) -> dict[str, object]:
        """
        Make layout dictionary for

        Parameters:
            color (str): color

        Returns:
            dict: dictionary of layout options

        Usage:
            make_dict_marker()
        """
        dict_marker = dict(
            color=color,
            size=self.marker_size,
            opacity=self.marker_opacity,
        )
        return dict_marker


    def make_dict_bar(self, color: str) -> dict[str, object]:
        """
        Make layout dictionary for

        Parameters:
            color (str): color

        Returns:
            dict: dictionary of layout options

        Usage:
            make_dict_marker()
        """
        dict_marker = dict(
            color=color,
            opacity=self.bar_opacity,
        )
        return dict_marker


    def make_dict_box(self, color: str) -> dict[str, object]:
        """
        Make layout dictionary for

        Parameters:
            color (str): color

        Returns:
            dict: dictionary of layout options

        Usage:
            make_dict_marker()
        """
        dict_marker = dict(
            color=color,
            opacity=self.box_opacity,
        )
        return dict_marker


    def getColours(self, n=None):
        """
        Generate list of (virtually) infinite colours
        Generates colours only if self.colours has not been defined (is None)

        Parameters:
            n (int): Number of colours to generate
        Returns:
            (list): Hex colours of length >n (where length%6 = 0)
        """
        maxi = 235
        mini = 100
        z = 50

        if (self.colours is None or n>len(self.colours)):
            if n<=10 and not self.overideColours:
                cols = [
                        "#5778a4",
                        "#e49444",
                        "#d1615d",
                        "#85b6b2",
                        "#6a9f58",
                        "#e7ca60",
                        "#a87c9f",
                        "#f1a2a9",
                        "#967662",
                        "#b8b0ac",
                        ]
            else:
                def appendHex(x, y, z):
                    x=hex(int(round(x)))[2:]
                    y=hex(int(round(y)))[2:]
                    if len(x) == 1:
                        x = f"{x}0"
                    if len(y) == 1:
                        y = f"{y}0"

                    z = hex(int(round(z)))[2:]
                    if len(z) == 1:
                        z = f"{z}0"
                    out = [
                        f"#{x}{y}{z}",
                        f"#{z}{x}{y}",
                        f"#{y}{z}{x}",
                    ]
                    return out

                cols = []

                #Get initial cols for 3 groups (rgb)
                x=maxi
                y=mini
                cols = cols + appendHex(x, y, z)
                x=maxi
                y=maxi
                cols = cols + appendHex(x, y, z)

                i = 6
                while i <= n:
                    a = x - ((x-mini)/2)
                    b = y
                    cols = cols + appendHex(a, b, z)
                    cols = cols + appendHex(b, a, z)

                    x = x - ((x-mini)/2)
                    y = y - ((y-mini)/2)
                    i+=6

        else:
            cols = self.colours

        return cols


    def make_dict_background(self, i1: Union[int, str], i2: Union[int, str] = 1) -> dict[str, object]:
        """
        Make layout dictionary for

        Parameters:
            i1 (int): i1
            i2 (int): i2

        Returns:
            dict: dictionary of layout options

        Usage:
            make_dict_background()
        """
        if i1 == 1:
            i1 = ""
        if i2 == 1:
            i2 = ""

        dict_fill = dict(
            type="rect",
            yref=f"y{i1} domain",
            xref=f"x{i2} domain",
            y0=self.background_y0,
            x0=self.background_x0,
            y1=self.background_y1,
            x1=self.background_x1,
            fillcolor=self.background_fillcolor,
            opacity=self.background_opacity,
            layer="below",
            line_width=0,
        )
        return dict_fill

    def make_dict_table(self):
        """
        Make layout dictionary for table

        Returns:
            dict: dictionary of layout options

        Usages:
            make_dict_background()
        """
        font = dict(
            family=self.font_family,
            size=self.table_font_size,
            color=self.font_color,
        )
        if self.table_font_color is not None:
            font["color"]=self.table_font_color

        dict_table = dict(
            header = dict(
                font = font.copy(),
                height=self.table_header_height*self.table_header_sizeMulti,
                align=self.table_header_align,
                fill_color=self.table_header_bgcolor,
            ),
            cells = dict(
                font = font.copy(),
                align=self.table_cells_align,
                height=self.table_cells_height,
                fill_color=self.table_cells_bgcolor,
            ),
            columnwidth=self.table_columnwidth,
        )
        dict_table["header"]["font"]["size"] = font["size"]*self.table_header_sizeMulti
        return dict_table


class Visualisation:
    """
    Class storing all plotting parameters (i.e. layout options) for use when \
    plotting the data.

    Ensure plotly version >=5.12.0 (although tested on 5.15.0)

    Note:
        In plot_scatter, the output of ylims_def is used in get_scatterLayout, which auto defines the y-axis range. Therefore, the actual y-axis limits are not reflected by ylims_def
    """

    def __init__(
        self,
        label_map = None,
        colours: Union[list[str], None] = None,
        dict_lay_changes: Union[dict[str, Union[str, float, int, bool]], None] = None,
    ) -> None:

        self.label_map = label_map
        self.sf: float = 1.0
        self.ylims_def = lambda x: [0, x.max() + x.max() / 6.5]
        self.ylimsNeg_def = lambda x: [x.min()+(x.min()/6.5), x.max()+(x.max()/6.5)]
        self.ignore = []
        self.legends = {}
        self.legends_dash = {}
        self.sepPlots = dict()
        self.defaultCName = "Overall"

        # Plot Formatting
        self.dict_lay = LayoutOpts(colours)


    def highlight_point(self, name_col, highlight, col, highlight_col=None,):
        """

        Args:
            name_col:
            highlight:
            col:
            highlight_col:

        Returns:
            Tuple(Dict, bool): Dictionary of marker layout, indicator to whether point has been highlighted

        Notes:
            If highlight is None, does the same as self.make_dict_marker.

        """
        dict_marker = self.dict_lay.make_dict_marker(color=col[name_col])
        highlightedPoint=False
        if highlight_col is None:
            highlight_col = name_col
        if highlight is not None:
            if not isinstance(highlightedPoint, str):
                #for type float nan values
                highlightedPoint = str(highlightedPoint)
            if isinstance(highlight_col, str):
                highlight_col = [highlight_col]#highlight_col.split(",")
            highlight_point = [
                    # is filter_ in column to highlight
                    "".join([e for e in filter_ if e.isalnum()]).lower() in [
                        "".join([e for e in x if e.isalnum()]).lower() for x in highlight_col
                    ] for filter_ in highlight]
            #highlight_point = all(any(highlight_point))
            highlight_point = any(highlight_point)

            if highlight_point and len(highlight) > 0:
                highlightedPoint=True
                #dict_marker["line"] = {"color": self.dict_lay["highlightMarker_colour"],
                                       #"size": self.dict_lay["highlightMarker_size"],}
                dict_marker["color"] = self.dict_lay.highlightMarker_colour
                dict_marker["size"] = self.dict_lay.highlightMarker_size

        return dict_marker, highlightedPoint


    def check_cname(self, data: pd.DataFrame, c_name="OVERALL"):
        """
        Generate uni-value column for input data
        Used where input data is not stratified
        Where a "stratified-by" column does not exist

        Parameters:
            data (pd.DataFrame): Input Analogy data
            c_name (str): Value for new column (defaults to "OVERALL")
        Returns:
            pd.DataFrame: Data with new column

        Notes:
            Used to make "OVERALL" inputs compatible with stratified inputs
        """
        if c_name not in data.columns:
            #print(f"Warning: {c_name} column does not exist. Making temp.")
            data[c_name] = [c_name] * len(data.index)

        return data


    def rm_NumNull(self, data: pd.DataFrame, metric_type: Union[str, None] = None) -> pd.DataFrame:
        """
        Remove any 0 or NA values in Numerator

        Parameters:
            data (pd.DataFrame): Input data
            metric_type (str): Name of column to plot on x-axis

        Returns:
            pd.DataFrame: Processed Data
        """
        if metric_type is not None:
            df_temp = data.loc[(data[metric_type].notna() & \
                    data[metric_type].notnull())]
        else:
            df_temp = data
        #numerator = [x if x != 0 else pd.NA for x in df_temp["Numerator"]]
        #df_temp_map = pd.notna(numerator)
        #df_temp = df_temp.loc[df_temp_map]

        return df_temp


    def updateLayout_subplots(self, plots, layout, out_row_n, title_text, shapes):
        """
        Update layout of subplots using output from plot_incprev()

        Parameters:
            plots (plotly.Figure): Plotly subplots
            layout_out (tuple): 1st element plotly.Layout
            out_row_n (int): Number of rows of subplots Figure
            subcat (str): Name of stratification factor (used for title)
            shapes (list): List of shapes to add to subplots
        Returns:
            plotly.Figure: Subplots with updated layout

        Notes:
            plot_incprev() returns a tuple, where the 1st element contains the\
            layout options. If using without plot_incprev(), pass in \
            layout_out as a tuple of 1 element (layout_out).
        """
        #Prevents overwriting x and y axes of all subplots
        layout.xaxis=None
        layout.yaxis=None
        layout.title=None

        plots.update_layout(
            layout,
            autosize=self.dict_lay.autosize,
            height=self.dict_lay.height * self.dict_lay.dpi * out_row_n,
            width=self.dict_lay.width * self.dict_lay.dpi,
            title_text=title_text,
            shapes=shapes,
        )

        return plots


    def make_dict_dash_type(self, data: pd.DataFrame,
                            c_name=None) -> dict[str, str]:
        """
        Make layout dictionary

        Parameters:
            data (pd.DataFrame): Input data

        Returns:
            dict: dictionary of layout options

        Usage:
            make_dict_dash_type()
        """
        if c_name is None:
            lay: list[str] = self.dict_lay.dash_types
            dash_type = dict(zip(set(data["results"]), lay))
        else:
            lay: list[str] = self.dict_lay.dash_types
            if len(set(data[c_name])) > len(lay):
                multiFact = ceil(len(set(data[c_name])) / len(lay))
                lay = lay*multiFact

            dash_type = dict(zip(set(data[c_name]), lay))

        return dash_type


    def make_dict_col(self, data: pd.DataFrame,
                      c_name: str,
                      l: float = 0.55,
                      s: float = 0.8) -> Union[dict[str, str], None]:
        """
        Make layout dictionary

        Parameters:
            data (pd.DataFrame): Input data
            c_name (str): c_name

        Returns:
            dict: dictionary of layout options

        Usage:
            make_dict_col()
        """
        #colours = self.getColours(n=len(set(data[c_name])))
        colours = husl_palette(len(set(data[c_name])),
                               l=l,
                               s=s).as_hex()
        col = dict(zip(set(data[c_name]), colours))
        return col


    def make_dict_error_y(
        self, array: pd.Series, arrayminus: pd.Series
    ) -> dict[str, Union[str, bool, pd.Series, float, int]]:
        """
        Make layout dictionary for ...

        Parameters:
            array (pd.Series): array
            arrayminus (pd.Series): arrayminus

        Returns:
            dict: dictionary of layout options

        Usage:
            make_dict_error_y()
        """
        dict_error_y = dict(
            type=self.dict_lay.error_y_type,
            symmetric=self.dict_lay.error_y_symmetric,
            array=array,
            arrayminus=arrayminus,
            thickness=self.dict_lay.error_y_thickness,
            width=self.dict_lay.error_y_width,
        )
        return dict_error_y


    def make_dict_axes(
        self,
        ylims: Optional[list[Union[float, int]]],
        y_var: Optional[str],
        xlims=None,
        x_var: str="x",
        x_start=None,
    ) -> tuple[dict[str, object], dict[str, object]]:
        """
        Make layout dictionary for

        Parameters:
            ylims (list): ylims

        Returns:
            dict: dictionary of layout options
            dict: dictionary of layout options

        Usage:
            make_dict_axes()
        """
        if x_start is not None:
            x_tick0 = x_start
        else:
            x_tick0 = self.dict_lay.x_axis_tick0
        if xlims is not None:
            x_tick0 = xlims[0]


        dict_xaxis = dict(
            title_font_size=self.dict_lay.axes_title_font_size,
            title_font_family=self.dict_lay.font_family,
            title_standoff=self.dict_lay.axes_title_standoff,
            autorange=self.dict_lay.x_axis_autorange,
            showgrid=self.dict_lay.x_axis_showgrid,
            showline=self.dict_lay.axes_showline,
            showspikes=self.dict_lay.axes_showspikes,
            gridcolor=self.dict_lay.axes_gridcolor,
            linecolor=self.dict_lay.axes_linecolor,
            gridwidth=self.dict_lay.axes_gridwidth,
            showticklabels=self.dict_lay.axes_showticklabels,
            tickfont_size=self.dict_lay.axes_tickfont_size,
            tickfont_family=self.dict_lay.font_family,
            tickangle=self.dict_lay.x_axis_tickangle,
            dtick=self.dict_lay.x_axis_dtick,
            exponentformat=self.dict_lay.axes_exponentformat,
        )

        dict_yaxis = dict_xaxis.copy()
        dict_yaxis["dtick"] = self.dict_lay.y_axis_dtick
        dict_yaxis["tickangle"] = self.dict_lay.y_axis_tickangle
        dict_yaxis["autorange"] = self.dict_lay.y_axis_autorange
        dict_yaxis["showgrid"] = self.dict_lay.y_axis_showgrid

        def xy_round(x, nticks=10, axis="y"):
            #Haven't tested effect of rndFactor yet
            rndFactor = getattr(self.dict_lay, f"{axis}_axis_rndFactor")
            n=0
            x = x*rndFactor
            if x != 0:
                while x>=10:
                    x = x/10
                    n+=1
                while x<1:
                    x = x*10
                    n-=1

            maxi = 0.5*ceil(x/0.5)
            x = maxi*(10**n) / nticks
            maxi = maxi*(10**n)

            return [x/rndFactor, maxi/rndFactor]

        if y_var is not None:
            ticks = xy_round(ylims[1]-ylims[0],
                            nticks=self.dict_lay.y_axis_nticks)
            subdivide_y = self.dict_lay.y_axis_ticklabelstep
            if self.dict_lay.y_axis_dtick is None:
                dict_yaxis["dtick"] = ticks[0] / subdivide_y
            dict_yaxis["range"] = [ylims[0], ylims[0]+ticks[1]]
            dict_yaxis["ticklabelstep"] = subdivide_y

        if xlims is not None:
            subdivide_x = self.dict_lay.x_axis_ticklabelstep
            ticks_x = xy_round(xlims[1]-xlims[0],
                        nticks=self.dict_lay.x_axis_nticks,
                               axis="x")
            dict_xaxis["dtick"] = ticks_x[0] / subdivide_x
            dict_xaxis["range"] = [xlims[0], xlims[0]+ticks_x[1]]
            dict_xaxis["ticklabelstep"] = subdivide_x
        else:
            dict_xaxis["tick0"]=x_tick0
            dict_xaxis["ticklabelstep"]=self.dict_lay.x_axis_ticklabelstep

        dict_xaxis["title_text"] = x_var
        if y_var is not None:
            dict_yaxis["title_text"] = y_var

        if y_var is not None:
            ylims[0] = dict_yaxis["range"][0]
            ylims[1] = dict_yaxis["range"][1]

        return dict_xaxis, dict_yaxis, ylims


    def outPlots(self,
                 traces,
                 layout,
                 label=None,
                 method: str = "trace",
                 toDisk=True):
        """
        method: [trace | figure | interactive | div | png]
        """
        if method == "trace":
            return traces, layout

        elif method == "figure" or method == "interactive":
            plot = Figure(
                data = traces,
                layout = layout
            )

            if method == "interactive":
                if not toDisk:
                    return plot
                else:
                    plot.write_html(f"{label}.html")
                    return label
            else:
                return plot

        elif method == "png":
            plot = Figure(
                data = traces,
                layout = layout
            )
            #for i, x in enumerate(plot.layout.annotations):
            #    x.update(
            #             font={"size": self.dict_lay["annotation_font_size"],
            #                   "family": self.dict_lay["font_family"]}
            #             )

            pio.kaleido.scope.mathjax = None
            dpi = self.dict_lay.dpi
            plot.write_image(f"{label}.png",
                             engine="kaleido",
                             width=self.dict_lay.width*dpi,
                             height=self.dict_lay.height*dpi,
                             scale=1,)
            return label

        elif method == "div":
            plot = Figure(
                data = traces,
                layout = layout
            )
            divObj = html.Div([
                dcc.Graph(figure=plot)
                ])

            return divObj



    def update_subgroupColours(self, data, c_name, subgrouping=None, useObjectColours=True):
        """

        Args:
            data:
            c_name:
            subgrouping (None|str): String to use as identifier of collection of labels in passed data c_name column
            makeNewDict (bool): Update and use self colour dict, or return new

        Returns:

        Notes:
            Subgrouping allows grouping of colour dictionaries under user-defined labels

        """
        if subgrouping is None:
            subgrouping = c_name

        if useObjectColours:
            if subgrouping in self.legends.keys() and self.legends[subgrouping] is not None:
                col = self.legends[subgrouping]
                dash_type = self.legends_dash[subgrouping]
                #check all levels are in dict
                #if not, add levels to c_name's colour dictionary
                levels = []
                for name_col, group_col in data.groupby(c_name):
                    levels.append(name_col)

                stored = set(col.keys())
                levels = set(levels)
                check = levels == col
                if not check:
                    diff = levels.difference(col)
                    for i, x in enumerate(diff):
                        col[x] = self.dict_lay.getColours(len(stored)+i)[len(stored)+i-1]
                        dash_type[x] = self.dict_lay.dash_types[(len(stored)+i) % len(self.dict_lay.dash_types)]
                    self.legends[subgrouping] = col

            elif subgrouping not in self.legends.keys():
                col = self.make_dict_col(data, c_name)
                dash_type = self.make_dict_dash_type(data, c_name)
                self.legends[subgrouping] = col
                self.legends_dash[subgrouping] = dash_type

        else:
            col = self.make_dict_col(data, c_name)
            dash_type = self.make_dict_dash_type(data, c_name)

        if col is None:
            print(f"Failed for {c_name}")
            return None, None

        return col, dash_type


    def traceFormatPrep(self,
                        data,
                        c_name,
                        col_colour,
                        useObjectColours=False,
                        colourDictionaryId=None,):

        if c_name is None:
            c_name = self.defaultCName

        if col_colour is not None:
            colourLabelMap = dict(zip(data[c_name].to_list(),
                                      data[col_colour].to_list()))
            colourCol = col_colour
        else:
            colourLabelMap = None
            colourCol = c_name

        col, dash_type = self.update_subgroupColours(data,
                                                     colourCol,
                                                     subgrouping=colourDictionaryId,
                                                     useObjectColours=useObjectColours)

        return colourLabelMap, col, dash_type


    def dataPrep(self,
                 data,
                 y_var: Optional[str],
                 x_var: Optional[str],
                 c_name,
                 ylims=None,
                 xlims=None,
                 ):

        #Rm missing values
        if y_var is not None:
            data = data[(data[y_var].notna() | ~data[y_var].apply(npisnan))]
        if x_var is not None:
            data = data[(data[x_var].notna() | ~data[x_var].apply(npisnan))]

        if y_var is not None:
            is_yNumeric = str(data[y_var].dtype).lower().startswith("int") or \
                    str(data[y_var].dtype).lower().startswith("float")
        if x_var is not None:
            is_xNumeric = str(data[x_var].dtype).lower().startswith("int") or \
                    str(data[x_var].dtype).lower().startswith("float")

        if y_var is not None and is_yNumeric:
            if data[y_var].min()>=0:
                ylims_def = self.ylims_def
            else:
                ylims_def = self.ylimsNeg_def
            if ylims is None:
                ylims = ylims_def(data[y_var])
        else:
            ylims=None

        if x_var is not None and is_xNumeric:
            if data[x_var].min()>=0:
                xlims_def = self.ylims_def
            else:
                xlims_def = self.ylimsNeg_def
            if xlims is None:
                xlims = xlims_def(data[x_var])
        else:
            xlims = None

        if y_var is not None:
            upper_limit = [x if x > ylims[1] else pd.NA for x in data[y_var]]
            data = data.assign(upper_limit=upper_limit)


        if "results" not in data.columns:
            data = data.assign(results=["NotRequired"] * len(data.index))

        dash_type = self.make_dict_dash_type(data)

        if c_name is None:
            c_name = self.defaultCName
            data = pd.concat([data.reset_index(),(pd.Series([c_name]*data.shape[0],
                                     name=c_name,
                                     dtype="string",))],
                             axis=1)
        else:
            try:
                data[c_name] = pd.to_numeric(data[c_name])
            except:
                pass
            else:
                data[c_name] = pd.to_numeric(data[c_name])

            data[c_name] = data[c_name].fillna("Null")
            data = data.sort_values(by=c_name)

        return data, xlims, ylims


    def plot_scatter(
        self,
        data: pd.DataFrame,
        y_var: str,
        x_var: str,
        c_name: [str,None] = None,
        ylims: Union[list[float], None] = None,
        xlims: Union[list[float], None] = None,
        legend: bool = True,
        dir: str = ".",
        interactive = False,
        meta_vars = None,
        out_type = "trace",
        highlight = None,
        is_errorY = False,
        cols_errorY = ["Lower", "Upper"],
        withPoints = True,
        withLine = False,
        toDisk = False,
        overrideColour = None,
        overrideDashType = None,
        col_colour = None,
        useObjectColours = False,
        colourDictionaryId = None,
    ) -> tuple[list[Scatter], Layout, dict[str, object], dict[str, object],]:
        """
        Plot a simple scatter graph
        Non-automated implementation of plot_incprev()

        Parameters:
            data (pd.DataFrame):
            y_var (str):
            x_var (str):
            c_name (str):
            is_numeric_x (bool):
            ylims (list):
            xlims (list):
            legend (bool):
            dir (str): path to dir to save graphs to disk (if applicable)
            interactive (bool): Output interactive? If returnTrace==False
            meta_vars (list): List of column names to show in hover labels\
                if interactive output
            returnPlot (bool): Return Figure object? If returnTrace==False
            returnTrace (bool): Return list of traces or save figure to disk

        Returns:
            list: List of traces
                if returnTrace is True

        Notes:
            returnTrace determines whether a list of traces is returned, or\
            the final figure. The final figure can be returned as a plotly \
            Figure object (returnPlot is True), saved to disk as .html \
            (interactive is True AND returnPlot is False) or saved to disk as \
            .png (interactive is True AND returnPlot is False)
            If x axis values represent years, but is in numeric format, then \
            the xaxis limits (unless otherwise specified) will automatically be\
             determined in the same way as the y-axis limits, meaning lower \
            limit will be 0 (causes an issue when tested on data 2000-2021).

            When plotting, and useObjectColours is True, to define unique colours,
            a colour dict will be defined, with each unique value in data c_name
            column being assigned a colour. These colours will be reused when
            plotting data points matching a label in this dict. To define
            seperate dictionaries, where one might pass in data filtered for
            specific values in c_name, and wish to reuse already used colours,
            but for a different setting (e.g. graphing data disaggregated by
            ethnicity, and graphing data disaggregated by location), one should
            define the parameter colourDictionaryId.
        """
        data, xlims, ylims = self.dataPrep(data,
                                           y_var,
                                           x_var,
                                           c_name,
                                           ylims,
                                           xlims,)

        colourLabelMap, col, dash_type = self.traceFormatPrep(data,
                                                              c_name,
                                                              col_colour,
                                                              useObjectColours,
                                                              colourDictionaryId,)

        dict_xaxis, dict_yaxis, ylims = self.make_dict_axes(ylims, y_var,
                                             xlims, x_var,)

        title_text = f"{y_var} trend by {x_var}"

        layout = self.get_scatterLayout(dict_yaxis,
                                        dict_xaxis,
                                        title_text,)

        #Plotting
        traces = []
        #May be better to alter this so that currently stored legend labels in this object are stored as an attr, not just stored at the level of the plotting method. But this will work for now.
        legendLabels = set()

        if c_name is None:
            c_name = self.defaultCName

        for name_col, group_col in data.groupby(c_name):

            #Determine whether to plot this point
            skip = False
            if name_col not in self.ignore:
                for x in self.ignore:
                    if str(name_col).find(f"'{x}'") != -1:
                        skip=True
                        break
            else:
                skip=True
            if skip==True:
                continue

            #Ensure data is ordered by x_axis
            group_col.sort_values(by=x_var, inplace=True)

            #Assign colour and dash type
            if col_colour is not None:
                nameLabel_colour = colourLabelMap[name_col]
            else:
                nameLabel_colour = name_col

            #Format marker/line
            dict_line = self.dict_lay.make_dict_line(color=col[nameLabel_colour],
                                            dash=dash_type[nameLabel_colour])

            dict_marker, skipOverrideCol = self.highlight_point(nameLabel_colour, highlight, col, highlight_col=name_col)

            if overrideColour is not None and not skipOverrideCol:
                dict_marker["color"] = overrideColour
                dict_line["color"] = overrideColour
            if overrideDashType is not None:
                dict_line["dash"] = overrideDashType

            #Set labels for interactive
            if meta_vars is not None:
                meta_label = group_col[meta_vars].applymap(str)
                meta_label = meta_label.agg(', '.join, axis=1).tolist()
            else:
                meta_label = None

            #Format error bars
            dict_error_y = self.formatErrorBar(is_errorY,
                                               group_col,
                                               cols_errorY,
                                               dict_line["color"],
                                               ylims,
                                               y_var,)

            if colourDictionaryId is not None:
                legendgroup = colourDictionaryId
            else:
                legendgroup = "group"

            #Create traces
            if withPoints and withLine:
                mode = "lines+markers"
            elif withPoints:
                mode = "markers"
            else:
                mode = 'lines'

            traces.append(
                Scatter(
                    x=group_col[x_var],
                    y=group_col[y_var],
                    mode=mode,
                    marker=dict_marker,
                    line=dict_line,
                    showlegend=legend,
                    legendgroup=nameLabel_colour,#legendgroup,
                    name=nameLabel_colour,
                    hoverinfo="text",
                    hovertext=meta_label,
                    error_y=dict_error_y,
                )
            )

            legendLabels.add(nameLabel_colour)


        return self.outPlots(traces, layout,
                             sub("[[:punct:] ]+", "", title_text),
                             out_type,
                             toDisk=toDisk)


    def formatErrorBar(self,
                       plotError:bool,
                       dat: pd.Series,
                       cols_error: list[str],
                       colour: str,
                       lims: list[float],
                       valueCol: str,):
        #Format error bars
        if plotError:
            #Ensure error bar is shown, even if beyond y limit
            withinBounds_upper = npvectorize(lambda x,y: x if x<=y else y)
            withinBounds_lower = npvectorize(lambda x,y: x if x>=y else y)
            error_upper = withinBounds_upper(dat[cols_error[1]], lims[1])
            error_lower = withinBounds_lower(dat[cols_error[0]], lims[0])

            dict_error_y = self.make_dict_error_y(
                array=error_upper - dat[valueCol], #Upper
                arrayminus=dat[valueCol] - error_lower, #Lower
            )

            dict_error_y["color"] = colour
        else:
            dict_error_y = None

        return dict_error_y


    def plot_bar(
        self,
        data: pd.DataFrame,
        y_var: str,
        x_var: str,
        c_name: [str,None] = None,
        ylims: Union[list[float], None] = None,
        xlims: Union[list[float], None] = None,
        legend: bool = True,
        dir: str = ".",
        interactive = False,
        meta_vars = None,
        out_type = "trace",
        highlight = None,
        is_errorY = False,
        cols_errorY = ["Lower", "Upper"],
        toDisk = False,
        overrideColour = None,
        col_colour = None,
        useObjectColours = False,
        colourDictionaryId = None,
    ) -> tuple[list[Scatter], Layout, dict[str, object], dict[str, object],]:
        """
        Plot a simple scatter graph
        Non-automated implementation of plot_incprev()

        Parameters:
            data (pd.DataFrame):
            y_var (str):
            x_var (str):
            c_name (str):
            is_numeric_x (bool):
            ylims (list):
            xlims (list):
            legend (bool):
            dir (str): path to dir to save graphs to disk (if applicable)
            interactive (bool): Output interactive? If returnTrace==False
            meta_vars (list): List of column names to show in hover labels\
                if interactive output
            returnPlot (bool): Return Figure object? If returnTrace==False
            returnTrace (bool): Return list of traces or save figure to disk

        Returns:
            list: List of traces
                if returnTrace is True

        Notes:
            returnTrace determines whether a list of traces is returned, or\
            the final figure. The final figure can be returned as a plotly \
            Figure object (returnPlot is True), saved to disk as .html \
            (interactive is True AND returnPlot is False) or saved to disk as \
            .png (interactive is True AND returnPlot is False)
            If x axis values represent years, but is in numeric format, then \
            the xaxis limits (unless otherwise specified) will automatically be\
             determined in the same way as the y-axis limits, meaning lower \
            limit will be 0 (causes an issue when tested on data 2000-2021).
        """
        data, xlims, ylims = self.dataPrep(data,
                                           y_var,
                                           x_var,
                                           c_name,
                                           ylims,
                                           xlims,)

        colourLabelMap, col, _ = self.traceFormatPrep(data,
                                                      c_name,
                                                      col_colour,
                                                      useObjectColours,
                                                      colourDictionaryId,)

        dict_xaxis, dict_yaxis, ylims = self.make_dict_axes(ylims, y_var,
                                             xlims, x_var,)

        title_text = f"{y_var} trend by {x_var}"
        layout = self.get_barLayout(dict_yaxis,
                                    dict_xaxis,
                                    title_text,)

        #Plotting
        traces = []
        #May be better to alter this so that currently stored legend labels in this object are stored as an attr, not just stored at the level of the plotting method. But this will work for now.
        legendLabels = set()

        if c_name is None:
            c_name = self.defaultCName

        for name_col, group_col in data.groupby(c_name):

            skip = False
            if name_col not in self.ignore:
                for x in self.ignore:
                    if str(name_col).find(f"'{x}'") != -1:
                        skip=True
                        break
            else:
                skip=True
            if skip==True:
                continue

            #group_col.sort_values(by=x_var, inplace=True)

            if col_colour is not None:
                nameLabel_colour = colourLabelMap[name_col]
            else:
                nameLabel_colour = name_col

            dict_bar = self.dict_lay.make_dict_bar(color=col[nameLabel_colour])

            if overrideColour is not None:
                dict_bar["color"] = overrideColour

            if meta_vars is not None:
                meta_label = group_col[meta_vars].applymap(str)
                meta_label = meta_label.agg(', '.join, axis=1).tolist()
            else:
                meta_label = None

            dict_error_y = self.formatErrorBar(is_errorY,
                                               group_col,
                                               cols_errorY,
                                               dict_bar["color"],
                                               ylims,
                                               y_var,)

            traces.append(
                Bar(
                    x=group_col[x_var],
                    y=group_col[y_var],
                    showlegend=all([legend,
                                    nameLabel_colour not in legendLabels]),
                    legendgroup=nameLabel_colour,#"group",
                    name=nameLabel_colour,
                    hoverinfo="text",
                    hovertext=meta_label,
                    error_y=dict_error_y,
                )
            )
            #Ensure not duplicating error bars if withLine==True
            legendLabels.add(nameLabel_colour)


        return self.outPlots(traces, layout,
                             sub("[[:punct:] ]+", "", title_text),
                             out_type,
                             toDisk=toDisk)


    def plot_hist(
        self,
        data: pd.DataFrame,
        x_var: str,
        c_name: Union[str,None] = None,
        nbinsx: Optional[int] = None,
        xlims: Union[list[float], None] = None,
        legend: bool = True,
        dir: str = ".",
        interactive = False,
        meta_vars = None,
        out_type = "trace",
        highlight = None,
        toDisk = False,
        overrideColour = None,
        col_colour = None,
        useObjectColours = False,
        colourDictionaryId = None,
    ) -> tuple[list[Scatter], Layout, dict[str, object], dict[str, object],]:
        """
        Plot a simple scatter graph
        Non-automated implementation of plot_incprev()

        Parameters:
            data (pd.DataFrame):
            y_var (str):
            x_var (str):
            c_name (str):
            is_numeric_x (bool):
            ylims (list):
            xlims (list):
            legend (bool):
            dir (str): path to dir to save graphs to disk (if applicable)
            interactive (bool): Output interactive? If returnTrace==False
            meta_vars (list): List of column names to show in hover labels\
                if interactive output
            returnPlot (bool): Return Figure object? If returnTrace==False
            returnTrace (bool): Return list of traces or save figure to disk

        Returns:
            list: List of traces
                if returnTrace is True

        Notes:
            returnTrace determines whether a list of traces is returned, or\
            the final figure. The final figure can be returned as a plotly \
            Figure object (returnPlot is True), saved to disk as .html \
            (interactive is True AND returnPlot is False) or saved to disk as \
            .png (interactive is True AND returnPlot is False)
            If x axis values represent years, but is in numeric format, then \
            the xaxis limits (unless otherwise specified) will automatically be\
             determined in the same way as the y-axis limits, meaning lower \
            limit will be 0 (causes an issue when tested on data 2000-2021).
        """
        data, xlims, _ = self.dataPrep(data,
                                           None,
                                           x_var,
                                           c_name,
                                           ylims = None,
                                           xlims = xlims,)

        colourLabelMap, col, _ = self.traceFormatPrep(data,
                                                      c_name,
                                                      col_colour,
                                                      useObjectColours,
                                                      colourDictionaryId,)

        dict_xaxis, dict_yaxis, ylims = self.make_dict_axes(None, None,
                                             xlims, x_var,)
        dict_yaxis["autorange"] = True

        title_text = f"Count by {x_var}"
        layout = self.get_histLayout(
                dict_yaxis,
                dict_xaxis,
                title_text,)

        #Plotting
        traces = []
        #May be better to alter this so that currently stored legend labels in this object are stored as an attr, not just stored at the level of the plotting method. But this will work for now.
        legendLabels = set()

        if c_name is None:
            c_name = self.defaultCName

        for name_col, group_col in data.groupby(c_name):

            skip = False
            if name_col not in self.ignore:
                for x in self.ignore:
                    if str(name_col).find(f"'{x}'") != -1:
                        skip=True
                        break
            else:
                skip=True
            if skip==True:
                continue

            #group_col.sort_values(by=x_var, inplace=True)

            if col_colour is not None:
                nameLabel_colour = colourLabelMap[name_col]
            else:
                nameLabel_colour = name_col

            dict_bar = self.dict_lay.make_dict_bar(color=col[nameLabel_colour])

            if overrideColour is not None:
                dict_bar["color"] = overrideColour

            if meta_vars is not None:
                meta_label = group_col[meta_vars].applymap(str)
                meta_label = meta_label.agg(', '.join, axis=1).tolist()
            else:
                meta_label = None

            traces.append(
                Histogram(
                    x=group_col[x_var],
                    showlegend=all([legend,
                                    nameLabel_colour not in legendLabels]),
                    legendgroup=nameLabel_colour,#"group",
                    name=nameLabel_colour,
                    hoverinfo="text",
                    hovertext=meta_label,
                    nbinsx=nbinsx,
                )
            )
            #Ensure not duplicating error bars if withLine==True
            legendLabels.add(nameLabel_colour)


        return self.outPlots(traces, layout,
                             sub("[[:punct:] ]+", "", title_text),
                             out_type,
                             toDisk=toDisk)


    def plot_box(
        self,
        data: pd.DataFrame,
        y_var: str,
        x_var: str,
        c_name: [str,None] = None,
        ylims: Union[list[float], None] = None,
        xlims: Union[list[float], None] = None,
        legend: bool = True,
        dir: str = ".",
        interactive = False,
        meta_vars = None,
        out_type = "trace",
        highlight = None,
        is_errorY = False,
        cols_errorY = ["Lower", "Upper"],
        toDisk = False,
        overrideColour = None,
        col_colour = None,
        useObjectColours = False,
        colourDictionaryId = None,
    ) -> tuple[list[Scatter], Layout, dict[str, object], dict[str, object],]:
        """
        Plot a box graph

        Parameters:
            data (pd.DataFrame):
            y_var (str):
            x_var (str):
            c_name (str):
            is_numeric_x (bool):
            ylims (list):
            xlims (list):
            legend (bool):
            dir (str): path to dir to save graphs to disk (if applicable)
            interactive (bool): Output interactive? If returnTrace==False
            meta_vars (list): List of column names to show in hover labels\
                if interactive output
            returnPlot (bool): Return Figure object? If returnTrace==False
            returnTrace (bool): Return list of traces or save figure to disk

        Returns:
            list: List of traces
                if returnTrace is True

        Notes:
            returnTrace determines whether a list of traces is returned, or\
            the final figure. The final figure can be returned as a plotly \
            Figure object (returnPlot is True), saved to disk as .html \
            (interactive is True AND returnPlot is False) or saved to disk as \
            .png (interactive is True AND returnPlot is False)
            If x axis values represent years, but is in numeric format, then \
            the xaxis limits (unless otherwise specified) will automatically be\
             determined in the same way as the y-axis limits, meaning lower \
            limit will be 0 (causes an issue when tested on data 2000-2021).
        """
        data, xlims, ylims = self.dataPrep(data,
                                           y_var,
                                           x_var,
                                           c_name,
                                           ylims,
                                           xlims,)

        colourLabelMap, col, _ = self.traceFormatPrep(data,
                                                      c_name,
                                                      col_colour,
                                                      useObjectColours,
                                                      colourDictionaryId,)

        dict_xaxis, dict_yaxis, ylims = self.make_dict_axes(ylims, y_var,
                                             xlims, x_var,)

        title_text = f"{y_var} trend by {x_var}"
        layout = self.get_boxLayout(dict_yaxis,
                                    dict_xaxis,
                                    title_text,)

        #Plotting
        traces = []
        #May be better to alter this so that currently stored legend labels in this object are stored as an attr, not just stored at the level of the plotting method. But this will work for now.
        legendLabels = set()

        if c_name is None:
            c_name = self.defaultCName

        for name_col, group_col in data.groupby(c_name):

            skip = False
            if name_col not in self.ignore:
                for x in self.ignore:
                    if str(name_col).find(f"'{x}'") != -1:
                        skip=True
                        break
            else:
                skip=True
            if skip==True:
                continue

            #group_col.sort_values(by=x_var, inplace=True)

            if col_colour is not None:
                nameLabel_colour = colourLabelMap[name_col]
            else:
                nameLabel_colour = name_col

            dict_box = self.dict_lay.make_dict_box(color=col[nameLabel_colour])

            if overrideColour is not None:
                dict_box["color"] = overrideColour

            if meta_vars is not None:
                meta_label = group_col[meta_vars].applymap(str)
                meta_label = meta_label.agg(', '.join, axis=1).tolist()
            else:
                meta_label = None

            dict_error_y = self.formatErrorBar(is_errorY,
                                               group_col,
                                               cols_errorY,
                                               dict_box["color"],
                                               ylims,
                                               y_var,)

            traces.append(
                Box(
                    x=group_col[x_var],
                    y=group_col[y_var],
                    showlegend=all([legend,
                                    nameLabel_colour not in legendLabels]),
                    legendgroup=nameLabel_colour,#"group",
                    name=nameLabel_colour,
                    hoverinfo="text",
                    hovertext=meta_label,
                    error_y=dict_error_y,
                )
            )
            #Ensure not duplicating error bars if withLine==True
            legendLabels.add(nameLabel_colour)


        return self.outPlots(traces, layout,
                             sub("[[:punct:] ]+", "", title_text),
                             out_type,
                             toDisk=toDisk)


    def get_scatterLayout(self,
                          dict_yaxis,
                          dict_xaxis,
                          title_text,):
        title_text = title_text
        layout = Layout(
            title_text=title_text,
            legend_title_text=self.dict_lay.legend_title_text,
            legend_title_font_size=self.dict_lay.axes_title_font_size,
            legend_font_size=self.dict_lay.legend_font_size,
            legend_valign=self.dict_lay.legend_valign,
            legend_x = self.dict_lay.legend_x,
            legend_y = self.dict_lay.legend_y,
            legend_xanchor = self.dict_lay.legend_xanchor,
            legend_yanchor = self.dict_lay.legend_yanchor,
            margin=self.dict_lay.margin,
            scattermode="group",
            scattergap=self.dict_lay.plot_scattergap,
            font=self.dict_lay.make_dict_font(),
            title=self.dict_lay.make_dict_title(),
            xaxis=dict_xaxis,
            yaxis=dict_yaxis,
            plot_bgcolor=self.dict_lay.plot_bgcolor,
        )
        return layout


    def get_barLayout(self,
                      dict_yaxis,
                      dict_xaxis,
                      title_text,):
        dict_xaxis["ticklabelstep"] = 1
        title_text = title_text
        layout = Layout(
            title_text=title_text,
            legend_title_text=self.dict_lay.legend_title_text,
            legend_title_font_size=self.dict_lay.axes_title_font_size,
            legend_font_size=self.dict_lay.legend_font_size,
            legend_valign=self.dict_lay.legend_valign,
            legend_x = self.dict_lay.legend_x,
            legend_y = self.dict_lay.legend_y,
            legend_xanchor = self.dict_lay.legend_xanchor,
            legend_yanchor = self.dict_lay.legend_yanchor,
            margin=self.dict_lay.margin,
            font=self.dict_lay.make_dict_font(),
            title=self.dict_lay.make_dict_title(),
            xaxis=dict_xaxis,
            yaxis=dict_yaxis,
            plot_bgcolor=self.dict_lay.plot_bgcolor,
        )
        return layout


    def get_histLayout(self,
                       dict_yaxis,
                       dict_xaxis,
                       title_text,):
        dict_xaxis["ticklabelstep"] = 1
        title_text = title_text
        layout = Layout(
            title_text=title_text,
            legend_title_text=self.dict_lay.legend_title_text,
            legend_title_font_size=self.dict_lay.axes_title_font_size,
            legend_font_size=self.dict_lay.legend_font_size,
            legend_valign=self.dict_lay.legend_valign,
            legend_x = self.dict_lay.legend_x,
            legend_y = self.dict_lay.legend_y,
            legend_xanchor = self.dict_lay.legend_xanchor,
            legend_yanchor = self.dict_lay.legend_yanchor,
            margin=self.dict_lay.margin,
            font=self.dict_lay.make_dict_font(),
            title=self.dict_lay.make_dict_title(),
            xaxis=dict_xaxis,
            yaxis=dict_yaxis,
            plot_bgcolor=self.dict_lay.plot_bgcolor,
        )
        return layout


    def get_boxLayout(self,
                      dict_yaxis,
                      dict_xaxis,
                      title_text,):
        dict_xaxis["ticklabelstep"] = 1
        title_text = title_text
        layout = Layout(
            title_text=title_text,
            legend_title_text=self.dict_lay.legend_title_text,
            legend_title_font_size=self.dict_lay.axes_title_font_size,
            legend_font_size=self.dict_lay.legend_font_size,
            legend_valign=self.dict_lay.legend_valign,
            legend_x = self.dict_lay.legend_x,
            legend_y = self.dict_lay.legend_y,
            legend_xanchor = self.dict_lay.legend_xanchor,
            legend_yanchor = self.dict_lay.legend_yanchor,
            margin=self.dict_lay.margin,
            font=self.dict_lay.make_dict_font(),
            title=self.dict_lay.make_dict_title(),
            xaxis=dict_xaxis,
            yaxis=dict_yaxis,
            plot_bgcolor=self.dict_lay.plot_bgcolor,
        )
        return layout


    def table(
            self,
            data: pd.DataFrame,
            columns: list[str],
            ndec: int = 2,
        ) -> Table:
        """
        """
        layout = self.dict_lay.make_dict_table()
        layout["header"]["values"]=columns
        layout["cells"]["values"]=[data[x].tolist() for x in data.columns]

        table = Table(**layout)

        return table


    def sequesteredGroups_plot(
        self,
        data: pd.DataFrame,
        studyName: str,
        groupings: list[list[str]],
        subgroups: bool = False,
        c_name: str = "OVERALL",
        ylims: Union[list[float], None] = None,
        incprev: bool = True,
        metric_type: Union[str, None] = None,
        legend: bool = True,
    ) -> tuple[list[Scatter], list[Union[Layout, dict[str, Union[str, int, float, bool]]]],]:
        """
        Generate a group of plots
        Generate a list of plots, where the levels in each plot are defined by a list of character vectors of levels in c_name.

        Parameters:
            data (pd.DataFrame): Data to be visualised
            studyName (str): One of c("Incidence", "Prevalence", "Numerator", "Denominator", "PersonYears")
            subgroups (bool): is data subgrouped?
            groupings (list): Each element a list of strings defining levels to plot
            c_name (str): column name of the subgroup levels
            ylims (list): vector of c(y_min, y_max), designating y-axis range.
                        If null, then will automatically set c(0, max(data[`studyName`])+(max(data[`studyName`])/5) )
            incprev (bool): Are incidence/prevelance values being plotted on the y-axis?
            metric_type (str): Name of column to plot on x-axis (if not studyName)
            legend (bool): Should traces include legends? (default True)

        Returns:
            plot (list): list of plotly.trace
            layout (Layout):

        Usage:
            sequesteredGroups_plot(data, studyName, groupings)
            sequesteredGroups_plot(data, studyName, groupings, TRUE, c_name=c_name, incPrev=FALSE, metric_type=metric_type)

        Notes:
            If is.null(ylims)==TRUE then function will automatically determine the max y-axis value across the groups; c(0, ymax)

            If subgroups==TRUE, must define c_name.
            Or if subgroups==False, and c_name!="OVERALL", then must define the c_name of input data

            If plotting numerator/denominator, then set incprev==FALSE.
            If incprev==TRUE, studyName must be one of c("Incidence", "Prevalence")
        """
        def ignoreRow(x, labs=self.ignore):
            skip = False
            if x in labs:
                skip = True
            else:
                for lab in labs:
                    if str(x).find(f"'{lab}'") != -1:
                        skip=True
                        break
            return skip


        if metric_type is None:
            metric_type = studyName

        # ensure consistent y-axis for grouped graphs if ylims==None
        if ylims is None:
            y_max = 0
            for i in range(0, len(groupings)):
                df_temp = self.rm_NumNull(data, metric_type=metric_type).copy()

                rem = df_temp[c_name].map(ignoreRow)
                rem = rem[rem].index
                df_temp.drop(labels=rem, axis=0, inplace=True)

                y_temp = self.ylims_def(df_temp[studyName])[1]
                if y_max < y_temp:
                    y_max = y_temp

            ylims = [0, y_max]

        plots = []
        for i in range(0, len(groupings)):
            levels = groupings[i]

            # remove na values in Numerator col
            df_temp = self.rm_NumNull(data, metric_type=metric_type)

            #if numeric c_name col, convert to int if possible, then to str
            #to int removes decimal if being treated as float
            #not pretty or efficient, but works
            try:
                df_temp[c_name] = df_temp[c_name].map(lambda x: int(x))
                df_temp[c_name] = df_temp[c_name].map(lambda x: str(x))
            except:
                df_temp[c_name] = df_temp[c_name].map(lambda x: str(x))


            # remove values not in current levels list
            def subLev(x, levels=None):
                if x in levels:
                    return x
                else:
                    return pd.NA
            subsetLevels = npvectorize(subLev, excluded=['levels'])
            numerator = df_temp[c_name].to_numpy()
            numerator = subsetLevels(numerator, levels=levels)
            #numerator = [x if x in levels else pd.NA for x in df_temp[c_name]]
            df_temp_map = pd.notna(numerator)
            df_temp = df_temp.loc[df_temp_map]
            p, layout = self.plot_incprev(
                df_temp, studyName, subgroups, c_name, ylims, incprev, legend=legend
            )
            if p is None:
                return None, None

            plots.append(p)

        return (plots, layout)


    def generateGroups(self, dat_levels: pd.Series, max_groups: int) -> list[list[str]]:
        """
        Group categorical data into a defined number of groups.
        Where there are many levels to graph, can return levels in groups,
        to distribute data across multiple graphs for clarity.
        Useful to automate input into sequesteredGroups_plot().

        Parameters:
            dat_levels (pd.Series): Subgroup column of incprev data.
            max_groups (int): Max number of levels per graph.

        Returns:
            list: Each element a list of strings consisting of a group of levels.

        Usage:
            generateGroups(dat, max_groups)
        """
        diffVals = dat_levels.unique()
        makeType = None
        def ListType(inp, ignore):
            try:
                out = [int(x) for x in inp if x not in ignore]
                out = sorted(set(out))
                return [str(x) for x in out]
            except:
                pass
            try:
                out = [float(x) for x in inp if x not in ignore]
                out = sorted(set(out))
                return [str(x) for x in out]
            except:
                pass
            out = [x for x in inp if x not in ignore]
            return sorted(set(out))

        diffVals = ListType(diffVals, self.ignore)

        diffVals_n = len(diffVals)
        group_n = ceil(diffVals_n / max_groups)

        group_levels = []
        if group_n > 1:
            for i in range(0, group_n):
                group_levels.append(
                    diffVals[(i * max_groups) - max_groups + 1 : i * max_groups]
                )
        else:
            group_levels.append(diffVals[0:diffVals_n])

        return group_levels

