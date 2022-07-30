#!/usr/bin/python
import os, sys, math
import numpy as np
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
plt.style.use(['grayscale'])

from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import rcParams
import matplotlib.cbook as cbook
import matplotlib.ticker as ticker
import matplotlib.lines as lines
from matplotlib.ticker import FuncFormatter
import matplotlib.patches as mpatches

import logging
logging.basicConfig()

from scipy.stats import gmean

from bitfusion.src.utils.utils import lookup_pandas_dataframe

class BarChart(object):
    def __init__(self):
        self.log = logging.getLogger('BarChart')
        # self.log.setLevel(logging.DEBUG)

        #
        # Configurable Parameters
        #
        self.LABEL_BAR = True # Put data label
        self.YAXIS_MIN = 0.0  # minimum value of Y-axis
        self.YAXIS_MAX = 6.0  # maximum value of Y-axis
        self.COLOR_MAX = np.array([255/255., 255/255., 255/255.])  # lightest color of bars
        self.COLOR_MIN = np.array([13/255., 31/255., 60/255.])  # darkest color of bars
        self.FIG_WIDTH = 11.5  # figure's width
        self.FIG_HEIGHT = 3.5  # figure's height
        self.BAR_WIDTH = 0.13
        self.BAR_GAP = 0.0
        self.BAR_LEFT_MARGIN = 1.
        self.GLOBAL_FONTSIZE = 12
        self.LEGEND_FONTSIZE = 12
        self.AXIS_TITLE_FONTSIZE = 18
        self.XAXIS_FONTSIZE = 14
        self.TOP_LABEL_FONTSIZE = 10
        self.YAXIS_PAD = 10
        self.ISRATES = False
        self.ISTIMES = False
        self.TOP_LABEL_SHIFT = 0.02
        self.LEGEND_LOCATION = 3
        self.TOP_ROTATE = False
        self.TOP_MARGIN = 0.85
        self.BOTTOM_MARGIN = 0.2
        self.LEFT_MARGIN = 0.14
        self.RIGHT_MARGIN = 0.95
        self.BENCH_NEWLINE = True
        self.LOG_SCALE = False
        self.LEGEND_NCOLUMN = 1
        self.MV_LEGEND_OUTSIDE_X = 0.0
        self.MV_LEGEND_OUTSIDE_Y = 1.0
        self.XAXIS_LABEL_ROTATE = 0
        #
        #
        self.COLOR = []

        self.xaxis = ''
        self.yaxis = ''
        self.mode = None

        rcParams['font.family'] = 'sans-serif'
        rcParams['font.sans-serif'] = ['Helvetica']
        rcParams['font.size'] = self.GLOBAL_FONTSIZE
        rcParams['text.usetex'] = True

        pgf_with_rc_fonts = {"pgf.texsystem": "pdflatex"}
        matplotlib.rcParams.update(pgf_with_rc_fonts)


    def to_percent(self, y, position):
        # Ignore the passed in position. This has the effect of scaling the default
        # tick locations.
        s = str(100 * y)
        s = s.split('.')[0]

        # The percent symbol needs escaping in latex
        if rcParams['text.usetex'] == True:
            return s + r'$\%$'
        else:
            return s + '%'

    def to_percent2(self, y, position):
        # Ignore the passed in position. This has the effect of scaling the default
        # tick locations.
        s = str(100 * y)

        # The percent symbol needs escaping in latex
        if rcParams['text.usetex'] == True:
            return s + r'$\%$'
        else:
            return s + '%'

    def to_text(self, y, position):
        if y >= 1.0 or y == 0:
            s = str(int(y))
        else:
            s = str(float(y))
        return r'\sf{%s}' % s

    def to_times2(self, y, position):
        s = str(float(y))
        return r'\sf{%s$\times$}' % s

    def autolabel(self, ax, rects):
        if self.ISRATES == True:
            formatter = self.to_percent
        elif self.ISTIMES == True:
            formatter = self.to_times2
        else:
            formatter = self.to_text
        for i in range(0, len(rects)):
            rect = rects[i]
            height = rect.get_height()

            if (self.LOG_SCALE == 1):
                overflow = math.log(height, 10) > math.log(self.YAXIS_MAX, 10) * 0.8
            else:
                overflow = height > (self.YAXIS_MAX * 0.95)

            if (self.LOG_SCALE == 1):
                if (overflow):
                    offset = self.YAXIS_MAX * 10**(0.1) - self.YAXIS_MAX
                else:
                    offset = height * 10**(0.1) - height
            else:
                offset = self.YAXIS_MAX * 0.03

            if (self.TOP_ROTATE == True):
                rotation = 90
            else:
                rotation = 0


            if height > 10:
                datalabel = r'{0}'.format(int(height))
            else:
                datalabel = r'{0:1.1f}'.format(round(height, 1))
            datalabel = formatter(datalabel, None)

            if overflow:
                ax.text(rect.get_x() + rect.get_width() / 2. + self.TOP_LABEL_SHIFT, self.YAXIS_MAX + offset, datalabel, ha='center',
                        va='bottom', fontsize=self.TOP_LABEL_FONTSIZE, rotation=rotation)
            elif self.LABEL_BAR:
                ax.text(rect.get_x() + rect.get_width() / 2. + self.TOP_LABEL_SHIFT, height + offset, datalabel, ha='center',
                        va='bottom', fontsize=self.TOP_LABEL_FONTSIZE, rotation=rotation)

    def readData(self, csvFilename):
        os.system('mac2unix ' + csvFilename + ' > /dev/null 2> /dev/null')
        f = open(csvFilename, 'r')
        lines = f.readlines()

        self.mode = lines[0].strip('\n').split(',')[0]

        xaxisLine = lines[1].strip('\n')
        self.xaxis = xaxisLine.split(',')[1]

        yaxisLine = lines[2].strip('\n')
        self.yaxis = yaxisLine.split(',')[1]

        legends = lines[3].strip('\n').split(',')[1:-1]

        data = {}
        for l in legends:
            data[l] = []

        bench = []

        for i in range(4, len(lines)):
            tokens = lines[i].strip('\n').split(',')
            curr_bench = tokens[0]
            bench.append(curr_bench)

            assert len(tokens) >= len(legends) + 1

            for j in range(len(legends)):
                l = legends[j]
                data[l].append(tokens[j+1])

        for l in legends:
            self.log.info('Legend: {}, Data: {}'.format(l, data[l]))

        return legends, bench, data

    def plot(self, outputFileName, dataframe, x_plot, y_plot, bar_plot, x_plot_list=None, y_plot_list=['1','2','3','4','5','6','7','8','9'], bar_list=None, lookup_dict=None, add_geomean=False, baseline=None):

        stacked = isinstance(y_plot, list) and len(y_plot) > 1
        if not stacked:
            y_plot = [y_plot]

        self.log.info('Plotting BarChart to file: {}'.format(outputFileName))
        self.log.info('bar_plot: {}'.format(bar_plot))
        self.log.info('x-axis: {}'.format(x_plot))
        self.log.info('y-axis: {}'.format(y_plot))

        assert bar_list is not None, 'Expected list of bar_plot'
        num_bar_plot = len(bar_list)

        if x_plot_list is None:
            x_plot_list = list(dataframe[x_plot].unique())

        if add_geomean:
            # Add an additional 'Gmean'
            x_plot_list.append('Gmean')

        if stacked:
            num_colors = len(y_plot)
        else:
            num_colors = num_bar_plot

        if self.COLOR is None or len(self.COLOR) != num_colors:
            self.COLOR = []
            diff = (self.COLOR_MAX - self.COLOR_MIN) / (num_colors - 1)
            for i in range(num_colors):
                self.COLOR.append(tuple(self.COLOR_MAX - i * diff))
            self.log.info('Using colors: {}'.format(self.COLOR))

        with PdfPages(outputFileName) as pdf:

            fig, ax = plt.subplots(figsize=(self.FIG_WIDTH, self.FIG_HEIGHT))

            num_benchmarks = len(x_plot_list)
            N = num_benchmarks
            ind = np.arange(N)

            legendLocation = []
            rectsList = []

            xbegin = np.arange(N)

            self.BAR_WIDTH = 1. / (num_bar_plot * (1+self.BAR_GAP) + self.BAR_LEFT_MARGIN)

            if lookup_dict is None:
                default_lookup = {}
            else:
                default_lookup = lookup_dict
            default_lookup[bar_plot] = bar_list[0]
            left_margin = float(self.BAR_LEFT_MARGIN * self.BAR_WIDTH)

            edgecolors = ['black'] * num_benchmarks

            for i in range(0, num_bar_plot):
                bottom = np.zeros(num_benchmarks)
                for _y in range(len(y_plot)):
                    _y_plot = y_plot[_y]

                    if lookup_dict is None:
                        data_lookup = {}
                        base_lookup = {}
                    else:
                        data_lookup = lookup_dict
                        base_lookup = {}

                    dataLine = []
                    data_lookup[bar_plot] = bar_list[i]
                    for j in range(num_benchmarks - add_geomean):
                        data_lookup[x_plot] = x_plot_list[j]
                        data_row = lookup_pandas_dataframe(dataframe, data_lookup)
                        
                        dataLine.append(float(data_row[_y_plot]))

                    if baseline is None:
                        baseLine = [1]*len(dataLine)
                    else:
                        base_lookup[bar_plot] = baseline
                        baseLine = []
                        for j in range(num_benchmarks - add_geomean):
                            base_lookup[x_plot] = x_plot_list[j]
                            base_row = lookup_pandas_dataframe(dataframe, base_lookup)
                            baseLine.append(float(base_row[_y_plot]))
                        assert len(baseLine) == len(dataLine)
                    dataLine = [x/y for x,y in zip(dataLine, baseLine)]

                    if add_geomean:
                        dataLine.append(gmean(dataLine))

                    if stacked:
                        color = self.COLOR[_y]
                    else:
                        color = self.COLOR[i]

                    if self.LOG_SCALE == False:
                        rects = ax.bar(ind + left_margin, dataLine, self.BAR_WIDTH, color=color, bottom=bottom, edgecolor=edgecolors)
                    else:
                        rects = ax.bar(ind + left_margin, dataLine, self.BAR_WIDTH, log=1, color=color, bottom=bottom, edgecolor=edgecolors)

                    legendLocation.append(rects[0])
                    rectsList.append(rects)

                    if stacked:
                        bottom += dataLine
                left_margin += self.BAR_WIDTH + self.BAR_GAP

            if self.LOG_SCALE == True:
                ax.set_yscale('log', nonposy='clip')
            ax.set_xlim(0, len(x_plot_list))
            ax.set_ylim(self.YAXIS_MIN, self.YAXIS_MAX)

            if stacked:
                num_minor_ticks = len(x_plot_list) * len(bar_plot)
                num_major_ticks = len(x_plot_list)
                minor_ticks = []
                minor_ind = []
                for i in range(len(x_plot_list)):
                    offset = self.BAR_LEFT_MARGIN * self.BAR_WIDTH + i
                    for j in range(len(bar_list)):
                        minor_ticks.append(bar_list[j])
                        minor_ind.append(offset)
                        offset += self.BAR_WIDTH + self.BAR_GAP
                ax.set_xticks(minor_ind, minor=True)
                ax.set_xticklabels(minor_ticks, rotation=self.XAXIS_LABEL_ROTATE, minor=True)
                ax.set_xticks(ind+0.5, minor=False)
                ax.set_xticklabels(x_plot_list, rotation=self.XAXIS_LABEL_ROTATE, minor=False)
                ax.tick_params(axis='x', which='major', direction='out', length=0)

                # vertical alignment of xtick labels
                va = [ -0.10 for i in range(len(x_plot_list))]
                for t, y in zip( ax.get_xticklabels( ), va ):
                        t.set_y( y )
            else:
                ax.set_xticks(ind+0.5, minor=False)
                ax.set_xticklabels(x_plot_list, rotation=self.XAXIS_LABEL_ROTATE)

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            fig.subplots_adjust(top=self.TOP_MARGIN)
            if self.BENCH_NEWLINE == True:
                fig.subplots_adjust(bottom=self.BOTTOM_MARGIN)
            else:
                fig.subplots_adjust(bottom=self.BOTTOM_MARGIN * 0.5)
            fig.subplots_adjust(left=self.LEFT_MARGIN)
            fig.subplots_adjust(right=self.RIGHT_MARGIN)

            if self.ISRATES == True:
                formatter = FuncFormatter(self.to_percent)
                ax.yaxis.set_major_formatter(formatter)
            elif self.ISTIMES == True:
                formatter = FuncFormatter(self.to_times2)
                ax.yaxis.set_major_formatter(formatter)
            else:
                formatter = FuncFormatter(self.to_text)
                ax.yaxis.set_major_formatter(formatter)

            if self.yaxis != '':
                ax.set_ylabel(self.yaxis, fontsize=self.AXIS_TITLE_FONTSIZE, labelpad=5.0)
            if self.xaxis != '':
                ax.set_xlabel(self.xaxis, fontsize=self.AXIS_TITLE_FONTSIZE, labelpad=20.0)

            # for i in range(0, len(x_plot_list)):
            #     loc = (1.0 / len(x_plot_list)) * (i + 0.5)
            #     space4line = 0.075
            #     if self.BENCH_NEWLINE == True:
            #         space4text = space4line * 2
            #     else:
            #         space4text = space4line
            #     ax.text(loc, -space4text, x_plot_list[i], horizontalalignment='center', fontsize=self.XAXIS_FONTSIZE,
            #             transform=ax.transAxes, rotation=self.XAXIS_LABEL_ROTATE)

            # is stacked, show the stacks as legends; otherwise show the bars as legend
            if stacked:
                legend_list = y_plot
            else:
                legend_list = bar_list
            if self.MV_LEGEND_OUTSIDE_X != 0.0 or self.MV_LEGEND_OUTSIDE_Y != 1.0:
                ax.legend(tuple(legendLocation), legend_list, loc=self.LEGEND_LOCATION, fontsize=self.LEGEND_FONTSIZE, borderpad=0.2,
                          borderaxespad=0.3, labelspacing=0.05, ncol=self.LEGEND_NCOLUMN,
                          bbox_to_anchor=(self.MV_LEGEND_OUTSIDE_X, self.MV_LEGEND_OUTSIDE_Y))
            else:
                ax.legend(tuple(legendLocation), legend_list, loc=self.LEGEND_LOCATION, fontsize=self.LEGEND_FONTSIZE, borderpad=0.2,
                          borderaxespad=0.3, labelspacing=0.05, ncol=self.LEGEND_NCOLUMN)

            length = len(rectsList)
            for i in range(0, length):
                rects = rectsList[i]
                self.autolabel(ax, rects)

            ax.set_axisbelow(True)

            for tick in ax.get_yaxis().get_major_ticks():
                tick.set_pad(self.YAXIS_PAD)
                tick.label1 = tick._get_text1()

            if x_plot_list[len(x_plot_list) - 1] == '\sf{Gmean}' or x_plot_list[len(x_plot_list) - 1] == 'Gmean' or x_plot_list[len(x_plot_list) - 1] == 'Avg' or x_plot_list[len(x_plot_list) - 1] == 'Average':
                plt.axvline(x=len(x_plot_list) - 1, c="black", lw=1.2)

            # plt.axhline(y=1.0, c="black", lw=2)

            pdf.savefig(fig, bbox_inches='tight', pad_inches=0.02)
            # plt.show()

if __name__ == "__main__":
    bc = BarChart()
    csvFilename = sys.argv[1]
    legends, bench, data = bc.readData(csvFilename)
    bc.ISTIMES = True
    bc.LEGEND_LOCATION = 0
    bc.TOP_ROTATE = True
    bc.plot('tmp.pdf', legends, bench, data)
