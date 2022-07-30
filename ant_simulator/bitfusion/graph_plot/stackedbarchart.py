#!/usr/bin/python
import os,sys,math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import rcParams
import matplotlib.cbook as cbook
import matplotlib.ticker as ticker
import matplotlib.lines as lines
from matplotlib.ticker import FuncFormatter
import matplotlib.patches as mpatches
from matplotlib.ticker import AutoMinorLocator
from matplotlib.artist import Artist
from matplotlib.patches import Rectangle

#
# Configurable Parameters 
#
YAXIS_MIN = 0.0 		# minimum value of Y-axis
YAXIS_MAX = 6.0 		# maximum value of Y-axis 
COLOR_MAX = 0.95		# lightest color of bars
COLOR_MIN = 0.15		# darkest color of bars
FIG_WIDTH = 25.5		# figure's width
FIG_HEIGHT = 3.5		# figure's height
BAR_WIDTH = 0.2
BAR_LEFT_MARGIN = 0.5
GLOBAL_FONTSIZE = 12
LEGEND_FONTSIZE = 12
AXIS_TITLE_FONTSIZE = 15
XAXIS_FONTSIZE = 12
XAXIS_YTEXT_OFFSET = -0.1
TOP_LABEL_FONTSIZE = 10
DATA_LABEL_FONTSIZE = 10
YAXIS_PAD = 10
XAXIS_PAD = 10
ISRATES = False
SPACE_BW_NEURALORACLE = False
TOP_LABEL_SPACE = 0.03
ISXAXIS = True
ISYAXIS = True
LOG = 0
LEGEND_LOCATION = 8
ROTATEXAXIS = 0
NOLEGEND = False
XAXISPADDING = 0.4
ISTIMES = False
DATALABEL = False
TOPLABEL_FORMAT = '%.1f'
DATALABEL_FORMAT = '%d'
TOPLABEL_ROTATE = 0
DATALABEL_ROTATE = 0
TOPLABEL_BOLD = 'normal'
DATALABEL_BOLD = 'normal'
LEGEND_XSPACE=0.5
LEGEND_YSPACE=1.25
LEGEND_NCOL=4
XAXIS_TEXTOFFSET=0.5
BAR_LABEL1 = ''
BAR_LABEL2 = ''
#
#

# Legend Location
# 'best'         : 0, (only implemented for axis legends)
# 'upper right'  : 1,
# 'upper left'   : 2,
# 'lower left'   : 3,
# 'lower right'  : 4,
# 'right'        : 5,
# 'center left'  : 6,
# 'center right' : 7,
# 'lower center' : 8,
# 'upper center' : 9,
# 'center'       : 10,



COLOR = []

def to_percent(y, position):
    # Ignore the passed in position. This has the effect of scaling the default
    # tick locations.
    s = str(100 * y)
    s = s.split('.')[0]

    # The percent symbol needs escaping in latex
    if rcParams['text.usetex'] == True:
        return '\sf{%s%s}' % (s, '\%')
    else:
        return s + '%'
def to_fixed(y, position):
    # Ignore the passed in position. This has the effect of scaling the default
    # tick locations
    s = str(y)
    return s

def change_font(y, position):
	x = int(y)
	if rcParams['text.usetex'] == True:
		return '\sf{%s}' % x

#def to_times(y, position):
#    # The percent symbol needs escaping in latex
#    if rcParams['text.usetex'] == True:
#        return s + r'$\X$'
#    else:
#        return s + 'X'

def plotBarGraph(outputFileName, legends, bench, data, std):

	with PdfPages(outputFileName) as pdf:

		N = len(bench)
		ind = np.arange(N) 
		y_offset = np.array([0.0] * N)

		fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
		plt.grid(b=True, which='both', color='0.65',linestyle='-')

		width = BAR_WIDTH

		legendLocation = []
		rectsList = []

		firstBarLoc = []
		secondBarLoc = []

		# Amir: I know it is Dumb but I want to do it!
		# Speedup
		dataLine = data[1]
		rects = ax.bar(ind + (BAR_LEFT_MARGIN) * width , dataLine, width, color=COLOR[2], log=LOG)
		for r in rects:
			firstBarLoc.append(r.get_x())
		legendLocation.append(rects[0])
		rectsList.append(rects)

		y_offset = y_offset + data[1]
		dataLine = data[2]
		rects = ax.bar(ind + (BAR_LEFT_MARGIN) * width, dataLine, width, color=COLOR[1], bottom=y_offset, log=LOG)
		legendLocation.append(rects[0])
		rectsList.append(rects)

		y_offset = y_offset + data[2]
		dataLine = data[3]
		rects = ax.bar(ind + (BAR_LEFT_MARGIN) * width, dataLine, width, color=COLOR[0], bottom=y_offset, log=LOG)
		legendLocation.append(rects[0])
		rectsList.append(rects)

		# y_offset = y_offset + data[3]
		# dataLine = data[4]
		# rects = ax.bar(ind + (BAR_LEFT_MARGIN) * width, dataLine, width, color=COLOR[3], bottom=y_offset, log=LOG)
		# legendLocation.append(rects[0])
		# rectsList.append(rects)

		# y_offset = y_offset + data[4]
		# dataLine = data[5]
		# rects = ax.bar(ind + (BAR_LEFT_MARGIN) * width, dataLine, width, color=COLOR[4], bottom=y_offset, log=LOG)
		# legendLocation.append(rects[0])
		# rectsList.append(rects)

		# y_offset = y_offset + data[5]
		# dataLine = data[6]
		# rects = ax.bar(ind + (BAR_LEFT_MARGIN) * width, dataLine, width, color=COLOR[5], bottom=y_offset, log=LOG)
		# legendLocation.append(rects[0])
		# rectsList.append(rects)

		# y_offset = y_offset + data[6]
		# dataLine = data[7]
		# rects = ax.bar(ind + (BAR_LEFT_MARGIN) * width, dataLine, width, color=COLOR[6], bottom=y_offset, log=LOG)
		# legendLocation.append(rects[0])
		# rectsList.append(rects)

		# Energy
		y_offset = np.array([0.0] * N)
		dataLine = data[4]
		rects = ax.bar(ind + (BAR_LEFT_MARGIN + 1.2) * width , dataLine, width, color=COLOR[2], log=LOG)
		legendLocation.append(rects[0])
		for r in rects:
			secondBarLoc.append(r.get_x())
		rectsList.append(rects)

		y_offset = y_offset + data[4]
		dataLine = data[5]
		rects = ax.bar(ind + (BAR_LEFT_MARGIN + 1.2) * width, dataLine, width, color=COLOR[1], bottom=y_offset, log=LOG)
		legendLocation.append(rects[0])
		rectsList.append(rects)

		y_offset = y_offset + data[5]
		dataLine = data[6]
		rects = ax.bar(ind + (BAR_LEFT_MARGIN + 1.2) * width, dataLine, width, color=COLOR[0], bottom=y_offset, log=LOG)
		legendLocation.append(rects[0])
		rectsList.append(rects)

		# y_offset = np.array([0.0] * N)
		# dataLine = data[8]
		# rects = ax.bar(ind + (BAR_LEFT_MARGIN + 1.2) * width , dataLine, width, color=COLOR[0], log=LOG)
		# legendLocation.append(rects[0])
		# for r in rects:
		# 	secondBarLoc.append(r.get_x())
		# rectsList.append(rects)

		# y_offset = y_offset + data[8]
		# dataLine = data[9]
		# rects = ax.bar(ind + (BAR_LEFT_MARGIN + 1.2) * width, dataLine, width, color=COLOR[1], bottom=y_offset, log=LOG)
		# legendLocation.append(rects[0])
		# rectsList.append(rects)

		# y_offset = y_offset + data[9]
		# dataLine = data[10]
		# rects = ax.bar(ind + (BAR_LEFT_MARGIN + 1.2) * width, dataLine, width, color=COLOR[2], bottom=y_offset, log=LOG)
		# legendLocation.append(rects[0])
		# rectsList.append(rects)

		# y_offset = y_offset + data[10]
		# dataLine = data[11]
		# rects = ax.bar(ind + (BAR_LEFT_MARGIN + 1.2) * width, dataLine, width, color=COLOR[3], bottom=y_offset, log=LOG)
		# legendLocation.append(rects[0])
		# rectsList.append(rects)

		# y_offset = y_offset + data[11]
		# dataLine = data[12]
		# rects = ax.bar(ind + (BAR_LEFT_MARGIN + 1.2) * width, dataLine, width, color=COLOR[4], bottom=y_offset, log=LOG)
		# legendLocation.append(rects[0])
		# rectsList.append(rects)

		# y_offset = y_offset + data[12]
		# dataLine = data[13]
		# rects = ax.bar(ind + (BAR_LEFT_MARGIN + 1.2) * width, dataLine, width, color=COLOR[5], bottom=y_offset, log=LOG)
		# legendLocation.append(rects[0])
		# rectsList.append(rects)

		# y_offset = y_offset + data[13]
		# dataLine = data[14]
		# rects = ax.bar(ind + (BAR_LEFT_MARGIN + 1.2) * width, dataLine, width, color=COLOR[6], bottom=y_offset, log=LOG)
		# legendLocation.append(rects[0])
		# rectsList.append(rects)


		ax.set_ylim(YAXIS_MIN, YAXIS_MAX)
		ax.set_xticks(ind + 1, minor=False)
		ax.set_xticklabels([])
		ax.set_xlim([0, len(bench)])

		minorLocator = AutoMinorLocator(2)
		ax.xaxis.set_minor_locator(minorLocator)
		pos1 = ax.get_position()	
		minorLocs = ax.xaxis.get_minorticklocs()
		plt.minorticks_off()

		if ISRATES == True:
			fig.subplots_adjust(left=0.14)
			formatter = FuncFormatter(to_percent)
			ax.yaxis.set_major_formatter(formatter)
		elif ISFIXED == True:
			fig.subplots_adjust(left=0.14)
			formatter = FuncFormatter(to_fixed)
			ax.yaxis.set_major_formatter(formatter)
		else:
			formatter = FuncFormatter(change_font)
			ax.yaxis.set_major_formatter(formatter)


		if yaxis != '' and ISYAXIS == True:
			ax.set_ylabel(yaxis, fontsize=AXIS_TITLE_FONTSIZE, labelpad=5.0)
		if xaxis != '' and ISXAXIS == True:
			ax.set_xlabel(xaxis, fontsize=AXIS_TITLE_FONTSIZE, labelpad=10.0)
		
		for i in range(0, len(firstBarLoc)):
			firstBarLoc[i] += 0.08
		for i in range(0, len(secondBarLoc)):
			secondBarLoc[i] += 0.08
		for i in range(0, len(bench)):
			if(ROTATEXAXIS == 0):
				loc = (1.0 / float(len(bench))) * (i + XAXIS_TEXTOFFSET)
				ax.text(loc, XAXIS_YTEXT_OFFSET, bench[i], horizontalalignment='center', fontsize=XAXIS_FONTSIZE, transform = ax.transAxes, rotation = ROTATEXAXIS)
			else:
				loc = minorLocs[i]
				ax.text(loc+0.1, XAXIS_YTEXT_OFFSET, bench[i], horizontalalignment='right', fontsize=XAXIS_FONTSIZE, transform = ax.transAxes, rotation = ROTATEXAXIS)
			ax.text(firstBarLoc[i], +1.27, BAR_LABEL1, horizontalalignment='left', fontsize=10, rotation = 90)
			ax.text(secondBarLoc[i], +1.31, BAR_LABEL2, horizontalalignment='left', fontsize=10, rotation = 90)
		if NOLEGEND == False:
			new_legends = legends[0:3]
			# ax.legend(new_legends, loc=LEGEND_LOCATION, bbox_to_anchor=(LEGEND_XSPACE, LEGEND_YSPACE), ncol=LEGEND_NCOL, fontsize=LEGEND_FONTSIZE, borderpad=0.25, borderaxespad=0.5, labelspacing=0.15)
			ax.legend(new_legends, loc=LEGEND_LOCATION, bbox_to_anchor=(LEGEND_XSPACE, LEGEND_YSPACE), ncol=LEGEND_NCOL, fontsize=LEGEND_FONTSIZE, borderpad=0.25, borderaxespad=0.5, labelspacing=0.15)
		length = len(rectsList)
		for i in range(0, length):
			tlMargin = TOP_LABEL_SPACE
			rects = rectsList[i]
			if float(length)/2.0 - int(float(length)/2.0) == 0: # even
				if i < int(float(length)/2.0): # minus
					tlMargin = -(tlMargin * (int(float(length)/2.0) - 1 - i + 1))
				else: # plus
					tlMargin = tlMargin * (i - (int(float(length)/2.0) - 1))
			else: # odd
				if i < int(float(length)/2.0): # minus
					tlMargin = -(tlMargin * (int(float(length)/2.0) - i + 1))
				elif i > int(float(length)/2.0): # plus
					tlMargin = tlMargin * (i - int(float(length)/2.0) + 1)
			autolabel(ax, rects, tlMargin)
			if DATALABEL == True:
				datalabel(ax, rects, tlMargin)

		ax.set_axisbelow(True)

		for tick in ax.get_yaxis().get_major_ticks():
		    tick.set_pad(YAXIS_PAD)
		    tick.label1 = tick._get_text1()
		for tick in ax.get_xaxis().get_major_ticks():
		    tick.set_pad(XAXIS_PAD)
		    tick.label1 = tick._get_text1()

		if bench[len(bench)-1] == 'geomean' or bench[len(bench)-1] == 'Gmean' or bench[len(bench)-1] == 'Geomean' or bench[len(bench)-1] == 'avg' or bench[len(bench)-1] == 'Avg' or bench[len(bench)-1] == 'average': 
			plt.axvline(x=len(bench)-1, c="black", lw=2)

		fig.subplots_adjust(left=0.3, top=0.9, bottom=XAXISPADDING)

		#pdf.savefig(fig)
		pdf.savefig(fig,bbox_inches = 'tight',pad_inches = 0.02)

def autolabel(ax, rects, tlMargin):
	for i in range(0, len(rects)):
		rect = rects[i]
		height = rect.get_height()
		if height > YAXIS_MAX:
			ax.text(rect.get_x() + rect.get_width()/2. + tlMargin, YAXIS_MAX * 1.01, TOPLABEL_FORMAT%float(height), ha='center', va='bottom', fontsize=TOP_LABEL_FONTSIZE, rotation=TOPLABEL_ROTATE, fontweight=TOPLABEL_BOLD)

def datalabel(ax, rects, tlMargin):
	for i in range(0, len(rects)):
		rect = rects[i]
		height = rect.get_height()
		# change to move slightly right for num-annotations
		minor = 0.0
		if height == 11:
			minor = 0.07
		if height == 40:
			minor = 0.11
		if height == 14:
			minor = 0.07
		if height == 109:
			minor = 0.16
		if height == 39:
			minor = 0.09
		ax.text(rect.get_x() + rect.get_width()/2. + tlMargin - 0.02 + minor, float(height) + YAXIS_MAX * 0.01, DATALABEL_FORMAT%int(height), ha='center', va='bottom', fontsize=DATA_LABEL_FONTSIZE, rotation=DATALABEL_ROTATE, fontweight=DATALABEL_BOLD)

def readData(csvFilename):

	os.system('mac2unix ' + csvFilename + ' > /dev/null 2> /dev/null')

	f = open(csvFilename, 'r')

	lines = f.readlines()

	global mode
	mode = lines[0].strip('\n').split(',')[0]

	xaxisLine = lines[1].strip('\n')
	global xaxis
	xaxis = xaxisLine.split(',')[1]
	
	yaxisLine = lines[2].strip('\n')
	global yaxis
	yaxis = yaxisLine.split(',')[1]

	legendLine = lines[3].strip('\n')
	legends = legendLine.split(',')[1:len(legendLine)]
	legends = tuple(legends)
	data = []
	std = []
	std.append([0.0])
	data.append([0.0])
	bench = []
	for i in range(4, len(lines)):
		tokens = lines[i].strip('\n').split(',')
		if BENCH_NEWLINE == True:
			if ' ' in tokens[0]:
				new = ''
				for ch in tokens[0]:
					if ch != ' ':
						new += ch
					else:
						new += '\n'
				bench.append(new)
			else:
				bench.append(tokens[0])
		else:
			bench.append(tokens[0])
		# print tokens
		if mode == 'baseline':
			for j in range(1, len(tokens)):
				if i == 4:
					data.append([float(tokens[j])])
				else:
					#print(( tokens[j]))
					data[j].append(float(tokens[j]))
		elif mode == 'confidence' or mode == 'baysian-confidence':
			for j in range(1, len(tokens)):
				if j%2 != 1:
					continue
				if i == 4:
					data.append([float(tokens[j])])
					std.append([float(tokens[j+1])])
				else:
					data[j/2+1].append(float(tokens[j]))
					std[j/2+1].append(float(tokens[j+1]))
		else:
			print( 'Unknown mode: ' + csvFilename)
			sys.exit(0)

	for i in range(0, len(data)):
		data[i] = tuple(data[i])

	lenLegends = 0
	for legend in legends:
		if legend == '':
			continue
		lenLegends = lenLegends + 1
	if lenLegends != 1:
		diff = (COLOR_MAX - COLOR_MIN) / (lenLegends/2 - 1)
	else:
		diff = 0

	for i in range(0, lenLegends/2):
		COLOR.append(str(COLOR_MAX - i * diff))

	
	rcParams['font.family'] = 'sans-serif'
	rcParams['font.sans-serif'] = ['Lucida Grande']
	rcParams['font.size'] = GLOBAL_FONTSIZE
	rcParams['text.usetex']=True

	return legends, bench, data, std

def main():

	usage = "Usage: ./bargraph.py [.csv file path] [output file path]\n\
	-ymin [y-axis minimum value]\n\
	-ymax [y-axis maximum value]\n\
	-width [figure width]\n\
	-height [figure height]\n\
	-barwidth [bar width]\n\
	-colormin [minimum color value from 0.0(black) to 1.0(white)]\n\
	-colormax [maximum color value from 0.0(black) to 1.0(white)]\n\
	-toplabelfontsize [font size for data label on top ]\n\
	-legendfontsize [font size for legends]\n\
	-axistitlefontsize [font size for titles of x-axis and y-axis]\n\
	-globalfontsize [font size (overried if defined separately)]"

	if len(sys.argv) < 3:
		print( usage)
		sys.exit()

	csvFilename = sys.argv[1]
	outputFileName = sys.argv[2]

	if os.path.isfile(csvFilename) == False:
		print( "Error: File [" + csvFilename + "] doesn't exist")
		sys.exit()

	if csvFilename[-4:] != ".csv":
		print( "Error: File [" + csvFilename + "] is not a csv file")
		sys.exit()

	name = csvFilename.strip('\n').split('.')[0]

	global YAXIS_MIN, YAXIS_MAX, FIG_WIDTH, FIG_HEIGHT, BAR_WIDTH, COLOR_MIN, COLOR_MAX, GLOBAL_FONTSIZE, LEGEND_FONTSIZE, \
			AXIS_TITLE_FONTSIZE, TOP_LABEL_FONTSIZE, XAXIS_FONTSIZE, YAXIS_PAD, XAXIS_PAD, ISRATES, BAR_LEFT_MARGIN, SPACE_BW_NEURALORACLE, TOP_LABEL_SPACE, \
			ISXAXIS, ISYAXIS, LOG, LEGEND_LOCATION, ROTATEXAXIS, NOLEGEND, XAXISPADDING, DATALABEL, DATA_LABEL_FONTSIZE, TOPLABEL_FORMAT, DATALABEL_FORMAT, \
			TOPLABEL_ROTATE, DATALABEL_ROTATE, TOPLABEL_BOLD, DATALABEL_BOLD, LEGEND_XSPACE, LEGEND_YSPACE, LEGEND_NCOL, XAXIS_TEXTOFFSET, ISFIXED, BAR_LABEL1, BAR_LABEL2, XAXIS_YTEXT_OFFSET, \
			BENCH_NEWLINE

	if len(sys.argv) >= 3:
		for i in range(3, len(sys.argv)):
			if '-' in sys.argv[i]:
				option = sys.argv[i]
				if option == '-ymin':
					if len(sys.argv) <= (i+1):
						print( "Error: -ymin option not specified")
						sys.exit(0)
					YAXIS_MIN = float(sys.argv[i+1])
				elif option == '-ymax':
					if len(sys.argv) <= (i+1):
						print( "Error: -ymax option not specified")
						sys.exit(0)
					YAXIS_MAX = float(sys.argv[i+1])
				elif option == '-width':
					if len(sys.argv) <= (i+1):
						print( "Error: -width option not specified")
						sys.exit(0)
					FIG_WIDTH = float(sys.argv[i+1])
				elif option == '-height':
					if len(sys.argv) <= (i+1):
						print( "Error: -height option not specified")
						sys.exit(0)
					FIG_HEIGHT = float(sys.argv[i+1])
				elif option == '-barwidth':
					if len(sys.argv) <= (i+1):
						print( "Error: -barwidth option not specified")
						sys.exit(0)
					BAR_WIDTH = float(sys.argv[i+1])
				elif option == '-colormin':
					if len(sys.argv) <= (i+1):
						print( "Error: -colormin option not specified")
						sys.exit(0)
					COLOR_MIN = float(sys.argv[i+1])
				elif option == '-colormax':
					if len(sys.argv) <= (i+1):
						print( "Error: -colormax option not specified")
						sys.exit(0)
					COLOR_MAX = float(sys.argv[i+1])
				elif option == '-toplabelfontsize':
					if len(sys.argv) <= (i+1):
						print( "Error: -toplabelfontsize option not specified")
						sys.exit(0)
					TOP_LABEL_FONTSIZE = float(sys.argv[i+1])
				elif option == '-legend_xspace':
					if len(sys.argv) <= (i+1):
						print( "Error: -legend_xspace option not specified")
						sys.exit(0)
					LEGEND_XSPACE = float(sys.argv[i+1])
				elif option == '-legend_yspace':
					if len(sys.argv) <= (i+1):
						print( "Error: -legend_yspace option not specified")
						sys.exit(0)
					LEGEND_YSPACE = float(sys.argv[i+1])
				elif option == '-legend_ncol':
					if len(sys.argv) <= (i+1):
						print( "Error: -legend_ncol option not specified")
						sys.exit(0)
					LEGEND_NCOL = int(sys.argv[i+1])
				elif option == '-datalabelfontsize':
					if len(sys.argv) <= (i+1):
						print( "Error: -datalabelfontsize option not specified")
						sys.exit(0)
					DATA_LABEL_FONTSIZE = float(sys.argv[i+1])
				elif option == '-legendfontsize':
					if len(sys.argv) <= (i+1):
						print( "Error: -legendfontsize option not specified")
						sys.exit(0)
					LEGEND_FONTSIZE = float(sys.argv[i+1])
				elif option == '-axistitlefontsize':
					if len(sys.argv) <= (i+1):
						print( "Error: -axistitlefontsize option not specified")
						sys.exit(0)
					AXIS_TITLE_FONTSIZE = float(sys.argv[i+1])
				elif option == '-globalfontsize':
					if len(sys.argv) <= (i+1):
						print( "Error: -globalfontsize option not specified")
						sys.exit(0)
					GLOBAL_FONTSIZE = float(sys.argv[i+1])
				elif option == '-xaxis_ytextoffset':
					if len(sys.argv) <= (i+1):
						print( "Error: -xaxis_ytextoffset option not specified")
						sys.exit(0)
					XAXIS_YTEXT_OFFSET = float(sys.argv[i+1])
				elif option == '-xaxisfontsize':
					if len(sys.argv) <= (i+1):
						print( "Error: -xaxisfontsize option not specified")
						sys.exit(0)
					XAXIS_FONTSIZE = float(sys.argv[i+1])
				elif option == '-yaxispad':
					if len(sys.argv) <= (i+1):
						print( "Error: -yaxispad option not specified")
						sys.exit(0)
					YAXIS_PAD = float(sys.argv[i+1])	
				elif option == '-xaxispad':
					if len(sys.argv) <= (i+1):
						print( "Error: -xaxispad option not specified")
						sys.exit(0)
					XAXIS_PAD = float(sys.argv[i+1])
				elif option == '-isrates':
					if len(sys.argv) <= (i+1):
						print( "Error: -isrates option not specified")
						sys.exit(0)
					if sys.argv[i+1] == 'false':
						ISRATES = False
					else:
						ISRATES = True
				elif option == '-bar_label1':
					if len(sys.argv) <= (i+1):
						print( "Error: -bar_label1 option not specified")
						sys.exit(0)
					BAR_LABEL1 = sys.argv[i+1]
				elif option == '-bar_label2':
					if len(sys.argv) <= (i+1):
						print( "Error: -bar_label2 option not specified")
						sys.exit(0)
					BAR_LABEL2 = sys.argv[i+1]
				elif option == '-isfixed':
					if len(sys.argv) <= (i+1):
						print( "Error: -isfixed option not specified")
						sys.exit(0)
					if sys.argv[i+1] == 'false':
						ISFIXED = False
					else:
						ISFIXED = True
				elif option == '-barleftmargin':
					if len(sys.argv) <= (i+1):
						print( "Error: -barleftmargin option not specified")
						sys.exit(0)
					BAR_LEFT_MARGIN = float(sys.argv[i+1])	
				elif option == '-spacebwneuraloracle':
					if len(sys.argv) <= (i+1):
						print( "Error: -spacebwneuraloracle option not specified")
						sys.exit(0)
					if sys.argv[i+1] == 'False':
						SPACE_BW_NEURALORACLE = False
					else:
						SPACE_BW_NEURALORACLE = True
				elif option == '-toplabelspace':
					if len(sys.argv) <= (i+1):
						print( "Error: -toplabelspace option not specified")
						sys.exit(0)
					TOP_LABEL_SPACE = float(sys.argv[i+1])	
				elif option == '-isxaxis':
					if len(sys.argv) <= (i+1):
						print( "Error: -isxaxis option not specified")
						sys.exit(0)
					if sys.argv[i+1] == 'False':
						ISXAXIS = False
					else:
						ISXAXIS = True
				elif option == '-log':
					if len(sys.argv) <= (i+1):
						print( "Error: -log option not specified")
						sys.exit(0)
					if sys.argv[i+1] == 'False':
						LOG = 0
					else:
						LOG = 1
				elif option == '-rotatexaxis':
					if len(sys.argv) <= (i+1):
						print( "Error: -rotatexaxis option not specified")
						sys.exit(0)
					if sys.argv[i+1] == 'False':
						ROTATEXAXIS = 0
					else:
						ROTATEXAXIS = float(sys.argv[i+1])
				elif option == '-isyaxis':
					if len(sys.argv) <= (i+1):
						print( "Error: -isyaxis option not specified")
						sys.exit(0)
					if sys.argv[i+1] == 'False':
						ISYAXIS = False
					else:
						ISYAXIS = True
				elif option == '-nolegend':
					if len(sys.argv) <= (i+1):
						print( "Error: -nolegend option not specified")
						sys.exit(0)
					if sys.argv[i+1] == 'False':
						NOLEGEND = False
					else:
						NOLEGEND = True
				elif option == '-legloc':
					if len(sys.argv) <= (i+1):
						print( "Error: -legloc option not specified")
						sys.exit(0)
					LEGEND_LOCATION = int(sys.argv[i+1])
				elif option == '-xaxispadding':
					if len(sys.argv) <= (i+1):
						print( "Error: -xaxispadding option not specified")
						sys.exit(0)
					XAXISPADDING = float(sys.argv[i+1])	
				elif option == '-datalabel':
					if len(sys.argv) <= (i+1):
						print( "Error: -datalabel option not specified")
						sys.exit(0)
					if sys.argv[i+1] == 'False':
						DATALABEL = False
					else:
						DATALABEL = True
				elif option == '-datalabelformat':
					if len(sys.argv) <= (i+1):
						print( "Error: -datalabelformat option not specified")
						sys.exit(0)
					DATALABEL_FORMAT = sys.argv[i+1]
				elif option == '-toplabelformat':
					if len(sys.argv) <= (i+1):
						print( "Error: -toplabelformat option not specified")
						sys.exit(0)
					TOPLABEL_FORMAT = sys.argv[i+1]
				elif option == '-toplabelrotate':
					if len(sys.argv) <= (i+1):
						print( "Error: -toplabelrotate option not specified")
						sys.exit(0)
					TOPLABEL_ROTATE = int(sys.argv[i+1])
				elif option == '-datalabelrotate':
					if len(sys.argv) <= (i+1):
						print( "Error: -datalabelrotate option not specified")
						sys.exit(0)
					DATALABEL_ROTATE = int(sys.argv[i+1])
				elif option == '-xaxis_textoffset':
					if len(sys.argv) <= (i+1):
						print( "Error: -xaxis_textoffset option not specified")
						sys.exit(0)
					XAXIS_TEXTOFFSET = float(sys.argv[i+1])
				elif option == '-toplabelbold':
					if len(sys.argv) <= (i+1):
						print( "Error: -toplabelformat option not specified")
						sys.exit(0)
					TOPLABEL_BOLD = sys.argv[i+1] 
				elif option == '-datalabelbold':
					if len(sys.argv) <= (i+1):
						print( "Error: -datalabelbold option not specified")
						sys.exit(0)
					DATALABEL_BOLD = sys.argv[i+1] 
				elif option == '-benchnewline':
					if len(sys.argv) <= (i+1):
						print "Error: -benchnewline option not specified"
						sys.exit(0)
					if sys.argv[i+1] == 'False':
						BENCH_NEWLINE = False
					else:
						BENCH_NEWLINE = True
				elif option == '-h' or option == '--help':
					print( usage)
					sys.exit()
	
	legends, bench, data, std = readData(csvFilename)
	plotBarGraph(outputFileName, legends, bench, data, std)
	os.system('pdfcrop ' + outputFileName + ' ' + outputFileName + ' > /dev/null 2> /dev/null')

if __name__ == "__main__":
    main()
