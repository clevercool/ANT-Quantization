from re import I
import pandas
import configparser
import os
import numpy as np
import bitfusion.src.benchmarks.benchmarks as benchmarks
from bitfusion.src.simulator.stats import Stats
from bitfusion.src.simulator.simulator import Simulator
from bitfusion.src.sweep.sweep import SimulatorSweep, check_pandas_or_run
from bitfusion.src.utils.utils import *
from bitfusion.src.optimizer.optimizer import optimize_for_order, get_stats_fast

def df_to_stats(df):
    stats = Stats()
    stats.total_cycles = float(df['Cycles'].iloc[0])
    stats.mem_stall_cycles = float(df['Memory wait cycles'].iloc[0])
    stats.reads['act'] = float(df['IBUF Read'].iloc[0])
    stats.reads['out'] = float(df['OBUF Read'].iloc[0])
    stats.reads['wgt'] = float(df['WBUF Read'].iloc[0])
    stats.reads['dram'] = float(df['DRAM Read'].iloc[0])
    stats.writes['act'] = float(df['IBUF Write'].iloc[0])
    stats.writes['out'] = float(df['OBUF Write'].iloc[0])
    stats.writes['wgt'] = float(df['WBUF Write'].iloc[0])
    stats.writes['dram'] = float(df['DRAM Write'].iloc[0])
    return stats

sim_sweep_columns = ['N', 'M',
        'Max Precision (bits)', 'Min Precision (bits)',
        'Network', 'Layer',
        'Cycles', 'Memory wait cycles',
        'WBUF Read', 'WBUF Write',
        'OBUF Read', 'OBUF Write',
        'IBUF Read', 'IBUF Write',
        'DRAM Read', 'DRAM Write',
        'Bandwidth (bits/cycle)',
        'WBUF Size (bits)', 'OBUF Size (bits)', 'IBUF Size (bits)',
        'Batch size']

batch_size = 64

results_dir = './results'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)


# ANT configuration file
config_file = 'conf_ant.ini'
# Create simulator object
bf_e_sim = Simulator(config_file, False)
bf_e_sim_sweep_csv = os.path.join(results_dir, 'ant_os.csv')
bf_e_sim_sweep_df = pandas.DataFrame(columns=sim_sweep_columns)
bf_e_results = check_pandas_or_run(bf_e_sim, bf_e_sim_sweep_df, bf_e_sim_sweep_csv, batch_size=batch_size, bench_type='ant')
bf_e_results = bf_e_results.groupby('Network',as_index=False).agg(np.sum)
bf_e_cycles_ant = []
bf_e_energy_ant = []
for name in benchmarks.benchlist:
    bf_e_stats = df_to_stats(bf_e_results.loc[bf_e_results['Network'] == name])
    bf_e_cycles_ant.append(bf_e_stats.total_cycles)
    bf_e_energy_ant.append(bf_e_stats.get_energy_breakdown(bf_e_sim.get_energy_cost()))


# ANT weight_stationary configuration file
config_file = 'conf_ant.ini'
# Create simulator object
bf_e_sim1 = Simulator(config_file, False)
bf_e_sim_sweep_csv_1 = os.path.join(results_dir, 'ant_ws.csv')
bf_e_sim_sweep_df_1 = pandas.DataFrame(columns=sim_sweep_columns)
bf_e_results = check_pandas_or_run(bf_e_sim1, bf_e_sim_sweep_df_1, bf_e_sim_sweep_csv_1, batch_size=batch_size, bench_type='ant_weight', weight_stationary=True)
bf_e_results = bf_e_results.groupby('Network',as_index=False).agg(np.sum)
# area_stats = bf_e_sim.get_area()
bf_e_cycles_ant_weight = []
bf_e_energy_ant_weight = []
for name in benchmarks.benchlist:
    bf_e_stats = df_to_stats(bf_e_results.loc[bf_e_results['Network'] == name])
    bf_e_cycles_ant_weight.append(bf_e_stats.total_cycles)
    bf_e_energy_ant_weight.append(bf_e_stats.get_energy_breakdown(bf_e_sim1.get_energy_cost()))


# BitFusion configuration file
config_file = 'conf_ant.ini'
# Create simulator object
bf_e_sim = Simulator(config_file, False)
bf_e_sim_sweep_csv = os.path.join(results_dir, 'bitfusion.csv')
bf_e_sim_sweep_df = pandas.DataFrame(columns=sim_sweep_columns)
bf_e_results = check_pandas_or_run(bf_e_sim, bf_e_sim_sweep_df, bf_e_sim_sweep_csv, batch_size=batch_size, bench_type='bit')
bf_e_results = bf_e_results.groupby('Network',as_index=False).agg(np.sum)
# area_stats = bf_e_sim.get_area()
bf_e_cycles_bit = []
bf_e_energy_bit = []
for name in benchmarks.benchlist:
    bf_e_stats = df_to_stats(bf_e_results.loc[bf_e_results['Network'] == name])
    bf_e_cycles_bit.append(bf_e_stats.total_cycles)
    bf_e_energy_bit.append(bf_e_stats.get_energy_breakdown(bf_e_sim.get_energy_cost()))


# OLAceel configuration file
config_file = 'conf_olaccel.ini'
# Create simulator object
bf_e_sim = Simulator(config_file, False)
bf_e_sim_sweep_csv = os.path.join(results_dir, 'olaceel.csv')
bf_e_sim_sweep_df = pandas.DataFrame(columns=sim_sweep_columns)
bf_e_results = check_pandas_or_run(bf_e_sim, bf_e_sim_sweep_df, bf_e_sim_sweep_csv, batch_size=batch_size, bench_type='ola')
bf_e_results = bf_e_results.groupby('Network',as_index=False).agg(np.sum)
# area_stats = bf_e_sim.get_area()
bf_e_cycles_ola = []
bf_e_energy_ola = []
for name in benchmarks.benchlist:
    bf_e_stats = df_to_stats(bf_e_results.loc[bf_e_results['Network'] == name])
    bf_e_cycles_ola.append(bf_e_stats.total_cycles)
    bf_e_energy_ola.append(bf_e_stats.get_energy_breakdown(bf_e_sim.get_energy_cost()))

# AdaFloat configuration file
config_file = 'conf_adafloat.ini'
# Create simulator object
bf_e_sim = Simulator(config_file, False)
bf_e_sim_sweep_csv = os.path.join(results_dir, 'adafloat.csv')
bf_e_sim_sweep_df = pandas.DataFrame(columns=sim_sweep_columns)
bf_e_results = check_pandas_or_run(bf_e_sim, bf_e_sim_sweep_df, bf_e_sim_sweep_csv, batch_size=batch_size, bench_type='ada')
bf_e_results = bf_e_results.groupby('Network',as_index=False).agg(np.sum)
# area_stats = bf_e_sim.get_area()
bf_e_cycles_ada = []
bf_e_energy_ada = []
for name in benchmarks.benchlist:
    bf_e_stats = df_to_stats(bf_e_results.loc[bf_e_results['Network'] == name])
    bf_e_cycles_ada.append(bf_e_stats.total_cycles)
    bf_e_energy_ada.append(bf_e_stats.get_energy_breakdown(bf_e_sim.get_energy_cost()))

# BiScaled configuration file
config_file = 'conf_biscaled.ini'
# Create simulator object
bf_e_sim = Simulator(config_file, False)
bf_e_sim_sweep_csv = os.path.join(results_dir, 'biscale.csv')
bf_e_sim_sweep_df = pandas.DataFrame(columns=sim_sweep_columns)
bf_e_results = check_pandas_or_run(bf_e_sim, bf_e_sim_sweep_df, bf_e_sim_sweep_csv, batch_size=batch_size, bench_type='bis')
bf_e_results = bf_e_results.groupby('Network',as_index=False).agg(np.sum)
# area_stats = bf_e_sim.get_area()
bf_e_cycles_bis = []
bf_e_energy_bis = []
for name in benchmarks.benchlist:
    bf_e_stats = df_to_stats(bf_e_results.loc[bf_e_results['Network'] == name])
    bf_e_cycles_bis.append(bf_e_stats.total_cycles)
    bf_e_energy_bis.append(bf_e_stats.get_energy_breakdown(bf_e_sim.get_energy_cost()))


all_cyc = []
cyc_1_mean = 0
cyc_2_mean = 0
cyc_3_mean = 0
cyc_4_mean = 0
cyc_5_mean = 0
cyc_6_mean = 0

# write to csv
model_name_dict = {'vgg16':'VGG16', 
                   'resnet18':'ResNet18',
                   'resnet50':'ResNet50',
                   'inceptionv3':'InceptionV3',
                   'vit':'ViT',
                   'mnli':'BERT-MNLI',
                   'cola':'BERT-CoLA',
                   'sst_2':'BERT-SST-2'}
ff = open(os.getcwd() + '/results/ant_res.csv', "a")
wr_line = "Time, "
wr_bench_name = ", "
wr_model_name = ", "
for i in range(len(bf_e_cycles_ant)):
    model_name = benchmarks.benchlist[i]

    cyc_4 = bf_e_cycles_ada[i]
    cyc_1 = bf_e_cycles_ant[i] / cyc_4
    cyc_1_mean += cyc_1
    cyc_5 = bf_e_cycles_ant_weight[i] / cyc_4
    cyc_5_mean += cyc_5
    cyc_2 = bf_e_cycles_bit[i] / cyc_4
    cyc_2_mean += cyc_2
    cyc_3 = bf_e_cycles_ola[i] / cyc_4
    cyc_3_mean += cyc_3
    cyc_6 = bf_e_cycles_bis[i] / cyc_4
    cyc_6_mean += cyc_6
    cyc_4 = cyc_4 / cyc_4
    cyc_4_mean += cyc_4
    
    all_cyc.append(cyc_1)
    all_cyc.append(cyc_5)
    all_cyc.append(cyc_2)
    all_cyc.append(cyc_3)
    all_cyc.append(cyc_4)
    all_cyc.append(cyc_6)

    if(model_name == 'vgg16' or model_name == 'resnet50'):
        wr_model_name += model_name_dict[model_name] + ", , , , , , "
        wr_bench_name += "ANT-OS, ANT-WS, BitFusion, OLAccel, BiScaled, AdaFloat, "
        wr_line += "%0.2f, %0.2f, %0.2f, %0.2f, %0.2f, %0.2f, " %(cyc_1, cyc_5, cyc_2, cyc_3, cyc_6, cyc_4)
        print("%0.2f, %0.2f, %0.2f, %0.2f, %0.2f, %0.2f, " %(cyc_1, cyc_5, cyc_2, cyc_3, cyc_6, cyc_4), end="")
    else:
        wr_model_name += model_name_dict[model_name] + ", , , , , "
        wr_bench_name += "ANT-OS, ANT-WS, BitFusion, OLAccel, AdaFloat, "
        wr_line += "%0.2f, %0.2f, %0.2f, %0.2f, %0.2f, " %(cyc_1, cyc_5, cyc_2, cyc_3, cyc_4)
        print("%0.2f, %0.2f, %0.2f, %0.2f, %0.2f, " %(cyc_1, cyc_5, cyc_2, cyc_3, cyc_4), end="")

cyc_1_mean /= len(bf_e_cycles_ant)
cyc_5_mean /= len(bf_e_cycles_ant)
cyc_2_mean /= len(bf_e_cycles_ant)
cyc_3_mean /= len(bf_e_cycles_ant)
cyc_4_mean /= len(bf_e_cycles_ant)
cyc_6_mean /= len(bf_e_cycles_ant)

wr_model_name += "Geomean, , , , , \n"
wr_bench_name += "ANT-OS, ANT-WS, BitFusion, OLAccel, BiScaled, AdaFloat\n"
wr_line += ("%0.2f, %0.2f, %0.2f, %0.2f, %0.2f, %0.2f, " %(cyc_1_mean, cyc_5_mean, cyc_2_mean, cyc_3_mean, cyc_6_mean, cyc_4_mean)) + "\n"
ff.write(wr_model_name)
ff.write(wr_bench_name)
ff.write(wr_line)
wr_line = ""
print("%0.2f, %0.2f, %0.2f, %0.2f, %0.2f, %0.2f, " %(cyc_1_mean, cyc_5_mean, cyc_2_mean, cyc_3_mean, cyc_6_mean, cyc_4_mean))
print()

all_energy1 = []
all_energy2 = []
all_energy3 = []
all_energy4 = []
all_energy5 = []
all_energy6 = []
for i in range(len(bf_e_cycles_ant)):

    model_name = benchmarks.benchlist[i]
    # print(model_name)

    energy_data_4 = bf_e_energy_ada[i]
    energy_data_total = energy_data_4[0] + energy_data_4[1] + energy_data_4[2] + energy_data_4[3]

    energy_data_4[0] /= energy_data_total
    energy_data_4[1] /= energy_data_total
    energy_data_4[2] /= energy_data_total
    energy_data_4[3] /= energy_data_total

    energy_data_1 = bf_e_energy_ant[i]
    energy_data_1[0] /= energy_data_total
    energy_data_1[1] /= energy_data_total
    energy_data_1[2] /= energy_data_total
    energy_data_1[3] /= energy_data_total

    energy_data_5 = bf_e_energy_ant_weight[i]
    energy_data_5[0] /= energy_data_total
    energy_data_5[1] /= energy_data_total
    energy_data_5[2] /= energy_data_total
    energy_data_5[3] /= energy_data_total

    energy_data_2 = bf_e_energy_bit[i]
    energy_data_2[0] /= energy_data_total
    energy_data_2[1] /= energy_data_total
    energy_data_2[2] /= energy_data_total
    energy_data_2[3] /= energy_data_total

    energy_data_3 = bf_e_energy_ola[i]
    energy_data_3[0] /= energy_data_total
    energy_data_3[1] /= energy_data_total
    energy_data_3[2] /= energy_data_total
    energy_data_3[3] /= energy_data_total

    energy_data_6 = bf_e_energy_bis[i]
    energy_data_6[0] /= energy_data_total
    energy_data_6[1] /= energy_data_total
    energy_data_6[2] /= energy_data_total
    energy_data_6[3] /= energy_data_total

    # print("cc", len(energy_data_6), len(bf_e_energy_bis))
    
    all_energy1.append(energy_data_1[0])
    all_energy1.append(energy_data_5[0])
    all_energy1.append(energy_data_2[0])
    all_energy1.append(energy_data_3[0])
    # biscale only test vg16 and resnet50
    if(model_name == 'vgg16' or model_name == 'resnet50'):
        all_energy1.append(energy_data_6[0])
    all_energy1.append(energy_data_4[0])


    all_energy2.append(energy_data_1[1])
    all_energy2.append(energy_data_5[1])
    all_energy2.append(energy_data_2[1])
    all_energy2.append(energy_data_3[1])
    if(model_name == 'vgg16' or model_name == 'resnet50'):
        all_energy2.append(energy_data_6[1])
    all_energy2.append(energy_data_4[1])


    all_energy3.append(energy_data_1[2])
    all_energy3.append(energy_data_5[2])
    all_energy3.append(energy_data_2[2])
    all_energy3.append(energy_data_3[2])
    if(model_name == 'vgg16' or model_name == 'resnet50'):
        all_energy3.append(energy_data_6[2])
    all_energy3.append(energy_data_4[2])


    all_energy4.append(energy_data_1[3])
    all_energy4.append(energy_data_5[3])
    all_energy4.append(energy_data_2[3])
    all_energy4.append(energy_data_3[3])
    if(model_name == 'vgg16' or model_name == 'resnet50'):
        all_energy4.append(energy_data_6[3])
    all_energy4.append(energy_data_4[3])


print()

wr_line = "Static, "
for i in all_energy1:
    wr_line += "%0.2f, " %(i)
    print("%0.2f, " %(i), end="")
energy_mean_1 = 0
energy_mean_2 = 0
energy_mean_3 = 0
energy_mean_4 = 0
energy_mean_5 = 0
energy_mean_6 = 0

biscale_offset = 0
for i in range(len(bf_e_cycles_ant)):
    model_name = benchmarks.benchlist[i]
    idx = i * 5 + biscale_offset
    energy_mean_1 += all_energy1[idx]
    energy_mean_5 += all_energy1[idx+1]
    energy_mean_2 += all_energy1[idx+2]
    energy_mean_3 += all_energy1[idx+3]
    if(model_name == 'vgg16' or model_name == 'resnet50'):
        biscale_offset += 1
        energy_mean_6 += all_energy1[idx+4]
        energy_mean_4 += all_energy1[idx+5]
    else:
        energy_mean_4 += all_energy1[idx+4]
energy_mean_1 /= len(bf_e_cycles_ant)
energy_mean_5 /= len(bf_e_cycles_ant)
energy_mean_2 /= len(bf_e_cycles_ant)
energy_mean_3 /= len(bf_e_cycles_ant)
energy_mean_6 /= 2
energy_mean_4 /= len(bf_e_cycles_ant)

wr_line += ("%0.2f, %0.2f, %0.2f, %0.2f, %0.2f, %0.2f," %(energy_mean_1, energy_mean_5, energy_mean_2, energy_mean_3, energy_mean_6, energy_mean_4)) + "\n"
ff.write("\n")
ff.write(wr_model_name)
ff.write(wr_bench_name)
ff.write(wr_line)
wr_line = ""
print("%0.2f, %0.2f, %0.2f, %0.2f, %0.2f, %0.2f," %(energy_mean_1, energy_mean_5, energy_mean_2, energy_mean_3, energy_mean_6, energy_mean_4))

wr_line = "Dram, "
for i in all_energy2:
    wr_line += "%0.2f, " %(i)
    print("%0.2f, " %(i), end="")
energy_mean_1 = 0
energy_mean_2 = 0
energy_mean_3 = 0
energy_mean_4 = 0
energy_mean_5 = 0
energy_mean_6 = 0

biscale_offset = 0
for i in range(len(bf_e_cycles_ant)):
    model_name = benchmarks.benchlist[i]
    idx = i * 5 + biscale_offset
    energy_mean_1 += all_energy2[idx]
    energy_mean_5 += all_energy2[idx+1]
    energy_mean_2 += all_energy2[idx+2]
    energy_mean_3 += all_energy2[idx+3]
    if(model_name == 'vgg16' or model_name == 'resnet50'):
        energy_mean_6 += all_energy2[idx+4]
        energy_mean_4 += all_energy2[idx+5]
        biscale_offset += 1 
    else:
        energy_mean_4 += all_energy2[idx+4]
energy_mean_1 /= len(bf_e_cycles_ant)
energy_mean_5 /= len(bf_e_cycles_ant)
energy_mean_2 /= len(bf_e_cycles_ant)
energy_mean_3 /= len(bf_e_cycles_ant)
energy_mean_4 /= len(bf_e_cycles_ant)
energy_mean_6 /= 2

wr_line += ("%0.2f, %0.2f, %0.2f, %0.2f, %0.2f, %0.2f," %(energy_mean_1, energy_mean_5, energy_mean_2, energy_mean_3, energy_mean_6, energy_mean_4)) + "\n"
ff.write(wr_line)
wr_line = ""
print("%0.2f, %0.2f, %0.2f, %0.2f, %0.2f, %0.2f," %(energy_mean_1, energy_mean_5, energy_mean_2, energy_mean_3, energy_mean_6, energy_mean_4))

wr_line = "Buffer, "
for i in all_energy3:
    wr_line += "%0.2f, " %(i)
    print("%0.2f, " %(i), end="")
energy_mean_1 = 0
energy_mean_2 = 0
energy_mean_3 = 0
energy_mean_4 = 0
energy_mean_6 = 0

biscale_offset = 0
for i in range(len(bf_e_cycles_ant)):
    model_name = benchmarks.benchlist[i]
    idx = i * 5 + biscale_offset
    energy_mean_1 += all_energy3[idx]
    energy_mean_5 += all_energy3[idx+1]
    energy_mean_2 += all_energy3[idx+2]
    energy_mean_3 += all_energy3[idx+3]
    if(model_name == 'vgg16' or model_name == 'resnet50'):
        energy_mean_6 += all_energy3[idx+4]
        energy_mean_4 += all_energy3[idx+5]
        biscale_offset += 1
    else:
        energy_mean_4 += all_energy3[idx+4]
energy_mean_1 /= len(bf_e_cycles_ant)
energy_mean_5 /= len(bf_e_cycles_ant)
energy_mean_2 /= len(bf_e_cycles_ant)
energy_mean_3 /= len(bf_e_cycles_ant)
energy_mean_4 /= len(bf_e_cycles_ant)
energy_mean_6 /= 2

wr_line += ("%0.2f, %0.2f, %0.2f, %0.2f, %0.2f, %0.2f," %(energy_mean_1, energy_mean_5, energy_mean_2, energy_mean_3, energy_mean_6, energy_mean_4)) + "\n"
ff.write(wr_line)
wr_line = ""
print("%0.2f, %0.2f, %0.2f, %0.2f, %0.2f, %0.2f," %(energy_mean_1, energy_mean_5, energy_mean_2, energy_mean_3, energy_mean_6, energy_mean_4))

wr_line = "Core, "
for i in all_energy4:
    wr_line += "%0.2f, " %(i)
    print("%0.2f, " %(i), end="")
energy_mean_1 = 0
energy_mean_2 = 0
energy_mean_3 = 0
energy_mean_4 = 0
energy_mean_5 = 0
energy_mean_6 = 0

biscale_offset = 0
for i in range(len(bf_e_cycles_ant)):
    model_name = benchmarks.benchlist[i]
    idx = i * 5 + biscale_offset
    energy_mean_1 += all_energy4[idx]
    energy_mean_5 += all_energy4[idx+1]
    energy_mean_2 += all_energy4[idx+2]
    energy_mean_3 += all_energy4[idx+3]
    if(model_name == 'vgg16' or model_name == 'resnet50'):
        energy_mean_6 += all_energy4[idx+4]
        energy_mean_4 += all_energy4[idx+5]
        biscale_offset += 1
    else:
        energy_mean_4 += all_energy4[idx+4]
energy_mean_1 /= len(bf_e_cycles_ant)
energy_mean_5 /= len(bf_e_cycles_ant)
energy_mean_2 /= len(bf_e_cycles_ant)
energy_mean_3 /= len(bf_e_cycles_ant)
energy_mean_4 /= len(bf_e_cycles_ant)
energy_mean_6 /= 2

wr_line += ("%0.2f, %0.2f, %0.2f, %0.2f, %0.2f, %0.2f," %(energy_mean_1, energy_mean_5, energy_mean_2, energy_mean_3, energy_mean_6, energy_mean_4)) + "\n"
ff.write(wr_line)
wr_line = ""
print("%0.2f, %0.2f, %0.2f, %0.2f, %0.2f, %0.2f," %(energy_mean_1, energy_mean_5, energy_mean_2, energy_mean_3, energy_mean_6, energy_mean_4))

print("Please see the results at ./results/ant_res.csv ")
ff.close()