import os
import pandas
from sram.sram_stats import get_sram_dataframe
from src.simulator.simulator import Simulator
from src.simulator.stats import Stats, get_energy_from_results
from src.utils.utils import *
import src.benchmarks.benchmarks as benchmarks
from src.sweep.sweep import check_pandas_or_run
from scipy.stats import gmean

def optimize_area_allocation(
        precision_list, benchmark_list,
        frequency, baseline_mult, results_dir='./results'):
    """
    Optimize the area allocated to the three scratchpad buffers and the systolic
    array. Assume spatial multiplier here.
    """

    print('Running Architecture optimization')

    ##################################################
    # Get stats for systolic array
    core_csv = os.path.join(results_dir, 'systolic_array_synth.csv')
    core_synth_data = pandas.read_csv(core_csv)

    ##################################################
    # Get stats for SRAM
    tech_node = 45
    voltage = 0.85
    sram_csv = 'hardware_sweep/sram_results.csv'
    sram_data = get_sram_dataframe(tech_node, voltage, int(frequency * 1.e-6), './sram/data',
                                   logpath='./sram/mcpat.sram/SampleScirpts/RunLog')

    ##################################################
    # Get stats for SRAM
    # Eyeriss uses 12.25 mm2 for 168 PEs
    # Tetris uses 3.5mm2 for 196 PEs
    # area_budget = 3.5 * 3.5 * tech_node * tech_node / (65. * 65.)
    area_budget = 3.5
    print('Area budget = {} mm^2'.format(area_budget))

    config_file = 'conf.ini'
    verbose = False
    sim_obj = Simulator(config_file, verbose)

    sim_obj.accelerator.N = 8
    sim_obj.accelerator.M = 8
    sim_obj.accelerator.pmax = 16
    sim_obj.accelerator.pmin = 16
    print('#' * 50)
    print('Baseline config:')
    print(sim_obj)
    print('#' * 50)

    ##################################################
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

    result_columns = ['Cycles', 'Memory wait cycles',
            'WBUF Read', 'WBUF Write',
            'OBUF Read', 'OBUF Write',
            'IBUF Read', 'IBUF Write',
            'DRAM Read', 'DRAM Write']

    group_columns = []
    for x in sim_sweep_columns:
        if x not in result_columns and x is not 'Layer':
            group_columns.append(x)

    # Generate baseline numbers
    sim_sweep_csv = os.path.join(results_dir, 'bitfusion-sim-sweep.csv')
    if os.path.exists(sim_sweep_csv):
        sim_sweep_df = pandas.read_csv(os.path.join(results_dir, 'bitfusion-sim-sweep.csv'))
    else:
        sim_sweep_df = pandas.DataFrame(columns=sim_sweep_columns)
    batch_size = 16
    df = check_pandas_or_run(sim_obj, sim_sweep_df, sim_sweep_csv, batch_size=batch_size)
    baseline_results = df.groupby(group_columns, as_index=False).sum()

    approx_dram_energy = lookup_pandas_dataframe(sram_data,
                                                 {'size (Bytes)': 128 * 1024, 'ports': 2, 'bits': 512})

    finished_sim = {}
    for pmax, pmin in precision_list:
        sim_obj.accelerator.pmax = pmax
        sim_obj.accelerator.pmin = pmin
        for log2_n in range(0, 6):
            for log2_m in range(0, 9):
                n = 1 << log2_n
                m = 1 << log2_m

                sim_obj.accelerator.N = n
                sim_obj.accelerator.M = m

                lookup_dict = {}
                lookup_dict['Max Precision (bits)'] = pmax
                lookup_dict['Min Precision (bits)'] = pmin
                lookup_dict['N'] = n
                lookup_dict['M'] = m

                core_data = lookup_pandas_dataframe(core_synth_data, lookup_dict)
                if len(core_data) == 0:
                    lookup_dict['N'] = 4
                    lookup_dict['M'] = 4
                    core_data = lookup_pandas_dataframe(core_synth_data, lookup_dict)
                    if len(core_data) == 0:
                        print(pmax, pmin)
                        print(n, m)
                    assert len(core_data) == 1
                    core_area = float(core_data['Area (um^2)']) * 1.e-6 * (n * m) / 16.
                    core_dyn_power = float(core_data['Dynamic Power (nW)']) * (n * m) / 16.
                    core_dyn_energy = core_dyn_power / float(core_data['Frequency'])
                    core_leak_power = float(core_data['Leakage Power (nW)']) * (n * m) / 16.
                    core_leak_energy = core_leak_power / float(core_data['Frequency'])
                else:
                    core_area = float(core_data['Area (um^2)']) * 1.e-6
                    core_dyn_power = float(core_data['Dynamic Power (nW)'])
                    core_dyn_energy = core_dyn_power / float(core_data['Frequency'])
                    core_leak_power = float(core_data['Leakage Power (nW)'])
                    core_leak_energy = core_leak_power / float(core_data['Frequency'])

                print(n, m)
                print(core_area)

                sram_area_budget = area_budget - core_area
                print('SRAM area budget: {}'.format(sram_area_budget))
                if sram_area_budget < 0:
                    continue

                wports = 2
                iports = 2
                oports = 2

                sim_config = (pmax,pmin,n,m)

                for log2_wbuf_size in range(20, 8, -1):
                    if sim_config in finished_sim:
                        continue

                    sim_obj.accelerator.sram['wgt'] = 1 << log2_wbuf_size
                    wbuf_size = sim_obj.accelerator.sram['wgt']
                    # double buffered, so 2x banks
                    wbanks = n * 2
                    wbuf_size_per_bank = wbuf_size / wbanks
                    wbuf_bits_per_bank = m * pmax * pmax / pmin
                    wbuf_results = lookup_pandas_dataframe(sram_data,
                            {'size (Bytes)': wbuf_size_per_bank, 'ports': wports,
                                'bits': wbuf_bits_per_bank})
                    if len(wbuf_results) == 0:
                        continue

                    wbuf_area = float(wbuf_results['area (mm^2)']) * wbanks
                    if wbuf_area > sram_area_budget:
                        continue
                    ibuf_area_budget = sram_area_budget - wbuf_area

                    for log2_ibuf_size in range(19, 7, -1):
                        if sim_config in finished_sim:
                            continue

                        sim_obj.accelerator.sram['act'] = 1 << log2_ibuf_size
                        ibuf_size = sim_obj.accelerator.sram['act']
                        # double buffered, so 2x banks
                        ibanks = n * 2
                        ibuf_size_per_bank = ibuf_size / ibanks
                        ibuf_bits_per_bank = pmax * pmax / pmin
                        ibuf_results = lookup_pandas_dataframe(sram_data,
                                                               {'size (Bytes)': ibuf_size_per_bank, 'ports': iports,
                                                                'bits': ibuf_bits_per_bank})
                        if len(ibuf_results) == 0:
                            continue
                        ibuf_area = float(ibuf_results['area (mm^2)']) * ibanks
                        if ibuf_area > ibuf_area_budget:
                            continue
                        obuf_area_budget = ibuf_area_budget - ibuf_area

                        for log2_obuf_size in range(19, 7, -1):
                            if sim_config in finished_sim:
                                continue
                            sim_obj.accelerator.sram['out'] = 1 << log2_obuf_size
                            obuf_size = sim_obj.accelerator.sram['out']
                            # double buffered, so 2x banks
                            obanks = 1 * 2
                            obuf_size_per_bank = obuf_size / obanks
                            obuf_bits_per_bank = 32 * m
                            obuf_results = lookup_pandas_dataframe(sram_data,
                                                                   {'size (Bytes)': obuf_size_per_bank, 'ports': oports,
                                                                    'bits': obuf_bits_per_bank})
                            if len(obuf_results) == 0:
                                continue
                            obuf_area = float(obuf_results['area (mm^2)']) * obanks
                            if obuf_area > obuf_area_budget + 0.1 * sram_area_budget:
                                continue

                            unused_area = obuf_area_budget - obuf_area
                            if unused_area > obuf_area or unused_area > 0.2 * area_budget:
                                continue

                            if max(log2_wbuf_size - log2_ibuf_size,
                                   log2_ibuf_size - log2_wbuf_size,
                                   log2_wbuf_size - log2_obuf_size,
                                   log2_obuf_size - log2_wbuf_size) > 2:
                                if sim_config in finished_sim:
                                    continue

                            sim_obj.accelerator.sram['out'] = 1 << log2_obuf_size
                            obuf_size = sim_obj.accelerator.sram['out']
                            # double buffered, so 2x banks
                            obanks = 1 * 2
                            obuf_size_per_bank = obuf_size / obanks
                            obuf_bits_per_bank = 32 * m
                            obuf_results = lookup_pandas_dataframe(sram_data,
                                                                   {'size (Bytes)': obuf_size_per_bank, 'ports': oports,
                                                                    'bits': obuf_bits_per_bank})

                            total_buffer_area = wbuf_area + ibuf_area + obuf_area
                            sram_systolic_ratio = total_buffer_area / core_area
                            print('#' * 50)
                            print('Area Breakdown= {:0.2f}% core {:0.2f}% sram'.format(core_area/area_budget * 100, sram_area_budget/area_budget * 100))
                            print('Unused area ratio : {}'.format(unused_area / area_budget))
                            print(sim_obj)
                            print('#' * 50)

                            results = check_pandas_or_run(sim_obj, sim_sweep_df, sim_sweep_csv, batch_size=batch_size).groupby(group_columns, as_index=False).sum()
                            perf_relative = []
                            energy_relative = []
                            for b in benchmarks.benchlist:
                                lookup_dict = {'Network': b}
                                basline_cycles = lookup_pandas_dataframe(baseline_results, lookup_dict)['Cycles']
                                current_cycles = lookup_pandas_dataframe(results, lookup_dict)['Cycles']
                                perf_relative.append(float(basline_cycles)/float(current_cycles))
                                baseline_energy = get_energy_from_results(lookup_pandas_dataframe(baseline_results, lookup_dict), sim_obj.get_energy_cost())
                                current_energy = get_energy_from_results(lookup_pandas_dataframe(results, lookup_dict), sim_obj.get_energy_cost())
                                energy_relative.append(float(baseline_energy) / float(current_energy))
                            results['Performance-Relative'] = gmean(perf_relative)
                            results['Energy-Relative'] = gmean(energy_relative)

                            config_data = [n, m, pmax, pmin, sim_obj.accelerator.mem_if_width, wbuf_size, wbuf_area, ibuf_size, ibuf_area, obuf_size, obuf_area, core_area, batch_size, gmean(perf_relative), gmean(energy_relative)]
                            config_data_columns = ['N', 'M', 'Max Precision (bits)', 'Min Precision (bits)', 'Bandwidth (bits/cycle)', 'WBUF Size (bits)', 'WBUF Area (mm^2)', 'IBUF Size (bits)', 'IBUF Area (mm^2)', 'OBUF Size (bits)', 'OBUF Area (mm^2)', 'Core Area (mm^2)', 'Batch size', 'Performance-Relative', 'Energy-Efficiency-Relative']

                            finished_sim[sim_config] = pandas.DataFrame([config_data], columns=config_data_columns)

    return finished_sim

