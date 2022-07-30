import os
import src.benchmarks.benchmarks as benchmarks
from collections import OrderedDict
import pandas
from src.utils.utils import lookup_pandas_dataframe

def ideal_benefits_variable_precision(mult_type, frequency, csv_file_name, default_dict={}):
    '''
    benefits of variable precision for the benchmarks
    Args:
        mult_type: Type of multiplier
        ...
    Returns:
        pandas.dataframe object
    '''
    benchlist = benchmarks.benchlist

    nn_list = OrderedDict()

    for b in benchlist:
        nn_list[b] = benchmarks.get_bench_nn(b)

    clock_delay = int(1e12 / frequency)

    synth_dataframe = pandas.read_csv(csv_file_name)

    result_columns = ['Multiplier_type', 'Max Precision', 'Min Precision',
                      'Frequency', 'Area (um^2)',
                      'Neural Network', 'Layer',
                      'Cycles', 'Static Energy (nJ)', 'Dynamic Energy (nJ)', 'Energy (nJ)']
    results = []

    for mtype in mult_type:
        lookup_dict = default_dict
        lookup_dict['Mult Type'] = mtype
        lookup_dict['Clock delay (ps)'] = clock_delay

        synth_results = lookup_pandas_dataframe(synth_dataframe, lookup_dict)
        if len(synth_results.values) > 1:
            raise ValueError('More than one match found')

        area_um2 = float(synth_results['Area (um^2)'])

        dynamic_power_nW = float(synth_results['Dynamic Power (nW)'])
        leakage_power_nW = float(synth_results['Leakage Power (nW)'])

        try:
            min_weight_prec = int(synth_results['Min Precision (bits)'])
            min_input_prec = int(synth_results['Min Precision (bits)'])
            mult_type = 'spatial'
            max_weight_prec = int(synth_results['Max Precision (bits)'])
            max_input_prec = int(synth_results['Max Precision (bits)'])
        except:
            min_weight_prec = int(synth_results['Weight Precision (bits)'])
            min_input_prec = int(synth_results['Activation Precision (bits)'])
            mult_type = 'temporal'
            max_weight_prec = min_weight_prec
            max_input_prec = min_input_prec

        for b in benchlist:
            nn = nn_list[b]

            bench_time = 0.0
            bench_cycles = 0
            bench_static_energy = 0.0
            bench_dynamic_energy = 0.0

            for l in nn:
                layer = nn[l]
                if not (isinstance(layer, ConvLayer) or isinstance(layer, FCLayer)):
                    continue

                ops = layer.get_num_ops()

                wprec = layer.wprec
                iprec = layer.iprec

                perf_factor = float(max_weight_prec * max_input_prec) / (max(min_weight_prec, wprec) * max(min_input_prec, iprec))
                cycles = ops / float(perf_factor)

                time_in_sec = clock_delay * 1e-12 * cycles
                static_energy = leakage_power_nW * 1e-9 * time_in_sec
                dynamic_energy = dynamic_power_nW * 1e-9 * time_in_sec

                max_precision = max(max_weight_prec, max_input_prec)
                min_precision = min(min_weight_prec, min_input_prec)

                total_energy = static_energy + dynamic_energy

                results_dict = {}
                results_dict['Layer'] = l
                results_dict['Multiplier_type'] = mtype
                results_dict['Max Precision'] = max(max_weight_prec, max_input_prec)
                results_dict['Min Precision'] = min(min_weight_prec, min_input_prec)
                results_dict['Frequency'] = 1.e-12 / clock_delay
                results_dict['Area (um^2)'] = area_um2
                results_dict['Neural Network'] = b

                results_dict['Cycles'] = cycles
                results_dict['Static Energy (nJ)'] = static_energy
                results_dict['Dynamic Energy (nJ)'] = dynamic_energy
                results_dict['Energy (nJ)'] = total_energy

                results.append(results_dict)

                bench_time += time_in_sec
                bench_cycles += cycles
                bench_static_energy += static_energy
                bench_dynamic_energy += dynamic_energy

            bench_total_energy = bench_static_energy + bench_dynamic_energy

            results_dict = {}

            results_dict['Multiplier_type'] = mtype
            results_dict['Max Precision'] = max(max_weight_prec, max_input_prec)
            results_dict['Min Precision'] = min(min_weight_prec, min_input_prec)
            results_dict['Frequency'] = 1.e-12 / clock_delay
            results_dict['Area (um^2)'] = area_um2
            results_dict['Neural Network'] = b

            results_dict['Layer'] = 'total'
            results_dict['Cycles'] = bench_cycles
            results_dict['Static Energy (nJ)'] = bench_static_energy
            results_dict['Dynamic Energy (nJ)'] = bench_dynamic_energy
            results_dict['Energy (nJ)'] = bench_total_energy

            results.append(results_dict)

    results = pandas.DataFrame(results, columns=result_columns)
    return results

def plot_ideal_benefits_variable_precision(
        frequency,
        temporal_csv_name, temporal_mult_type,
        temporal_dict,
        spatial_csv_name, spatial_mult_type,
        results_csv):
    if not os.path.exists(results_csv):
        spatial_mult_results = ideal_benefits_variable_precision(
            mult_type=spatial_mult_type, frequency=frequency,
            csv_file_name=spatial_csv_name)

        temporal_mult_results = ideal_benefits_variable_precision(
            mult_type=temporal_mult_type,
            frequency=frequency,
            csv_file_name=temporal_csv_name,
            default_dict=temporal_dict)
        results = pandas.concat([spatial_mult_results, temporal_mult_results])

        mult_type = spatial_mult_type + temporal_mult_type

        baseline_mult = '8:8 mult'
        max_ppa = 0
        max_ppw = 0

        plot_columns = ['Benchmark', 'Multiplier', 'Area-normalized-Performance', 'Power-normalized-Performance']
        plot_results = []

        for m in mult_type:
            ppa_list = []
            ppw_list = []
            for b in benchmarks.benchlist:
                lookup_dict = {'Multiplier_type': m,
                               'Neural Network': b,
                               'Layer': 'total'}
                baseline_lookup_dict = {'Multiplier_type': baseline_mult,
                                        'Neural Network': b,
                                        'Layer': 'total'}
                curr_results = lookup_pandas_dataframe(results, lookup_dict)
                baseline_results = lookup_pandas_dataframe(results, baseline_lookup_dict)

                time_baseline = int(baseline_results['Cycles'])
                time_current = int(curr_results['Cycles'])
                performance = time_baseline / float(time_current)

                area_baseline = float(baseline_results['Area (um^2)'])
                area_current = float(curr_results['Area (um^2)'])

                energy_baseline = float(baseline_results['Energy (nJ)'])
                energy_current = float(curr_results['Energy (nJ)'])

                power_baseline = energy_baseline / float(time_baseline)
                power_current = energy_current / float(time_current)

                ppa = performance * area_baseline / area_current
                ppw = performance * power_baseline / power_current

                ppa_list.append(ppa)
                ppw_list.append(ppw)

                results_dict = {
                    'Benchmark': b,
                    'Multiplier': m,
                    'Area-normalized-Performance': ppa,
                    'Power-normalized-Performance': ppw}
                plot_results.append(results_dict)

            results_dict = {
                'Benchmark': 'Gmean',
                'Multiplier': m,
                'Area-normalized-Performance': scipy.stats.gmean(ppa_list),
                'Power-normalized-Performance': scipy.stats.gmean(ppw_list)}
            plot_results.append(results_dict)

        plot_dataframe = pandas.DataFrame(plot_results, columns=plot_columns)
    else:
        plot_dataframe = pandas.read_csv(results_csv)

    bc1 = BarChart()
    bc1.yaxis = 'Area-normalized-Performance'
    bc1.BENCH_NEWLINE = False
    bc1.BOTTOM_MARGIN = 0.4
    bc1.ISTIMES = True
    bc1.TOP_ROTATE = True
    bc1.LEGEND_LOCATION = 0
    bc1.XAXIS_LABEL_ROTATE = 20
    bench = []
    for b in benchmarks.benchlist:
        bench.append("\sf{" + b + "}")
    bench.append("\sf{Gmean}")
    pdf_name = os.path.join(fig_dir, 'area-normalized-performance.pdf')
    bc1.plot(pdf_name, plot_dataframe, x_plot='Benchmark', legends='Multiplier', y_plot='Area-normalized-Performance')

    bc1.yaxis = 'Power-normalized-Performance'
    pdf_name = os.path.join(fig_dir, 'power-normalized-performance.pdf')
    bc1.plot(pdf_name, plot_dataframe, x_plot='Benchmark', legends='Multiplier', y_plot='Power-normalized-Performance')

