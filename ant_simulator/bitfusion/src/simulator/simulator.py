import logging
import math
import configparser
import numpy as np

from bitfusion.src.utils.utils import ceil_a_by_b, log2, lookup_pandas_dataframe
from bitfusion.src.simulator.stats import Stats
from bitfusion.src.simulator.loop_stack import LoopStack
from bitfusion.src.optimizer.optimizer import optimize_for_order, get_stats_fast
from bitfusion.src.simulator.accelerator import Accelerator
from bitfusion.src.simulator.energy import EnergyTuple

from bitfusion.sram.cacti_sweep import CactiSweep
import os
import pandas

from dnnweaver2.tensorOps.cnn import Convolution, MatMul

class Simulator(object):
    """
    Simulator class
    """

    def __init__(self, config_file='conf.ini', verbose=False, energy_costs=None):

        # custom energy cost
        self.energy_costs = energy_costs

        self.config_file = config_file

        self.config = configparser.ConfigParser()
        self.config.read(config_file)

        systolic_dim = [self.config.getint('accelerator', 'a'),
                             1,
                             self.config.getint('accelerator', 'c')]

        if verbose:
            log_level = logging.DEBUG
        else:
            log_level = logging.INFO

        # logging.basicConfig(level=log_level)
        self.logger = logging.getLogger('{}.{}'.format(__name__, 'Simulator'))
        self.logger.setLevel(log_level)
        self.logger.debug("Creating Simulator Object")
        self.logger.debug("Systolic Array dimentions: {}".format(systolic_dim))

        mem_if_width = self.config.getint('system', 'if_width')
        self.logger.debug("Memory Interface Bit-Width: {}-bits".format(mem_if_width))

        pmax = self.config.getint('accelerator', 'high_prec')
        pmin = self.config.getint('accelerator', 'low_prec')
        self.logger.debug("High Precision: {}-bits".format(pmax))
        self.logger.debug("Low Precision: {}-bits".format(pmin))

        # Using half the size assuming double buffering
        sram = {}

        sram['act'] = self.config.getint('accelerator', 'Act_SRAM')
        self.logger.debug("Activation SRAM size: {:,} Bytes".format(sram['act']))

        sram['wgt'] = self.config.getint('accelerator', 'Wgt_SRAM')
        self.logger.debug("Weight SRAM size: {:,} Bytes".format(sram['wgt']))

        sram['out'] = self.config.getint('accelerator', 'Out_SRAM')
        self.logger.debug("Output SRAM size: {:,} Bytes".format(sram['out']))

        frequency = self.config.getint('accelerator', 'frequency')
        self.logger.debug('Frequency: {:,} Hz'.format(frequency))

        hp_peak_throughput = systolic_dim[0] * \
                             systolic_dim[1] * \
                             systolic_dim[2]
        peak_throughput = hp_peak_throughput * \
                               (int(pmax / pmin) ** 2)
        self.logger.debug('Lowest  precision: Peak Throughput: {:,} Ops/cycle'.format(peak_throughput))
        self.logger.debug('Highest precision: Peak Throughput: {:,} Ops/cycle'.format(hp_peak_throughput))

        N = systolic_dim[0]
        beta = systolic_dim[1]
        M = systolic_dim[2]

        assert beta == 1

        self.accelerator = Accelerator(N, M, pmax, pmin, sram, mem_if_width, frequency)

        ##################################################
        # Get stats for SRAM
        frequency = self.accelerator.frequency
        tech_node = 28
        sram_csv = 'hardware_sweep/sram_results.csv'
        sram_opt_dict = {'technology (u)': tech_node*1.e-3}
        dir_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../sram')
        self.sram_obj = CactiSweep(
                bin_file=os.path.join(dir_path, 'cacti/cacti'),
                csv_file=os.path.join(dir_path, 'cacti_sweep.csv'),
                default_json=os.path.join(dir_path, 'default.json'),
                default_dict=sram_opt_dict)

    def get_area(self):
        frequency = self.accelerator.frequency
        ##################################################
        N = self.accelerator.N
        M = self.accelerator.M
        pmax = self.accelerator.pmax
        pmin = self.accelerator.pmin
        wbuf_size = self.accelerator.sram['wgt'] * 8
        ibuf_size = self.accelerator.sram['act'] * 8
        obuf_size = self.accelerator.sram['out'] * 8
        wbuf_bank = N
        ibuf_bank = N
        obuf_bank = N * M
        wbuf_bits = (pmax * pmax / pmin) * M
        ibuf_bits = (pmax * pmax / pmin) * M
        obuf_bits = 32 
        wbuf_word = ceil_a_by_b(wbuf_size, wbuf_bank * wbuf_bits)
        ibuf_word = ceil_a_by_b(ibuf_size, ibuf_bank * ibuf_bits)
        obuf_word = ceil_a_by_b(obuf_size, obuf_bank * obuf_bits)
        wbuf_bank_size = wbuf_word * wbuf_bits
        ibuf_bank_size = ibuf_word * ibuf_bits
        obuf_bank_size = obuf_word * obuf_bits

        assert wbuf_bank_size * wbuf_bank == wbuf_size
        assert ibuf_bank_size * ibuf_bank == ibuf_size
        assert obuf_bank_size * obuf_bank == obuf_size


        ##################################################
        cfg_dict = {'size (bytes)': wbuf_bank_size /8., 'block size (bytes)': wbuf_bits/8., 'read-write port': 0}
        wbuf_data = self.sram_obj.get_data_clean(cfg_dict)
        wbuf_read_energy = float(wbuf_data['read_energy_nJ'].iloc[0]) / wbuf_bits
        wbuf_write_energy = float(wbuf_data['write_energy_nJ'].iloc[0]) / wbuf_bits
        wbuf_leak_power = float(wbuf_data['leak_power_mW'].iloc[0]) * wbuf_bank
        wbuf_area = float(wbuf_data['area_mm^2'].iloc[0]) * wbuf_bank

        self.logger.debug('WBUF :')
        self.logger.debug('\tBanks                       : {0:>8}'.format(wbuf_bank))
        self.logger.debug('\tBitWidth                    : {0:>8} bits'.format(wbuf_bits))
        self.logger.debug('\tWords                       : {0:>8}'.format(wbuf_word))
        self.logger.debug('\tTotal Size                  : {0:>8} kBytes'.format(wbuf_size/8./1024.))
        self.logger.debug('\tTotal Area                  : {0:>8.2f} mm^2'.format(wbuf_area))
        self.logger.debug('\tLeak Energy (per clock)     : {0:>8.4f} mWatt'.format(wbuf_leak_power))
        self.logger.debug('\tRead Energy                 : {0:>8.4f} pJ/bit'.format(wbuf_read_energy * 1.e3))
        self.logger.debug('\tWrite Energy                : {0:>8.4f} pJ/bit'.format(wbuf_write_energy * 1.e3))
        ##################################################
        cfg_dict = {'size (bytes)': ibuf_bank_size /8., 'block size (bytes)': ibuf_bits/8., 'read-write port': 0}
        ibuf_data = self.sram_obj.get_data_clean(cfg_dict)
        ibuf_read_energy = float(ibuf_data['read_energy_nJ'].iloc[0]) / ibuf_bits
        ibuf_write_energy = float(ibuf_data['write_energy_nJ'].iloc[0]) / ibuf_bits
        ibuf_leak_power = float(ibuf_data['leak_power_mW'].iloc[0]) * ibuf_bank
        ibuf_area = float(ibuf_data['area_mm^2'].iloc[0]) * ibuf_bank

        self.logger.debug('IBUF :')
        self.logger.debug('\tBanks                       : {0:>8}'.format(ibuf_bank))
        self.logger.debug('\tBitWidth                    : {0:>8} bits'.format(ibuf_bits))
        self.logger.debug('\tWords                       : {0:>8}'.format(ibuf_word))
        self.logger.debug('\tTotal Size                  : {0:>8} kBytes'.format(ibuf_size/8./1024.))
        self.logger.debug('\tTotal Area                  : {0:>8.2f} mm^2'.format(ibuf_area))
        self.logger.debug('\tLeak Energy (per clock)     : {0:>8.4f} mWatt'.format(ibuf_leak_power))
        self.logger.debug('\tRead Energy                 : {0:>8.4f} pJ/bit'.format(ibuf_read_energy * 1.e3))
        self.logger.debug('\tWrite Energy                : {0:>8.4f} pJ/bit'.format(ibuf_write_energy * 1.e3))
        ##################################################
        cfg_dict = {'size (bytes)': obuf_bank_size /8., 'block size (bytes)': obuf_bits/8., 'read-write port': 1}
        obuf_data = self.sram_obj.get_data_clean(cfg_dict)
        obuf_read_energy = float(obuf_data['read_energy_nJ'].iloc[0]) / obuf_bits
        obuf_write_energy = float(obuf_data['write_energy_nJ'].iloc[0]) / obuf_bits
        obuf_leak_power = float(obuf_data['leak_power_mW'].iloc[0]) * obuf_bank
        obuf_area = float(obuf_data['area_mm^2'].iloc[0]) * obuf_bank

        self.logger.debug('OBUF :')
        self.logger.debug('\tBanks                       : {0:>8}'.format(obuf_bank))
        self.logger.debug('\tBitWidth                    : {0:>8} bits'.format(obuf_bits))
        self.logger.debug('\tWords                       : {0:>8}'.format(obuf_word))
        self.logger.debug('\tTotal Size                  : {0:>8} kBytes'.format(obuf_size/8./1024.))
        self.logger.debug('\tTotal Area                  : {0:>8.2f} mm^2'.format(obuf_area))
        self.logger.debug('\tLeak Energy (per clock)     : {0:>8.4f} mWatt'.format(obuf_leak_power))
        self.logger.debug('\tRead Energy                 : {0:>8.4f} pJ/bit'.format(obuf_read_energy * 1.e3))
        self.logger.debug('\tWrite Energy                : {0:>8.4f} pJ/bit'.format(obuf_write_energy * 1.e3))
        ##################################################
        # Get stats for systolic array
        core_csv = os.path.join('./results', 'systolic_array_synth.csv')
        core_synth_data = pandas.read_csv(core_csv)

        lookup_dict = {}
        lookup_dict['Max Precision (bits)'] = pmax
        lookup_dict['Min Precision (bits)'] = pmin
        lookup_dict['N'] = N
        lookup_dict['M'] = M
        core_data = lookup_pandas_dataframe(core_synth_data, lookup_dict)
        if len(core_data) == 0:
            lookup_dict['N'] = 1
            lookup_dict['M'] = 1
            core_data = lookup_pandas_dataframe(core_synth_data, lookup_dict)
            assert len(core_data) == 1
            core_area = float(core_data['Area (um^2)'].iloc[0]) * 1.e-6 * (N * M) / 1.
            core_dyn_power = float(core_data['Dynamic Power (nW)'].iloc[0]) * (N * M) / 1.
            core_dyn_energy = core_dyn_power / float(core_data['Frequency'].iloc[0])
            core_leak_power = float(core_data['Leakage Power (nW)'].iloc[0]) * (N * M) / 1.
            core_leak_energy = core_leak_power / float(core_data['Frequency'].iloc[0])
        else:
            core_area = float(core_data['Area (um^2)'].iloc[0]) * 1.e-6
            core_dyn_power = float(core_data['Dynamic Power (nW)'].iloc[0])
            core_dyn_energy = core_dyn_power / float(core_data['Frequency'].iloc[0])
            core_leak_power = float(core_data['Leakage Power (nW)'].iloc[0])
            core_leak_energy = core_leak_power / float(core_data['Frequency'].iloc[0])
        self.logger.debug('Core :')
        self.logger.debug('\tDimensions              : {0}x{1}-systolic array'.format(N, M))
        self.logger.debug('\tMax-Precision           : {}'.format(pmax))
        self.logger.debug('\tMin-Precision           : {}'.format(pmin))
        self.logger.debug('\tLeak power              : {} (nW)'.format(core_leak_energy))
        self.logger.debug('\tDynamic Energy (nJ)     : {}'.format(core_dyn_energy))
        self.logger.debug('\tArea (mm^2)             : {}'.format(core_area))
        ##################################################

        return core_area, wbuf_area, ibuf_area, obuf_area

    def get_energy_cost(self):

        if self.energy_costs is not None:
            return self.energy_costs

        frequency = self.accelerator.frequency
        ##################################################
        N = self.accelerator.N
        M = self.accelerator.M
        pmax = self.accelerator.pmax
        pmin = self.accelerator.pmin
        wbuf_size = self.accelerator.sram['wgt'] * 8
        ibuf_size = self.accelerator.sram['act'] * 8
        obuf_size = self.accelerator.sram['out'] * 8
        wbuf_bank = 32
        ibuf_bank = 32
        obuf_bank = 32 * 32
        # wbuf_bits = (pmax * pmax / pmin) * 32
        # ibuf_bits = (pmax * pmax / pmin) * 32
        wbuf_bits = 16 * 32
        ibuf_bits = 16 * 32
        obuf_bits = 32 
        wbuf_word = ceil_a_by_b(wbuf_size, wbuf_bank * wbuf_bits)
        ibuf_word = ceil_a_by_b(ibuf_size, ibuf_bank * ibuf_bits)
        obuf_word = ceil_a_by_b(obuf_size, obuf_bank * obuf_bits)
        wbuf_bank_size = wbuf_word * wbuf_bits
        ibuf_bank_size = ibuf_word * ibuf_bits
        obuf_bank_size = obuf_word * obuf_bits

        assert wbuf_bank_size * wbuf_bank == wbuf_size
        assert ibuf_bank_size * ibuf_bank == ibuf_size
        assert obuf_bank_size * obuf_bank == obuf_size


        ##################################################
        cfg_dict = {'size (bytes)': wbuf_bank_size /8., 'block size (bytes)': wbuf_bits/8., 'read-write port': 0}
        
        wbuf_data = self.sram_obj.get_data_clean(cfg_dict)
        wbuf_read_energy = float(wbuf_data['read_energy_nJ'].iloc[0]) / wbuf_bits
        wbuf_write_energy = float(wbuf_data['write_energy_nJ'].iloc[0]) / wbuf_bits
        wbuf_leak_power = float(wbuf_data['leak_power_mW'].iloc[0]) * wbuf_bank
        wbuf_area = float(wbuf_data['area_mm^2'].iloc[0]) * wbuf_bank

        self.logger.debug('WBUF :')
        self.logger.debug('\tBanks                       : {0:>8}'.format(wbuf_bank))
        self.logger.debug('\tBitWidth                    : {0:>8} bits'.format(wbuf_bits))
        self.logger.debug('\tWords                       : {0:>8}'.format(wbuf_word))
        self.logger.debug('\tTotal Size                  : {0:>8} kBytes'.format(wbuf_size/8./1024.))
        self.logger.debug('\tTotal Area                  : {0:>8.2f} mm^2'.format(wbuf_area))
        self.logger.debug('\tLeak Energy (per clock)     : {0:>8.4f} mWatt'.format(wbuf_leak_power))
        self.logger.debug('\tRead Energy                 : {0:>8.4f} pJ/bit'.format(wbuf_read_energy * 1.e3))
        self.logger.debug('\tWrite Energy                : {0:>8.4f} pJ/bit'.format(wbuf_write_energy * 1.e3))
        ##################################################
        cfg_dict = {'size (bytes)': ibuf_bank_size /8., 'block size (bytes)': ibuf_bits/8., 'read-write port': 0}
        ibuf_data = self.sram_obj.get_data_clean(cfg_dict)
        ibuf_read_energy = float(ibuf_data['read_energy_nJ'].iloc[0]) / ibuf_bits
        ibuf_write_energy = float(ibuf_data['write_energy_nJ'].iloc[0]) / ibuf_bits
        ibuf_leak_power = float(ibuf_data['leak_power_mW'].iloc[0]) * ibuf_bank
        ibuf_area = float(ibuf_data['area_mm^2'].iloc[0]) * ibuf_bank

        self.logger.debug('IBUF :')
        self.logger.debug('\tBanks                       : {0:>8}'.format(ibuf_bank))
        self.logger.debug('\tBitWidth                    : {0:>8} bits'.format(ibuf_bits))
        self.logger.debug('\tWords                       : {0:>8}'.format(ibuf_word))
        self.logger.debug('\tTotal Size                  : {0:>8} kBytes'.format(ibuf_size/8./1024.))
        self.logger.debug('\tTotal Area                  : {0:>8.2f} mm^2'.format(ibuf_area))
        self.logger.debug('\tLeak Energy (per clock)     : {0:>8.4f} mWatt'.format(ibuf_leak_power))
        self.logger.debug('\tRead Energy                 : {0:>8.4f} pJ/bit'.format(ibuf_read_energy * 1.e3))
        self.logger.debug('\tWrite Energy                : {0:>8.4f} pJ/bit'.format(ibuf_write_energy * 1.e3))
        ##################################################
        cfg_dict = {'size (bytes)': obuf_bank_size /8., 'block size (bytes)': obuf_bits/8., 'read-write port': 1}
        obuf_data = self.sram_obj.get_data_clean(cfg_dict)
        obuf_read_energy = float(obuf_data['read_energy_nJ'].iloc[0]) / obuf_bits
        obuf_write_energy = float(obuf_data['write_energy_nJ'].iloc[0]) / obuf_bits
        obuf_leak_power = float(obuf_data['leak_power_mW'].iloc[0]) * obuf_bank
        obuf_area = float(obuf_data['area_mm^2'].iloc[0]) * obuf_bank

        self.logger.debug('OBUF :')
        self.logger.debug('\tBanks                       : {0:>8}'.format(obuf_bank))
        self.logger.debug('\tBitWidth                    : {0:>8} bits'.format(obuf_bits))
        self.logger.debug('\tWords                       : {0:>8}'.format(obuf_word))
        self.logger.debug('\tTotal Size                  : {0:>8} kBytes'.format(obuf_size/8./1024.))
        self.logger.debug('\tTotal Area                  : {0:>8.2f} mm^2'.format(obuf_area))
        self.logger.debug('\tLeak Energy (per clock)     : {0:>8.4f} mWatt'.format(obuf_leak_power))
        self.logger.debug('\tRead Energy                 : {0:>8.4f} pJ/bit'.format(obuf_read_energy * 1.e3))
        self.logger.debug('\tWrite Energy                : {0:>8.4f} pJ/bit'.format(obuf_write_energy * 1.e3))
        ##################################################
        # Get stats for systolic array
        core_csv = os.path.join('./results', 'systolic_array_synth.csv')
        core_synth_data = pandas.read_csv(core_csv)

        lookup_dict = {}
        lookup_dict['Max Precision (bits)'] = pmax
        lookup_dict['Min Precision (bits)'] = pmin
        lookup_dict['N'] = N
        lookup_dict['M'] = M
        core_data = lookup_pandas_dataframe(core_synth_data, lookup_dict)
        if len(core_data) == 0:
            lookup_dict['N'] = 1
            lookup_dict['M'] = 1
            core_data = lookup_pandas_dataframe(core_synth_data, lookup_dict)
            assert len(core_data) == 1
            core_area = float(core_data['Area (um^2)'].iloc[0]) * 1.e-6 * (N * M) / 1.
            core_dyn_power = float(core_data['Dynamic Power (nW)'].iloc[0]) * (N * M) / 1.
            core_dyn_energy = core_dyn_power / float(core_data['Frequency'].iloc[0])
            core_leak_power = float(core_data['Leakage Power (nW)'].iloc[0]) * (N * M) / 1.
            core_leak_energy = core_leak_power / float(core_data['Frequency'].iloc[0])
        else:
            core_area = float(core_data['Area (um^2)'].iloc[0]) * 1.e-6
            core_dyn_power = float(core_data['Dynamic Power (nW)'].iloc[0])
            core_dyn_energy = core_dyn_power / float(core_data['Frequency'].iloc[0])
            core_leak_power = float(core_data['Leakage Power (nW)'].iloc[0])
            core_leak_energy = core_leak_power / float(core_data['Frequency'].iloc[0])
        self.logger.debug('Core :')
        self.logger.debug('\tDimensions              : {0}x{1}-systolic array'.format(N, M))
        self.logger.debug('\tMax-Precision           : {}'.format(pmax))
        self.logger.debug('\tMin-Precision           : {}'.format(pmin))
        self.logger.debug('\tLeak power              : {} (nW)'.format(core_leak_energy))
        self.logger.debug('\tDynamic Energy (nJ)     : {}'.format(core_dyn_energy))
        self.logger.debug('\tArea (mm^2)             : {}'.format(core_area))
        ##################################################

        sram_leak_energy = (wbuf_leak_power + ibuf_leak_power + obuf_leak_power) * 1e6 / float(core_data['Frequency'].iloc[0])
        energy_tuple = EnergyTuple(core_leak_energy, core_dyn_energy, wbuf_read_energy, wbuf_write_energy, ibuf_read_energy, ibuf_write_energy, obuf_read_energy, obuf_write_energy, sram_leak_energy)

        return energy_tuple


    def __str__(self):
        ret = ''
        ret += 'Simulator object'
        ret += '\n'
        ret += '\tMax supported precision: {}'.format(self.accelerator.pmax)
        ret += '\n'
        ret += '\tMin supported precision: {}'.format(self.accelerator.pmin)
        ret += '\n'
        ret += '\tSystolic array size: {} -inputs x {} -outputs'.format(
                self.accelerator.N,
                self.accelerator.M)

        ret += '\n'
        ret += '\tWbuf size: {:,} Bytes'.format(self.accelerator.sram['wgt'])
        ret += '\n'
        ret += '\tIbuf size: {:,} Bytes'.format(self.accelerator.sram['act'])
        ret += '\n'
        ret += '\tObuf size: {:,} Bytes'.format(self.accelerator.sram['out'])
        ret += '\n'
        ret += 'Double buffering enabled. Sizes of SRAM are halved'
        return ret

    def loop_estimate_stats(self, loop_instruction, verbose=False):
        """
        args:
            loop_instruction: Loops for the NN.
                index 0 = outer loop
                index -1 = inner loop
        """

        # The following loop promotes Memory accesses to improve reuse
        loop_instruction.promote_mem_ops(self.accelerator.sram)
        # get stats
        stats = loop_instruction.get_stats(self.accelerator, verbose)

        return stats

    def get_FC_cycles(self, Ni, No,
                      iprec, wprec,
                      batch_size=1,
                      weight_stationary = False):
        """
        Get number of cycles required for Fully-Connected Layer.

        args:
            Ni: Input neurons
            No: Output neurons
            batch_size: Batch size for FC layer
            iprec: Precision for activations (bits)
            wprec: Precision for weights (bits)
            batch_size: Batch size for the layer

        description:
            This function calls the get_conv_cycles function
        """
        total_cycles = self.get_conv_cycles(1, 1, 1, Ni, No, iprec, wprec, batch_size, im2col=True, weight_stationary = weight_stationary)

        return total_cycles


    def get_perf_factor(self, iprec, wprec):
        iprec = max(iprec, self.accelerator.pmin)
        wprec = max(wprec, self.accelerator.pmin)
        return int(self.accelerator.pmax / iprec) * int(self.accelerator.pmax / wprec)

    def get_perf_factor_2(self, iprec, wprec):
        iprec = max(iprec, self.accelerator.pmin)
        wprec = max(wprec, self.accelerator.pmin)
        return int(self.accelerator.pmax / iprec), int(self.accelerator.pmax / wprec)

    def get_conv_cycles(self, K, O, S, IC, OC, iprec, wprec, batch_size=1, im2col=True, weight_stationary = False):
        """
        Get number of cycles required for Convolution layer.

        description:
            This functions does an exhaustive search for finding the optimal
            Tiling and Ordering parameters
        """
        B = batch_size
        I = (O - 1) * S + K

        # We do not tile the "K" dimension and compute an entire 2-D conv at a
        # time
        num_O_tiles = int(math.ceil(log2(O))) + 1
        num_IC_tiles = int(math.ceil(log2(IC))) + 1
        num_OC_tiles = int(math.ceil(log2(math.ceil(float(OC)/self.accelerator.M)))) + 1
        num_B_tiles = int(math.ceil(log2(B))) + 1

        # print("*"*30)
        # print(num_O_tiles)
        # print(num_IC_tiles)
        # print(num_OC_tiles)
        # print(num_B_tiles)

        self.logger.debug('Number of O Tiles: {}'.format(num_O_tiles))
        self.logger.debug('Number of IC Tiles: {}'.format(num_IC_tiles))
        self.logger.debug('Number of OC Tiles: {}'.format(num_OC_tiles))
        self.logger.debug('Number of B Tiles: {}'.format(num_B_tiles))

        best_instructions_dict = {}
        conv_params = self.accelerator, K, O, S, IC, OC, B, iprec, wprec, im2col, weight_stationary, self.get_energy_cost()

        best_instructions, best_tiling, best_order = optimize_for_order(conv_params)
        # print(best_instructions)
        # print(best_tiling)
        # print(best_order)


        stats = get_stats_fast(conv_params, best_tiling, best_order, verbose=False)

        act_reads = stats.reads['act']
        wgt_reads = stats.reads['wgt']
        out_reads = stats.reads['out']
        dram_reads = stats.reads['dram']
        out_writes = stats.writes['out']
        dram_writes = stats.writes['dram']
        best_cycles = stats.total_cycles

        num_ops = O * O * K * K * IC * OC * B

        # self.logger.debug('Best Operations: {}'.format(best_operations))

        self.logger.debug('Conv Layer')
        self.logger.debug('Num of ops: {}'.format(num_ops))
        self.logger.debug('Kernel Size: {}x{}x{}x{}'.format(K, K, IC, OC))
        self.logger.debug('Output Size: {}x{}x{}'.format(O, O, OC))
        self.logger.debug('Stride Size: {}x{}'.format(S, S))
        self.logger.debug('Input  Size: {}x{}x{}'.format(I, I, IC))

        self.logger.debug('Max Precision: {}'.format(self.accelerator.pmax))
        self.logger.debug('Min Precision: {}'.format(self.accelerator.pmin))

        self.logger.debug('Activation Precision: {}'.format(iprec))
        self.logger.debug('Weight Precision: {}'.format(wprec))
        self.logger.debug('Performance Factor: {}'.format(self.get_perf_factor(iprec, wprec)))

        self.logger.debug('Total Cycles: {:,}'.format(best_cycles))
        cycles_per_batch = ceil_a_by_b(best_cycles, B)
        self.logger.debug('Total Cycles per batch: {:,}'.format(cycles_per_batch))
        ops_per_cycle = float(num_ops) / best_cycles
        self.logger.debug('Ops/Cycle: {:,.2f}'.format(ops_per_cycle))
        ops_per_cycle_per_pe = float(ops_per_cycle) / (self.accelerator.N * self.accelerator.M)
        self.logger.debug('Ops/Cycle/PE: {:,.4}'.format(ops_per_cycle_per_pe))

        return stats, best_instructions

    def get_cycles(self, op, im2col=True, weight_stationary = False):
        if isinstance(op, Convolution):
            B, I, _, IC = op.data.shape
            _, O, _, OC = op.output_tensors.shape
            _, K, _, _  = op.weights.shape
            _, S, _, _  = op.stride

            iprec = op.data.dtype.bits
            wprec = op.weights.dtype.bits
            # if op.data.op is None:
            #     im2col = True # im2col for first layer
            # else:
            #     im2col = False
            im2col = True
            return self.get_conv_cycles(K,
                                        O,
                                        S,
                                        IC,
                                        OC,
                                        iprec,
                                        wprec,
                                        B,
                                        im2col,
                                        weight_stationary)
        elif isinstance(op, MatMul):
            B = op.data.shape[0]
            OC, IC  = op.weights.shape
            iprec = op.data.dtype.bits
            wprec = op.weights.dtype.bits
            return self.get_FC_cycles(IC, OC, iprec, wprec, batch_size=B, weight_stationary = weight_stationary)
