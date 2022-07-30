import math
import functools
import time
import logging

from itertools import permutations
from multiprocessing import Pool, cpu_count

from bitfusion.src.utils.utils import ceil_a_by_b, log2
from bitfusion.src.simulator.loop_stack import LoopStack
from bitfusion.src.simulator.stats import Stats

import numpy as np

logger = logging.getLogger('{}.{}'.format(__name__, 'Optimizer'))
logger.setLevel(logging.DEBUG)

tile_deps = {}
tile_deps['B/b']   = {'act': True, 'wgt': False, 'out': True}
tile_deps['OW/ow'] = {'act': True, 'wgt': False, 'out': True}
tile_deps['OH/oh'] = {'act': True, 'wgt': False, 'out': True}
tile_deps['IC/ic'] = {'act': True, 'wgt': True, 'out': False}
tile_deps['OC/oc'] = {'act': False, 'wgt': True, 'out': True}

# inner_loop = {}
# inner_loop['b']  = {'act': True, 'wgt': False, 'out': True}
# inner_loop['ow'] = {'act': True, 'wgt': False, 'out': True}
# inner_loop['oh'] = {'act': True, 'wgt': False, 'out': True}
# inner_loop['ic'] = {'act': True, 'wgt': True, 'out': False}
# inner_loop['oc'] = {'act': False, 'wgt': True, 'out': True}
# inner_loop['kh'] = {'act': True, 'wgt': True, 'out': False}
# inner_loop['kw'] = {'act': True, 'wgt': True, 'out': False}

def get_stats_fast(conv_params, tiling, order_type, verbose=False):
    """
    Returns cycles and memory accesses to DRAM, IBUF, OBUF, and WBUF
        TODOs: Without im2col, the calculation of weight and act size is inexact
    """
    acc_obj, K, O, S, IC, OC, B, iprec, wprec, im2col, weight_stationary, energy_cost = conv_params

    num_b, b = tiling['B/b']
    num_ow, ow = tiling['OW/ow']
    num_oh, oh = tiling['OH/oh']
    num_ic, ic = tiling['IC/ic']
    num_oc, oc = tiling['OC/oc']

    kw = kh = K
    
    # perf_factor = acc_obj.get_perf_factor(iprec, wprec)       
        
    writes = {}
    reads = {}

    writes['wgt'] = \
            ceil_a_by_b(K * K * ic, acc_obj.N * acc_obj.get_perf_factor(wprec)) * acc_obj.N * acc_obj.get_perf_factor(wprec) * \
            oc * \
            wprec

    writes['act'] = ow * oh * \
            ceil_a_by_b(K * K * ic, acc_obj.M * acc_obj.get_perf_factor(iprec)) * acc_obj.M * acc_obj.get_perf_factor(iprec) * \
            b * iprec

    oprec = 16
    writes['out'] = 0# ow * oh * oc * b * oprec
    reads['out'] = ow * oh * oc * b * oprec

    # Skip if overutilizing resources
    # TODO check bytes/bits
    overflow = False
    if writes['wgt'] > acc_obj.sram['wgt']*8/2:
        if verbose:
            print('wgt overflow: {}'.format(writes['wgt']))
            print(b, ow, oh, ic, oc)
        overflow = True
    if writes['act'] > acc_obj.sram['act']*8/2:
        if verbose:
            print('act overflow')
            print(b, ow, oh, ic, oc)
        overflow = True
    if writes['out'] > acc_obj.sram['out']*8/2:
        if verbose:
            print('out overflow')
            print(b, ow, oh, ic, oc)
        overflow = True
    if overflow:
        if verbose:
            print('Activation size: {} bytes'.format(writes['act']/8.))
            print('Weights size: {} bytes'.format(writes['wgt']/8.))
            print('Output size: {} bytes'.format(writes['out']/8.))
        return

    max_write_size = {}
    max_read_size = {}
    for namespace in writes:
        max_write_size[namespace] = writes[namespace]
    for namespace in reads:
        max_read_size[namespace] = reads[namespace]

    # First the loop block optimizations
    stats = Stats()
    write_promote = {'wgt': True, 'act': True, 'out': True}
    read_promote = {'out': True}
    if verbose:
        logger.debug('Initialize reads/writes')
        logger.debug('\tim2col: {}'.format(im2col))
        logger.debug('\tTiling: {}'.format(tiling))
        logger.debug('\tReads : {}'.format(reads))
        logger.debug('\tWrites: {}'.format(writes))
    for loop in reversed(order_type):
        num_tiles, tile_size = tiling[loop]
        # promote all writes
        for namespace in writes:
            # promote is true
            if write_promote[namespace]:
                # If tile loop depends on the namespace index, make the read size larger
                if tile_deps[loop][namespace]:
                    writes[namespace] *= num_tiles
                    # If tile size is larger than the SRAM, set promote to False
                    if writes[namespace] > acc_obj.sram[namespace]*8./2:
                        write_promote[namespace] = False
                    else:
                        max_write_size[namespace] = writes[namespace]
            else:
                writes[namespace] *= num_tiles

        # promote all reads
        for namespace in reads:
            # promote is true
            if read_promote[namespace]:
                # Tile loop depends on the namespace index
                if tile_deps[loop][namespace]:
                    reads[namespace] *= num_tiles
                    # Tile size is now larger than the SRAM, set promote to False
                    if reads[namespace] > acc_obj.sram[namespace]*8./2:
                        read_promote[namespace] = False
                    else:
                        max_read_size[namespace] = writes[namespace]
            else:
                reads[namespace] *= num_tiles

        if verbose:
            logger.debug('Loop: {}'.format(loop))
            logger.debug('\tLoop range: {}'.format(tiling[loop]))
            logger.debug('\tMax write size: {}'.format(max_write_size))
            logger.debug('\tMax read size: {}'.format(max_read_size))
            logger.debug('\tLoop Dependencies: {}'.format(tile_deps[loop]))
            logger.debug('\tLoop Promote: {}'.format(write_promote))
            logger.debug('\tReads : {}'.format(reads))
            logger.debug('\tWrites: {}'.format(writes))


    for namespace in writes:
        stats.writes[namespace] = writes[namespace]
        stats.reads['dram'] += writes[namespace]
    for namespace in reads:
        stats.reads[namespace] = reads[namespace]
        stats.writes['dram'] += reads[namespace]

    # Next the inner loop optimizations
    num_tiles = num_b * num_ow * num_oh * num_ic * num_oc

    if weight_stationary:
        if verbose:
            logger.debug('SRAM access order: Weight Stationary')
        stats.reads['act'] += num_tiles * (kw * kh * ic * oc) * (b * ow * oh) * iprec
        stats.reads['out'] += num_tiles * (kw * kh * ic * oc) * (b * ow * oh) * oprec
        stats.writes['out'] += num_tiles * (kw * kh * ic * oc) * (b * ow * oh) * oprec
        stats.reads['wgt'] += num_tiles * (kw * kh * ic * oc) * wprec
    else:
        if verbose:
            logger.debug('SRAM access order: Output Stationary')
        stats.reads['act'] += num_tiles * (oc * oh * ow * b) * (kw * kh * ic) * iprec
        stats.writes['out'] += num_tiles * (oc * oh * ow * b) * oprec
        stats.reads['wgt'] += num_tiles * (oc * oh * ow * b) * (kw * kh * ic) * wprec
        stats.reads['out'] += num_tiles * (oc * oh * ow * b) * oprec

    initial_dram_reads = 0
    final_dram_writes = 0
    for namespace in max_write_size:
        initial_dram_reads += max_write_size[namespace]
    for namespace in max_read_size:
        final_dram_writes += max_read_size[namespace]
    latency = acc_obj.get_mem_read_cycles('dram', initial_dram_reads) + \
            acc_obj.get_mem_write_cycles('dram', final_dram_writes)

    total_dram_accesses = stats.reads['dram'] + stats.writes['dram']
    middle_dram_accesses = total_dram_accesses - initial_dram_reads - final_dram_writes


    if weight_stationary:
        compute_cycles = num_tiles * acc_obj.get_compute_cycles(ic, oc, ow, oh, b, kw, kh, iprec, wprec, im2col)
    else:
        compute_cycles = num_tiles * acc_obj.get_compute_cycles_output_stationary(ic, oc, ow, oh, b, kw, kh, iprec, wprec, im2col)

    memory_cycles_required = ceil_a_by_b(middle_dram_accesses, acc_obj.mem_if_width)

    memory_stalls = max(0, memory_cycles_required - compute_cycles) + latency
    stats.total_cycles = compute_cycles + memory_stalls
    stats.mem_stall_cycles = memory_stalls

    if verbose:
        logger.debug('Compute cycles : {:>20,}'.format(compute_cycles))
        logger.debug('Memory cycles  : {:>20,}'.format(memory_cycles_required + latency))
        logger.debug('Memory stalls  : {:>20,}'.format(memory_stalls))

    return stats

def optimize_for_order(conv_params):
    # Generate permutations for the order
    loops = ['B/b', 'OW/ow', 'OH/oh', 'IC/ic', 'OC/oc']
    order = set(permutations(loops))

    return_dict = {}
    acc_obj, K, O, S, IC, OC, B, iprec, wprec, im2col, weight_stationary, energy_cost = conv_params

    _bound_optimizer_method = functools.partial(_optimize_for_order, conv_params)

    try:
        pool = Pool(cpu_count())
        results = pool.map_async(_bound_optimizer_method, order).get(10000)
        pool.close()
        pool.join()

        # results = []
        # for order_ in order:
        #     results.append(_optimize_for_order(conv_params, order_))


        # for o in order:
        #     _bound_optimizer_method(o)
        # exit()

        best_cycles = None
        best_energy = None
        
        ##  By GC
        # min_cycles = min([x[-4] for x in results])
        # min_energy = min([x[-3] for x in results])
        # cycles_list = [x[-2] for x in results]
        # energy_list = [x[-1] for x in results]
        for r in results:
            tiling, order_type, cycles, energy = r
            if best_cycles is None or best_cycles > cycles or (best_cycles == cycles and best_energy > energy):
                best_cycles = cycles
                best_energy = energy
                best_tiling = tiling
                best_order = order_type
        return get_loop_instructions(conv_params, best_tiling, best_order), best_tiling, best_order

    except KeyboardInterrupt:
        pool.terminate()
        pool.join()
        return


def get_loop_instructions(conv_params, tiling, order_type):
    acc_obj, K, O, S, IC, OC, B, iprec, wprec, im2col, weight_stationary, energy_cost = conv_params
    I = (O - 1) * S + K

    num_b, b = tiling['B/b']
    num_ow, ow = tiling['OW/ow']
    num_oh, oh = tiling['OH/oh']
    num_ic, ic = tiling['IC/ic']
    num_oc, oc = tiling['OC/oc']

    instructions = {}
    instructions['B/b'] = [num_b, I * I * IC * b, 0, O * O * OC * b]
    instructions['OW/ow'] = [num_ow, ow * S, 0, ow]
    instructions['OH/oh'] = [num_oh, I * S, 0, O]
    instructions['IC/ic'] = [num_ic, I * I * ic, K * K * ic, 0]
    instructions['OC/oc'] = [num_oc, 0, K * K * IC * oc, O * O * oc]

    instruction_ordered = LoopStack()
    wgt_stride = []
    act_stride = []
    out_stride = []
    count = 0
    for o in order_type:
        ins = instructions[o]
        if ins[0] > 1:
            stride = {'wgt': ins[2], 'act': ins[1], 'out': ins[3]}
            instruction_ordered.insert_loop(ins[0], stride=stride, level=count, name=o)
            wgt_stride.append(stride['wgt'])
            act_stride.append(stride['act'])
            out_stride.append(stride['out'])
            count += 1
    if count == 0:
        ins = instructions[o]
        stride = {'wgt': ins[2], 'act': ins[1], 'out': ins[3]}
        instruction_ordered.insert_loop(ins[0], stride=stride, level=count, name=o)
        wgt_stride.append(stride['wgt'])
        act_stride.append(stride['act'])
        out_stride.append(stride['out'])
        count += 1

    iw = K + (ow - 1) * S
    ih = K + (oh - 1) * S

    I = K + (O - 1) * S

    if im2col:
        wgt_read_size = \
                ceil_a_by_b(K * K * ic, acc_obj.N) * acc_obj.N * oc * \
                wprec
        max_wgt_size = \
                ceil_a_by_b(K * K * IC, acc_obj.N) * acc_obj.N * OC * wprec
    else:
        wgt_read_size = \
                ceil_a_by_b(K * K * ic, acc_obj.N) * acc_obj.N * \
                ceil_a_by_b(oc, acc_obj.M) * acc_obj.M * \
                wprec
        max_wgt_size = \
                ceil_a_by_b(K * K * IC, acc_obj.N) * acc_obj.N * \
                ceil_a_by_b(OC, acc_obj.M) * acc_obj.M * wprec


    if im2col:
        act_read_size = ow * oh * \
                ceil_a_by_b(K * K, acc_obj.N) * \
                b * iprec * acc_obj.N
        max_act_size = B * O * O * \
                ceil_a_by_b(K * K, acc_obj.N) * acc_obj.N * \
                iprec
    else:
        act_read_size = iw * ih * ic * b * iprec
        max_act_size = B * I * I * IC * iprec


    oprec = 32
    out_read_size = ow * oh * oc * b * oprec
    max_out_size = O * O * OC * B * oprec


    # Skip if overutilizing resources (consider double buffering)
    if wgt_read_size > acc_obj.sram['wgt'] * 8 / 2.0:
        print('error')
        return
    if act_read_size > acc_obj.sram['act'] * 8 / 2.0:
        return
    if out_read_size > acc_obj.sram['out'] * 8 / 2.0:
        return

    # Skip tiling if underutilizing resources
    # underutilization_count = 0
    # if act_read_size < 0.5 * acc_obj.sram['act'] and max_act_size >= 0.5 * acc_obj.sram['act']:
    #     underutilization_count += 1
    # if out_read_size < 0.5 * acc_obj.sram['out'] and max_out_size >= 0.5 * acc_obj.sram['out']:
    #     underutilization_count += 1
    # if wgt_read_size < 0.5 * acc_obj.sram['wgt'] and max_wgt_size >= 0.5 * acc_obj.sram['wgt']:
    #     underutilization_count += 1
    # if underutilization_count > 1:
    #     return

    # Memory Instructions
    instruction_ordered.insert_mem_read(name='Wgt RD', namespace='wgt', addr=0,
                                        size=wgt_read_size, stride=wgt_stride, level=count - 0)
    instruction_ordered.insert_mem_read(name='Act RD', namespace='act', addr=0,
                                        size=act_read_size, stride=act_stride, level=count - 0)
    instruction_ordered.insert_mem_read(name='Out RD', namespace='out', addr=0,
                                        size=out_read_size, stride=out_stride, level=count - 0)
    instruction_ordered.insert_mem_write(name='Out WR', namespace='out', addr=0,
                                         size=out_read_size, stride=out_stride, level=count - 0)
    ni = K * K * ic
    no = oh * ow * oc
    b = b

    instruction_ordered.insert_compute(acc_obj.get_compute_stats, ic, oc, ow, oh, b, K, K, iprec, wprec, im2col, weight_stationary)

    # stats = acc_obj.loop_estimate_stats(instruction_ordered)
    instruction_ordered.promote_mem_ops(acc_obj.sram)

    return instruction_ordered


def _optimize_for_order(conv_params, order_type, verbose=False):
    """
    For a given ordering, optimizes tiling
    Args:
        conv_params: A tuple with convolution params
        order_type: ordering loop
    """
    acc_obj, K, O, S, IC, OC, B, iprec, wprec, im2col, weight_stationary, energy_cost = conv_params
    I = (O - 1) * S + K

    # We do not tile the "K" dimension and compute an entire 2-D conv at a
    # time
    num_O_tiles = int(math.ceil(log2(O))) + 1
    num_IC_tiles = int(math.ceil(log2(IC))) + 1

    # TODO: Fix?
    if im2col:
        num_OC_tiles = int(math.ceil(log2(OC))) + 1
    else:
        num_OC_tiles = int(math.ceil(log2(math.ceil(float(OC)/acc_obj.M)))) + 1

    num_B_tiles = int(math.ceil(log2(B))) + 1

    best_cycles = None
    best_energy = None
    best_tiling = None

    for _b in range(num_B_tiles):
        b = min(1 << _b, B)
        num_b = ceil_a_by_b(B, b)

        for _o in range(num_O_tiles):
            ow = min(1 << _o, O)
            oh = ow
            num_ow = ceil_a_by_b(O, ow)
            num_oh = ceil_a_by_b(O, oh)

            for _ic in range(num_IC_tiles):
                ic = min(1 << _ic, IC)
                num_ic = ceil_a_by_b(IC, ic)

                for _oc in range(num_OC_tiles):

                    if im2col:
                        oc = min((1 << _oc), OC)
                    else:
                        oc = min((1 << _oc) * acc_obj.M, OC)

                    num_oc = ceil_a_by_b(OC, oc)

                    iw = K + (ow - 1) * S
                    ih = K + (oh - 1) * S

                    tiling = {}
                    tiling['B/b'] = (num_b, b)
                    tiling['OW/ow'] = (num_ow, ow)
                    tiling['OH/oh'] = (num_oh, oh)
                    tiling['IC/ic'] = (num_ic, ic)
                    tiling['OC/oc'] = (num_oc, oc)

                    stats = get_stats_fast(conv_params, tiling, order_type, verbose=verbose)

                    if stats is None:
                        continue

                    cycles = stats.total_cycles
                    energy = stats.get_energy(energy_cost)
                    mem_cycles = stats.mem_stall_cycles

                    if best_cycles is None or best_cycles > cycles or (best_cycles == cycles and best_energy > energy):
                    # if best_energy is None or best_energy > energy or (best_energy == energy and best_cycles > cycles):
                        best_energy = energy
                        best_cycles = cycles
                        best_mem_cycles = mem_cycles
                        best_order = order_type
                        best_tiling = tiling


    # if best_cycles is None:
        # print('Not found')
        # print(conv_params)
        # stats = get_stats_fast(conv_params, tiling, order_type, verbose=True)

    return (best_tiling, order_type, best_cycles, best_energy)
