import argparse
import logging

from dnnweaver2.graph import Graph, get_default_graph
from dnnweaver2.tensorOps.cnn import conv2D, maxPool, flatten, matmul, addBias, batch_norm, reorg, concat, leakyReLU, add
from dnnweaver2 import get_tensor
import logging
from dnnweaver2.scalar.dtypes import FQDtype, FixedPoint

import bitfusion.src.benchmarks.ant_bench as ant
import bitfusion.src.benchmarks.adafloat_bench as ada
import bitfusion.src.benchmarks.bitfusion_bench as bit
import bitfusion.src.benchmarks.olaccel_bench as ola
import bitfusion.src.benchmarks.ant_weight_bench as ant_weight
import bitfusion.src.benchmarks.biscaled_bench as biscaled

import os

def fc(tensor_in, output_channels=1024,
        f_dtype=None, w_dtype=None,
        act='linear'):
    input_channels = tensor_in.shape[-1]
    weights = get_tensor(shape=(output_channels, input_channels),
            name='weights',
            dtype=w_dtype)
    biases = get_tensor(shape=(output_channels,),
            name='biases',
            dtype=FixedPoint(32,w_dtype.frac_bits + tensor_in.dtype.frac_bits))
    _fc = matmul(tensor_in, weights, biases, dtype=f_dtype)

    if act == 'leakyReLU':
        with get_default_graph().name_scope(act):
            act = leakyReLU(_fc, dtype=_fc.dtype)
    elif act == 'linear':
        with get_default_graph().name_scope(act):
            act = _fc
    else:
        raise (ValueError, 'Unknown activation type {}'.format(act))

    return act

def conv(tensor_in, filters=32, stride=None, kernel_size=3, pad='SAME',
        c_dtype=None, w_dtype=None,
        act='linear'):

    if stride is None:
        stride = (1,1,1,1)

    input_channels = tensor_in.shape[-1]

    weights = get_tensor(shape=(filters, kernel_size, kernel_size, input_channels),
                         name='weights',
                         dtype=w_dtype)
    biases = get_tensor(shape=(filters),
                         name='biases',
                         dtype=FixedPoint(32,w_dtype.frac_bits + tensor_in.dtype.frac_bits))
    _conv = conv2D(tensor_in, weights, biases, stride=stride, pad=pad, dtype=c_dtype)

    if act == 'leakyReLU':
        with get_default_graph().name_scope(act):
            act = leakyReLU(_conv, dtype=_conv.dtype)
    elif act == 'linear':
        with get_default_graph().name_scope(act):
            act = _conv
    else:
        raise (ValueError, 'Unknown activation type {}'.format(act))

    return act


def get_precision(precision):
    if precision == 16:
        return FQDtype.FXP16
    if precision == 8:
        return FQDtype.FXP8
    if precision == 4:
        return FQDtype.FXP4
    if precision == 6:
        return FQDtype.FXP6

def create_net(net_name, net_list, batch_size):
    g = Graph(net_name, dataset='imagenet', log_level=logging.INFO)
    with g.as_default():
        for idx, op in enumerate(net_list):
            input_size, kernel_size, output_size, kernel_stride, padding, precision, op_type =  op
            input_size[0] = input_size[0] * batch_size
            output_size[0] = output_size[0] * batch_size
            precision = get_precision(precision)

            if op_type == 0:
                with g.name_scope('conv'+str(idx)):
                    out = create_conv(input_size, kernel_size, stride_size=kernel_stride, pad=padding, c_dtype=FQDtype.FXP16, w_dtype=precision)
                    # print(idx, op, out.shape)
                    assert out.shape[0] == output_size[0]
                    assert out.shape[1] == output_size[2]
                    assert out.shape[2] == output_size[3]
                    assert out.shape[3] == output_size[1]
            else:
                with g.name_scope('fc'+str(idx)):
                    out = create_fc(input_size, kernel_size, c_dtype=precision, w_dtype=precision)
                    # print(idx, op, out.shape)
                    assert out.shape[0] == output_size[0]
                    assert out.shape[1] == output_size[1]
    return g

def create_conv(input_size, weight_size, stride_size=None, pad=None, c_dtype=None, w_dtype=None):

    if stride_size is None:
        stride = (1,1,1,1)
    else:
        stride = (1,stride_size[0],stride_size[1],1)

    batch_size = input_size[0]
    output_channels = weight_size[0]
    input_channels = weight_size[1]
    kernel_size = (weight_size[2], weight_size[3])

    input = get_tensor(shape=(batch_size, input_size[2], input_size[3], input_size[1]), name='data', dtype=w_dtype, trainable=False)
    weights = get_tensor(shape=(output_channels, kernel_size[0], kernel_size[1], input_channels), name='weights', dtype=w_dtype)
    biases = get_tensor(shape=(output_channels), name='biases', dtype=c_dtype)
    _conv = conv2D(input, weights, biases, stride=stride, pad=pad, dtype=c_dtype)
    return _conv

def create_fc(input_size, weight_size, c_dtype=None, w_dtype=None):
    batch_size = input_size[0]
    output_channels = weight_size[0]
    input_channels = weight_size[1]

    input = get_tensor(shape=(batch_size, input_size[1]), name='data', dtype=w_dtype, trainable=False)
    weights = get_tensor(shape=(output_channels, input_channels), name='weights', dtype=w_dtype)
    biases = get_tensor(shape=(output_channels,), name='biases', dtype=c_dtype)
    _fc = matmul(input, weights, biases, dtype=c_dtype)
    return _fc

benchlist = [\
             'vgg16',
             'resnet18',
             'resnet50',
             'inceptionv3',
             'vit', 
             'mnli',
             'cola', 
             'sst_2',
            ]


def get_bench_nn_ant(bench_name, batch_size):
    if bench_name == 'vgg16':
        return create_net(bench_name, ant.vgg16, batch_size)
    elif bench_name == 'resnet18':
        return create_net(bench_name, ant.resnet18, batch_size)
    elif bench_name == 'resnet50':
        return create_net(bench_name, ant.resnet50, batch_size)
    elif bench_name == 'inceptionv3':
        return create_net(bench_name, ant.inceptionv3, batch_size)
    elif bench_name == 'vit':
        return create_net(bench_name, ant.vit, batch_size)
    elif bench_name == 'rte':
        return create_net(bench_name, ant.rte, batch_size)
    elif bench_name == 'wnli':
        return create_net(bench_name, ant.wnli, batch_size)
    elif bench_name == 'mrpc':
        return create_net(bench_name, ant.mrpc, batch_size)
    elif bench_name == 'cola':
        return create_net(bench_name, ant.cola, batch_size)
    elif bench_name == 'sst_2':
        return create_net(bench_name, ant.sst_2, batch_size)
    elif bench_name == 'qnli':
        return create_net(bench_name, ant.qnli, batch_size)
    elif bench_name == 'qqp':
        return create_net(bench_name, ant.qqp, batch_size)
    elif bench_name == 'mnli':
        return create_net(bench_name, ant.mnli, batch_size)

def get_bench_nn_ant_weight(bench_name, batch_size):
    if bench_name == 'vgg16':
        return create_net(bench_name, ant_weight.vgg16, batch_size)
    elif bench_name == 'resnet18':
        return create_net(bench_name, ant_weight.resnet18, batch_size)
    elif bench_name == 'resnet50':
        return create_net(bench_name, ant_weight.resnet50, batch_size)
    elif bench_name == 'inceptionv3':
        return create_net(bench_name, ant_weight.inceptionv3, batch_size)
    elif bench_name == 'vit':
        return create_net(bench_name, ant_weight.vit, batch_size)
    elif bench_name == 'rte':
        return create_net(bench_name, ant_weight.rte, batch_size)
    elif bench_name == 'wnli':
        return create_net(bench_name, ant_weight.wnli, batch_size)
    elif bench_name == 'mrpc':
        return create_net(bench_name, ant_weight.mrpc, batch_size)
    elif bench_name == 'cola':
        return create_net(bench_name, ant_weight.cola, batch_size)
    elif bench_name == 'sst_2':
        return create_net(bench_name, ant_weight.sst_2, batch_size)
    elif bench_name == 'qnli':
        return create_net(bench_name, ant_weight.qnli, batch_size)
    elif bench_name == 'qqp':
        return create_net(bench_name, ant_weight.qqp, batch_size)
    elif bench_name == 'mnli':
        return create_net(bench_name, ant_weight.mnli, batch_size)

def get_bench_nn_bit(bench_name, batch_size):
    if bench_name == 'vgg16':
        return create_net(bench_name, bit.vgg16, batch_size)
    elif bench_name == 'resnet18':
        return create_net(bench_name, bit.resnet18, batch_size)
    elif bench_name == 'resnet50':
        return create_net(bench_name, bit.resnet50, batch_size)
    elif bench_name == 'inceptionv3':
        return create_net(bench_name, bit.inceptionv3, batch_size)
    elif bench_name == 'vit':
        return create_net(bench_name, bit.vit, batch_size)
    elif bench_name == 'rte':
        return create_net(bench_name, bit.rte, batch_size)
    elif bench_name == 'wnli':
        return create_net(bench_name, bit.wnli, batch_size)
    elif bench_name == 'mrpc':
        return create_net(bench_name, bit.mrpc, batch_size)
    elif bench_name == 'cola':
        return create_net(bench_name, bit.cola, batch_size)
    elif bench_name == 'sst_2':
        return create_net(bench_name, bit.sst_2, batch_size)
    elif bench_name == 'qnli':
        return create_net(bench_name, bit.qnli, batch_size)
    elif bench_name == 'qqp':
        return create_net(bench_name, bit.qqp, batch_size)
    elif bench_name == 'mnli':
        return create_net(bench_name, bit.mnli, batch_size)


def get_bench_nn_ada(bench_name, batch_size):
    if bench_name == 'vgg16':
        return create_net(bench_name, ada.vgg16, batch_size)
    elif bench_name == 'resnet18':
        return create_net(bench_name, ada.resnet18, batch_size)
    elif bench_name == 'resnet50':
        return create_net(bench_name, ada.resnet50, batch_size)
    elif bench_name == 'inceptionv3':
        return create_net(bench_name, ada.inceptionv3, batch_size)
    elif bench_name == 'vit':
        return create_net(bench_name, ada.vit, batch_size)
    elif bench_name == 'rte':
        return create_net(bench_name, ada.rte, batch_size)
    elif bench_name == 'wnli':
        return create_net(bench_name, ada.wnli, batch_size)
    elif bench_name == 'mrpc':
        return create_net(bench_name, ada.mrpc, batch_size)
    elif bench_name == 'cola':
        return create_net(bench_name, ada.cola, batch_size)
    elif bench_name == 'sst_2':
        return create_net(bench_name, ada.sst_2, batch_size)
    elif bench_name == 'qnli':
        return create_net(bench_name, ada.qnli, batch_size)
    elif bench_name == 'qqp':
        return create_net(bench_name, ada.qqp, batch_size)
    elif bench_name == 'mnli':
        return create_net(bench_name, ada.mnli, batch_size)


def get_bench_nn_ola(bench_name, batch_size):
    if bench_name == 'vgg16':
        return create_net(bench_name, ola.vgg16, batch_size)
    elif bench_name == 'resnet18':
        return create_net(bench_name, ola.resnet18, batch_size)
    elif bench_name == 'resnet50':
        return create_net(bench_name, ola.resnet50, batch_size)
    elif bench_name == 'inceptionv3':
        return create_net(bench_name, ola.inceptionv3, batch_size)
    elif bench_name == 'vit':
        return create_net(bench_name, ola.vit, batch_size)
    elif bench_name == 'rte':
        return create_net(bench_name, ola.rte, batch_size)
    elif bench_name == 'wnli':
        return create_net(bench_name, ola.wnli, batch_size)
    elif bench_name == 'mrpc':
        return create_net(bench_name, ola.mrpc, batch_size)
    elif bench_name == 'cola':
        return create_net(bench_name, ola.cola, batch_size)
    elif bench_name == 'sst_2':
        return create_net(bench_name, ola.sst_2, batch_size)
    elif bench_name == 'qnli':
        return create_net(bench_name, ola.qnli, batch_size)
    elif bench_name == 'qqp':
        return create_net(bench_name, ola.qqp, batch_size)
    elif bench_name == 'mnli':
        return create_net(bench_name, ola.mnli, batch_size)

def get_bench_nn_bis(bench_name, batch_size):
    if bench_name == 'vgg16':
        return create_net(bench_name, biscaled.vgg16, batch_size)
    elif bench_name == 'resnet18':
        return create_net(bench_name, biscaled.resnet18, batch_size)
    elif bench_name == 'resnet50':
        return create_net(bench_name, biscaled.resnet50, batch_size)
    elif bench_name == 'inceptionv3':
        return create_net(bench_name, biscaled.inceptionv3, batch_size)
    elif bench_name == 'vit':
        return create_net(bench_name, biscaled.vit, batch_size)
    elif bench_name == 'rte':
        return create_net(bench_name, biscaled.rte, batch_size)
    elif bench_name == 'wnli':
        return create_net(bench_name, biscaled.wnli, batch_size)
    elif bench_name == 'mrpc':
        return create_net(bench_name, biscaled.mrpc, batch_size)
    elif bench_name == 'cola':
        return create_net(bench_name, biscaled.cola, batch_size)
    elif bench_name == 'sst_2':
        return create_net(bench_name, biscaled.sst_2, batch_size)
    elif bench_name == 'qnli':
        return create_net(bench_name, biscaled.qnli, batch_size)
    elif bench_name == 'qqp':
        return create_net(bench_name, biscaled.qqp, batch_size)
    elif bench_name == 'mnli':
        return create_net(bench_name, biscaled.mnli, batch_size)

def write_to_csv(csv_name, fields, stats, graph, csv_path='./'):
    if not os.path.exists(csv_path):
        os.makedirs(csv_path)

    for l in stats:
        print(l)
        print(stats[l]['total'])

    bench_csv_name = os.path.join(csv_path, csv_name)
    with open(bench_csv_name, 'w') as f:
        f.write(', '.join(fields+['\n']))
        for l in network:
            if isinstance(network[l], ConvLayer):
                f.write('{}, {}\n'.format(l, ', '.join(str(x) for x in stats[l]['total'])))

def get_bench_numbers(graph, sim_obj, batch_size=1, weight_stationary = False):
    stats = {}
    for opname, op in graph.op_registry.items():
        out = sim_obj.get_cycles(op, batch_size, weight_stationary = weight_stationary)
        if out is not None:
            s, l = out
            stats[opname] = s
    return stats

if __name__ == "__main__":
    # parser object
    argp = argparse.ArgumentParser()

    # parser arguments
    argp.add_argument("-c", "--config_file", dest='config_file', default='conf.ini', type=str)
    argp.add_argument("-v", "--verbose", dest='verbose', default=False, action='store_true')

    # parse
    args = argp.parse_args()

    if args.verbose:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO

    logging.basicConfig(level=log_level)
    logger = logging.getLogger(__name__)

    # Read config file
    logger.info('Creating benchmarks')

    sim_obj = Simulator(args.config_file, args.verbose)
    fields = ['Layer', 'Total Cycles', 'Memory Stall Cycles', \
              'Activation Reads', 'Weight Reads', 'Output Reads', \
              'DRAM Reads', 'Output Writes', 'DRAM Writes']
    csv_dir = 'csv'
    if not os.path.isdir(csv_dir):
        os.makedirs(csv_dir)

    for bench in benchlist:
        print(bench)
        nn = get_bench_nn(bench)
        print(nn)
        stats = get_bench_numbers(nn, sim_obj, weight_stationary = False)
        write_to_csv(os.path.join(csv_dir, bench+'.csv'), fields, stats, nn)
