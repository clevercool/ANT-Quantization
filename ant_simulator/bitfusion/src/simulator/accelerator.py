from bitfusion.src.utils.utils import ceil_a_by_b, log2
from bitfusion.src.simulator.stats import Stats

class Accelerator(object):
    def __init__(self, N, M, pmax, pmin, sram, mem_if_width, frequency):
        """
        accelerator object
        """
        self.N = N
        self.M = M
        self.sram = sram
        self.mem_if_width = mem_if_width
        self.frequency = frequency
        self.pmax = pmax
        self.pmin = pmin

    def get_mem_read_cycles(self, dst, size):
        """
        Read instruction
        args:
            src_idx: index of source address
            dst: destination address
            size: size of data in bits
        """
        return ceil_a_by_b(size, self.mem_if_width)

    def get_mem_write_cycles(self, src, size):
        """
        Write instruction
        args:
            src_idx: index of source address
            src: destination address
            size: size of data in bits
        """
        return ceil_a_by_b(size, self.mem_if_width)


    def get_compute_stats(self, ic, oc, ow, oh, b, kw, kh, iprec, wprec, im2col=False, weight_stationary = False):
        """
        Compute instruction
        args:
            ic: Input Channels
            oc: Output Channels
            ow: Output Width
            oh: Output Height
            kw: Output Height
            kh: Output Height
            b: Batch Size
            im2col: boolean. If true, we assume the cpu does im2col. Otherwise,
                    we do convolutions channel-wise
        """
        compute_stats = Stats()
        if weight_stationary:
            compute_stats.total_cycles = self.get_compute_cycles(ic, oc, ow, oh,
                                                                b, kw, kh,
                                                                iprec,
                                                                wprec,
                                                                im2col)
        else:
            compute_stats.total_cycles = self.get_compute_cycles_output_stationary(ic, oc, ow, oh,
                                                                b, kw, kh,
                                                                iprec,
                                                                wprec,
                                                                im2col)
                                    
        return compute_stats


    def get_perf_factor(self, iprec, wprec):
        iprec = max(iprec, self.pmin)
        wprec = max(wprec, self.pmin)
        return int(self.pmax / iprec) * int(self.pmax / wprec)

    def get_perf_factor(self, prec):
        prec = max(prec, self.pmin)
        return int(self.pmax / prec)


    def get_compute_cycles(self, ic, oc, ow, oh, b, kw, kh, iprec, wprec, im2col=False):
        """
        Compute instruction
        args:
            ic: Input Channels
            oc: Output Channels
            ow: Output Width
            oh: Output Height
            kw: Output Height
            kh: Output Height
            b: Batch Size
            im2col: boolean. If true, we assume the cpu does im2col. Otherwise,
                    we do convolutions channel-wise
        """
        
        if im2col:
            ni = kw * kh * ic
            no = oc
            batch = b * oh * ow
            compute_cycles = batch * ceil_a_by_b(no, self.M * self.get_perf_factor(wprec)) * \
                    (ceil_a_by_b(ni, self.N * self.get_perf_factor(iprec)))
            # compute_cycles = ceil_a_by_b(batch, self.N * self.get_perf_factor(iprec)) * ceil_a_by_b(no, self.M * self.get_perf_factor(wprec)) * \
            #         ni
        else:
            raise (ValueError, 'Not supported!')

        return compute_cycles

    def get_compute_cycles_output_stationary(self, ic, oc, ow, oh, b, kw, kh, iprec, wprec, im2col=False):
        """
        Compute instruction
        args:
            ic: Input Channels
            oc: Output Channels
            ow: Output Width
            oh: Output Height
            kw: Output Height
            kh: Output Height
            b: Batch Size
            im2col: boolean. If true, we assume the cpu does im2col. Otherwise,
                    we do convolutions channel-wise
        """
        if im2col:
            ni = kw * kh * ic
            no = oc
            batch = b * oh * ow
            compute_cycles = ceil_a_by_b(batch, self.N * self.get_perf_factor(iprec)) * ceil_a_by_b(no, self.M * self.get_perf_factor(wprec)) * \
                    ni
        else:
            raise (ValueError, 'Not supported!')

        return compute_cycles

    def get_area(self):
        return None
