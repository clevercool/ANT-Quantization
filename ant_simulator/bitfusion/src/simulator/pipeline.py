class PipeStage(object):
    def __init__(self, rd, compute, wr):
        self.rd = rd
        self.compute = compute
        self.wr = wr

    def get_cycles(self):
        return max(self.rd + self.wr, self.compute)

    def __str__(self):
        if self.rd is not None and self.wr is not None:
            return 'Pipe: ' + str(int(max(self.rd + self.wr, self.compute)))
        else:
            return 'RD {}, C {}, WR {}'.format(self.rd, self.compute, self.wr)

class Pipeline(object):
    def __init__(self, pipe, rd, wr, pipe_len):
        # print('Creating Pipeline')
        if isinstance(pipe, int):
            self.i0 = PipeStage(rd, 0, 0)
            self.i1 = PipeStage(rd, pipe, 0)
            self.mid = max(pipe, rd + wr) * (pipe_len - 2)
            self.f0 = PipeStage(0, pipe, wr)
            self.f1 = PipeStage(0, 0, wr)
        elif isinstance(pipe, Pipeline):
            self.i0 = PipeStage(pipe.i0.rd + rd, 0, 0)
            self.i1 = pipe.i1

            self.mid = pipe.mid
            self.mid += (pipe_len-1) * \
                    (PipeStage(pipe.i0.rd + rd, pipe.f0.compute, pipe.f0.wr).get_cycles() +\
                     PipeStage(pipe.i1.rd, pipe.i1.compute, pipe.f1.wr + wr).get_cycles() +\
                     pipe.mid)

            self.f0 = pipe.f0
            self.f1 = PipeStage(0, 0, pipe.f1.wr + wr)

    def get_cycles(self):
        return self.i0.get_cycles() + \
                self.i1.get_cycles() + \
                self.mid + \
                self.f0.get_cycles() + \
                self.f1.get_cycles()

    def __str__(self):
        ret  = 'Init0 : {}'.format(self.i0)
        ret += ', Init1 : {}'.format(self.i1)
        ret += ', Mid  : {}'.format(self.mid)
        ret += ', Final0: {}'.format(self.f0)
        ret += ', Final1: {}'.format(self.f1)
        return ret

if __name__ == "__main__":

    rk = 1
    ck = 3
    wk = 1

    kmax = 4

    c_pipe = PipeStage(None, ck, None)
    print(c_pipe)

    k_pipe = Pipeline(c_pipe, rk, wk, kmax)
    print(k_pipe)

    rj = 5
    cj = ck
    wj = 5

    jmax = 4
    j_pipe = Pipeline(k_pipe, rj, wj, jmax)
    print(j_pipe)
