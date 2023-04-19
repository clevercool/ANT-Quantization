import subprocess
import pandas
import os
import re
import json

class CactiSweep(object):
    def __init__(self, bin_file='./cacti/cacti', csv_file='cacti_sweep.csv', default_json='./default.json', default_dict=None):
        if not os.path.isfile(bin_file):
            print("Can't find binary file {}. Please clone and compile cacti first".format(bin_file))
            self.bin_file = None
        else:
            self.bin_file = os.path.abspath(bin_file)
        self.csv_file = os.path.abspath(os.path.join(os.path.dirname(__file__), csv_file))
        self.default_dict = json.load(open(default_json))
        self.cfg_file = os.path.join(os.path.dirname(os.path.abspath(self.csv_file)), 'sweep.cfg')
        if default_dict is not None:
            self.default_dict.update(default_dict)
        if os.path.isfile(self.csv_file) and False:
            self._df = pandas.read_csv(csv_file)
        else:
            output_dict = {
                    'Access time (ns)': 'access_time_ns',
                    'Total dynamic read energy per access (nJ)': 'read_energy_nJ',
                    'Total dynamic write energy per access (nJ)': 'write_energy_nJ',
                    'Total leakage power of a bank (mW)': 'leak_power_mW',
                    'Total gate leakage power of a bank (mW)': 'gate_leak_power_mW',
                    'Cache height (mm)': 'height_mm',
                    'Cache width (mm)': 'width_mm',
                    'Cache area (mm^2)': 'area_mm^2',
                    }
            cols = list(self.default_dict)
            cols.extend(output_dict.keys())
            self._df = pandas.DataFrame(columns=cols)

    def update_csv(self):
        self._df = self._df.drop_duplicates()
        self._df.to_csv(self.csv_file, index=False)

    def _create_cfg(self, cfg_dict, filename):
        with open(filename, 'w') as f:
            cfg_dict['output/input bus width'] = cfg_dict['block size (bytes)'] * 8
            for key in cfg_dict:
                if cfg_dict[key] is not None:
                    f.write('-{} {}\n'.format(key, cfg_dict[key]))

    def _parse_cacti_output(self, out):
        output_dict = {
                'Access time (ns)': 'access_time_ns',
                'Total dynamic read energy per access (nJ)': 'read_energy_nJ',
                'Total dynamic write energy per access (nJ)': 'write_energy_nJ',
                'Total leakage power of a bank (mW)': 'leak_power_mW',
                'Total gate leakage power of a bank (mW)': 'gate_leak_power_mW',
                # 'Cache height (mm)': 'height_mm',
                # 'Cache width (mm)': 'width_mm',
                # 'Cache area (mm^2)': 'area_mm^2',
                'Cache height x width (mm)': 'height_mm',
                }
        parsed_results = {}
        for line in out:
            line = line.rstrip()
            line = line.lstrip()
            if line:
                for o in output_dict:
                    key = output_dict[o]
                    o = o.replace('(', '\(')
                    o = o.replace(')', '\)')
                    o = o.replace('^', '\^')
                    regex = r"{}\s*:\s*([\d\.]*)".format(o)
                    m = re.match(regex, line.decode('utf-8'))
                    if m:
                        parsed_results[key] = m.groups()[0]
                        if key == "height_mm":
                            regex = r"{}\s*x\s*([\d\.]*)".format(o + ": " + m.groups()[0])
                            m = re.match(regex, line.decode('utf-8'))
                            parsed_results['width_mm'] = m.groups()[0]
        return parsed_results

    def _run_cacti(self, index_dict):
        """
        Get data from cacti
        """
        assert self.bin_file is not None, 'Can\'t run cacti, no binary found. Please clone and compile cacti first.'
        cfg_dict = self.default_dict.copy()
        cfg_dict.update(index_dict)
        self._create_cfg(cfg_dict, self.cfg_file)
        args = (self.bin_file, "-infile", self.cfg_file)
        print(args)
        popen = subprocess.Popen(args, stdout=subprocess.PIPE, cwd=os.path.dirname(self.bin_file))
        popen.wait()
        output = popen.stdout
        cfg_dict.update(self._parse_cacti_output(output))
        return cfg_dict

    def locate(self, index_dict):
        self._df = self._df.drop_duplicates()
        data = self._df

        for key in index_dict:
            data = data.loc[data[key] == index_dict[key]]
        return data

    def get_data(self, index_dict):
        data = self.locate(index_dict)
        if len(data) == 0:
            print('No entry found in {}, running cacti'.format(self.csv_file))
            row_dict = index_dict.copy()
            row_dict.update(self._run_cacti(index_dict))
            row_dict["area_mm^2"] = float(row_dict["height_mm"]) * float(row_dict["width_mm"])
            self._df = pandas.concat([self._df, pandas.DataFrame([row_dict])], ignore_index=True)
            self.update_csv()
            return self.locate(index_dict)
        else:
            return data

    def get_data_clean(self, index_dict):
        data = self.get_data(index_dict)
        cols = [
                'size (bytes)',
                'block size (bytes)',
                'access_time_ns',
                'read_energy_nJ',
                'write_energy_nJ',
                'leak_power_mW',
                'gate_leak_power_mW',
                'height_mm',
                'width_mm',
                'area_mm^2',
                'technology (u)',
                ]
        return data[cols]

if __name__ == "__main__":
    cache_sweep_data = CactiSweep()
    # for log2_size in range(8, 19):
    #     for log2_width in range(1, log2_size-5):
    #         width = 1<<log2_width
    #         cfg_dict = {'block size (bytes)': width, 'size (bytes)': (1<<log2_size), 'technology (u)': 0.045}
    #         if log2_size > 20:
    #             print('size: {} mBytes'.format(1<<(log2_size-20)))
    #         elif log2_size > 10:
    #             print('size: {} kBytes'.format(1<<(log2_size-10)))
    #         else:
    #             print('size: {} Bytes'.format(1<<log2_size))
    #         print('technology: {}'.format(float(cache_sweep_data.get_data_clean(cfg_dict)['technology (u)'])))
    #         print('block size (bytes): {}'.format(float(cache_sweep_data.get_data_clean(cfg_dict)['block size (bytes)'])))
    #         print('read energy per bit: {} pJ'.format(float(cache_sweep_data.get_data_clean(cfg_dict)['read_energy_nJ'])/width/8*1.e3))
    #         print('write energy per bit: {} pJ'.format(float(cache_sweep_data.get_data_clean(cfg_dict)['write_energy_nJ'])/width/8*1.e3))
    print('*'*50)
    tech_node = 0.065
    print('Eyeriss @ {:1.0f}nm'.format(tech_node*1.e3))
    eyeriss_banks = 27
    eyeriss_entries = 512
    eyeriss_line_size = 8
    cfg_dict = {'block size (bytes)': eyeriss_line_size, 'size (bytes)': eyeriss_line_size * eyeriss_entries, 'technology (u)': tech_node}
    eyeriss_read_energy = float(cache_sweep_data.get_data_clean(cfg_dict)['read_energy_nJ'])/eyeriss_line_size*1.e3
    eyeriss_write_energy = float(cache_sweep_data.get_data_clean(cfg_dict)['write_energy_nJ'])/eyeriss_line_size*1.e3
    eyeriss_avg_energy = (eyeriss_read_energy + eyeriss_write_energy) / 2.
    eyeriss_area = float(cache_sweep_data.get_data_clean(cfg_dict)['area_mm^2'])
    eyeriss_leak_power = float(cache_sweep_data.get_data_clean(cfg_dict)['leak_power_mW'])
    print('area: {} mm^2'.format(eyeriss_area * eyeriss_banks))
    print('leakage power: {} mWatt'.format(eyeriss_leak_power * eyeriss_banks))
    print('read energy per bit: {} pJ'.format(eyeriss_read_energy))
    print('write energy per bit: {} pJ'.format(eyeriss_write_energy))
    print('avg energy per bit: {} pJ'.format(eyeriss_avg_energy))

    print('*'*50)
    tech_node = 0.045
    N = 16
    M = 32
    print('BitFusion @ {:1.0f}nm'.format(tech_node*1.e3))
    total_size = 64*1024 #bytes
    line_size = 4 #bytes
    entries = total_size / (N*M*line_size)
    cfg_dict = {'block size (bytes)': line_size, 'size (bytes)': line_size * entries, 'technology (u)': tech_node}
    read_energy = float(cache_sweep_data.get_data_clean(cfg_dict)['read_energy_nJ'])/line_size*1.e3/8
    area = float(cache_sweep_data.get_data_clean(cfg_dict)['area_mm^2'])
    write_energy = float(cache_sweep_data.get_data_clean(cfg_dict)['write_energy_nJ'])/line_size*1.e3/8
    avg_energy = (read_energy + write_energy) / 2.
    print(cfg_dict)
    print(cache_sweep_data.get_data_clean(cfg_dict))
    print('area: {} mm^2'.format(area))
    print('size: {} bytes'.format(entries * line_size))
    print('total area: {} mm^2'.format(area * N * M))
    print('total size: {} bytes'.format(entries * line_size * 512))
    print('read energy per bit: {} pJ'.format(read_energy))
    print('write energy per bit: {} pJ'.format(write_energy))
    print('avg energy per bit: {} pJ'.format(avg_energy))

    print('*'*10)
    total_size = 32*1024 #bytes
    line_size = 4 #bytes
    entries = total_size / (N*line_size)
    cfg_dict = {'block size (bytes)': line_size, 'size (bytes)': line_size * entries, 'technology (u)': tech_node}
    read_energy = float(cache_sweep_data.get_data_clean(cfg_dict)['read_energy_nJ'])/line_size*1.e3/8
    area = float(cache_sweep_data.get_data_clean(cfg_dict)['area_mm^2'])
    write_energy = float(cache_sweep_data.get_data_clean(cfg_dict)['write_energy_nJ'])/line_size*1.e3/8
    avg_energy = (read_energy + write_energy) / 2.
    print('area: {} mm^2'.format(area))
    print('size: {} bytes'.format(entries * line_size))
    print('total area: {} mm^2'.format(area * N))
    print('total size: {} bytes'.format(total_size))
    print('read energy per bit: {} pJ'.format(read_energy))
    print('write energy per bit: {} pJ'.format(write_energy))
    print('avg energy per bit: {} pJ'.format(avg_energy))

    print('*'*10)
    total_size = 16*1024 #bytes
    line_size = 4 #bytes
    entries = total_size / (M*line_size)
    cfg_dict = {'block size (bytes)': line_size, 'size (bytes)': line_size * entries, 'technology (u)': tech_node, 'read-write port': 1}
    read_energy = float(cache_sweep_data.get_data_clean(cfg_dict)['read_energy_nJ'])/line_size*1.e3/8
    area = float(cache_sweep_data.get_data_clean(cfg_dict)['area_mm^2'])
    write_energy = float(cache_sweep_data.get_data_clean(cfg_dict)['write_energy_nJ'])/line_size*1.e3/8
    avg_energy = (read_energy + write_energy) / 2.
    print('area: {} mm^2'.format(area))
    print('size: {} bytes'.format(entries * line_size))
    print('total area: {} mm^2'.format(area * M))
    print('total size: {} bytes'.format(total_size))
    print('read energy per bit: {} pJ'.format(read_energy))
    print('write energy per bit: {} pJ'.format(write_energy))
    print('avg energy per bit: {} pJ'.format(avg_energy))

    cache_sweep_data.update_csv()
