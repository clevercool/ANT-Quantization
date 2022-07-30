from collections import namedtuple

BaseEnergyTuple = namedtuple('BaseEnergyTuple',
        ['core_leak_energy',
         'core_dynamic_energy',
         'wbuf_read_energy',
         'wbuf_write_energy',
         'ibuf_read_energy',
         'ibuf_write_energy',
         'obuf_read_energy',
         'obuf_write_energy',
         'sram_leak_energy',
        ])

class EnergyTuple(BaseEnergyTuple):
    def __str__(self):
        dram_leak_energy = 484.615 / 500
        ret = ''
        ret+='Energy costs for BitFusion\n'
        ret+='Core leak energy : {:.3f} pJ/cycle (for entire systolic array)\n'.format(self.core_leak_energy*1.e3)
        ret+='Core dynamic energy : {:.3f} pJ/cycle (for entire systolic array)\n'.format(self.core_dynamic_energy*1.e3)
        ret+='WBUF Read energy    : {:.3f} pJ/bit\n'.format(self.wbuf_read_energy*1.e3)
        ret+='WBUF Write energy   : {:.3f} pJ/bit\n'.format(self.wbuf_write_energy*1.e3)
        ret+='IBUF Read energy    : {:.3f} pJ/bit\n'.format(self.ibuf_read_energy*1.e3)
        ret+='IBUF Write energy   : {:.3f} pJ/bit\n'.format(self.ibuf_write_energy*1.e3)
        ret+='OBUF Read energy    : {:.3f} pJ/bit\n'.format(self.obuf_read_energy*1.e3)
        ret+='OBUF Write energy   : {:.3f} pJ/bit\n'.format(self.obuf_write_energy*1.e3)
        ret+='SRAM leak energy   : {:.3f} pJ/bit\n'.format(self.sram_leak_energy*1.e3)
        ret+='DRAM leak energy   : {:.3f} pJ/bit\n'.format(dram_leak_energy*1.e3)
        return ret
