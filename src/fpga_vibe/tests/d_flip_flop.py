import random

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, ReadOnly, RisingEdge, Timer


@cocotb.test()
async def d_flip_flop(dut):
    clock = Clock(dut.clk, 1, unit="ns")
    cocotb.start_soon(clock.start())

    rng = random.Random(42)

    for _ in range(100):
        expected = rng.randint(0, 1)

        await FallingEdge(dut.clk)
        dut.d.value = expected

        await RisingEdge(dut.clk)
        await ReadOnly()

        assert int(dut.q.value) == expected
