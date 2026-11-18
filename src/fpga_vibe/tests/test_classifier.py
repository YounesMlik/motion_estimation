import cocotb
from cocotb.triggers import Timer


@cocotb.test()
async def test_classifier(dut):
    cases = [
        (10, 12, 5, 1),
        (20, 30, 5, 0),
        (50, 50, 0, 1),
        (0, 255, 255, 1),
        (100, 90, 10, 1),
        (100, 89, 10, 0),
    ]

    for pixel, sample, threshold, expected in cases:
        dut.pixel.value = pixel
        dut.sample.value = sample
        dut.threshold.value = threshold

        await Timer(1, unit="ns")

        actual = int(dut.is_match.value)
        assert actual == expected, (
            f"pixel={pixel}, sample={sample}, "
            f"threshold={threshold}: "
            f"expected {expected}, got {actual}"
        )
