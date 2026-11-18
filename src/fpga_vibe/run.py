from pathlib import Path

from cocotb_tools.runner import get_runner

ROOT = Path(__file__).resolve().parents[0]


def simulate() -> None:
    runner = get_runner("ghdl")
    print(ROOT)

    runner.build(
        sources=[ROOT / "rtl/classifier.vhd"],
        hdl_toplevel="classifier",
        build_dir=ROOT / "build/classifier",
        always=True,
    )

    runner.test(
        hdl_toplevel="classifier",
        test_module="test_classifier",
        test_dir=ROOT / "sim/tests",
    )


if __name__ == "__main__":
    simulate()
