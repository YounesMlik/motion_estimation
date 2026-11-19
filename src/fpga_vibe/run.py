from collections.abc import Sequence
from pathlib import Path

from cocotb_tools.runner import get_runner

ROOT = Path(__file__).resolve().parents[0]
TEST_DIR = ROOT / "tests"
BUILD_DIR = ROOT / "build"


def simulate(
    top_level: str,
    sources: Sequence[str | Path],
    test_module: str,
    *,
    always: bool = False,
) -> None:
    runner = get_runner("ghdl")
    build_dir = BUILD_DIR / top_level
    # print(TEST_DIR)

    runner.build(
        sources=[
            Path(source) if Path(source).is_absolute() else ROOT / source
            for source in sources
        ],
        hdl_toplevel=top_level,
        build_dir=build_dir,
        always=always,
        # verbose=True,
    )

    runner.test(
        test_module=test_module,
        hdl_toplevel=top_level,
        # build_dir=build_dir,
        test_args=[f"--workdir={build_dir}"],
        test_dir=TEST_DIR,
        # verbose=True,
    )
