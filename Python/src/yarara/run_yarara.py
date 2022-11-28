#!/usr/bin/env python3

import logging
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Union

import configpile as cp
import numpy as np
import pandas as pd
from typing_extensions import Annotated

import yarara.stages
from yarara.sts import spec_time_series
from yarara.util import print_iter

try:
    warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
except Exception:
    pass


@dataclass(frozen=True)
class Task(cp.Config):
    """Runs the YARARA pipeline"""

    prog = Path(__file__).stem

    #: Stage at which to start the processing
    stage_start: Annotated[
        int, cp.Param.store(cp.parsers.int_parser, short_flag_name="-b", default_value="0")
    ]

    #: Stage before which to stop the processing
    stage_break: Annotated[
        int, cp.Param.store(cp.parsers.int_parser, short_flag_name="-e", default_value="99")
    ]

    #: TODO: reference
    reference: Annotated[
        Optional[str],
        cp.Param.store(
            cp.parsers.stripped_str_parser.empty_means_none(),
            short_flag_name="-r",
            default_value="",
        ),
    ]

    #: Whether to close plots
    close_figure: Annotated[
        bool, cp.Param.store(cp.parsers.bool_parser, short_flag_name="-d", default_value="true")
    ]

    #: Path to the data to process
    #:
    #: The path must have the format "STARNAME/data/s1d/INSTRUMENT/"
    #: It must also contain a WORKSPACE/ subfolder.
    data_folder: Annotated[
        Path,
        cp.Param.store(
            cp.parsers.path_parser.map(lambda p: p.resolve()),
            long_flag_name=None,
            positional=cp.Positional.ONCE,
        ),
    ]

    #: Path to the SIMBAD database folder
    simbad_folder: Annotated[
        Path,
        cp.Param.store(
            cp.parsers.path_parser.validated(
                lambda p: (p / "table_stars.p").is_file(),
                "Database folder must contain table_stars.p",
            ),
            default_value=str(Path(__file__).parent.parent.parent / "database" / "SIMBAD"),
        ),
    ]

    #: Path to the mask_ccf folder
    mask_ccf_folder: Annotated[
        Path,
        cp.Param.store(
            cp.parsers.path_parser.validated(lambda p: p.is_dir(), "Path does not exist"),
            default_value=str(Path(__file__).parent.parent.parent / "MASK_CCF"),
        ),
    ]

    #: Path to the material folder
    material_folder: Annotated[
        Path,
        cp.Param.store(
            cp.parsers.path_parser.validated(lambda p: p.is_dir(), "Path does not exist"),
            default_value=str(Path(__file__).parent.parent.parent / "Material"),
        ),
    ]

    def validate_data_folder(self) -> Optional[cp.Err]:
        if not self.data_folder.exists():
            return cp.Err.make(f"The data folder {self.data_folder} does not exist")
        if not self.data_folder.is_dir():
            return cp.Err.make(f"The path {self.data_folder} is not a directory")
        parts = self.data_folder.parts
        if len(parts) <= 5:  # including /
            return cp.Err.make(
                f"The path is too short, must have the structure STAR/data/s1d/INSTRUMENT at the end"
            )
        if parts[-1] != "HARPN":
            logging.warning(f"This code has only been tested with HARPN, not {parts[-1]}")
        if parts[-2] != "s1d" or parts[-3] != "data":
            return cp.Err.make(
                f"The path must have the structure STAR/data/s1d/INSTRUMENT at the end"
            )
        if not (self.data_folder / "WORKSPACE").is_dir():
            return cp.Err.make(
                f"The provided path {self.data_folder} must contain a WORKSPACE subfolder with a copy of the RASSINE processed files."
            )


def run(t: Task) -> None:
    # planted_activated is false during our runs
    # Disable the new Pandas performance warnings due to fragmentation
    parts = t.data_folder.parts
    ins = parts[-1]
    instrument = ins[0:5]
    star = parts[-4]
    logging.info(f"Reducing data from star {star} and instrument {instrument}")
    directory_workspace = str(t.data_folder / "WORKSPACE") + "/"
    begin = time.time()
    all_time = [begin]
    time_step: Dict[Union[str, int], Union[int, float]] = {"begin": 0}

    def get_time_step(step):
        now = time.time()
        time_step[step] = now - all_time[-1]
        all_time.append(now)

    sts = spec_time_series(
        directory_workspace,
        starname=star,
        instrument=ins,
        simbad_folder=t.simbad_folder,
        mask_ccf_folder=t.mask_ccf_folder,
        material_folder=t.material_folder,
    )
    # file_test = sts.import_spectrum()

    print(f"------------------------\n STAR LOADED : {sts.starname} \n------------------------")

    print(
        f"\n [INFO] Complete analysis {ins} {['with', 'without'][int(t.reference is None)]} reference spectrum launched...\n"
    )

    for stage in range(t.stage_start, t.stage_break):
        if stage == 0:
            yarara.stages.preprocessing(sts, t.close_figure)
            get_time_step("preprocessing")

        if stage == 1:
            yarara.stages.statistics(sts, ins, t.close_figure)
            get_time_step("statistics")

        if stage == 2:
            yarara.stages.matching_cosmics(
                sts=sts, reference=t.reference, close_figure=t.close_figure
            )
            get_time_step("matching_cosmics")

        if stage == 3:
            yarara.stages.matching_telluric(
                sts=sts, reference=t.reference, close_figure=t.close_figure
            )
            get_time_step("matching_telluric")

        if stage == 4:
            yarara.stages.matching_oxygen(
                sts=sts, reference=t.reference, close_figure=t.close_figure
            )
            get_time_step("matching_oxygen")

        if stage == 5:
            yarara.stages.matching_contam(
                sts,
                reference=t.reference,
                frog_file=str(sts.material_folder / "Contam_HARPN.p"),
                close_figure=t.close_figure,
            )
            get_time_step("matching_contam")

        if stage == 6:
            yarara.stages.matching_pca(sts=sts, reference=t.reference, close_figure=t.close_figure)
            get_time_step("matching_pca")

        ref: Union[Literal["master"], Literal["median"]] = ["master", "median"][t.reference is None]  # type: ignore

        if stage == 7:
            yarara.stages.matching_activity(
                sts=sts,
                reference=t.reference,
                ref=ref,
                close_figure=t.close_figure,
                input_dico="matching_pca",
            )
            get_time_step("matching_activity")

        if stage == 8:
            yarara.stages.matching_ghost_a(
                sts=sts,
                ref=ref,
                close_figure=t.close_figure,
                frog_file=str(t.material_folder / ("Ghost_" + ins + ".p")),
            )
            get_time_step("matching_ghost_a")

        if stage == 9:
            yarara.stages.matching_ghost_b(sts=sts, ref=ref, close_figure=t.close_figure)
            get_time_step("matching_ghost_b")

        if stage == 10:
            yarara.stages.matching_fourier(
                sts=sts, reference=t.reference, close_figure=t.close_figure
            )
            get_time_step("matching_fourier")

        if stage == 11:
            yarara.stages.matching_smooth(
                sts=sts, ref=ref, reference=t.reference, close_figure=t.close_figure
            )
            get_time_step("matching_smooth")

        if stage == 12:
            yarara.stages.matching_mad(sts=sts, reference=t.reference, close_figure=t.close_figure)
            get_time_step("matching_mad")

        if stage == 13:
            yarara.stages.stellar_atmos1(
                sts=sts, reference=t.reference, close_figure=t.close_figure
            )
            get_time_step("stellar_atmos1")

        if stage == 14:
            yarara.stages.matching_brute(
                sts=sts, reference=t.reference, close_figure=t.close_figure
            )
            get_time_step("matching_brute")

    # =============================================================================
    # SAVE INFO TIME
    # =============================================================================

    print_iter(time.time() - begin)

    if True:
        table_time = pd.DataFrame(
            time_step.values(), index=time_step.keys(), columns=["time_step"]
        )
        table_time["frac_time"] = 100 * table_time["time_step"] / np.sum(table_time["time_step"])
        table_time["time_step"] /= 60  # convert in minutes

        filename_time = (
            sts.dir_root
            + f"REDUCTION_INFO/Time_informations_reduction_{time.strftime('%Y-%m-%dT%H:%M:%S', time.gmtime())}.csv"
        )
        table_time.to_csv(filename_time)


def cli() -> None:
    run(Task.from_command_line_())
