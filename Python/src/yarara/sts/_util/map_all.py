from __future__ import annotations

import os
from typing import TYPE_CHECKING, Literal, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from .. import spec_time_series


def yarara_map_all(
    self: spec_time_series,
    wave_min: Optional[float] = None,
    wave_max: Optional[float] = None,
    index: str = "index",
    reference: Union[
        int, NDArray[np.float64], Literal["snr", "median", "master", "zeros"]
    ] = "median",
):

    planet = self.planet
    self.import_dico_tree()

    dico_tree = self.dico_tree
    all_dico = self.all_dicos

    directory_images = "/".join(self.directory.split("/")[0:-2]) + "/IMAGES/"

    if not os.path.exists(directory_images):
        os.system("mkdir " + directory_images)

    if not os.path.exists(directory_images + f"{wave_min}_{wave_max}"):
        os.system("mkdir " + directory_images + f"{wave_min}_{wave_max}")

    self.import_info_reduction()
    test_file = self.info_reduction

    crossed_dico = np.array(all_dico)[np.in1d(all_dico, list(test_file.keys()))]

    dico_found = np.array(dico_tree["dico"])[np.in1d(np.array(dico_tree["dico"]), crossed_dico)]

    # dico_chain = np.hstack(['matching_anchors',dico_found])
    dico_chain = dico_found

    # last = np.array(crossed_dico)[~np.in1d(crossed_dico,dico_chain)][0]

    # nb_plot = len(crossed_dico)+1

    # counter=nb_plot
    for counter, last in enumerate(dico_chain):
        if test_file[last]["valid"]:
            print(f"sub dico {last} being processed...")
            plt.figure(figsize=(15, 5))
            self.yarara_map(
                sub_dico=last,
                planet=planet,
                wave_min=wave_min,
                wave_max=wave_max,
                index=index,
                reference=reference,
                new=True,
            )
            plt.savefig(
                directory_images
                + f"{wave_min}_{wave_max}/"
                + f"Map_{wave_min}_{wave_max}_"
                + str(counter + 1).zfill(2)
                + f"_{last}.png"
            )
            # counter-=1
            plt.close("all")
            # if counter!=0:
            #    last = np.array(dico_chain)[np.where(np.array(crossed_dico)==last)[0]][0]

    plt.figure(figsize=(15, 10))
    plt.subplot(2, 1, 1)
    plt.title("Before YARARA processing", fontsize=14)
    self.yarara_map(
        sub_dico="matching_diff",
        wave_min=wave_min,
        wave_max=wave_max,
        index=index,
        reference=reference,
        new=False,
    )
    ax = plt.gca()
    plt.subplot(2, 1, 2, sharex=ax, sharey=ax)
    plt.title("After YARARA processing", fontsize=14)
    self.yarara_map(
        sub_dico="matching_mad",
        wave_min=wave_min,
        wave_max=wave_max,
        index=index,
        reference=reference,
        new=False,
    )
    plt.subplots_adjust(top=0.97, right=0.97, left=0.05, bottom=0.05)
    plt.savefig(directory_images + f"{wave_min}_{wave_max}/" + "Before_after.png")
