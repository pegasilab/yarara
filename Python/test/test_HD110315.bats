setup_file() {
    TEST_DIR="$(cd "$(dirname "$BATS_TEST_FILENAME")" >/dev/null 2>&1 && pwd)"
    export TEST_DIR
    cd "$TEST_DIR"/../..
    git submodule update --init spectra/HD110315
    cd "$TEST_DIR"/../../spectra/HD110315
    git reset --hard HEAD
    git clean -fdx
    cd "$TEST_DIR"/..
}

@test "HD110315 import" {
    ./run_rassine.sh -l WARNING -c harpn.ini ../spectra/HD110315/data/s1d/HARPN import
}

@test "HD110315 reinterpolate" {
    ./run_rassine.sh -l WARNING -c harpn.ini ../spectra/HD110315/data/s1d/HARPN reinterpolate
}

@test "HD110315 stacking" {
    ./run_rassine.sh -l WARNING -c harpn.ini ../spectra/HD110315/data/s1d/HARPN stacking
}

@test "HD110315 rassine" {
    ./run_rassine.sh -l WARNING -c harpn.ini ../spectra/HD110315/data/s1d/HARPN rassine
}

@test "HD110315 matching_anchors" {
    ./run_rassine.sh -l WARNING -c harpn.ini ../spectra/HD110315/data/s1d/HARPN matching_anchors
}

@test "HD110315 matching_diff" {
    ./run_rassine.sh -l WARNING -c harpn.ini ../spectra/HD110315/data/s1d/HARPN matching_diff
}

@test "Preparing YARARA run" {
    mkdir ../spectra/HD110315/data/s1d/HARPN/WORKSPACE
    cp ../spectra/HD110315/data/s1d/HARPN/STACKED/RASSINE_* ../spectra/HD110315/data/s1d/HARPN/WORKSPACE
}

@test "YARARA" {
    python trigger_yarara_harpn.py -b 0 -e 15
}
