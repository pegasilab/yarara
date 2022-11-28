setup_file() {
    TEST_DIR="$(cd "$(dirname "$BATS_TEST_FILENAME")" >/dev/null 2>&1 && pwd)"
    export TEST_DIR
    cd "$TEST_DIR"/..
    git submodule update --init spectra/HD110315
    cd "$TEST_DIR"/../spectra/HD110315
    git reset --hard HEAD
    git clean -fdx
    cd "$TEST_DIR"/..
}

@test "HD110315 import" {
    ./run_rassine.sh -l WARNING -c harpn.ini spectra/HD110315/data/s1d/HARPN import
}

@test "HD110315 reinterpolate" {
    ./run_rassine.sh -l WARNING -c harpn.ini spectra/HD110315/data/s1d/HARPN reinterpolate
}

@test "HD110315 stacking" {
    ./run_rassine.sh -l WARNING -c harpn.ini spectra/HD110315/data/s1d/HARPN stacking
}

@test "HD110315 rassine" {
    ./run_rassine.sh -l WARNING -c harpn.ini spectra/HD110315/data/s1d/HARPN rassine
}

@test "HD110315 matching_anchors" {
    ./run_rassine.sh -l WARNING -c harpn.ini spectra/HD110315/data/s1d/HARPN matching_anchors
}

@test "HD110315 matching_diff" {
    ./run_rassine.sh -l WARNING -c harpn.ini spectra/HD110315/data/s1d/HARPN matching_diff
}

@test "Preparing YARARA run" {
    mkdir spectra/HD110315/data/s1d/HARPN/WORKSPACE
    cp spectra/HD110315/data/s1d/HARPN/STACKED/RASSINE_* spectra/HD110315/data/s1d/HARPN/WORKSPACE
}

@test "YARARA step 0" {
    run_yarara -b 0 -e 1 spectra/HD110315/data/s1d/HARPN
}

@test "YARARA step 1" {
    run_yarara -b 1 -e 2 spectra/HD110315/data/s1d/HARPN
}

@test "YARARA step 2" {
    run_yarara -b 2 -e 3 spectra/HD110315/data/s1d/HARPN
}

@test "YARARA step 3" {
    run_yarara -b 3 -e 4 spectra/HD110315/data/s1d/HARPN
}

@test "YARARA step 4" {
    run_yarara -b 4 -e 5 spectra/HD110315/data/s1d/HARPN
}

@test "YARARA step 5" {
    run_yarara -b 5 -e 6 spectra/HD110315/data/s1d/HARPN
}

@test "YARARA step 6" {
    run_yarara -b 6 -e 7 spectra/HD110315/data/s1d/HARPN
}

@test "YARARA step 7" {
    run_yarara -b 7 -e 8 spectra/HD110315/data/s1d/HARPN
}

@test "YARARA step 8" {
    run_yarara -b 8 -e 9 spectra/HD110315/data/s1d/HARPN
}

@test "YARARA step 9" {
    run_yarara -b 9 -e 10 spectra/HD110315/data/s1d/HARPN
}

@test "YARARA step 10" {
    run_yarara -b 10 -e 11 spectra/HD110315/data/s1d/HARPN
}

@test "YARARA step 11" {
    run_yarara -b 11 -e 12 spectra/HD110315/data/s1d/HARPN
}

@test "YARARA step 12" {
    run_yarara -b 12 -e 13 spectra/HD110315/data/s1d/HARPN
}

@test "YARARA step 13" {
    run_yarara -b 13 -e 14 spectra/HD110315/data/s1d/HARPN
}

@test "YARARA step 14" {
    run_yarara -b 14 -e 15 spectra/HD110315/data/s1d/HARPN
}
