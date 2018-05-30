#!/bin/bash

# paths to programs
CUBIT_PATH="/opt/cubit-13.2/cubit"
BACI_PATH="/home/ivo/baci/work/release"

# path to cubit journal file
CUBIT_JOURNAL="../input-solid-mesh/tube-py.jou"

# create mesh
$CUBIT_PATH -nographics -batch -nojournal $CUBIT_JOURNAL

echo "CUBIT DONE!"

# create dat file
$BACI_PATH/pre_exodus --exo=tube.exo --head=../input-solid-mesh/tube.head --bc=../input-solid-mesh/tube.bc

echo "DONE!"
