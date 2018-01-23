#!/bin/bash
source /etc/profile.d/modules.sh
set -ex

source ./ci_test_scripts/.ci-env-vars
module list
echo $1

make -j $1

