#/bin/bash

mkdir -p /data/elpa/gitlab/coverage/`git log | head -n 1 | sed -r s/commit\ // | cut -c1-16`
lcov  -q --capture --directory src/.libs  --directory src/elpa2_kernels --output-file coverage_all.info && lcov -q ./coverage_all.info /usr/lib64/\* -r ./coverage_all.info /afs/ipp-garching.mpg.de/common/soft/gcc/4.9.3/@sys/lib/gcc/x86_64-unknown-linux-gnu/4.9.3/include/* > ./coverage_all_cleaned_$(git log | head -n 1 | sed -r s/commit\ // | cut -c1-16)_"$(pidof pidof)".info
mv coverage_all_cleaned_*  /data/elpa/gitlab/coverage/`git log | head -n 1 | sed -r s/commit\ // | cut -c1-16`
lcov $(for f in /data/elpa/gitlab/coverage/`git log | head -n 1 | sed -r s/commit\ // | cut -c1-16`/*; do echo "-a $f"; done) -o info.out
lcov --summary info.out 2>&1 | awk '/lines|functions/ {gsub(/\.*:$/, "", $1); gsub(/^./, "", $3); printf "%s: %s (%s of %s), ", $1, $2, $3, $5; } /branches/ {print "";}' | sed 's/^/__COVERAGE__:/; s/, $//;'


