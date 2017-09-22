# @ shell=/bin/bash
#
# Sample script for LoadLeveler
#
## @ class = test
# @ task_affinity = core(1)
# @ error = _OUTPUT_DIR_/$(jobid).err
# @ output = _OUTPUT_DIR_/$(jobid).out
# @ job_type = parallel
# @ node_usage= not_shared
# @ node = _NUM_NODES_·
# @ tasks_per_node = 20
# xxx @ first_node_tasks = 20
#### @ resources = ConsumableCpus(1)
##### @ node_resources = ConsumableMemory(2GB)
# @ network.MPI = sn_all,not_shared,us
# @ wall_clock_limit = 00:20:00
# @ notification = never
# @ notify_user = $(user)@rzg.mpg.de
# @ queue·

# run the program

cd ..
OUTPUT_FILE=run/_OUTPUT_DIR_/${LOADL_STEP_ID}.txt

cat $0 | grep "# @" >> ${OUTPUT_FILE}
echo BUILD_DIR= `pwd` >> $OUTPUT_FILE

echo "Modules loaded at config-time" >> $OUTPUT_FILE
cat modules_config.log >> $OUTPUT_FILE
source ./load_modules.sh
echo "Modules loaded at run-time" >> $OUTPUT_FILE
module list >> $OUTPUT_FILE 2>&1
#echo "ulimit -s" >> $OUTPUT_FILE
#ulimit -s >> $OUTPUT_FILE
#echo "List of hosts" >> $OUTPUT_FILE
#cat $LOADL_HOSTFILE >> $OUTPUT_FILE

#echo "Content of config.log" >> $OUTPUT_FILE
#cat config.log >> $OUTPUT_FILE
#echo "Output of configure script" >> $OUTPUT_FILE
#cat config_output.log >> $OUTPUT_FILE

echo _PRE_RUN_ >> $OUTPUT_FILE
_PRE_RUN_

echo "Running elpa command: " >> $OUTPUT_FILE
COMMAND="poe ./_EXECUTABLE_ _MAT_SIZE_ _NUM_EIGEN_ _BLOCK_SIZE_ "
echo $COMMAND >> $OUTPUT_FILE
${COMMAND} >> $OUTPUT_FILE 2>&1

