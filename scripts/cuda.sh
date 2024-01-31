#!/bin/bash
WORK_DIR=`mktemp -d`
if [[ ! "$WORK_DIR" || ! -d "$WORK_DIR" ]]; then
    echo "Could not create temp dir $WORK_DIR" >&2
    exit 1
fi
function cleanup {
    rm -rf "$WORK_DIR"
}
trap cleanup EXIT
nvidia-smi --query-gpu=index,gpu_bus_id --format=csv,noheader,nounits > $WORK_DIR/index.csv
nvidia-smi --query-compute-apps=pid,gpu_bus_id --format=csv,noheader,nounits > $WORK_DIR/pid.csv
join -t, -o 1.1 -1 2 -2 2 $WORK_DIR/index.csv $WORK_DIR/pid.csv > $WORK_DIR/running.csv
indexes=""
max=$1
if [[ ! $max -eq 0 ]]
then
    for index in $(cat $WORK_DIR/index.csv | cut -d, -f1)
    do
        if ! grep -q $index $WORK_DIR/running.csv
        then
            indexes="$indexes,$index"
            num=$(echo $indexes | tr -cd , | wc -c)
            if [[ $max -eq $num ]]
            then
                break
            fi
        fi
    done
fi
indexes=${indexes:1}
echo "Found $indexes" >&2
if [[ ! $max -eq $num && ! -z $max ]]
then
    echo "Not enough free GPUs" >&2
    exit 1
fi
echo $indexes