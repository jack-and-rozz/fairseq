#!/bin/bash
usage() {
    echo "Usage:$0 target_dir mode"
    exit 1
}
if [ $# -lt 2 ];then
    usage;
fi

target_dir=$1
mode=$2


case $mode in
    aspec.baseline)
    jesc.baseline)
    * ) echo "invalid mode"
        exit 1 ;;
esac
