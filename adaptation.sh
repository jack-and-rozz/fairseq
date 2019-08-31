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

