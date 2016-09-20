#!/bin/bash

SCRIPT_ROOT="${BASH_SOURCE[0]}"
if [ ! `uname` = "Darwin" ]; then
  SCRIPT_ROOT=`readlink -f "${SCRIPT_ROOT}"`
else
  while [ -L "${SCRIPT_ROOT}" ]; do
    SCRIPT_ROOT=`readlink "${SCRIPT_ROOT}"`
  done
fi
SCRIPT_ROOT=`dirname "$SCRIPT_ROOT"`
pushd "${SCRIPT_ROOT}"  > /dev/null
SCRIPT_ROOT=`pwd`
SCRIPT_ROOT="$( echo ${SCRIPT_ROOT} | sed s+/cygdrive/c+c:+ )"

while read worker_ip; do
    echo "Killing BIDMach worker on ${worker_ip}"
    ssh -T "ec2-user@${worker_ip}" >/dev/null << EOS

    ps aux | grep '[t]estrecv' | awk '{print \$2}' | xargs -I% kill -9 %

EOS
done < ${SCRIPT_ROOT}/hosts.txt
echo 'done'
