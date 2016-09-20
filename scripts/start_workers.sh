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
    echo "Starting BIDMach worker on ${worker_ip}"
    ssh -T "ec2-user@${worker_ip}" >/dev/null << EOS
    nohup bidmach /opt/BIDMach/scripts/testrecv.ssc </dev/null >/tmp/bidmach_worker.log 2>&1 &
EOS
done < ${SCRIPT_ROOT}/hosts.txt
echo 'done'
