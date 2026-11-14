#!/bin/sh
# apt-get update and install for CI, retried: each try has its own time limit, so a
# stalled mirror or download fails the try instead of the step, and later tries switch
# from the runners' Azure mirror to archive.ubuntu.com. Uses sudo unless run as root.
#
#   tools/dev/apt_install.sh [APT_GET_INSTALL_OPTION...] PACKAGE...
tries=3
sudo=
[ "$(id -u)" -eq 0 ] || sudo=sudo
apt="$sudo env DEBIAN_FRONTEND=noninteractive apt-get -qq -o Acquire::Retries=3
     -o Acquire::http::Timeout=30 -o Acquire::https::Timeout=30 -o DPkg::Lock::Timeout=60"

# man-db rebuilds its index after every install, which takes minutes on the runners
$sudo rm -f /var/lib/man-db/auto-update

try=1
while :; do
    # shellcheck disable=SC2086
    timeout 120 $apt update && timeout 300 $apt install -y "$@" && exit 0
    [ "$try" -lt "$tries" ] || break
    echo "::warning::apt-get failed or timed out (try $try of $tries); retrying"
    $sudo dpkg --configure -a
    for f in /etc/apt/sources.list /etc/apt/sources.list.d/*; do
        [ -f "$f" ] && $sudo sed -i 's|azure\.archive\.ubuntu\.com|archive.ubuntu.com|g' "$f"
    done
    sleep $((try * 15))
    try=$((try + 1))
done
echo "::error::apt-get failed $tries times: $*"
exit 1
