#!/bin/sh
# apt-get update and install for CI, retried: each try has its own time limit, so a
# stalled mirror or download fails the try instead of the step, and later tries drop the
# runners' Azure mirror for archive.ubuntu.com. Uses sudo unless run as root. With
# APT_ARCHIVES set, the downloaded packages are kept in that directory (owned by the
# caller, for actions/cache), and packages already there are not downloaded again.
#
#   tools/dev/apt_install.sh [APT_GET_INSTALL_OPTION...] PACKAGE...
tries=3
sudo=
[ "$(id -u)" -eq 0 ] || sudo=sudo
apt="$sudo env DEBIAN_FRONTEND=noninteractive apt-get -o Acquire::Retries=3
     -o Acquire::http::Timeout=30 -o Acquire::https::Timeout=30 -o DPkg::Lock::Timeout=60"
if [ -n "$APT_ARCHIVES" ]; then
    mkdir -p "$APT_ARCHIVES/partial"
    apt="$apt -o Dir::Cache::Archives=$APT_ARCHIVES"
fi

# man-db rebuilds its index after every install, which takes minutes on the runners
$sudo rm -f /var/lib/man-db/auto-update

try=1
while :; do
    # -q, not -qq: the log then shows each download, which mirror served it, and its size
    # shellcheck disable=SC2086
    if timeout 120 $apt -qq update && timeout 300 $apt -q install -y "$@"; then
        [ -z "$APT_ARCHIVES" ] || $sudo chown -R "$(id -u):$(id -g)" "$APT_ARCHIVES"
        exit 0
    fi
    [ "$try" -lt "$tries" ] || break
    echo "::warning::apt-get failed or timed out (try $try of $tries); retrying"
    $sudo dpkg --configure -a
    # The runner images list the mirrors in apt-mirrors.txt (mirror+file: sources), which
    # tries the next mirror only when one fails, not when it stalls; containers name the
    # archive in their sources
    [ -f /etc/apt/apt-mirrors.txt ] && $sudo sed -i '/azure\.archive\.ubuntu\.com/d' /etc/apt/apt-mirrors.txt
    for f in /etc/apt/sources.list /etc/apt/sources.list.d/*; do
        [ -f "$f" ] && $sudo sed -i 's|azure\.archive\.ubuntu\.com|archive.ubuntu.com|g' "$f"
    done
    sleep $((try * 15))
    try=$((try + 1))
done
echo "::error::apt-get failed $tries times: $*"
exit 1
