#!/bin/sh
# Posts CI commit statuses `CI / <job>` on the commit under test, linking to the job's log.
#
#   tools/ci_status.sh running|success|failure|cancelled   (this job, named by $STATUS_NAME)
#   tools/ci_status.sh queued NAME...                       (jobs that will run later)
#
# Needs GH_TOKEN; JOB_ID (the job's check run id) makes the link point at the job's log.
mode=$1
shift
run_url="$GITHUB_SERVER_URL/$GITHUB_REPOSITORY/actions/runs/$GITHUB_RUN_ID"

post() {
    curl -fsS -o /dev/null -X POST -H "Authorization: Bearer $GH_TOKEN" \
        -H "Accept: application/vnd.github+json" \
        "$GITHUB_API_URL/repos/$GITHUB_REPOSITORY/statuses/$GITHUB_SHA" \
        -d "{\"state\":\"$2\",\"context\":\"CI / $1\",\"description\":\"$3\",\"target_url\":\"$4\"}" ||
        echo "Could not post the status of $1" >&2
}

if [ "$mode" = queued ]; then
    for name in "$@"; do post "$name" pending Queued "$run_url"; done
    exit 0
fi

url=$run_url
[ -n "$JOB_ID" ] && url="$run_url/job/$JOB_ID"
case "$mode" in
    running) post "$STATUS_NAME" pending Running "$url" ;;
    success) post "$STATUS_NAME" success Passed "$url" ;;
    failure) post "$STATUS_NAME" failure Failed "$url" ;;
    *) post "$STATUS_NAME" error Cancelled "$url" ;;
esac
