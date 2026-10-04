#!/bin/bash
# Clear IRIS alerts raised only by recovery from a forced shutdown.
#
# Run after IRIS starts, e.g. `/iris-main --after <this script>`.
#
# When a container is killed without `iris stop`, the next start logs
# severity-2 events such as "Previous system shutdown was abnormal" and
# "Preserving journal files ... for journal recovery". Each one raises the
# system monitor state (0 ok, 1 warn, 2 alert). The image healthcheck
# (/irisHealth.sh) fails on "alert", so the container reports unhealthy
# although IRIS recovered normally.
#
# This resets the monitor state only when every alert since startup is one
# of those recovery messages. Any other alert is left alone and still makes
# the healthcheck fail.

INSTANCE="${ISC_PACKAGE_INSTANCENAME:-IRIS}"
MGR="${ISC_PACKAGE_INSTALLDIR:-/usr/irissys}/mgr"
WAIT_SECONDS="${WAIT_SECONDS:-90}"

# The log monitor copies severity 2+ entries from messages.log to alerts.log
# (and raises the monitor state) on a ~10s cycle, so right after `iris start`
# the alerts may not be posted yet. Wait until every severity 2+ entry logged
# since this startup has been posted, so a later alert can't undo the reset.
start_line=$(grep -n "Instance '$INSTANCE' starting" "$MGR/messages.log" | tail -1 | cut -d: -f1)
expected=$(tail -n +"${start_line:-1}" "$MGR/messages.log" \
    | grep -cE '^[0-9/]+-[0-9:]+ \([0-9]+\) [23] ')
for _ in $(seq "$WAIT_SECONDS"); do
    posted=$(cat "$MGR/alerts.log" 2>/dev/null | wc -l)
    [ "$posted" -ge "$expected" ] && break
    sleep 1
done

# The IRIS terminal runs each input line on its own, so each command is one line.
iris session "$INSTANCE" -U %SYS <<'EOF'
set sc=$SYSTEM.Monitor.GetAlerts(.n,.msgs,.last),n=+$get(n),other=0
for i=1:1:n { set m=$get(msgs(i)) if m'["Previous system shutdown was abnormal",m'["for journal recovery" { set other=other+1 } }
if n=0 { write "no startup alerts",! } elseif other { write other," non-recovery alert(s); monitor state left at ",$SYSTEM.Monitor.State(),! } else { do $SYSTEM.Monitor.Clear() write "cleared ",n," shutdown-recovery alert(s)",! }
halt
EOF
