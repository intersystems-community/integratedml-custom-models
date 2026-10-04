#!/bin/bash
# Post-start setup for intersystemsdc/iris-community when /iris-main is run
# directly:
#
#   docker run ... -e IRIS_USERNAME=demo -e IRIS_PASSWORD=demo \
#       --entrypoint /tini intersystemsdc/iris-community \
#       -- /iris-main --after /opt/irisapp/docker/iris-after-start.sh
#
# The image's own /docker-entrypoint.sh exits with status 1 on IRIS 2026.1:
# its namespace setup runs `irispython -m irissqlcli`, which calls
# iris.dbapi.connect, and the embedded `iris` module has no dbapi. This
# script does the parts of that setup the demos need, without irissqlcli:
#
#   1. enable %Service_CallIn (OS + password auth), needed by embedded Python
#   2. create or update the IRIS_USERNAME login with IRIS_PASSWORD, if set,
#      for external clients such as DB-API (new users don't carry the
#      default accounts' "change password at next login" flag)
#   3. clear alerts raised only by recovery from a forced shutdown, so the
#      image healthcheck doesn't report a recovered instance as unhealthy
#
# The password is read from the environment inside IRIS, never interpolated
# into the ObjectScript.

INSTANCE="${ISC_PACKAGE_INSTANCENAME:-IRIS}"
DIR="$(cd "$(dirname "$0")" && pwd)"

# The IRIS terminal runs each input line on its own, so each command is one line.
iris session "$INSTANCE" -U %SYS <<'EOF'
set p("Enabled")=1,p("AutheEnabled")=48,sc=##class(Security.Services).Modify("%Service_CallIn",.p) write "CallIn enabled: ",sc,!
set user=$SYSTEM.Util.GetEnviron("IRIS_USERNAME"),pw=$SYSTEM.Util.GetEnviron("IRIS_PASSWORD")
if user'="",pw'="" { if ##class(Security.Users).Exists(user,.u) { set u.PasswordExternal=pw,u.ChangePassword=0,sc=u.%Save() } else { set sc=##class(Security.Users).Create(user,"%All",pw) } write "login ",user,": ",$select(sc=1:"ok",1:$SYSTEM.Status.GetErrorText(sc)),! }
halt
EOF

"$DIR/iris-clear-recovery-alerts.sh"
