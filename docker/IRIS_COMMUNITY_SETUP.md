# Running the foundation-model demos on IRIS Community

Setup used to verify the TabFM and Kumo Tabular demos end to end, and what
to watch out for.

Tested with `intersystemsdc/iris-community:latest` (digest
`sha256:228654b1…cdbdd1`): IRIS 2026.1.0.234.1com on Ubuntu 24.04.4 LTS,
embedded Python 3.12.3, `intersystems-iris-automl` 1.0.3. The DB-API client
was `intersystems-irispython` 5.4.0.

## Start the container

From the repo root:

```bash
docker run -d --name iris --stop-timeout 60 -w /home/irisowner \
    -p 1972:1972 -p 52773:52773 -v "$PWD":/opt/irisapp \
    -e IRIS_USERNAME=demo -e IRIS_PASSWORD=demo \
    --entrypoint /tini intersystemsdc/iris-community \
    -- /iris-main --after /opt/irisapp/docker/iris-after-start.sh
```

Why each part is there:

* **`/iris-main` instead of the image's entrypoint.** The image's
  `/docker-entrypoint.sh` exits with status 1 on this version. Its namespace
  setup runs `irispython -m irissqlcli`, which calls `iris.dbapi.connect`,
  and the embedded `iris` module has no `dbapi` ("Cannot call an
  iris.package wrapper ... dbapi.connect"). `/tini` stays in front, as in
  the original entrypoint, to forward signals and reap child processes.
* **`-w /home/irisowner`.** The image's working directory is `/opt/irisapp`,
  where the repo is mounted, and `/iris-main` crashes if it can't create
  `iris-main.log` in its working directory.
* **`--after docker/iris-after-start.sh`** does the setup the entrypoint
  would have done:
  1. enables `%Service_CallIn`, needed by embedded Python;
  2. creates the `IRIS_USERNAME` login with `IRIS_PASSWORD` for DB-API
     clients (the image's default accounts must change their password at
     first login, so they can't be used directly);
  3. runs `docker/iris-clear-recovery-alerts.sh` (see Healthcheck below).
* **`--stop-timeout 60`** gives `iris stop` time to shut down cleanly on
  `docker stop`.

## Healthcheck

The image's healthcheck (`/irisHealth.sh`, every 60s) fails when the
`iris qlist` system status is `alert`. After a container is killed without
`iris stop`, the next start logs severity-2 events such as "Previous system
shutdown was abnormal" and "Preserving journal files ... for journal
recovery". Two of them raise the status to `alert`, so the container
reported unhealthy although IRIS had recovered normally.

`iris-clear-recovery-alerts.sh` waits until IRIS's log monitor has posted
this startup's alerts (it copies them from `messages.log` on a ~10s cycle),
then resets the monitor state **only** if every alert is one of those
recovery messages. Any other alert is left in place and still fails the
healthcheck. Verified by force-killing and restarting the container
(status back to `ok`, healthcheck exit 0) and with an injected unrelated
alert (left alone).

## Install the AutoML provider and the demo models

```bash
# AutoML provider for TRAIN MODEL, into IRIS's Python path
docker exec iris /usr/irissys/bin/irispython -m pip install \
    --index-url https://registry.intersystems.com/pypi/simple \
    --extra-index-url https://pypi.org/simple \
    --target /usr/irissys/mgr/python intersystems-iris-automl
# (no outbound access in the container: `pip download` the same package on
#  the host for Python 3.12 and install with --no-index --find-links)

# Optional, recommended: stop AutoML from dropping features
docker exec iris /usr/irissys/bin/irispython \
    /opt/irisapp/scripts/automl_keep_features/install.py

# Per demo: stage the models, and copy the wrapper modules where PREDICT()
# can import them in a new process (repeat after editing them)
python demos/kumo_tabular_foundation/scripts/deploy_models.py
docker exec iris cp \
    /opt/irisapp/demos/kumo_tabular_foundation/iris_models/kumo_classifier.py \
    /opt/irisapp/demos/kumo_tabular_foundation/iris_models/kumo_regressor.py \
    /usr/irissys/mgr/python/
```

## Load data and run the SQL

`scripts/run_sql.py` runs a `.sql` file statement by statement, printing
OK/FAIL, timings and the first rows of each result, and exits 1 if anything
failed. `scripts/iris_sql.py` is the connection layer it shares with the
demos' `load_data.py`. Both connect over **DB-API** from the host
(`pip install intersystems-irispython`; settings from IRIS_HOST, IRIS_PORT,
IRIS_NAMESPACE, IRIS_USERNAME, IRIS_PASSWORD, defaulting to
`localhost:1972/USER` as `demo`/`demo`), or through **embedded Python** when
run under `irispython` inside the container.

```bash
# from the host, over DB-API
python demos/kumo_tabular_foundation/scripts/load_data.py
python scripts/run_sql.py demos/kumo_tabular_foundation/sql/03_evaluation.sql

# inside the container, with embedded Python
docker exec -e IRISNAMESPACE=USER iris /usr/irissys/bin/irispython \
    /opt/irisapp/scripts/run_sql.py \
    /opt/irisapp/demos/kumo_tabular_foundation/sql/02_create_models.sql
```

Embedded mode checks SQLCODE after fetching rows, because `iris.sql.exec`
can drop an error raised during the fetch (for example a failing
`PREDICT()`) and return an empty result instead.

### Known issue: `TRAIN MODEL` over DB-API ends the server process

On this setup, `TRAIN MODEL` sent over DB-API fails on the client with
`<COMMUNICATION LINK ERROR> ... Communication timed out`, and the
connection is dead afterwards. This happens with IntegratedML's built-in
AutoML models too, so the custom models are not the cause.

What is known:

* Training completes: `INFORMATION_SCHEMA.ML_TRAINING_RUNS` shows the run
  as `completed` and the trained model exists.
* The DB-API server process then exits on its own (`exit_group(0)`, with no
  signal, no Python exception, nothing in `messages.log`, the audit log or
  the application error log) without sending a reply. Just before exiting
  it loads IRIS's `libzstd.so`, apparently to build the reply.
* The same `TRAIN MODEL` from embedded Python or an `iris session` terminal
  succeeds and never loads `libzstd.so`.
* It is not caused by AutoML's stdout logging (also fails with the log
  redirected to a file via `AUTOML_HOME`), by the keep-features patch (also
  fails with it uninstalled), by client timeouts (a 10-second statement
  works), or by native thread pools (also fails with `OMP_NUM_THREADS=1`).
* Pillow, which AutoML's dependencies install, bundles its own `libzstd`
  1.5.7, while IRIS ships 1.5.4. That clash is a possible cause, not a
  confirmed one.

Workaround: run `TRAIN MODEL` (the demos' `02_create_models.sql`) with
embedded Python inside the container, as above. Everything else, including
`PREDICT()` and the evaluation queries, works over DB-API.
