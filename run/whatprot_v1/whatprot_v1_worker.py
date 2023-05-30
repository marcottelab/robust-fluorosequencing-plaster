# Historically, things like sim, sigproc, vfs, existed already.  In migrating to Flyte, those existing
# functions were placed in (or called from) this "worker" module, and the worker module was in  turn
# called from the @task function.
#
# At present, all of the whatwork work is being done by the tasks in whatprot_v1_task.py -
# and this file only exists to match the pattern of every other job-type in plaster.
# And to hold these comments.  :)
#
# Is it useful to move the biz logic into this module to separate more from Flyte?


# The following comment is just a note about the jumble of things associated with
# the legacy pipeline and the migration to flyte, and the need to clean this up.
#
# TODO: remove Pipeline-based classes/functions & cleanup naming.
#
# The codebase is confusing with respect to result types (among others) that are a
# product of moving to Flyte from Pipeline.  Some older job types like sigproc_v2 had
# both a pipeline-based result class, and a newer dataclass-based result class for the
# move to flyte.  But newer versions of job types, like SimV3, never needed the
# Pipeline-based result classes, and so have only the dataclass-based version, but it no
# longer carry the DC suffix to differentiate.  It would be cleanest at this point
# to just retain the DC suffix for clarity until all Pipeline-based code is removed.
