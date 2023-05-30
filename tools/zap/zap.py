"""
This is a wrapper for executing various parallel map routines in a consistent way.

The primary routines are:
    work_orders(): Calls a function with args or kwargs.
        def _do_work(i, foo=None):
            print(i, foo)

        results = zap.work_orders([
            dict(fn=_do_work, args=(i,), foo=i*10)
            for i in range(10)
        ])

    arrays():
        def _do_work_on_an_arrays(a, b, foo=None):
            # a and b are np.ndarrays
            print(a, b, foo)

        array0 = np.zeros((10, 100))
        array1 = np.zeros((10, 1000))
        results = zap.arrays(_do_work_on_an_arrays, dict(a=array0, b=array1), foo=1)

    df_rows(): Split along Dataframe rows
        def _do_work_on_df_row(row, foo=None):
            # row is a row of a dataframe
            print(row, foo)

        df = pd.DataFrame(dict(a=[1,2], b=[3,4]))
        results = zap.df_rows(_do_work_on_df_row, df, foo=1)

    df_groups(): Split along Dataframe groups
        def _do_work_on_df_group(group, foo=None):
            # group is a pandas groupby object
            print(group, foo)

        df = pd.DataFrame(dict(a=[1,1,2,2], b=[3,4,4,5]))
        results = zap.df_groups(_do_work_on_df_group, df.groupby("a"), foo=1)


Contexts
    A zap context establishes how parallelism will be allowed:

        # Exmaple, run _do_work in, at most, 5 sub-processes
        with zap.Context(cpu_limit=5, mode="process", progress=progress):
            results = zap.work_orders([
                dict(fn=_do_work, args=(i,), foo=i*10)
                for i in range(10)
            ])

Debugging run-away processes.

    Sometimes you can get into a situation where processes seem
    to be stranded and are still running after a ^C.
    This is complicated by running docker under OSX.
    Docker under OSX is actually running under a Linux VM called
    "com.docker.hyperkit". The OSX pid of that process
    has nothing to do with the pid of the processes that are
    running inside the VM and the pids running insider the
    container (inside the VM).

    You can drop into the hyperkit VM with the following command
    from the an OSX shell using Erisyon's "p" helper.
        $ OSX_VM=1 ./p

    Once in there you can "top" or "htop" and see what processes are
    running. Let's say that you see that pid 5517 taking 100% cpu.
    You can then find the pid INSIDE the container with this:
        $ cat /proc/5517/status | grep NSpid
        > NSpid:	5517	832

    The second number of which is the pid INSIDE the container (832).
"""

import gc
import logging
import os
import random
import signal
import sys
import time
import traceback
from concurrent.futures import (
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
    thread,
)
from concurrent.futures.process import BrokenProcessPool
from contextlib import contextmanager
from multiprocessing import cpu_count

import numpy as np
import pandas as pd
import psutil
from munch import Munch

log = logging.getLogger(__name__)
from plaster.tools.utils import utils

_context_depth = 0
_cpu_limit = None
_mode = "process"
_progress = None
_allow_inner_parallelism = False
_trap_exceptions = True
_thread_name_prefix = "zap_"
_zap_verbose = False

if os.environ.get("ZAP_DEBUG_MODE") == "True":
    _mode = "debug"


@contextmanager
def Context(
    cpu_limit=None,
    mode=None,
    progress=None,
    allow_inner_parallelism=False,
    trap_exceptions=True,
    thread_name_prefix=None,
    mem_per_workorder=None,
    verbose=None,
    _force_mode=False,  # Used for testing purposes
):
    """
    Arguments:
        cpu_limit: int (default None==all). Maximum number of CPUs to use
            None: all
            positive numbers: that many cpus
            negative numbers: all cpus except this many. eg: -2 = all cpus less two
            default: all
        mode: str. (Default "process")
            "process": Run in sub-processes
            "thread": Run as threads
            "debug" : Run the work orders serially and blocking (ie no threads or processes)
                This is useful both for debugging and to prevent inner contexts from
                parallelizing. See allow_inner_parallelism.
        allow_inner_parallelism: bool (default False)
            If True, allow inner contexts to parallelize normally.
            Usually this is a bad idea as it can lead to serious contention
            wherein a group or parallel work order each tries to allocate
            all cpus for themselves and causes CPU and/or Memory contention.
        progress: function pointer
            If non None will callback with args (work_order_i, n_total_work_orders, extra_info)
        trap_exceptions: bool (default True)
            If true, exceptions are trapped and returned as a result.
            When false, any worker execption bubbles up immediately to
            the caller and other workers will die when they die.
            The default is True because there's nothing more annoying than
            running a long-running parallel job only to find that
            after hours of execution there was one rare exception stopped
            the whole run!
        thread_name_prefix: str (default "zap_")
            Set the thread names for easier debugging
        mem_per_workorder: int
            If not None, use this to estimate the cpu_limit
        verbose: bool
            If True, emit debugging traces
    """

    global _cpu_limit, _mode, _progress, _allow_inner_parallelism, _context_depth, _trap_exceptions, _thread_name_prefix, _zap_verbose
    _context_depth += 1

    orig_cpu_limit = _cpu_limit
    orig_mode = _mode
    orig_progress = _progress
    orig_allow_inner_parallelism = _allow_inner_parallelism
    orig_trap_exceptions = _trap_exceptions
    orig_thread_name_prefix = _thread_name_prefix
    orig_zap_verbose = _zap_verbose

    if _context_depth > 1 and not _allow_inner_parallelism:
        # In a nested context if inner_parallelism is not allowed then
        # the zaps are kicked into mode = "debug" meaning that
        # work_orders will execute serially in the current thread & process.
        mode = "debug"

    if cpu_limit is not None:
        _cpu_limit = cpu_limit
    elif mem_per_workorder is not None:
        gb = 2**30
        vmm = psutil.virtual_memory().total - (8 * gb)  # Reserve 8 GB for other factors
        _cpu_limit = max(1, min(vmm // mem_per_workorder, _cpu_count()))

    if mode is not None and _mode != "debug":
        # debug mode will not allow any inner contexts to be anything other than debug
        _mode = mode

    if _force_mode:
        _mode = mode

    _progress = progress
    _allow_inner_parallelism = allow_inner_parallelism
    _trap_exceptions = trap_exceptions
    _thread_name_prefix = thread_name_prefix

    if verbose is not None:
        _zap_verbose = verbose

    try:
        yield
    finally:
        _cpu_limit = orig_cpu_limit
        _mode = orig_mode
        _progress = orig_progress
        _allow_inner_parallelism = orig_allow_inner_parallelism
        _trap_exceptions = orig_trap_exceptions
        _thread_name_prefix = orig_thread_name_prefix
        _zap_verbose = orig_zap_verbose
        _context_depth -= 1


def _cpu_count():
    """mock-point"""
    return cpu_count()


def _show_work_order_exception(e):
    """Mock-point"""
    log.exception(f"Exception raised by a work order {e.work_order}")


def _mock_BrokenProcessPool_exception():
    """mock_point"""
    pass


def _set_zap(**kwargs):
    """
    Creates a global variable with the zap information to bypass
    the serialization that multiprocessing would otherwise do.
    """
    zap_id = int(time.time() * 1000000)
    zap = Munch(id=zap_id, **kwargs)
    globals()[f"__zap_{zap_id}"] = zap
    return zap


def _get_zap(zap_id):
    """
    Fetches zap data from global. See _set_zap.
    """
    return globals()[f"__zap_{zap_id}"]


def _del_zap(zap_id):
    del globals()[f"__zap_{zap_id}"]


def _run_work_order_fn(zap_id, work_order_i):
    """
    Wrap the function to handle args, kwargs, capture exceptions, and re-seed RNG.
    Note: This may run in the sub-process or thread and therefore should not use stdio.
    """
    start_time = time.time()
    try:
        work_order = _get_zap(zap_id).work_orders[work_order_i]

        # RE-INITIALIZE the random seed because numpy resets the seed in sub-processes.
        np.random.seed(seed=int(time.time() * 100_000) % int(2**32))
        random.seed()

        args = work_order.pop("args", ())
        fn = work_order.pop("fn")
        assert callable(fn)
        result = fn(*args, **work_order)

        # GARBAGE collect.
        # A future improvement might be allow this to be configurable
        gc.collect()
    except Exception as e:
        formatted = traceback.format_exception(
            etype=type(e), value=e, tb=e.__traceback__
        )
        result = e
        result.exception_lines = formatted

    return result, time.time() - start_time


def _call_progress(zap, i, retry=False):
    if zap.progress is not None:
        try:
            retry_msg = f"retry" if retry else ""
            zap.progress(i + 1, zap.n_work_orders, retry_msg)
        except Exception as e:
            log.exception(e, "Warning: progress function exceptioned; ignoring.")


def _dump_exception(result):
    """Mock-point"""
    exception_name = f"{result.__class__.__name__}({result})"
    log.error(
        f"Work order generated un-trapped exception: '{exception_name}'. exception_lines were: {result.exception_lines}"
    )
    print(f"Work order generated un-trapped exception: '{exception_name}'.")
    print("".join(result.exception_lines))


def _examine_result(zap, result, work_order):
    if isinstance(result, Exception):
        result.work_order = work_order
        if not zap.trap_exceptions:
            _dump_exception(result)
            raise result
    return result


def _warn_about_retries(n_retries, zap):
    """Mock-point"""
    log.warning(
        f"There were {n_retries} processes killed in a zap (id={zap.id} fn_name={zap.fn_name})."
        f"This was likely caused by running out of memory. The zap executor will "
        f"now try to re-run each zap serially to reduce memory pressure but this may be very slow.\n"
        f"If you are running under docker or a VM you might consider trying to raise memory limits."
    )


def _do_zap_with_executor(executor, zap):
    """
    Execute work_orders through a thread or process pool executor
    """
    retry_iz = []

    wo_i_by_future = {}
    for i, work_order in enumerate(zap.work_orders):
        # Important: the executor submit must not be passed
        # the actual work_order to bypass serialization.
        future = executor.submit(_run_work_order_fn, zap.id, i)
        wo_i_by_future[future] = i

    results = [None] * zap.n_work_orders
    timings = [None] * zap.n_work_orders

    n_done = 0
    for future in as_completed(wo_i_by_future):
        i = wo_i_by_future[future]
        work_order = zap.work_orders[i]
        try:
            result, duration = future.result()
            _mock_BrokenProcessPool_exception()  # Used for testing
            _call_progress(zap, n_done)
            n_done += 1
            results[i] = _examine_result(zap, result, work_order)
            timings[i] = duration
        except BrokenProcessPool as e:
            # This can happen if the child process(es) run out
            # of memory. In that case, we need to retry those
            # work_orders.
            retry_iz += [i]

    n_retries = len(retry_iz)
    if n_retries > 0:
        _warn_about_retries(n_retries, zap)
    for i in retry_iz:
        # These retries are likely a result of running out memory
        # and we don't know how many of those processes we can support
        # so the only safe thing to do is to run them one at a time.
        # If this becomes a constant issue then we could try some
        # sort of exponential back-off on number of concurrent processes.
        # Sometimes this can create a 'queue.Full' exception in the babysitter
        # threads that are not handled gracefully by Python.
        # See:
        #   https://bugs.python.org/issue8426
        #   https://docs.python.org/3/library/multiprocessing.html#multiprocessing.Queue
        #   https://github.com/python/cpython/pull/3895
        try:
            _call_progress(zap, i, retry=True)
            result, duration = _run_work_order_fn(zap.id, i)
            results[i] = _examine_result(zap, result, zap.work_orders[i])
            timings[i] = duration
        except Exception as e:
            results[i] = e
            timings[i] = None

    return results, timings


def _do_work_orders_process_mode(zap):
    if sys.platform != "linux":
        raise Exception(
            "Process mode zap is not working under non-linux at moment sue to differences in globals() under non-linux."
        )

    with ProcessPoolExecutor(max_workers=zap.max_workers) as executor:
        try:
            return _do_zap_with_executor(executor, zap)
        except KeyboardInterrupt:
            # If I do not os.kill the processes then it seems
            # that will gracefully send kill signals and wait
            # for the children. I typically want to just abandon
            # anything that the process is doing and have it end instantly.
            # Thus, I just reach in to get the child pids and kill -9 them.
            for k, v in executor._processes.items():
                try:
                    os.kill(v.pid, signal.SIGKILL)
                except ProcessLookupError:
                    log.info(f"{v.pid} had already died")
            raise


def _do_work_orders_thread_mode(zap):
    with ThreadPoolExecutor(
        max_workers=zap.max_workers, thread_name_prefix=zap.thread_name_prefix
    ) as executor:
        try:
            return _do_zap_with_executor(executor, zap)
        except BaseException as e:
            # Any sort of exception needs to clear all threads.
            # Note that KeyboardInterrupt inherits from BaseException not
            # Exception so using BaseException to include KeyboardInterrupts
            # Unlike above with os.kill(), the thread clears are not so destructive,
            # so we want to call them in any situation in which we're bubbling up the
            # exception.
            executor._threads.clear()
            thread._threads_queues.clear()
            raise e


def _do_work_orders_debug_mode(zap):
    """
    debug_mode skips all multi-processing so that console-based debuggers are happy
    """
    results = [None] * zap.n_work_orders
    timings = [None] * zap.n_work_orders
    for i, work_order in enumerate(zap.work_orders):
        result, duration = _run_work_order_fn(zap.id, i)
        results[i] = _examine_result(zap, result, work_order)
        timings[i] = duration
        _call_progress(zap, i)

    return results, timings


def get_cpu_limit(_cpu_limit=None):
    if _cpu_limit is None:
        _cpu_limit = _cpu_count()

    if _cpu_limit < 0:
        _cpu_limit = _cpu_count() + _cpu_limit  # eg: 4 cpu + (-1) is 3

    assert _cpu_limit > 0
    return _cpu_limit


def work_orders(_work_orders, _return_timings=False, _fn_name=None):
    """
    Runs work_orders in parallel.

    work_orders: List[Dict]
        Each work_order should have a "fn" element that points to the fn to run
        If the work_order has an "args" element those will be passed as *args
        all other elements of the work_order will be passed as **kwargs
    _return_timings:
        If True, then returns a tuple of results, timings
        otherwise just returns results
    """

    if _fn_name is None:
        try:
            _fn_name = _work_orders[0]["fn"].__name__
        except:
            _fn_name = "Unknown"
            pass

    zap = _set_zap(
        work_orders=_work_orders,
        n_work_orders=len(_work_orders),
        progress=_progress,
        thread_name_prefix=_thread_name_prefix,
        trap_exceptions=_trap_exceptions,
        max_workers=get_cpu_limit(_cpu_limit),
        fn_name=_fn_name,
    )

    if _zap_verbose:
        log.info(
            (
                f"\nStarting zap.id {zap.id & 0xFFFF:4x} on worker function '{_fn_name}'' "
                f"with {zap.n_work_orders} work_orders in mode '{_mode}'"
            )
            + (f" (using up to {zap.max_workers} workers)." if _mode != "debug" else "")
        )

    if _mode not in ("debug", "process", "thread"):
        raise ValueError(f"Unknown zap mode '{_mode}'")

    results, timings = None, None
    try:
        if _mode == "debug":
            # debug_mode takes precedence; ie over-rides any multi-processing
            results, timings = _do_work_orders_debug_mode(zap)
        elif _mode == "process":
            results, timings = _do_work_orders_process_mode(zap)
        elif _mode == "thread":
            results, timings = _do_work_orders_thread_mode(zap)
    except Exception as e:
        if hasattr(e, "exception_lines"):
            _show_work_order_exception(e)
        raise e
    finally:
        if _zap_verbose:
            log.info(f"\nDone zap.id {zap.id & 0xFFFF:4x} {_fn_name}.")
        _del_zap(zap.id)

    if _return_timings:
        return results, timings

    return results


def make_batch_slices(
    n_rows: int, _batch_size=None, _limit_slice=None, _batch_updates=3
):
    """
    Given some number of rows, create a list of batches using various methods
    for grouping the batches.

    Example:
        batches = make_batch_slices(10, _batch_size=2)
        batches == [(0, 5), (5, 10)]

    Note that the tuples of each batch are start and stop where stop is EXCLUSIVE!
    So the correct handling of a batch is:
        for i in range(start, stop):
    Or equivilent in C:
        for(i=start; i < stop; i++) ...

    If _batch_size is None:
        Then the batches are based on the number of cpus but so that the
        user will get some progress, the n_cpus is multiplied by _batch_updates
        so that for eaxmaple there are 3 updates per CPU.

    If _limit_slice is not None then the batches range over that
    slice instead of 0:n_rows
    """

    assert isinstance(n_rows, int)

    if _limit_slice is None:
        _limit_slice = slice(0, n_rows, 1)

    if isinstance(_limit_slice, int):
        _limit_slice = slice(0, _limit_slice, 1)

    _limit_slice = [_limit_slice.start, _limit_slice.stop, _limit_slice.step]

    if _limit_slice[2] is None:
        _limit_slice[2] = 1

    if _limit_slice[1] is None:
        _limit_slice[1] = n_rows

    assert _limit_slice[2] == 1  # Until I have time to think this through
    n_rows = _limit_slice[1] - _limit_slice[0]
    if n_rows == 0:
        return []

    if _batch_size is None:
        # If not specified, base it on the number of cpus.
        # Note, if n_batches is only as big as the _cpu_count then there won't
        # be any output on the progress bar until it is done so it is scaled
        # by _batch_updates here to ensure the progress bar will at least move
        # that number of times.
        n_batches = min(n_rows, _batch_updates * _cpu_count())
        batch_size = max(1, (n_rows // n_batches) + 1)
    else:
        batch_size = _batch_size
        n_batches = max(
            1, (n_rows // batch_size) + (0 if n_rows % batch_size == 0 else 1)
        )

    if batch_size <= 0:
        raise ValueError(f"illegal batch_size {batch_size}")

    assert batch_size * n_batches >= n_rows

    batch_slices = []
    for batch_i in range(n_batches):
        start = _limit_slice[0] + batch_i * batch_size
        stop = _limit_slice[0] + min((batch_i + 1) * batch_size, n_rows)
        if stop > start:
            batch_slices += [(start, stop)]
    return batch_slices


def _run_arrays(inner_fn, slice, arrays_dict, **kwargs):
    """
    Assumes that the lengths of the value arrays are all the same.
    """
    # SETUP the re-usable kwargs with parameters and arrays and then poke values one row at a time
    res = []
    for row_i in range(slice[0], slice[1]):
        for field_i, (key, array) in enumerate(arrays_dict.items()):
            kwargs[key] = array[row_i]
        val = inner_fn(**kwargs)
        if isinstance(val, tuple):
            res += [val]
        else:
            res += [(val,)]

    return res


def arrays(
    fn,
    arrays_dict,
    _batch_size=None,
    _stack=False,
    _limit_slice=None,
    **kwargs,
):
    """
    Split an array by its first dimension and send each row to fn.
    The array_dict is one or more parallel arrays that will
    be passed to fn(). **kwargs will end up as (constant) kwargs
    to fn().

    Example:
        def myfn(a, b, c):
            return a + b + c

        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])

        res = zap.arrays(
            myfn,
            dict(a=a, b=b),
            c=1
        )

        # This will call:
        #   myfn(1, 4, 1)
        #   myfn(2, 5, 1)
        #   myfn(3, 6, 1)
        # and res == [1+4+1, 2+5+1, 3+6+1]

    These calls are batched into parallel processes (or _process_mode is False)
    where the _batch_size is set or if None it will be chosen to use all cpus.

    When fn returns a tuple of fields, these return fields
    will be maintained.

    Example:
        def myfn(a, b, c):
            return a, b+c

        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])

        res = zap.arrays(
            myfn,
            dict(a=a, b=b),
            c=1
        )

        # This will call as before but now:
        #   res == ([1, 2, 3], [4+1, 5+1, 6+1])

    If _stack is True then _each return field_ will be wrapped
    with a np.array() before it is returned.  If _stack is a list
    then you can selective wrap the np.array only to the return
    fields of your choice.

    Example:
        def myfn(a, b, c):
            return a, b+c

        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])

        res = zap.arrays(
            myfn,
            dict(a=a, b=b),
            c=1,
            _stack=True
        )

        # This will call as before but now:
        #   res == (np.array([1, 2, 3]), np.array([4+1, 5+1, 6+1]))
        # Of called with _stack=[True, False]
        #   res == (np.array([1, 2, 3]), [4+1, 5+1, 6+1])
    """

    n_rows = len(list(arrays_dict.values())[0])
    assert all([len(a) == n_rows for a in arrays_dict.values()])

    batch_slices = make_batch_slices(n_rows, _batch_size, _limit_slice)

    result_batches = work_orders(
        _work_orders=[
            Munch(
                fn=_run_arrays,
                inner_fn=fn,
                slice=batch_slice,
                arrays_dict=arrays_dict,
                **kwargs,
            )
            for batch_slice in batch_slices
        ],
        _fn_name=fn.__name__,
    )

    if len(result_batches) == 0:
        raise ValueError("No batches were returned")
    first_batch = result_batches[0]
    if isinstance(first_batch, Exception):
        raise first_batch
    if len(first_batch) == 0:
        raise ValueError("First batch had no elements")
    first_return = first_batch[0]
    if isinstance(first_return, Exception):
        raise first_return

    assert isinstance(first_return, tuple)
    n_fields = len(first_return)

    unbatched = []
    for field_i in range(n_fields):
        field_rows = []
        for batch in result_batches:
            field_rows += utils.listi(batch, field_i)
        unbatched += [field_rows]

    if _stack is not None:
        if isinstance(_stack, bool):
            _stack = [_stack] * n_fields

        if isinstance(_stack, (list, tuple)):
            assert all([isinstance(s, bool) for s in _stack])
            assert len(_stack) == n_fields

        # If requested, wrap the return field in np.array()
        for field_i in range(n_fields):
            if _stack[field_i]:
                unbatched[field_i] = np.array(unbatched[field_i])

    if n_fields == 1:
        return unbatched[0]
    else:
        return tuple(unbatched)


def _run_df_rows(inner_fn, slice, df, **kwargs):
    """
    Assumes that the lengths of the value arrays are all the same.
    """
    # SETUP the re-usable kwargs with parameters and arrays and then poke values one row at a time
    res = []
    for row_i in range(slice[0], slice[1]):
        args = (df.iloc[row_i : row_i + 1],)
        val = inner_fn(*args, **kwargs)
        res += [val]

    return res


def df_rows(
    fn,
    df,
    _batch_size=None,
    _limit_slice=None,
    **kwargs,
):
    """
    Split a dataframe along its rows. I do not want to actually
    split it because I want to minimize what is serialized.
    """
    n_rows = len(df)

    batch_slices = make_batch_slices(n_rows, _batch_size, _limit_slice)

    result_batches = work_orders(
        _work_orders=[
            Munch(
                fn=_run_df_rows,
                inner_fn=fn,
                slice=batch_slice,
                df=df,
                **kwargs,
            )
            for batch_slice in batch_slices
        ],
        _fn_name=fn.__name__,
    )

    unbatched = []
    for batch in result_batches:
        for ret in batch:
            if not isinstance(ret, pd.DataFrame):
                raise TypeError(
                    "return values from the fn of df_rows must be DataFrames"
                )
            unbatched += [ret]
    return pd.concat(unbatched).reset_index(drop=True)


def df_groups(fn, df_group, **kwargs):
    """
    Run function on each group of groupby

    There is a lot of complexity to the way that groupby handles return
    values from the functions so I use the apply to accumulate the
    work orders and then use apply again to return the results and
    let the apply whatever magic it wants to to reformat the result
    """

    def _do_get_calls(group, **kwargs):
        return Munch(args=(group.copy(),), _index=tuple(group.index.values), **kwargs)

    # 3/15/2021 DHW: GroupBy.apply is doing something that makes the result of the above group.copy() get mangled when passed to zap workers.
    # _work_orders = df_group.apply(_do_get_calls)
    _work_orders = [_do_get_calls(group) for i, group in df_group]

    wo_kwargs = {}
    non_wo_kwargs = {}
    for k, v in kwargs.items():
        if k.startswith("_"):
            non_wo_kwargs[k] = v
        else:
            wo_kwargs[k] = v

    _work_orders_with_fn = []
    for wo in _work_orders:
        del wo["_index"]
        wo["fn"] = fn
        wo.update(wo_kwargs)
        _work_orders_with_fn += [wo]

    results = work_orders(_work_orders_with_fn, _fn_name=fn.__name__, **non_wo_kwargs)

    # results is a list. One element per work order which in this case is
    # one work_order per group.
    #
    # Each WO result is the return value of the function; if multiple
    # return values then it is a tuple.

    return results
