import gzip
import os
import pickle
import time
import typing as t
from dataclasses import dataclass

import numpy as np

from . import dytchunks as ds
from . import dytsim_opt as do

lib_ = None
rng_ = None


@dataclass
class EvalRecord:
    eval_time: float
    eval_dur: float
    n_samples: int
    err: float
    x0: np.array
    dyt: list[str]
    dytc: list[int]
    pep_recalls: list[float]
    src: str


@dataclass
class HistoryRecord:
    eval_time: float
    eval_dur: float
    it: int
    xk: np.array
    err: float
    extra: t.Any


class DyetrackGen:
    def __init__(self) -> None:
        # self.lib = None
        # self.rng = None
        pass

    def gen_dyetracks_plaster(
        self,
        args: list[float],
        n_samples: int,
        marker_labels: list[str],
        n_edmans: int,
        dyetides: list[str],
        verbose: bool = False,
        allow_edman_cterm: bool = False,
        p_label_failure: float = 0.0,
    ):
        import ctypes

        import plaster.run.sim_v3.c_dytsim.dytsim as dytsim
        from plaster.tools.c_common import c_common_tools

        (
            p_initial_block,
            p_cyclic_block,
            p_detach,
            p_edman_failure,
        ) = args[:4]

        p_bleach = [args[4 + i * 2 + 0] for i in range(len(marker_labels))]
        p_dud = [args[4 + i * 2 + 1] for i in range(len(marker_labels))]

        config = {
            "marker_labels": marker_labels,
            "p_dud": p_dud,
            "p_bleach": p_bleach,
            "p_initial_block": p_initial_block,
            "p_cyclic_block": p_cyclic_block,
            "p_detach": p_detach,
            "p_edman_failure": p_edman_failure,
            "allow_edman_cterm": allow_edman_cterm,
            "n_edmans": n_edmans,
            "p_label_failure": p_label_failure,
            "n_samples": n_samples,
        }

        sim_params, synth_pcbs = do.config_plaster(config, dyetides)

        def dytsim_chunk(sim_params, pcbs, n_samples, start_idx, stop_idx):
            global lib_, rng_

            assert stop_idx <= pcbs[-1, 0] + 1

            cbbs = sim_params.cbbs()

            with dytsim.context(
                pcbs,
                cbbs,
                n_samples,
                sim_params.n_channels,
                len(sim_params.labels),
                sim_params.cycles_array,
                sim_params.seq_params.p_cyclic_block,
                sim_params.seq_params.p_initial_block,
                sim_params.seq_params.p_detach,
                sim_params.seq_params.p_edman_failure,
                sim_params.seq_params.p_label_failure,
                sim_params.seq_params.allow_edman_cterm,
            ) as ctx:
                if lib_ is None:
                    lib_ = dytsim.load_lib()

                    # ARGH! These functions also effect the numpy prng.
                    seed = np.random.randint(0, high=1_000_000)
                    rng_ = c_common_tools.RNG(seed)

                # Not sure what ctx.count_only=1 does.
                # It's a pre-dyeseq-count vs. a pre-dyeseq-peptide count
                ctx.count_only = 0

                error = lib_.dytsim_batch(ctx, rng_, start_idx, stop_idx)

                n_chcy = ctx.n_channels * ctx.n_cycles
                dytmat = np.zeros(
                    (ctx._dytrecs.n_rows + 1, n_chcy), dtype=dytsim.DytType
                )
                dytpeps = np.zeros(
                    (ctx._dytpeps.n_rows + 1, 3), dtype=dytsim.DytPepType
                )

                lib_.copy_results(
                    ctx,
                    dytmat.ctypes.data_as(
                        ctypes.POINTER(c_common_tools.typedef_to_ctype("DytType"))
                    ),
                    dytpeps.ctypes.data_as(
                        ctypes.POINTER(c_common_tools.typedef_to_ctype("DytPepType"))
                    ),
                )

                return dytmat, dytpeps, ctx._pep_recalls

        def run_plaster(config, dyetides, synth_pcbs, sim_params):
            return dytsim_chunk(
                sim_params,
                synth_pcbs,
                config["n_samples"],
                0,
                int(synth_pcbs[-1, 0]) + 1,
            )

        nretries = 5
        for i in range(nretries):
            dyetracks, dytracks_count, pep_recalls = run_plaster(
                config, dyetides, synth_pcbs, sim_params
            )

            if n_samples in dytracks_count:
                print(f"WARNING: ONLY ONE DYETRACK (RETRY {i+1}/{nretries})", config)
            else:
                break

        d = {}
        for r in dytracks_count:
            a, b, c = r
            if a not in d:
                d[a] = 0
            d[a] += c

        for x in range(len(dyetracks)):
            if x not in d:
                d[x] = 0
        dytracks_count = [d[x] for x in range(len(dyetracks))]

        trial_dyetracks = ["".join(x) for x in dyetracks.astype(str)]
        trial_dyetracks_count = dytracks_count

        aa = sorted(zip(trial_dyetracks, trial_dyetracks_count))
        trial_dyetracks = [x[0] for x in aa]
        trial_dyetracks_count = [x[1] for x in aa]

        return trial_dyetracks, trial_dyetracks_count, pep_recalls

    def gen_dyetracks(
        self, args, n_samples, marker_labels, n_edmans, dyetides, verbose=False
    ):
        return self.gen_dyetracks_plaster(
            args, n_samples, marker_labels, n_edmans, dyetides, verbose
        )


class ObjectiveFunction:
    # A class that stores the target distribution and keeps track of all evaluation points and values.
    def __init__(
        self,
        true_dyetracks_raw: list[str],
        true_dyetracks_count_raw: list[int],
        marker_labels: list[str],
        n_edmans: int,
        dyetides: list[str],
        bounds: t.Optional[
            list[t.Optional[t.Union[float, tuple[float, float]]]]
        ] = None,
        n_samples: int = 1_000_000,
    ) -> None:
        # For RMSE calculation, dyetracks are assumed to be sorted
        lst = sorted(zip(true_dyetracks_raw, true_dyetracks_count_raw))
        self.true_dyetracks_raw = [x[0] for x in lst]
        self.true_dyetracks_count_raw = [x[1] for x in lst]

        self.true_dyetracks = self.true_dyetracks_raw
        self.true_dyetracks_count = self.true_dyetracks_count_raw

        self.marker_labels = marker_labels[:]
        self.dyetides = dyetides[:]

        self.n_edmans = n_edmans
        self.n_samples = n_samples

        # Record evaluated pts and results.
        self.all_tests: list[EvalRecord] = []
        # A label to associate with entries in all_tests to track the point's source.
        self.src = "unset"

        # Unless bounds passed to set_defaults, the instance's bounds
        # only contains bounds on variables being optimized. Constant
        # variables are marked in the instance's defaults.
        self.bounds: list[tuple[float, float]] = []
        # Variables held constant are marked in defaults. Variables
        # being optimized are marked as None.
        self.defaults: list[t.Optional[float]] = []

        # Names and expected order of variables in the expanded point.
        # TODO: it would be convenient if these names matched the names used in the Erisyon
        # simulator -- see comments per line
        self.nms: list[str] = [
            "p_initial_block",
            "p_cyclic_block",
            "p_detach",
            "p_edman_failure",
        ]

        # For consistency with WhatProt, name parameters using channel number rather
        # than AA label.
        for i, c in enumerate(self.marker_labels):
            self.nms += [f"p_bleach_ch{i}", f"p_dud_ch{i}"]
        # Names and expected order of variables in the trail point.
        self.pnms: list[str] = self.nms

        # Used to print once the expansion of a trail point to a point
        # with all variables specified (trial + defaults).
        self.show_expand: bool = True

        # Call only after nms and pnms are initialized.
        self.set_bounds(bounds)

        self.random_res: list[float] = []
        self.sample_res: list[float] = []

        self.dyt_gen = DyetrackGen()

    def set_bounds(self, bounds: list[t.Optional[t.Union[float, tuple[float, float]]]]):
        self.show_expand = True
        if bounds is None:
            self.pnms = self.nms
            self.bounds = [(0.0, 1.0)] * len(self.nms)
            self.defaults = [None] * len(self.nms)
        else:
            assert len(self.nms) == len(bounds)

            self.pnms = []
            self.bounds = []
            self.defaults = []

            for nm, b in zip(self.nms, bounds):
                if isinstance(b, (int, float)):
                    self.defaults += [float(b)]
                else:
                    self.defaults += [None]
                    if b is None:
                        self.bounds += [(0.0, 1.0)]
                    else:
                        self.bounds += [b]
                    self.pnms += [nm]

    def set_n_samples(self, n_samples):
        self.n_samples = n_samples

    def set_defaults(self, defaults):
        if defaults is None:
            self.defaults = [None] * len(self.nms)
        else:
            self.defaults = defaults[:]

    def objective_func_plaster(
        self, args: list[float], verbose=False
    ) -> tuple[float, list[str], list[int], list[float]]:
        (
            trial_dyetracks,
            trial_dyetracks_count,
            pep_recalls,
        ) = self.dyt_gen.gen_dyetracks(
            args,
            self.n_samples,
            self.marker_labels,
            self.n_edmans,
            self.dyetides,
            verbose,
        )

        err = ds.compare_opt(
            self.true_dyetracks,
            self.true_dyetracks_count,
            trial_dyetracks,
            trial_dyetracks_count,
        )

        err = np.sqrt(err) * 10_000

        return err, trial_dyetracks, trial_dyetracks_count, pep_recalls

    def objective_func(
        self, x0: t.Union[list[float], tuple[float], np.array], do_expand=True
    ) -> float:
        if do_expand:
            x0 = self.expand_x(x0)

        for a, b in zip(x0, self.bounds):
            # if a < b[0] - .0001 or b[1] + .0001 < a:
            if a < 0.0 or 1.0 < a:
                print("To infinity", x0, self.bounds)
                return float("inf")

        t0 = time.time()
        err, dyt, dytc, pep_recalls = self.objective_func_plaster(x0)
        dt = time.time() - t0

        self.all_tests += [
            EvalRecord(
                t0,
                dt,
                self.n_samples,
                err,
                np.array(x0),
                dyt,
                dytc,
                pep_recalls,
                self.src,
            )
        ]
        return err

    def random_x(self) -> list[float]:
        return [np.random.uniform(l, h) for l, h in self.bounds]

    def expand_x(self, x: list[float]) -> list[float]:
        # The simulator expects all values to be specified regardless if they
        # are part of the optimization. Here the potentially sparse point being
        # evaluated is expanded to contain any held constant parameters.
        assert len(x) == len(self.bounds)

        it = iter(x)
        x_ = []
        for d in self.defaults:
            if d is None:
                x_.append(next(it))
            else:
                x_.append(d)

        if self.show_expand and len(x_) != len(x):
            self.show_expand = False
            print(f"Expanded {x} to {x_}")

        return x_

    def mask_x(self, x: list[float]) -> list[float]:
        # Mask off constant values from X, returning only the values
        # that are from optimization.
        if len(x) == len(self.bounds):
            # Already masked.
            return x

        assert len(x) == len(self.defaults)

        x_ = []
        for d, x2 in zip(self.defaults, x):
            if d is None:
                x_.append(x2)

        return x_

    def resample(self, reset=False) -> None:
        if reset:
            self.true_dyetracks = self.true_dyetracks_raw
            self.true_dyetracks_count = self.true_dyetracks_count_raw
        else:
            import random
            from collections import Counter

            x = random.choices(
                self.true_dyetracks_raw,
                weights=self.true_dyetracks_count_raw,
                k=np.sum(self.true_dyetracks_count_raw),
            )
            x = sorted(Counter(x).items())
            self.true_dyetracks = [x[0] for x in x]
            self.true_dyetracks_count = [x[1] for x in x]

    def report(
        self, x0: list[float]
    ) -> tuple[float, list[str], list[int], list[str], list[int]]:
        x = self.expand_x(x0)
        err, dyt, dytc, pep_recalls = self.objective_func_plaster(x)

        print("RMSE:", err)

        aa = ds.compare(self.true_dyetracks, self.true_dyetracks_count, dyt, dytc)
        aa = sorted(aa, key=lambda x: x[1] + x[2], reverse=True)
        at, bt = sum(self.true_dyetracks_count), sum(dytc)
        for s, ac, bc in aa:
            print(f"{s} {ac / at * 100:6.2f} {bc / bt * 100:6.2f} {ac:6d} {bc:6d}")

        return (
            err,
            self.true_dyetracks,
            self.true_dyetracks_count,
            dyt,
            dytc,
        )

    def build_expected_value(
        self, n_it: int = 100, reset: bool = False, random=True
    ) -> tuple[float, float, float]:
        t = self.src

        if reset:
            if random:
                self.random_res = []
            else:
                self.sample_res = []

        if random:
            self.random_res = []
            self.src = "sample"
        else:
            self.sample_res = []
            self.src = "sample_x"

        for _ in range(n_it):
            x = self.random_x() if random else self.x

            f = self.objective_func(x)

            if random:
                self.random_res += [f]
            else:
                self.sample_res += [f]

        self.src = t

    def likelihood(self, x_: float) -> tuple[float, float, float]:
        m = self.random_res

        if len(m) == 0:
            return float("inf"), float("inf"), float("inf")

        likelihood = len([x for x in m if x < x_]) / len(m)
        mm = np.min(m)

        return likelihood, mm, (x_ - mm) / mm


class OptimizeBase:
    def __init__(
        self,
        obj_fnc: t.Callable[[list[float]], float],
        obj_fnc_class: ObjectiveFunction,
        bounds: t.Optional[list[tuple[float, float]]] = None,
        x0: t.Optional[list[float]] = None,
    ) -> None:
        if bounds is None:
            self.bounds: list[tuple[float, float]] = obj_fnc_class.bounds[:]
        else:
            self.bounds = bounds[:]
        self.res: t.Any = None
        self.obj_fnc: t.Callable[[list[float]], float] = obj_fnc
        self.obj_fnc_class: ObjectiveFunction = obj_fnc_class
        self.x: list[float] = None
        self.fun: float = None
        self.fun_polish: float = None
        self.history: list[HistoryRecord] = []
        self.x0: t.Optional[list[float]] = None
        self.t0: float = None
        self.go_runtime: float = 0
        self.it: int = 0

        self.set_x0(x0)

    def set_x0(self, x0: t.Optional[list[float]]):
        self.x0 = None
        if x0 is not None:
            self.x0 = x0[:]

    def set_x0_not_supported(self, x0: t.Optional[list[float]]):
        if x0 is not None:
            # print("x0 is not supported")
            x0 = None

    def go(self):
        self.t0 = time.time()
        self.it = 0

    def polish(self, n=100):
        self.fun_polish = [self.fun]
        for i in range(1, n):
            self.fun_polish.append(self.obj_fnc(self.x))
        self.fun = np.median(self.fun_polish)

    def add_history(self, xk: list[float], err: float, extra: t.Optional[t.Any] = None):
        dt = time.time() - self.t0
        if isinstance(xk, np.ndarray):
            xk = xk.tolist()
        else:
            xk = xk[:]
        self.history += [HistoryRecord(time.time(), dt, self.it, xk, err, extra)]

    def cb_base(self, xk: list[float], err: float, extra: t.Optional[t.Any] = None):
        likelihood, mm, _ = self.obj_fnc_class.likelihood(err)

        self.add_history(xk, err, extra)

        dt = time.time() - self.t0
        xks = ", ".join([f"{x:.6f}" for x in xk])

        extra_s = f"   {extra}" if extra is not None else ""

        print(
            f"{self.it:6d} {dt:8.2f} [{xks}] {err:9.2f}   = min({mm:9.2f}) * {err / mm * 100:6.2f}%   p={likelihood*100:5.2f}%"
            + extra_s
        )

        self.it += 1

        return False

    def cb(self, xk: list[float]):
        err = self.obj_fnc(xk)
        return self.cb_base(xk, err)


class OptimizeDirect(OptimizeBase):
    def __init__(
        self,
        obj_fnc,
        obj_fnc_class,
        bounds: t.Optional[list[tuple[float, float]]] = None,
        x0=None,
    ) -> None:
        self.set_x0 = self.set_x0_not_supported
        super().__init__(obj_fnc, obj_fnc_class, bounds, x0)

    def go(self) -> tuple[list[float], float]:
        super().go()

        from scipy.optimize import direct

        self.res = direct(
            self.obj_fnc,
            self.bounds,
            callback=self.cb,
            locally_biased=False,
            #   eps=0.001
        )

        self.x = [x for x in self.res.x]
        self.fun = self.res.fun
        self.go_runtime = time.time() - self.t0

        if not self.res.success:
            print("Warning: optimizer did not converge")

        return self.x, self.fun


class OptimizeSciPyMinimizePowell(OptimizeBase):
    def __init__(
        self,
        obj_fnc,
        obj_fnc_class,
        bounds: t.Optional[list[tuple[float, float]]] = None,
        x0=None,
    ) -> None:
        super().__init__(obj_fnc, obj_fnc_class, bounds, x0)

    def go(self) -> tuple[list[float], float]:
        super().go()

        self.cb(self.x0)
        print()

        from scipy.optimize import minimize

        # Can specify initial direction vectors with option direc:ndarry.
        self.res = minimize(
            self.obj_fnc,
            self.x0,
            method="powell",
            bounds=self.bounds,
            callback=self.cb,
            options={"disp": True, "return_all": True},
        )

        if not self.res.success:
            print("Minimize did not succeed:", self.res.message)

        best = sorted(self.history, key=lambda x: x.err)[0]
        self.x = best.xk[:]
        self.fun = best.err

        self.go_runtime = time.time() - self.t0

        return self.x, self.fun
