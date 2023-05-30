import copy
import gzip
import itertools
import typing as t

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from . import dytchunks as ds
from . import dytsim_opt as do
from . import opt_routines as dr

tqdm_disabled = False


def silence_progress_bar(disable=True) -> None:
    from functools import partialmethod

    tqdm.__init__ = partialmethod(tqdm.__init__, disable=disable)
    global tqdm_disabled
    tqdm_disabled = disable


class ParameterScan:
    def __init__(
        self,
        x0: list[float],
        obj_fnc: t.Callable[[list[float]], float],
        obj_fnc_class: dr.ObjectiveFunction,
    ) -> None:
        self.x0: list[float] = []
        self.obj_fnc: t.Callable[[list[float]], float] = obj_fnc
        self.obj_fnc_class: dr.ObjectiveFunction = obj_fnc_class

        self.bounds = self.obj_fnc_class.bounds

        self.random_res_0d: list[float] = []

        self.cache = {}

        self.set_x0(x0)

    def set_x0(self, x0: t.Optional[list[float]]) -> None:
        self.x0 = copy.deepcopy(x0)
        if x0 is not None:
            k = tuple(x0)
            if k not in self.cache:
                self.cache[k] = {}

    def parameter_scan(self, args, report_progress=True) -> np.array:
        sizes = [len(x) for x in args]
        res = np.zeros(sizes, dtype=float)

        for a in tqdm(
            list(itertools.product(*[range(x) for x in sizes])),
            desc="Evaluating Parameters",
            ncols=80,
            disable=tqdm_disabled or not report_progress,
        ):
            aa = [args[i][a[i]] for i in range(len(args))]
            res[a] = self.obj_fnc(aa)
            # print(a, aa, res[a])

        return res

    def _plot_expected(self, m, nbins: int = 20) -> None:
        plt.hist(m, density=True, bins=nbins)
        plt.ylabel("Frequency")
        plt.xlabel(
            "Error value\n"
            + r" $\mu$="
            + f"{np.mean(m):.1f}"
            + r", $\sigma$="
            + f"{np.std(m):.1f}"
        )
        plt.title(f"Expected Value (n={len(m)})")
        # plt.yscale('log')
        plt.show()

    def random_zero_d_plot(self, nbins: int = 20) -> None:
        m = self.obj_fnc_class.random_res

        if len(m) == 0:
            print("Need to first run obj_fnc_class.build_expected_value.")
            return

        self._plot_expected(m, nbins)

    def zero_d_gen(self, n_it: int = 100, reset: bool = False) -> None:
        x0 = self.x0

        if False:
            # More elegant, but self.parameter_scan needs a rep=n argument
            # else report_progress=False to avoid n_it messages of one-unit
            #  work being completed.

            self.cache[k]["args_0d"] = [[x] for x in x0]

            res0 = [
                self.parameter_scan(self.cache[k]["args_0d"], report_progress=False)
                for x in range(n_it)
            ]
            res0 = np.squeeze(res0)
        else:
            res0 = []
            for _ in tqdm(
                list(range(n_it)),
                desc="Evaluating Parameters",
                ncols=80,
                # disable=not report_progress,
            ):
                res0 += [self.obj_fnc(x0)]

        k = tuple(self.x0)
        # Accumulate results until satisfied with the distribution, unless reset.
        if "res_0d" not in self.cache[k] or reset:
            self.cache[k]["res_0d"] = []
        self.cache[k]["res_0d"] += res0

    def zero_d_plot(self, nbins: int = 20) -> None:
        k = tuple(self.x0)
        m = self.cache[k]["res_0d"]

        if len(m) == 0:
            print("Need to first run zero_d_gen.")
            return

        self._plot_expected(m, nbins)

    def one_d_gen(
        self, n_it: int = 100, bounds: t.Optional[list[tuple[float, float]]] = None
    ) -> None:
        x0 = self.x0

        args_1d0 = [[x] for x in x0]
        # args_1d0 = [[np.random.random() * .3] for _ in range(len(x0))]

        res0_1d_all = []
        res_1d_all = []
        res_1d_nms = self.obj_fnc_class.pnms

        if bounds is None:
            bounds = self.bounds[:]

        for i in range(len(args_1d0)):
            args_1d = args_1d0[:]
            args_1d[i] = np.linspace(*bounds[i], n_it)

            res0 = self.parameter_scan(args_1d)
            res = res0

            # res = np.sqrt(res0)
            # res = (res - np.min(res)) / (np.max(res) - np.min(res))

            res0_1d_all.append(np.squeeze(res0))
            res_1d_all.append(np.squeeze(res))

        # for i in range(len(res_1d_all)):
        #     res_1d_all[i] = (res_1d_all[i] - np.min(res_1d_all)) / (np.max(res_1d_all) - np.min(res_1d_all))

        k = tuple(self.x0)
        self.cache[k]["args_1d0"] = args_1d0
        self.cache[k]["res_1d_range"] = bounds[:]
        self.cache[k]["res0_1d_all"] = res0_1d_all
        self.cache[k]["res_1d_all"] = res_1d_all
        self.cache[k]["res_1d_nms"] = res_1d_nms

    def one_d_plot(self) -> tuple[t.Any, t.Any]:
        k = tuple(self.x0)
        args_1d0 = self.cache[k]["args_1d0"]
        res_1d_range = self.cache[k]["res_1d_range"]
        res0_1d_all = self.cache[k]["res0_1d_all"]
        res_1d_all = self.cache[k]["res_1d_all"]
        res_1d_nms = self.cache[k]["res_1d_nms"]

        # fig, axs = plt.subplots(2, 4, figsize=(14, 10), sharey="row", tight_layout=True)
        fig, axs = plt.subplots(2, 4, figsize=(7, 5), sharey="row", tight_layout=True)
        for i in range(len(res_1d_nms)):
            x = i % 4
            y = i // 4
            x_pts = np.linspace(*res_1d_range[i], len(res_1d_all[i]))
            axs[y][x].plot(x_pts, res_1d_all[i])

            xx = args_1d0[i][0]
            # axs[y][x].axvline(x=xx, c="r")
            yy = np.interp(xx, x_pts, res_1d_all[i])
            axs[y][x].plot(xx, yy, "ro", markersize=1)

            if x == 0:
                axs[y][x].set_ylabel("Error")
            axs[y][x].set_xlabel(res_1d_nms[i])

        plt.suptitle("Parameter Scans")
        plt.show()

        return fig, axs

    def three_d_scan_gen(
        self,
        n_it: int = 4,
        axes: list[str] = [
            "p_bleach_A",
            "p_initial_blocking",
            "p_edman_failure",
        ],
        bounds: t.Optional[list[tuple[float, float]]] = None,
        normalize: bool = False,
    ) -> None:
        if bounds is None:
            bounds = self.bounds[:]

        x0 = self.obj_fnc_class.expand_x(self.x0)

        args_3d = []
        for x, nm in zip(x0, self.obj_fnc_class.nms):
            if nm in axes:
                args_3d += np.linspace(0.005, 1.0, n_it)
            else:
                args_3d += [[x]]

        res0 = self.parameter_scan(args_3d)
        res = res0

        if normalize:
            res = (res - np.min(res)) / (np.max(res) - np.min(res))

        k = tuple(self.x0)
        self.cache[k]["args_3d"] = args_3d
        self.cache[k]["res0_3d"] = res0
        self.cache[k]["res_3d"] = res
        self.cache[k]["res_3d_nms"] = axes[:]

    def three_d_scan_plot(self) -> tuple[t.Any, t.Any]:
        k = tuple(self.x0)
        args_3d = self.cache[k]["args_3d"]
        res = self.cache[k]["res_3d"]

        (
            p_bleach_A_,
            p_dud_A_,
            p_initial_blocking_,
            p_per_cycle_blocking_,
            p_detach_,
            p_edman_failure_,
        ) = args_3d

        fig, axs = plt.subplots(
            1,
            4,
            figsize=(14, 4),
            sharex=True,
            sharey=True,
            tight_layout=True,
        )
        for jj in range(4):
            # for ii in range(3):
            # aa = res[:, 0, :, 0, 0, jj]
            aa = res[:, :, jj]
            axs[jj].imshow(aa, vmin=np.min(res), vmax=np.max(res), cmap="hot")
            axs[jj].set_xticks(np.arange(0, 4, 1))
            axs[jj].set_xticklabels([f"{x:.3f}" for x in p_initial_blocking_])
            axs[jj].set_yticks(np.arange(0, 4, 1))
            axs[jj].set_yticklabels([f"{x:.3f}" for x in p_bleach_A_])

            axs[0].set_ylabel(f"p_bleach_A")

            axs[jj].set_xlabel(
                f"p_initial_blocking w/ p_edman_failure = {p_edman_failure_[jj]:.3f}"
            )

            for (j, i), label in np.ndenumerate(aa):
                axs[jj].text(
                    i,
                    j,
                    f"{label:.02f}",
                    ha="center",
                    va="center",
                    bbox=dict(facecolor="white", edgecolor="black", boxstyle="round"),
                )

        # plt.suptitle("Parameter Scan (Normalized Error)")
        plt.suptitle("Parameter Scan")
        plt.show()

        return fig, axs

    def four_d_scan_gen(
        self,
        n_it: int = 4,
        axes: list[str] = [
            "p_bleach_A",
            "p_per_cycle_blocking",
            "p_detach",
            "p_edman_failure",
        ],
        bounds: t.Optional[list[tuple[float, float]]] = None,
        normalize: bool = False,
    ) -> None:
        x0 = self.obj_fnc_class.expand_x(self.x0)
        if bounds is None:
            bounds = self.bounds[:]

        args_4d = []
        for x, bound, nm in zip(x0, bounds, self.obj_fnc_class.nms):
            if nm in axes:
                args_4d += [np.linspace(*bound, n_it)]
            else:
                args_4d += [[x]]

        res0 = self.parameter_scan(args_4d)
        res = res0

        if normalize:
            # Move normalize to plot?
            res = (res - np.min(res)) / (np.max(res) - np.min(res))

        k = tuple(self.x0)
        self.cache[k]["args_4d"] = args_4d
        self.cache[k]["res0_4d"] = res0
        self.cache[k]["res_4d"] = res
        self.cache[k]["res_4d_nms"] = axes[:]

    def four_d_scan_plot(self) -> tuple[t.Any, t.Any]:
        k = tuple(self.x0)
        args_4d = self.cache[k]["args_4d"]
        res = self.cache[k]["res_4d"]

        (
            p_bleach_A_,
            p_dud_A_,
            p_initial_blocking_,
            p_per_cycle_blocking_,
            p_detach_,
            p_edman_failure_,
        ) = args_4d

        fig, axs = plt.subplots(
            4,
            4,
            figsize=(6 * 3, 8 * 2 + 3),
            sharex=True,
            sharey=True,
            tight_layout=True,
        )
        for jj in range(4):
            for ii in range(4):
                aa = res[:, 0, 0, :, ii, jj]
                axs[jj][ii].imshow(aa, vmin=np.min(res), vmax=np.max(res), cmap="hot")
                axs[jj][ii].set_xticks(np.arange(0, 4, 1))
                axs[jj][ii].set_xticklabels([f"{x:.3f}" for x in p_bleach_A_])
                axs[jj][ii].set_yticks(np.arange(0, 4, 1))
                axs[jj][ii].set_yticklabels([f"{x:.3f}" for x in p_per_cycle_blocking_])

                if ii == 0:
                    axs[jj][ii].set_ylabel(
                        f"p_per_cycle_blocking w/ p_detach = {p_detach_[jj]:.3f}"
                    )

                if jj == 4 - 1:
                    axs[jj][ii].set_xlabel(
                        f"p_bleach_A w/ p_edman_failure = {p_edman_failure_[ii]:.3f}"
                    )

                for (j, i), label in np.ndenumerate(aa):
                    axs[jj][ii].text(
                        i,
                        j,
                        f"{label:.02f}",
                        ha="center",
                        va="center",
                        bbox=dict(
                            facecolor="white", edgecolor="black", boxstyle="round"
                        ),
                    )

        # plt.suptitle("Parameter Scan (Normalized Error)")
        plt.suptitle("Parameter Scan")
        plt.show()

        return fig, axs

    def random_1d(self, n_it=40) -> None:
        if self.obj_fnc_class is None:
            return

        random_x = self.obj_fnc_class.random_x()

        print(random_x, flush=True)

        ps = ParameterScan(random_x, self.obj_fnc, self.obj_fnc_class)
        ps.one_d_gen(n_it)
        ps.one_d_plot()
        plt.show()


class OtherGraphs:
    def __init__(
        self,
        opt: dr.OptimizeBase,
        x0: list[float],
        obj_fnc_class: t.Optional[dr.ObjectiveFunction] = None,
        exp_name: t.Optional[str] = None,
    ) -> None:
        self.opt: dr.OptimizeBase = opt
        self.x0 = copy.deepcopy(x0)
        self.obj_fnc_class: t.Optional[dr.ObjectiveFunction] = obj_fnc_class
        self.exp_name: t.Optional[str] = exp_name
        self.param_nms: list[str] = self.obj_fnc_class.pnms
        self.pca = None
        self.pca_y = None
        self.evar = None

    def plot_param_freq(self, filter=None) -> None:
        all_x = np.array(
            [
                x.x0
                for x in self.obj_fnc_class.all_tests
                if filter is None or x.src == filter
            ]
        )

        # if self.obj_fnc_class.one_count:
        #     all_x = [(x[0], x[2], x[5]) for x in all_x]
        #     all_x = np.array(all_x)
        #     #   x = [x[0], 0., x[1], 0., 0., x[2]]

        from scipy.stats import gaussian_kde

        nms = self.obj_fnc_class.nms
        n = all_x.shape[1]

        fig, axs = plt.subplots(
            n,
            n,
            figsize=(12, 10),
            # sharex=True,
            # sharey=True,
            tight_layout=True,
        )

        for i in range(n):
            for j in range(n):
                if i == j:
                    axs[i][j].hist(all_x[:, i])
                    axs[i][j].set_xlim((0.0, 1.0))
                    axs[i][j].set_yscale("log")
                    if i != n - 1:
                        axs[i][j].set_xticklabels([])
                    axs[i][j].set_yticklabels([])
                else:
                    x, y = all_x[:, j], all_x[:, i]
                    xy = np.vstack([x, y])
                    try:
                        z = gaussian_kde(xy)(xy)
                    except np.linalg.LinAlgError as e:
                        if "Singular matrix" in str(e):
                            print("Singular matrix")
                        else:
                            print(e)
                        continue

                    idx = z.argsort()
                    x, y, z = x[idx], y[idx], z[idx]

                    axs[i][j].scatter(x, y, c=z, s=20)
                    axs[i][j].set_xlim((0.0, 1.0))
                    axs[i][j].set_ylim((0.0, 1.0))

                    if i != n - 1:
                        axs[i][j].set_xticklabels([])

                    axs[i][j].set_yticklabels([])

                if i == n - 1:
                    axs[i][j].set_xlabel(nms[j])

                if j == 0:
                    axs[i][j].set_ylabel(nms[i])

        if filter is None:
            plt.suptitle("Parameter Test Coverage")
        else:
            plt.suptitle(f"Parameter Test Coverage ({filter})")
        plt.show()

    def g1(self) -> None:
        y = [x.err for x in self.opt.history]
        ym = np.minimum.accumulate(y)
        plt.plot(y, label="Fit")
        plt.plot(ym, label="Best Fit")

        plt.xlabel("Run")
        plt.ylabel("Best fit")
        plt.title(f"Objective fit by Run  {self.exp_name}")
        plt.legend()
        plt.show()

    def g2(self) -> None:
        all_params = [x.xk for x in self.opt.history]
        for i, nm in enumerate(self.param_nms):
            plt.plot(sorted([x[i] for x in all_params]), label=nm)
        plt.legend()
        plt.ylabel("Parameter value")
        plt.xlabel("Parameter sample")
        plt.title(f"Distribution of Parameter Values  {self.exp_name}")
        plt.ylim([0.0, 1.0])

    def g3(self) -> None:
        all_params = [x.xk for x in self.opt.history]
        nms = self.param_nms

        fig, axs = plt.subplots(1, len(nms), figsize=(10, 5))
        for i, nm in enumerate(nms):
            d_ = [x[i] for x in all_params]
            axs[i].hist(d_, label=nm, bins=15, density=False)
            mean = np.mean(d_)
            std = np.std(d_)
            axs[i].set_xlabel(f"{nm}\n$\\mu={mean:.3f}$\n$\\sigma={std:.3f}$")
            axs[i].set_xlim([0.0, 1.0])
            axs[i].set_yticklabels([])
            # axs[i].set_yscale('log')
        # plt.legend()
        # plt.ylabel('Parameter value')
        # plt.xlabel('Parameter sample')
        plt.suptitle(f"Parameter Histograms {self.exp_name}")

    def g4(self) -> None:
        all_params = [x.xk for x in self.opt.history]
        fns = [x.err for x in self.opt.history]
        nms = self.param_nms

        fig, axs = plt.subplots(
            len(nms), len(nms), sharex=False, sharey=False, tight_layout=True
        )
        for i, ynm in enumerate(nms):
            axs[i][0].set_ylabel(ynm, fontsize=8)
            for j, xnm in enumerate(nms):
                if i == j:
                    axs[i][j].hist([x[i] for x in all_params])
                    axs[i][j].set_yscale("log")
                else:
                    c = [x for x in fns]
                    # axs[i][j].scatter([x[0][i] for x in accums2_], [x[0][j] for x in accums2_], s=1, c=c, cmap='hot')
                    axs[i][j].scatter(
                        [x[i] for x in all_params], [x[j] for x in all_params], s=1
                    )
                    axs[i][j].set_xlim([0, 1.0])
                    axs[i][j].set_ylim([0, 1.0])
                axs[-1][j].set_xlabel(xnm, fontsize=8)
                # import matplotlib.ticker as ticker
                # axs[i][j].xaxis.set_major_locator(ticker.NullLocator())
                # axs[i][j].yaxis.set_major_locator(ticker.NullLocator())
                if j != 0:
                    axs[i][j].yaxis.set_visible(False)
                if i != len(nms) - 1:
                    axs[i][j].xaxis.set_visible(False)
                axs[i][j].set_xticks([])
                axs[i][j].set_yticks([])
                axs[i][j].set_xticklabels([])
                axs[i][j].set_yticklabels([])
        plt.suptitle("Parameters Correlation Matrix")
        # plt.legend()
        # plt.ylabel('Parameter value')
        # plt.xlabel('Parameter sample')
        # plt.title('Distribution of parameter values')

    def g5(self) -> None:
        all_params = [x.xk for x in self.opt.history]
        fns = [x.err for x in self.opt.history]
        nms = self.param_nms

        # all_params = all_params[5:]
        # fns = fns[5:]

        x = np.vstack(all_params)
        c = fns
        from sklearn.decomposition import PCA

        npc = len(nms)

        pca = PCA(n_components=npc, whiten=False)
        pca.fit(x)
        print(pca.singular_values_)
        pca.score_samples(x).shape
        y = pca.transform(x)
        pca.get_params()
        evar = pca.explained_variance_ratio_

        self.pca = pca
        self.evar = evar
        self.pca_y = y

        print(pca.explained_variance_ratio_)
        print(pca.components_)
        plt.imshow(pca.components_, cmap="coolwarm", vmin=-1, vmax=1)
        plt.xticks(range(len(nms)), nms, fontsize=6)
        plt.yticks(
            range(len(evar)),
            [f"{i+1}:{x*100:.1f}%" for i, x in enumerate(evar)],
            fontsize=8,
        )
        plt.ylabel("PCA Component")
        plt.xlabel("Parameter")
        ax = plt.gca()
        for (j, i), label in np.ndenumerate(pca.components_):
            ax.text(i, j, f"{label:.03f}", ha="center", va="center")

        plt.show()

    def g5b(self) -> None:
        pca = self.pca
        evar = self.evar
        y = self.pca_y
        fns = [x.err for x in self.opt.history]
        c = fns
        nms = self.param_nms

        fig, axs = plt.subplots(
            len(nms),
            len(nms),
            sharex=False,
            sharey=False,
            tight_layout=True,
            figsize=(15, 15),
        )
        npc = y.shape[1]
        for pc1 in range(npc):
            for pc2 in range(npc):
                if pc1 == pc2:
                    axs[pc2][pc1].hist(y[:, pc1])
                else:
                    # plt.scatter(y[:, pc1], y[:, pc2], s=10, c=c, cmap='hot', linewidth=.4, edgecolor='black')
                    # pca.get_params()
                    # evar = pca.explained_variance_ratio_
                    # # plt.title(f'PCA of Fitted Parameters ({sum(evar[:2])*100:.01f}%)')
                    # plt.title(f'PCA of Fitted Parameters ({(evar[pc1] + evar[pc2])*100:.01f}%)')
                    # plt.xlabel(f'Component {pc1+1} ({evar[pc1]*100:.01f}%)')
                    # plt.ylabel(f'Component {pc2+1} ({evar[pc2]*100:.01f}%)')
                    # plt.show()

                    axs[pc2][pc1].scatter(
                        y[:, pc1],
                        y[:, pc2],
                        s=10,
                        c=c,
                        cmap="hot",
                        linewidth=0.4,
                        edgecolor="black",
                    )
                    # plt.title(f'PCA of Fitted Parameters ({sum(evar[:2])*100:.01f}%)')
                    # axs[pc2][pcq].title(f'PCA of Fitted Parameters ({(evar[pc1] + evar[pc2])*100:.01f}%)')

        plt.suptitle(f"PCA Cross Matrix of Fitted Parameters {self.exp_name}")
        for pc1 in range(npc):
            axs[npc - 1][pc1].set_xlabel(f"Component {pc1+1} ({evar[pc1]*100:.01f}%)")
            axs[pc1][0].set_ylabel(f"Component {pc1+1} ({evar[pc1]*100:.01f}%)")
        plt.show()

    def g6(self) -> None:
        pca = self.pca
        evar = self.evar
        y = self.pca_y
        npc = y.shape[1]
        fns = [x.err for x in self.opt.history]
        c = fns

        if npc < 2:
            return

        pc1, pc2 = 0, 1
        plt.scatter(
            y[:, pc1],
            y[:, pc2],
            s=10,
            c=c,
            cmap="hot",
            linewidth=0.4,
            edgecolor="black",
        )
        pca.get_params()
        self.evar = pca.explained_variance_ratio_
        # plt.title(f'PCA of Fitted Parameters ({sum(evar[:2])*100:.01f}%)')
        plt.title(f"PCA of Fitted Parameters ({(evar[pc1] + evar[pc2])*100:.01f}%)")
        plt.xlabel(f"Component {pc1+1} ({evar[pc1]*100:.01f}%)")
        plt.ylabel(f"Component {pc2+1} ({evar[pc2]*100:.01f}%)")
        plt.show()

        # axs[pc2][pc1].scatter(y[:, pc1], y[:, pc2], s=10, c=c, cmap='hot', linewidth=.4, edgecolor='black')
        # plt.title(f'PCA of Fitted Parameters ({sum(evar[:2])*100:.01f}%)')
        # axs[pc2][pcq].title(f'PCA of Fitted Parameters ({(evar[pc1] + evar[pc2])*100:.01f}%)')

    def g7(self) -> None:
        pca = self.pca
        evar = self.evar
        y = self.pca_y
        npc = y.shape[1]
        fns = [x.err for x in self.opt.history]
        c = fns

        if npc < 3:
            return

        fig = plt.figure(figsize=(15, 15))
        ax = fig.add_subplot(projection="3d")

        pc1, pc2, pc3 = 0, 1, 2

        xs = y[:, pc1]
        ys = y[:, pc2]
        zs = y[:, pc3]
        m = "o"
        ax.scatter(xs, ys, zs, s=80, c=c, cmap="hot", linewidth=0.4, edgecolor="black")

        ax.set_xlabel(f"Component {pc1+1} ({evar[pc1]*100:.01f}%)")
        ax.set_ylabel(f"Component {pc2+1} ({evar[pc2]*100:.01f}%)")
        ax.set_zlabel(f"Component {pc3+1} ({evar[pc3]*100:.01f}%)")


def zero_d_compare(dat, labels=None) -> None:
    dat_filtered = [
        [x for x in a if np.mean(a) - 3 * np.std(a) < x < np.mean(a) + 3 * np.std(a)]
        for a in dat
    ]
    ff = [len(a) - len(af) for a, af in zip(dat, dat_filtered)]
    if any(ff):
        print(f"Filtered: {ff}")

    chrs = [chr(ord("a") + i) for i in range(len(dat))]
    if labels is None:
        labels = chrs

    for af, nm in zip(dat_filtered, labels):
        plt.hist(af, density=True, label=nm)

    plt.ylabel("Frequency")

    x_label = "Error value\n"
    for a, nm in zip(dat, chrs):
        x_label += (
            r" $"
            + nm
            + r"_\mu$="
            + f"{np.mean(a):.1f}"
            + r", $"
            + nm
            + r"_\sigma$="
            + f"{np.std(a):.1f}"
            + r" $"
            + nm
            + r"_{min}$="
            + f"{np.min(a):.1f}"
            + r", $"
            + nm
            + r"_{max}$="
            + f"{np.max(a):.1f}\n"
        )
    plt.xlabel(x_label)

    title = f"Expected Error ("
    title += ", ".join([r"$" + nm + r"_n=$" + f"{len(a)}" for a, nm in zip(dat, chrs)])
    title += ")"
    plt.title(title)

    # plt.yscale('log')
    plt.legend()
    plt.show()
