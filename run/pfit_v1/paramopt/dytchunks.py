import numpy as np


def compare(a_: list[str], ac_: list[int], b_: list[str], bc_: list[int]) -> float:
    a = zip(a_, ac_)
    b = zip(b_, bc_)

    try:
        aa = next(a)
    except StopIteration:
        print("Warning: empty during compare", len(a_), len(b_))
        aa = None

    try:
        bb = next(b)
    except StopIteration:
        print("Warning: empty during compare", len(a_), len(b_))
        bb = None

    err = []
    while True:
        if aa is not None and bb is None:
            err += [(aa[0], aa[1], 0)]
            aa = next(a, None)
        elif aa is None and bb is not None:
            err += [(bb[0], 0, bb[1])]
            bb = next(b, None)
        elif aa is None and bb is None:
            break
        else:
            if aa[0] == bb[0]:
                err += [(aa[0], aa[1], bb[1])]
                aa = next(a, None)
                bb = next(b, None)
            elif aa[0] < bb[0]:
                err += [(aa[0], aa[1], 0)]
                aa = next(a, None)
            else:  # bb < aa
                err += [(bb[0], 0, bb[1])]
                bb = next(b, None)

    return err


def compare_opt(a_: list[str], ac_: list[int], b_: list[str], bc_: list[int]) -> float:
    # Divide by n_samples here to allow comparison of differing number of samples.
    def f(x_, xc_):
        if np.sum(xc_) == 0:
            print("Warning: zero count during compare_opt", np.sum(ac_), np.sum(bc_))
            x, xx = None, None
        else:
            x = zip(x_, xc_ / np.sum(xc_))
            try:
                xx = next(x)
            except StopIteration:
                print("Warning: empty during compare_opt", len(a_), len(b_))
                xx = None
        return x, xx

    a, aa = f(a_, ac_)
    b, bb = f(b_, bc_)

    err = []
    while True:
        if aa is not None and bb is None:
            err += [aa[1]]
            aa = next(a, None)
        elif aa is None and bb is not None:
            err += [bb[1]]
            bb = next(b, None)
        elif aa is None and bb is None:
            break
        else:
            if aa[0] == bb[0]:
                err += [aa[1] - bb[1]]
                aa = next(a, None)
                bb = next(b, None)
            elif aa[0] < bb[0]:
                err += [aa[1]]
                aa = next(a, None)
            else:  # bb < aa
                err += [bb[1]]
                bb = next(b, None)

    # print(np.sum(np.array(err) ** 2))
    # print(err)

    return np.sum(np.array(err) ** 2)
