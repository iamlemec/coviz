import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from datetime import timedelta

mpl.style.use('clean.mplstyle')

eps = 1e-8

def log1(x, base=np.e):
    y = np.log(x)/np.log(base)
    y[np.isinf(y)] = np.nan
    return y

def get_start(ser, start):
    pos = ser > start
    if pos.any():
        return pos.idxmax()
    else:
        return None

def get_shifted(pan, dst):
    return pd.concat({c: pan[c].loc[s:, :].reset_index(drop=True) for c, s in dst.items() if s is not None}, axis=1)

def get_aligned(panel, cutoff, cutvar):
    start = {c: get_start(panel[cutvar, c], cutoff) for c in panel[cutvar]}
    panel = panel.swaplevel(axis=1).sort_index(axis=1)
    shift = get_shifted(panel, start)
    shift = shift.swaplevel(axis=1).sort_index(axis=1)
    return shift

def round_125(x):
    if x < 2:
        return 1
    elif x < 5:
        return 2
    else:
        return 5

def gen_ticks_log(ymin, ymax, per):
    ppow = np.floor(np.log10(ymin))
    pnum0 = ymin/np.power(10.0, ppow)
    pnum = round_125(pnum0)

    while (yval := pnum*(10**ppow)) <= ymax:
        yield yval

        if pnum == 1:
            pnum = 2
        elif pnum == 2:
            pnum = 5
        else:
            pnum = 1
            ppow += 1

    yield yval

def gen_ticks_lin(ymin0, ymax0, per):
    ymin = per*ymin0
    ymax = per*ymax0

    ppow = np.floor(np.log10(ymax))
    pval = np.power(10.0, ppow)
    pnum = round_125(ymax/pval)

    if pnum == 1:
        pnum = 2
        pval /= 10
    elif pnum == 2:
        pnum = 5
        pval /= 10
    elif pnum == 5:
        pnum = 1
    step = pnum*pval

    yval = 0
    while yval < ymax:
        yield yval/per
        yval += step
    yield yval/per

class FixedLogScale(mpl.scale.ScaleBase):
    name = 'fixed_log'

    def __init__(self, axis, per=1e6):
        super().__init__(axis)
        self.per = per

    def get_transform(self):
        return mpl.scale.FuncTransform(log1, np.exp)

    def set_default_locators_and_formatters(self, axis):
        class InverseFormatter(mpl.ticker.Formatter):
            def __init__(self, per):
                self.per = per

            def __call__(self, x, pos=None):
                d = self.per*x
                if d == 0:
                    return '0'
                elif d >= 1:
                    return '%d' % int(d+eps)
                else:
                    return '%.2f' % d

        pmin = 1/self.per
        ymin, ymax = axis.get_view_interval()
        if ymin < pmin:
            ymin = pmin
            axis.axes.set_ylim(bottom=ymin)

        ticks = list(gen_ticks_log(ymin, ymax, self.per))
        loc = mpl.ticker.FixedLocator(ticks)

        axis.set_major_locator(loc)
        axis.set_major_formatter(InverseFormatter(self.per))
        axis.set_minor_locator(mpl.ticker.NullLocator())

mpl.scale.register_scale(FixedLogScale)

class FixedLinScale(mpl.scale.ScaleBase):
    name = 'fixed_lin'

    def __init__(self, axis, per=1e6):
        super().__init__(axis)
        self.per = per

    def get_transform(self):
        return mpl.scale.IdentityTransform()

    def set_default_locators_and_formatters(self, axis):
        class InverseFormatter(mpl.ticker.Formatter):
            def __init__(self, per):
                self.per = per

            def __call__(self, x, pos=None):
                d = self.per*x
                if d == 0:
                    return '0'
                elif d >= 1:
                    return '%d' % int(d+eps)
                else:
                    return '%.2f' % d

        ymin, ymax = axis.get_view_interval()
        ticks = list(gen_ticks_lin(ymin, ymax, self.per))
        loc = mpl.ticker.FixedLocator(ticks)

        axis.set_major_locator(loc)
        axis.set_major_formatter(InverseFormatter(self.per))
        axis.set_minor_locator(mpl.ticker.NullLocator())

mpl.scale.register_scale(FixedLinScale)

def plot_progress(data, names=None, figsize=(8, 5), xylabel=(5, -4), per=1e6, log=False, smooth=7):
    # get correct labels
    if names is None:
        codes = list(data)
        names = {c: c for c in codes}
    elif type(names) is list:
        codes = names
        names = {c: c for c in codes}
    else:
        codes = list(names)
    data = data[codes].rename(columns=names)

    # smooth data
    if smooth is not None:
        data = data.rolling(smooth).mean()

    # kill off zeros in log mode
    if log:
        data[data<=0] = np.nan

    # get last valid point
    dlast, vlast = zip(*[next(data[c].dropna().iloc[[-1]].iteritems()) for c in data])

    # plot core data
    fig, ax = plt.subplots(figsize=figsize)
    data.plot(linewidth=2, ax=ax)

    # annotate endpoints
    colors = [l.get_color() for l in ax.get_lines()]
    ax.scatter(dlast, vlast, c=colors, s=20, zorder=10)
    for s, c, d, v in zip(codes, colors, dlast, vlast):
        n = names[s]
        ax.annotate(n, (d, v), xytext=xylabel, color=c, fontsize=14, textcoords='offset pixels')

    # set reasonable bounds
    vmin = data.min().min()
    vmax = data.max().max()
    dmin = data.index[0]
    dmax = data.index[-1]

    # 125 log axes
    if log:
        ax.set_yscale('fixed_log', per=per)
        # ax.set_ylim(0.7*vmin, 2.3*vmax)
    else:
        ax.set_yscale('fixed_lin', per=per)
        ax.set_ylim(0, 1.1*vmax)

    # extend right xlim
    ax.set_xlim(dmin + timedelta(days=smooth), dmax + timedelta(days=5))

    # set up axes
    ax.grid(axis='y', linewidth=1, alpha=0.3)
    ax.legend([])

    return fig, ax

def plot_aligned(names, data=None, diff=False, start=1e-6, per=1e6):
    if type(names) is list:
        names = {n: n for n in names}
    codes = list(names)

    if type(start) is str:
        align = data.loc[start:]
    elif start is not None:
        align = get_aligned(data, start)[codes]
    else:
        align = data

    if diff:
        align = align.diff(axis=0).where(lambda x: x > 0)

    fig, ax = plot_progress(codes, names=names, data=align, per=per)

    return fig, ax
