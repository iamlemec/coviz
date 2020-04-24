import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.style.use('clean.mplstyle')

eps = 1e-8

def get_start(ser, start):
    pos = ser > start
    if pos.any():
        return pos.idxmax()
    else:
        return None

def get_shifted(ser, start):
    date = get_start(ser, start)
    if date is not None:
        return ser[date:]
    else:
        return None

def get_aligned(sel, start):
    net = {c: get_shifted(sel[c], start) for c in sel}
    pos = pd.concat({c: s.reset_index(drop=True) for c, s in net.items() if s is not None}, axis=1)
    return pos

def gen_ticks(ymin, ymax, per):
    if ymin <= 0:
        yield 0
        ymin = 1/per

    ppow = np.floor(np.log10(ymin))
    pnum = ymin/np.power(10.0, ppow)

    if pnum < 2:
        pnum = 1
    elif pnum < 5:
        pnum = 2
    else:
        pnum = 5

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

class FixedLogScale(mpl.scale.ScaleBase):
    name = 'fixed_log'

    def __init__(self, axis, per=1e6):
        super().__init__(axis)
        self.per = per

    def get_transform(self):
        return mpl.scale.FuncTransform(np.log, np.exp)

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
                    return '%.1f' % d

        ymin, ymax = axis.get_view_interval()
        ticks = list(gen_ticks(ymin, ymax, self.per))
        loc = mpl.ticker.FixedLocator(ticks)

        axis.set_major_locator(loc)
        axis.set_major_formatter(InverseFormatter(self.per))
        axis.set_minor_locator(mpl.ticker.NullLocator())

mpl.scale.register_scale(FixedLogScale)

def plot_progress(codes, names=None, data=None, figsize=(8, 5), xylabel=(6, -4), per=1e6):
    # get correct labels
    if names is None:
        names = {c: c for c in codes}
    data = data[codes].rename(columns=names)

    # get last valid point
    dlast, vlast = zip(*[next(data[c].dropna().iloc[[-1]].iteritems()) for c in data])

    # plot core data
    fig, ax = plt.subplots(figsize=figsize)
    data.plot(linewidth=2, ax=ax)

    # annotate endpoints
    colors = [l.get_color() for l in ax.get_lines()]
    ax.scatter(dlast, vlast, c=colors, zorder=10)
    for n, c, d, v in zip(names.values(), colors, dlast, vlast):
        ax.annotate(n, (d, v), xytext=xylabel, color=c, fontsize=12, textcoords='offset pixels')

    # set reasonable bounds
    vmin = data.min().min()
    vmax = data.max().max()
    dmin = data.index[0]
    dmax = data.index[-1]
    ax.set_ylim(0.7*vmin, 2.3*vmax)
    ax.set_xlim(dmin-2, dmax+5)

    # set up axes
    ax.set_yscale('fixed_log', per=per)
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
