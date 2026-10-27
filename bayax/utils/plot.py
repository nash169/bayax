"""Plotting utilities."""

from typing import Optional, Tuple
import jax.numpy as jnp
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from itertools import combinations

from bayax.utils.types import Callable, Vector


def plot(
    fn: Callable | Tuple,
    fn_between: Optional[Callable | Vector] = None,
    range=[-1, 1],
    res=100,
    axes_lim=None,
    ax=None,
    fig=None,
    **kwargs
):
    if ax is None:
        if fig is not None:
            ax = fig.get_axes()[0]
        else:
            fig, ax = plt.subplots()
            fig.tight_layout()

    if isinstance(fn, Callable):
        x = jnp.linspace(range[0], range[1], res)
        p_args = [x, fn(x.reshape(-1, 1)).squeeze()]
    else:
        p_args = fn

    im = ax.plot(*p_args, **kwargs)

    if axes_lim is None:
        ax.set_xlim(p_args[0].min(), p_args[0].max())
    else:
        ax.set_xlim(*axes_lim[0])
        ax.set_ylim(*axes_lim[1])

    if not im[0].get_label().startswith("_"):
        ax.legend()

    if fn_between is not None:
        if isinstance(fn_between, Callable):
            fn_between_eval = fn_between(p_args[0].reshape(-1, 1)).squeeze()
            f_args = [p_args[0], p_args[1] - fn_between_eval, p_args[1] + fn_between_eval]
        else:
            f_args = [p_args[0], p_args[1] - fn_between, p_args[1] + fn_between]
        ax.fill_between(
            *f_args,
            alpha=0.2,
            color=im[0].get_color()
        )

    return fig if fig is not None else ax


def scatter(
        data,
        ranges=None,
        axes_labels=None,
        axes_aspect=None,
        cbar=None,
        ax=None,
        fig=None,
        **kwargs
):
    dim = data.shape[0] if len(data.shape) == 1 else data.shape[1]

    if ax is None:
        if fig is None:
            fig = plt.figure()
        # fig.patch.set_visible(False)
        # fig.tight_layout()
        try:
            ax = fig.get_axes()[-1]
        except:
            ax = fig.add_subplot(111, projection='3d' if dim == 3 else None)

    im = ax.scatter(*data.T, **kwargs)

    if axes_aspect is not None:
        ax.set_aspect(*axes_aspect)

    if not im.get_label().startswith("_"):
        ax.legend()

    if axes_labels is not None:
        ax.set_xlabel(axes_labels[0])
        ax.set_ylabel(axes_labels[1])
        if dim == 3:
            ax.set_zlabel(axes_labels[2])

    if ranges is not None:
        ax.set_xlim(*ranges[0])
        ax.set_ylim(*ranges[1])
        if dim == 3:
            ax.set_zlim(*ranges[2])

    if cbar is not None and fig is not None:
        fig.colorbar(im, ax=ax, **cbar)

    return fig if fig is not None else ax


def quiver(vecs, origin=None, ranges=None, fig=None, **kwargs):
    dim = vecs.shape[0] if len(vecs.shape) == 1 else vecs.shape[0]
    if fig is not None:
        ax = fig.get_axes()[0]
    else:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d') if dim == 3 else fig.add_subplot(111)
    if origin is not None:
        ax.quiver(*origin.T, *vecs.T, **kwargs)
    else:
        ax.quiver(*jnp.zeros_like(vecs.T), *vecs.T, **kwargs)
    if ranges is not None:
        ax.set_xlim(*ranges[0])
        ax.set_ylim(*ranges[1])
        if dim == 3:
            ax.set_zlim(*ranges[2])
    return fig


def contour(
    fn,
    ranges=None,
    res=50,
    axes_labels=None,
    isolines=None,
    cbar=None,
    reduce_fn=jnp.sum,
    ax=None,
    fig=None,
    **kwargs
):
    if isinstance(fn, Callable):
        assert ranges is not None, "provide grid ranges"
        dim = len(ranges)
        grids = [jnp.linspace(start, end, res) for (start, end) in ranges]
        mesh = jnp.meshgrid(*grids, indexing='ij')
        p_args = [*mesh, fn(jnp.stack([g.ravel() for g in mesh], axis=-1)).reshape(mesh[0].shape)]
    else:
        dim = len(fn) - 1
        p_args = fn

    if dim > 2:
        pairs = list(combinations(range(dim), 2))
        n_plots = len(pairs)
        ncols = int(jnp.ceil(jnp.sqrt(n_plots)))
        nrows = int(jnp.ceil(n_plots / ncols))
        if fig is None:
            fig, axs = plt.subplots(nrows, ncols)
            fig.patch.set_visible(False)
            fig.tight_layout()
        axs = [item for row in axs for item in row]

        for idx, (i, j) in enumerate(pairs):
            ax = axs[idx]

            reduce_axes = tuple(k for k in range(dim) if k not in (i, j))
            F_proj = reduce_fn(p_args[-1], axis=reduce_axes)

            Xi, Xj = jnp.meshgrid(grids[i], grids[j], indexing='ij')
            im = ax.contourf(Xi, Xj, F_proj, **kwargs)
            if isolines is not None:
                ax.contour(Xi, Xj, F_proj, cmap=None, colors='#f2e68f', **isolines)
            ax.set_aspect('equal', 'box')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.set_xticks([Xi.min(), Xi.max()])
            ax.set_yticks([Xj.min(), Xj.max()])
            ax.set_xlim([Xi.min(), Xi.max()])
            ax.set_ylim([Xj.min(), Xj.max()])
            # ax.yaxis.set_label_position("right")
            # ax.yaxis.tick_right()
            if axes_labels is not None:
                ax.set_xlabel(axes_labels[i], fontsize=20, labelpad=-15)
                ax.set_ylabel(axes_labels[j], fontsize=20, labelpad=-20)
            if cbar is not None:
                colorbar(im, fig, ax, pad=0.1, **cbar)

        for ax in axs[n_plots:]:
            ax.axis('off')
    else:
        if ax is None:
            if fig is None:
                fig = plt.figure()
            fig.patch.set_visible(False)
            fig.tight_layout()
            try:
                ax = fig.get_axes()[-1]
            except:
                ax = fig.add_subplot(111)

        if isolines is not None:
            ax.contour(*p_args, cmap=None, colors='#f2e68f', **isolines)
        im = ax.contourf(*p_args, **kwargs)
        ax.set_aspect('equal', 'box')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_xticks([p_args[0].min(), p_args[0].max()])
        ax.set_yticks([p_args[1].min(), p_args[1].max()])
        ax.set_xlim([p_args[0].min(), p_args[0].max()])
        ax.set_ylim([p_args[1].min(), p_args[1].max()])
        # ax.yaxis.set_label_position("right")
        # ax.yaxis.tick_right()
        if axes_labels is not None:
            ax.set_xlabel(axes_labels[0], fontsize=20, labelpad=-10)
            ax.set_ylabel(axes_labels[1], fontsize=20, labelpad=-10)
        if cbar is not None:
            assert fig is not None, "provide fiure handle to add colorbar"
            colorbar(im, fig, ax, **cbar)

    return fig if fig is not None else ax


def surf(
    fn,
    ranges=None,
    res=50,
    axes_labels=None,
    cbar=None,
    reduce_fn=jnp.sum,
    ax=None,
    fig=None,
    **kwargs
):
    if isinstance(fn, Callable):
        assert ranges is not None, "provide grid ranges"
        dim = len(ranges)
        grids = [jnp.linspace(start, end, res) for (start, end) in ranges]
        mesh = jnp.meshgrid(*grids, indexing='ij')
        p_args = [*mesh, fn(jnp.stack([g.ravel() for g in mesh], axis=-1)).reshape(mesh[0].shape)]
    else:
        dim = len(fn) - 1
        p_args = fn

    if dim > 2:
        pairs = list(combinations(range(dim), 2))
        n_plots = len(pairs)
        ncols = int(jnp.ceil(jnp.sqrt(n_plots)))
        nrows = int(jnp.ceil(n_plots / ncols))
        if fig is None:
            fig = plt.figure()
            fig.patch.set_visible(False)
            fig.tight_layout()

        for idx, (i, j) in enumerate(pairs):
            ax = fig.add_subplot(nrows, ncols, idx + 1, projection='3d')

            reduce_axes = tuple(k for k in range(dim) if k not in (i, j))
            F_proj = reduce_fn(p_args[-1], axis=reduce_axes)

            Xi, Xj = jnp.meshgrid(grids[i], grids[j], indexing='ij')
            im = ax.plot_surface(Xi, Xj, F_proj, cmap="Spectral", antialiased=True, alpha=0.8,)
            # ax.axis('off')
            ax.set_box_aspect((jnp.ptp(Xi), jnp.ptp(Xj), jnp.ptp(F_proj)))
            ax.set_xticks([Xi.min(), Xi.max()])
            ax.set_yticks([Xj.min(), Xj.max()])
            if axes_labels is not None:
                ax.set_xlabel(axes_labels[i], fontsize=20, labelpad=-15)
                ax.set_ylabel(axes_labels[j], fontsize=20, labelpad=-20)
                ax.set_ylabel(axes_labels[-1], fontsize=20, labelpad=-20)
            if cbar is not None:
                colorbar(im, fig, ax, pad=0.1, **cbar)
    else:
        if ax is None:
            if fig is None:
                fig = plt.figure()
            fig.patch.set_visible(False)
            fig.tight_layout()
            try:
                ax = fig.get_axes()[-1]
            except:
                ax = fig.add_subplot(111, projection='3d', computed_zorder=False)

        im = ax.plot_surface(*p_args, cmap="Spectral", antialiased=True, alpha=0.8,)
        # ax.axis('off')
        ax.set_box_aspect((jnp.ptp(p_args[0]), jnp.ptp(p_args[1]), jnp.ptp(p_args[2])))
        ax.set_xticks([p_args[0].min(), p_args[0].max()])
        ax.set_yticks([p_args[1].min(), p_args[1].max()])
        if axes_labels is not None:
            ax.set_xlabel(axes_labels[0], fontsize=20)
            ax.set_ylabel(axes_labels[1], fontsize=20)
            ax.set_zlabel(axes_labels[2], fontsize=20)
        if cbar is not None:
            assert fig is not None, "provide fiure handle to add colorbar"
            colorbar(im, fig, ax, **cbar)

    return fig if fig is not None else ax


def colorbar(im, fig, ax, pos="right", size="8%", pad=0.4, label=None, labelpad=5, ticks=None):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes(pos, size=size, pad=pad)
    if ticks is None:
        vmin, vmax = im.get_clim()
        ticks = [jnp.round(vmin, 2), jnp.round(vmax, 2)]
    cbar = fig.colorbar(im, cax=cax, ticks=ticks, orientation='vertical')
    cax.yaxis.set_ticks_position(pos)
    cbar.outline.set_visible(False)
    # cbar.set_ticks([0.0])
    if label is not None:
        cbar.set_label('$' + label + '$', fontsize=18, rotation=0, labelpad=labelpad)


def contour_plot(
        fn,
        grid=None,
        min=[-1, -1],
        max=[1, 1],
        res=100,
        xlabel=None,
        ylabel=None,
        iso=False,
        cbar=False,
        fig=None,
        **kwargs
):
    if fig is not None:
        ax = fig.get_axes()
    else:
        fig, ax = plt.subplots()

    if grid is None:
        x, y, z = _get_xyz(fn, min, max, res)
    else:
        z = fn
        x, y = grid

    im = ax.contourf(x, y, z, 500, cmap="Spectral")
    if iso:
        ax.contour(x, y, z, 10, cmap=None, colors='#f2e68f')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=20)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=20)
    ax.set_xlim([min[0], max[0]])
    ax.set_ylim([min[1], max[1]])
    ax.set_xticks([min[0], max[0]])
    ax.set_yticks([min[1], max[1]])
    ax.set_aspect('equal', 'box')

    if cbar:
        _colorbar(im, fig, ax, **kwargs)

    # fig.patch.set_visible(False)
    fig.tight_layout()
    return fig


def contour_nd_plot(
        fn,
        ranges,
        res=50,
        reduce_fn=jnp.sum,
        iso=False,
        cbar=False,
        axes_label=None,
        **kwargs
):
    dim = len(ranges)
    grids = [jnp.linspace(start, end, res) for (start, end) in ranges]
    mesh = jnp.meshgrid(*grids, indexing='ij')
    # F = fn(*mesh)
    F = fn(jnp.stack([g.ravel() for g in mesh], axis=-1)).reshape(mesh[0].shape)

    pairs = list(combinations(range(dim), 2))
    n_plots = len(pairs)
    ncols = int(jnp.ceil(jnp.sqrt(n_plots)))
    nrows = int(jnp.ceil(n_plots / ncols))

    fig, axs = plt.subplots(nrows, ncols)
    axs = [item for row in axs for item in row]
    for idx, (i, j) in enumerate(pairs):
        ax = axs[idx]
        reduce_axes = tuple(k for k in range(dim) if k not in (i, j))
        F_proj = reduce_fn(F, axis=reduce_axes)

        Xi, Xj = jnp.meshgrid(grids[i], grids[j], indexing='ij')
        im = ax.contourf(Xi, Xj, F_proj, levels=500, cmap="Spectral")
        if iso:
            ax.contour(Xi, Xj, F_proj, 10, cmap=None, colors='#f2e68f')

        # ax.set_title(f"Projection onto axes ({i}, {j})")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_xticks([Xi.min(), Xi.max()])
        ax.set_yticks([Xj.min(), Xj.max()])
        ax.set_xlabel(f"${axes_label}_1$", labelpad=-5)
        ax.set_ylabel(f"${axes_label}_2$", labelpad=-20)
        # ax.yaxis.set_label_position("right")
        # ax.yaxis.tick_right()
        ax.axis('equal')

        if cbar:
            _colorbar(im, fig, ax, **kwargs)

    for ax in axs[n_plots:]:
        ax.axis('off')

    # fig.patch.set_visible(False)
    fig.tight_layout()
    return fig


def surf_plot(fn, min=[-1, -1], max=[1, 1], res=100, fig_ax=None, **kwargs):
    if fig_ax is not None:
        fig, ax = fig_ax
    else:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d', computed_zorder=False)
    x, y, z = get_xyz(fn, min, max, res)
    im = ax.plot_surface(x, y, z, cmap="Spectral", antialiased=True, alpha=0.8,)
    ax.set_box_aspect((jnp.ptp(x), jnp.ptp(y), jnp.ptp(z)))
    ax.axis('off')
    # ax.view_init(elev=30, azim=-95)

    fig.patch.set_visible(False)
    fig.tight_layout()

    return fig


def _colorbar(im, fig, ax, pos="right", size="8%", pad=0.4, label=None, labelpad=-50, ticks=None):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes(pos, size=size, pad=pad)
    cbar = fig.colorbar(im, cax=cax, ticks=ticks, orientation='vertical')
    cax.yaxis.set_ticks_position(pos)
    cbar.outline.set_visible(False)
    cbar.set_ticks([0.0])
    if label is not None:
        cbar.set_label('$' + label + '$', fontsize=18, rotation=90, labelpad=labelpad)


def _get_xyz(fn, min, max, res):
    if isinstance(fn, Callable):
        x, y = jnp.meshgrid(jnp.linspace(min[0], max[0], res), jnp.linspace(min[1], max[1], res))
        eval_fn = fn(jnp.vstack((x.ravel(), y.ravel())).transpose())
        if len(eval_fn.shape) == 1:
            z = eval_fn.reshape(res, res)
        else:
            x, y, z = eval_fn[:, 0].reshape(res, res), eval_fn[:, 1].reshape(res, res), eval_fn[:, 2].reshape(res, res)
    else:
        res = int(jnp.sqrt(fn.shape[0]))
        x, y, z = fn[:, 0].reshape(res, res), fn[:, 1].reshape(res, res), fn[:, 2].reshape(res, res)
    return x, y, z


def plot_regression(model, X_test, y_test=None, X_train=None, y_train=None, fig=None):
    if fig is None:
        fig, ax = plt.subplots()
    else:
        ax = fig.axes[0]
    pred = model(X_test)
    ax.plot(X_test, pred, label='map')
    if y_test is not None:
        ax.plot(X_test, y_test, label='target')
    if X_train is not None and y_train is not None:
        ax.scatter(X_train, y_train, label='samples', color='green', alpha=0.5)
    ax.legend()

    return fig
