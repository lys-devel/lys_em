import math
import numpy as np
import jax
import jax.numpy as jnp

Host, Dev = jax.memory.Space.Host, jax.memory.Space.Device


def _pvary(x, axes):
    if not axes:
        return x
    if hasattr(jax.lax, "pcast"):
        return jax.lax.pcast(x, axes, to="varying")
    return jax.lax.pvary(x, axes)


def _propagate(phi, t, Pk, nvalid=None):
    """
    Propagate phi of one parameter (nprobe, Nx, Ny) through the slices t (nslices, Nx, Ny). Pk: (Nx, Ny)
    If nvalid is given, only the first nvalid slices are used (the rest is padding).
    The slices are split into about sqrt(nslices) segments, and each segment is checkpointed, so that
    the backward pass keeps about 2*sqrt(nslices) wave functions instead of nslices.
    """
    def step(i, phi, t, offset):
        new = jnp.fft.ifft2(jnp.fft.fft2(phi * t[i]) * Pk)
        return new if nvalid is None else jnp.where(offset + i < nvalid, new, phi)

    def segment(phi, t, offset):
        return jax.lax.fori_loop(0, t.shape[0], lambda i, phi: step(i, phi, t, offset), phi)
    n = t.shape[0]
    size = max(s for s in range(1, math.isqrt(n) + 1) if n % s == 0)
    if size == 1:
        return segment(phi, t, 0)
    segments = t.reshape(n // size, size, *t.shape[1:])
    return jax.lax.fori_loop(0, n // size, lambda j, phi: jax.checkpoint(segment)(phi, segments[j], j * size), phi)


def _propagate_all(phi, t, pk, batch, nvalid=None, pk_fn=None):
    """
    _propagate for all parameters. phi: (nparams, nprobe, Nx, Ny). batch parameters are computed at once.
    pk is Pk (nparams, Nx, Ny), or (x, spd) if pk_fn is given: Pk of each parameter is then calculated
    as pk_fn(spd, x[i]) when it is used, so that Pk of all parameters is not kept.
    """
    if pk_fn is None:
        f = jax.checkpoint(lambda a: _propagate(a[0], t, a[1], nvalid))  # in backward, recompute the propagation for each batch
        return jax.lax.map(f, (phi, pk), batch_size=batch)
    x, spd = pk
    f = jax.checkpoint(lambda a: _propagate(a[0], t, pk_fn(spd, a[1]), nvalid))
    return jax.lax.map(f, (phi, x), batch_size=batch)


def _scan_chunks(f, carry, t_r, size, xs=None, reverse=False):
    """
    jax.lax.scan of f over chunks of slices: carry, y = f(carry, t, x, start), where t is the next `size` slices
    of t_r (Nz, Nx, Ny) moved to device, start is the index of its first slice, and x is the corresponding element of xs (if given).
    """
    t_chunks = t_r.reshape(-1, size, *t_r.shape[1:])
    starts = jnp.arange(len(t_chunks)) * size
    carry, ys = jax.lax.scan(lambda c, a: f(c, jax.device_put(a[0], Dev), a[1], a[2]), carry, (t_chunks, xs, starts), reverse=reverse)
    return carry, ys


def _forward(phi0, Pk, t_r, size, batch, nslices, save=False, pk_fn=None):
    """Propagate through all slices t_r. If save, phi at the beginning of each chunk is stored on host."""
    def over_chunk(phi, t, _, start):
        new = _propagate_all(phi, t, Pk, batch, None if nslices is None else nslices - start, pk_fn)
        return new, (jax.device_put(phi, Host) if save else None)
    return _scan_chunks(over_chunk, phi0, t_r, size)


def make_propagator(chunk=None, batch=1, nslices=None, pk_fn=None):
    """
    Return prop(phi0, Pk, t_r) -> phi, which propagates the incident waves through all slices.

    Args:
        chunk (int): Number of slices transferred to device at once. Default (None) is all slices.
        batch (int): Number of parameters computed at once.
        nslices (int): Number of slices. If given, t_r may be padded at the end to a multiple of chunk.
        pk_fn (callable): If given, Pk is calculated as pk_fn(spd, x[i]) for each parameter when it is used,
            and prop takes (x, spd) instead of Pk. Only the gradient w.r.t. x (not spd) is calculated.

    prop arguments (per device):
        phi0: (nparams, nprobe, Nx, Ny), Pk: (nparams, Nx, Ny) or (x, spd), t_r: (Nz, Nx, Ny) on device or on (pinned) host.

    If t_r is on host, the gradient is calculated by custom_vjp, in which phi at the beginning of each chunk is
    stored on host and the inside of the chunk is recomputed, so that device memory does not depend on Nz.
    The gradient of t_r is also placed on host.
    """
    size = lambda t_r: len(t_r) if chunk is None else min(chunk, len(t_r))

    # pk is Pk (aux is None), or the parameters x (aux is spd) if pk_fn is given. aux is not differentiated.
    @jax.custom_vjp
    def prop(phi0, pk, aux, t_r):
        return _forward(phi0, pk if pk_fn is None else (pk, aux), t_r, size(t_r), batch, nslices, pk_fn=pk_fn)[0]

    def prop_fwd(phi0, pk, aux, t_r):
        phi, ck = _forward(phi0, pk if pk_fn is None else (pk, aux), t_r, size(t_r), batch, nslices, save=True, pk_fn=pk_fn)   # ck: phi at the beginning of each chunk (on host)
        return phi, (ck, pk, aux, t_r)

    def prop_bwd(res, g):
        ck, Pk, aux, t_r = res
        # Only the inexact leaves of pk (e.g. not integer parameters) are differentiated.
        leaves, tree = jax.tree.flatten(Pk)
        inexact = [jnp.issubdtype(v.dtype, jnp.inexact) for v in leaves]

        def merge(d, other=lambda v: v):    # pytree of pk from the inexact leaves d, other leaves from other(v)
            it = iter(d)
            return jax.tree.unflatten(tree, [next(it) if b else other(v) for b, v in zip(inexact, leaves)])
        f = lambda phi, t, d, nvalid: _propagate_all(phi, t, merge(d) if pk_fn is None else (merge(d), aux), batch, nvalid, pk_fn)
        Pk = [v for b, v in zip(inexact, leaves) if b]
        def over_chunk(carry, t, phi, start):
            dphi, dPk = carry
            # t is replicated over devices. Make it device-varying here, so that dt is summed over devices
            # only once per chunk (below), not for each batch inside vjp.
            axes = tuple(jax.typeof(dphi).vma - jax.typeof(t).vma)
            t = _pvary(t, axes)
            nvalid = None if nslices is None else nslices - start
            _, vjp = jax.vjp(lambda phi, t, pk: f(phi, t, pk, nvalid), jax.device_put(phi, Dev), t, Pk)
            dphi, dt, dpk = vjp(dphi)   # dt is summed over parameters
            return (dphi, jax.tree.map(jnp.add, dPk, dpk)), jax.device_put(jax.lax.psum(dt, axes), Host)
        (dphi0, dPk), dt = _scan_chunks(over_chunk, (g, jax.tree.map(jnp.zeros_like, Pk)), t_r, size(t_r), xs=ck, reverse=True)
        return dphi0, merge(dPk, lambda v: np.zeros(v.shape, jax.dtypes.float0)), None, dt.reshape(t_r.shape)

    prop.defvjp(prop_fwd, prop_bwd)

    def propagate(phi0, pk, t_r, spd=None):
        if jax.typeof(t_r).memory_space == Host:
            return prop(phi0, pk, spd, t_r)
        return _forward(phi0, pk if pk_fn is None else (pk, spd), t_r, size(t_r), batch, nslices, pk_fn=pk_fn)[0]
    return propagate


def make_sharded_propagator(nz, mesh, batch=1, pk_fn=None):
    """
    Return prop(phi0, Pk, t) -> phi for slices distributed over the devices (Slices mode "sharded"), used inside shard_map.
    t: (m, Nx, Ny) on each device, where device d has the slices d, d + ndev, ... of the nz slices.
    Slice k is sent from its device to all devices (psum) when it is used, and in the backward pass its gradient is summed
    (psum) and kept only on that device, so that no device has all the slices or their gradients.
    """
    axis, ndev = mesh.axis_names[0], mesh.size

    def propagate_one(phi, t, Pk):
        def fetch(k):
            mine = jax.lax.axis_index(axis) == k % ndev
            return _pvary(jax.lax.psum(jnp.where(mine, t[k // ndev], 0), axis), (axis,))   # device-varying like phi

        def segment(phi, offset, size):
            return jax.lax.fori_loop(0, size, lambda i, phi: jnp.fft.ifft2(jnp.fft.fft2(phi * fetch(offset + i)) * Pk), phi)
        size = max(s for s in range(1, math.isqrt(nz) + 1) if nz % s == 0)    # checkpointed segments of about sqrt(nz) slices
        return jax.lax.fori_loop(0, nz // size, lambda j, phi: jax.checkpoint(segment, static_argnums=2)(phi, j * size, size), phi)

    def prop(phi0, pk, t, spd=None):
        # pk is Pk (nparams, Nx, Ny), or the parameters x if pk_fn is given (Pk = pk_fn(spd, x[i]) when it is used)
        to_pk = (lambda p: p) if pk_fn is None else (lambda p: pk_fn(spd, p))
        f = jax.checkpoint(lambda a: propagate_one(a[0], t, to_pk(a[1])))
        return jax.lax.map(f, (phi0, pk), batch_size=batch)
    return prop
