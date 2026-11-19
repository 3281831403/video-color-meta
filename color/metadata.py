"""Color metadata normalization and pixel-conversion contracts.

A metadata mapping has four required fields, all enum strings after
normalization::

    {"primaries": ..., "transfer": ..., "matrix": ..., "range": ...}

Supported normalized values and the aliases accepted by :func:`normalize`:

* ``primaries``: ``bt709`` (``bt.709``, ``1``, ``srgb``), ``bt2020``
  (``bt.2020``, ``9``), ``dci_p3`` (``dci-p3``, ``p3``, ``12``).
* ``transfer``: ``bt709`` (``bt.709``, ``1``), ``srgb`` (``sRGB``,
  ``iec61966-2-1``, ``13``), ``pq`` (``smpte2084``, ``16``), ``hlg``
  (``arib-std-b67``, ``18``), ``linear`` (``8``).
* ``matrix``: ``bt709`` (``bt.709``, ``1``), ``bt601`` (``bt.601``,
  ``5``, ``6``), ``bt2020_ncl`` (``bt.2020-ncl``, ``9``), ``identity``
  (``gbr``, ``0``).
* ``range``: ``limited`` (``tv``, ``mpeg``), ``full`` (``pc``,
  ``jpeg``).

Enum matching is case-insensitive after stripping surrounding whitespace.
Unknown or missing values raise :class:`ColorError`.  Numeric contracts are
checked with an absolute tolerance of ``1e-6`` unless a function says
otherwise.
"""

from __future__ import annotations

__all__ = [
    "ColorError",
    "compatible",
    "convert_batch",
    "convert_pixel",
    "gamut_map",
    "normalize",
    "range_scale",
    "tone_map",
    "yuv_to_rgb_matrix",
]


class ColorError(ValueError):
    """Raised for missing, unknown, or inconsistent color metadata."""


def normalize(meta: dict) -> dict:
    """Normalize a metadata mapping and return a new dict.

    The four required keys are ``primaries``, ``transfer``, ``matrix`` and
    ``range``.  Their accepted normalized values are the enum strings listed
    in the module docstring; aliases such as ``bt.709``, ``1`` and ``sRGB``
    collapse to one canonical spelling.  Matching is case-insensitive.

    Extra keys are copied unchanged.  Missing keys, unknown enum values and
    non-string/non-integer values raise :class:`ColorError`.  The returned
    dict is independent of the input mapping.
    """
    raise NotImplementedError


def compatible(a: dict, b: dict) -> bool:
    """Return whether two metadata mappings may share the same samples.

    Both arguments are normalized first.  Compatibility requires identical
    ``primaries``, ``transfer`` and ``matrix`` values; ``range`` is ignored,
    so a limited/full pair with the same three sample properties is directly
    copyable.  A transfer mismatch (for example PQ versus sRGB) is never
    compatible.  Unknown metadata raises :class:`ColorError`.
    """
    raise NotImplementedError


def yuv_to_rgb_matrix(meta: dict) -> list[list[float]]:
    """Return the 3x3 YUV-to-RGB matrix for ``meta``.

    The returned matrix is row-major and expects ``[Y, U, V]`` with Y in
    ``[0, 1]`` and U/V centered at zero (roughly ``[-0.5, 0.5]``).  The
    ``matrix`` field selects the coefficients; ``primaries`` is validated but
    does not change the coefficients:

    * ``bt601``: R = Y + 1.4020 V; G = Y - 0.3441 U - 0.7141 V;
      B = Y + 1.7720 U.
    * ``bt709``: R = Y + 1.5748 V; G = Y - 0.1873 U - 0.4681 V;
      B = Y + 1.8556 U.
    * ``bt2020_ncl``: R = Y + 1.4746 V; G = Y - 0.1646 U - 0.5714 V;
      B = Y + 1.8814 U.
    * ``identity``: the 3x3 identity matrix.

    Equivalence of coefficients is checked with an absolute tolerance of
    ``1e-6``.  Unknown metadata raises :class:`ColorError`.
    """
    raise NotImplementedError


def range_scale(meta: dict) -> tuple:
    """Return the 8-bit luma range ``(low, high)`` selected by ``meta``.

    ``limited`` returns ``(16, 235)`` and ``full`` returns ``(0, 255)``.
    The chroma planes use 16..240 for limited range and 0..255 for full range;
    :func:`convert_pixel` applies those bounds separately.  Unknown metadata
    raises :class:`ColorError`.
    """
    raise NotImplementedError


def tone_map(
    hdr: list[float],
    target_nits: float,
    source_nits: float,
    method: str,
) -> list[float]:
    """Tone-map HDR luminance values to a display target.

    ``hdr`` contains source-referred luminance in nits; ``target_nits`` and
    ``source_nits`` must be positive and finite.  Values are normalized by
    ``source_nits`` and passed through one of the supported curves:

    * ``reinhard``: ``x / (1 + x)``.
    * ``hable``: the Uncharted 2 filmic curve, normalized at 11.2.
    * ``aces``: the Narkowicz ACES approximation.

    The curve result is multiplied by ``target_nits`` and clamped to
    ``[0, target_nits]``.  Output has the same length as ``hdr``, contains only
    finite values, and never exceeds ``target_nits``.  Negative input samples
    map to 0.  An unknown ``method`` raises :class:`ColorError`; a non-positive
    target or source also raises :class:`ColorError`.  Floating-point bounds
    are checked at a tolerance of ``1e-6``.
    """
    raise NotImplementedError


def gamut_map(
    rgb: list[float],
    from_primaries: str,
    to_primaries: str,
) -> list[float]:
    """Convert linear RGB between two gamuts through XYZ (D65).

    ``rgb`` is a three-component linear-light RGB triple.  Both primaries
    arguments accept the normalized names or aliases documented in the module
    docstring (for example ``bt2020`` and ``bt.709``).  The conversion is
    ``XYZ = M_from @ rgb`` followed by ``M_to^-1 @ XYZ``; it is not an RGB
    lerp.

    Each result component is clamped to ``[0.0, 1.0]``.  A component that
    lands exactly on 0.0 or 1.0 is the marker that clipping occurred.  The
    return value is a three-element ``list[float]``.  In-gamut results agree
    with an absolute tolerance of ``1e-6``.  Unknown primaries, a wrong-length
    triple, or non-finite input raise :class:`ColorError`.
    """
    raise NotImplementedError


def convert_pixel(
    y: float,
    u: float,
    v: float,
    from_meta: dict,
    to_meta: dict,
) -> tuple:
    """Convert one 8-bit YUV pixel and return a quantized 8-bit tuple.

    Both metadata mappings are normalized first.  Input components are 8-bit
    code values in ``[0, 255]``; values outside that range or non-finite
    values raise :class:`ColorError`.  The conversion order is fixed:

    1. decode the source range (Y and chroma separately),
    2. apply the source YUV-to-RGB matrix,
    3. map the RGB gamut from source primaries to target primaries,
    4. apply the inverse target RGB-to-YUV matrix,
    5. encode the target range,
    6. round to the nearest integer and clamp to ``0..255``.

    No implicit tone mapping is performed.  The result is a ``tuple`` of three
    integers, so no 16-bit or unquantized float leaks into an 8-bit pipeline.
    Intermediate values use the usual absolute tolerance of ``1e-6``.
    """
    raise NotImplementedError


def convert_batch(
    frames: list[tuple],
    from_meta: dict,
    to_meta: dict,
) -> dict:
    """Convert a batch of 8-bit YUV frames atomically.

    ``frames`` is a list of three-component ``(y, u, v)`` tuples.  The return
    value is ``{"frames": [converted tuples...], "count": len(frames)}`` with
    the input order preserved.  An empty batch returns
    ``{"frames": [], "count": 0}``.

    Validation is all-or-nothing: unknown metadata or any invalid frame raises
    :class:`ColorError` and no partial/mixed conversion result is returned.
    The input list and its frames are never mutated.
    """
    raise NotImplementedError
