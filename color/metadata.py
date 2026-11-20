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

import math

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


# Canonical value -> accepted aliases (all matched case-insensitively
# after stripping whitespace).
_ALIASES = {
    "primaries": {
        "bt709": ("bt709", "bt.709", "1", "srgb"),
        "bt2020": ("bt2020", "bt.2020", "9"),
        "dci_p3": ("dci_p3", "dci-p3", "p3", "12"),
    },
    "transfer": {
        "bt709": ("bt709", "bt.709", "1"),
        "srgb": ("srgb", "iec61966-2-1", "13"),
        "pq": ("pq", "smpte2084", "16"),
        "hlg": ("hlg", "arib-std-b67", "18"),
        "linear": ("linear", "8"),
    },
    "matrix": {
        "bt709": ("bt709", "bt.709", "1"),
        "bt601": ("bt601", "bt.601", "5", "6"),
        "bt2020_ncl": ("bt2020_ncl", "bt.2020-ncl", "9"),
        "identity": ("identity", "gbr", "0"),
    },
    "range": {
        "limited": ("limited", "tv", "mpeg"),
        "full": ("full", "pc", "jpeg"),
    },
}

_LOOKUP = {
    field: {alias: canonical for canonical, aliases in table.items() for alias in aliases}
    for field, table in _ALIASES.items()
}

_REQUIRED_KEYS = ("primaries", "transfer", "matrix", "range")


def _normalize_value(field: str, value) -> str:
    if isinstance(value, bool):
        raise ColorError(f"{field}: invalid value {value!r}")
    if isinstance(value, int):
        value = str(value)
    if not isinstance(value, str):
        raise ColorError(f"{field}: invalid value {value!r}")
    key = value.strip().lower()
    canonical = _LOOKUP[field].get(key)
    if canonical is None:
        raise ColorError(f"{field}: unknown value {value!r}")
    return canonical


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
    if not isinstance(meta, dict):
        raise ColorError(f"metadata must be a dict, got {type(meta).__name__}")
    result = dict(meta)
    for field in _REQUIRED_KEYS:
        if field not in meta:
            raise ColorError(f"missing required metadata field {field!r}")
        result[field] = _normalize_value(field, meta[field])
    return result


def compatible(a: dict, b: dict) -> bool:
    """Return whether two metadata mappings may share the same samples.

    Both arguments are normalized first.  Compatibility requires identical
    ``primaries``, ``transfer`` and ``matrix`` values; ``range`` is ignored,
    so a limited/full pair with the same three sample properties is directly
    copyable.  A transfer mismatch (for example PQ versus sRGB) is never
    compatible.  Unknown metadata raises :class:`ColorError`.
    """
    na = normalize(a)
    nb = normalize(b)
    return all(na[field] == nb[field] for field in ("primaries", "transfer", "matrix"))


# Standard row-major [Y, U, V] -> [R, G, B] matrices.  Y is in [0, 1] and
# U/V are centered at zero (roughly [-0.5, 0.5]).
_YUV_TO_RGB_STANDARD = {
    "bt601": (
        (1.0, 0.0, 1.4020),
        (1.0, -0.3441, -0.7141),
        (1.0, 1.7720, 0.0),
    ),
    "bt709": (
        (1.0, 0.0, 1.5748),
        (1.0, -0.1873, -0.4681),
        (1.0, 1.8556, 0.0),
    ),
    "bt2020_ncl": (
        (1.0, 0.0, 1.4746),
        (1.0, -0.1646, -0.5714),
        (1.0, 1.8814, 0.0),
    ),
    "identity": (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    ),
}


def _public_matrix(standard) -> tuple:
    """Lay out a standard YUV->RGB matrix for :func:`yuv_to_rgb_matrix`.

    The public layout keeps the R and G rows as [Y, U, V] coefficient
    triples and rotates the B row to [U coef, V coef, Y coef].  The
    identity matrix is returned unchanged.
    """
    if all(
        standard[i][j] == (1.0 if i == j else 0.0)
        for i in range(3)
        for j in range(3)
    ):
        return tuple(tuple(row) for row in standard)
    r, g, b = standard
    return (tuple(r), tuple(g), (b[1], b[2], b[0]))


_YUV_TO_RGB = {
    name: _public_matrix(matrix) for name, matrix in _YUV_TO_RGB_STANDARD.items()
}


def yuv_to_rgb_matrix(meta: dict) -> list[list[float]]:
    """Return the 3x3 YUV-to-RGB coefficient matrix for ``meta``.

    The returned matrix is row-major with one row per output channel
    ``[R, G, B]``.  The R and G rows hold the ``[Y, U, V]`` coefficients
    for Y in ``[0, 1]`` and U/V centered at zero (roughly ``[-0.5, 0.5]``);
    the B row is laid out as ``[U coef, V coef, Y coef]``.  The ``matrix``
    field selects the coefficients; ``primaries`` is validated but does not
    change the coefficients:

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
    normalized = normalize(meta)
    return [list(row) for row in _YUV_TO_RGB[normalized["matrix"]]]


def range_scale(meta: dict) -> tuple:
    """Return the 8-bit luma range ``(low, high)`` selected by ``meta``.

    ``limited`` returns ``(16, 235)`` and ``full`` returns ``(0, 255)``.
    The chroma planes use 16..240 for limited range and 0..255 for full range;
    :func:`convert_pixel` applies those bounds separately.  Unknown metadata
    raises :class:`ColorError`.
    """
    normalized = normalize(meta)
    if normalized["range"] == "limited":
        return (16, 235)
    return (0, 255)


def _chroma_scale(range_name: str) -> tuple:
    return (16, 240) if range_name == "limited" else (0, 255)


def _hable(x: float) -> float:
    # Uncharted 2 filmic curve (John Hable).
    a, b, c, d, e, f = 0.15, 0.50, 0.10, 0.20, 0.02, 0.30
    return ((x * (a * x + c * b) + d * e) / (x * (a * x + b) + d * f)) - e / f


_HABLE_WHITE = _hable(11.2)


def _tone_curve(method: str, x: float) -> float:
    if method == "reinhard":
        return x / (1.0 + x)
    if method == "hable":
        return _hable(x) / _HABLE_WHITE
    if method == "aces":
        # Narkowicz ACES approximation.
        return (x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14)
    raise ColorError(f"unknown tone-map method {method!r}")


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
    if not (isinstance(target_nits, (int, float)) and math.isfinite(target_nits) and target_nits > 0):
        raise ColorError(f"target_nits must be positive and finite, got {target_nits!r}")
    if not (isinstance(source_nits, (int, float)) and math.isfinite(source_nits) and source_nits > 0):
        raise ColorError(f"source_nits must be positive and finite, got {source_nits!r}")
    if not isinstance(method, str):
        raise ColorError(f"unknown tone-map method {method!r}")
    method_key = method.strip().lower()
    if method_key not in ("reinhard", "hable", "aces"):
        raise ColorError(f"unknown tone-map method {method!r}")

    target = float(target_nits)
    source = float(source_nits)
    mapped = []
    for value in hdr:
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ColorError(f"non-finite luminance sample {value!r}")
        x = max(float(value), 0.0) / source
        y = _tone_curve(method_key, x) * target
        mapped.append(min(max(y, 0.0), target))
    return mapped


# Linear RGB -> XYZ (D65) matrices for the supported primaries.
_RGB_TO_XYZ = {
    "bt709": (
        (0.41239079926595948, 0.35758433938387796, 0.18048078840183429),
        (0.21263900587151036, 0.71516867876775593, 0.07219231536073372),
        (0.01933081871559185, 0.11919477979462599, 0.95053215224966058),
    ),
    "bt2020": (
        (0.6369580483012914, 0.14461690358620832, 0.16888097516417205),
        (0.2627002120112671, 0.6779980715188708, 0.05930171646986196),
        (0.0, 0.028072693049087428, 1.0609850577107909),
    ),
    "dci_p3": (
        (0.4865709486482162, 0.26566769316909306, 0.1982172852343625),
        (0.2289745640697488, 0.6917385218365064, 0.079286914093745),
        (0.0, 0.04511338185890264, 1.043944368900976),
    ),
}


def _mat3_inverse(m) -> list[list[float]]:
    a, b, c = m[0]
    d, e, f = m[1]
    g, h, i = m[2]
    det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    if det == 0.0:
        raise ColorError("singular color matrix")
    inv = [
        [(e * i - f * h) / det, (c * h - b * i) / det, (b * f - c * e) / det],
        [(f * g - d * i) / det, (a * i - c * g) / det, (c * d - a * f) / det],
        [(d * h - e * g) / det, (b * g - a * h) / det, (a * e - b * d) / det],
    ]
    return inv


def _mat3_mul_vec(m, v) -> list[float]:
    return [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]


def _normalize_primaries(value) -> str:
    return _normalize_value("primaries", value)


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
    src = _normalize_primaries(from_primaries)
    dst = _normalize_primaries(to_primaries)
    if not isinstance(rgb, (list, tuple)) or len(rgb) != 3:
        raise ColorError(f"rgb must be a three-component triple, got {rgb!r}")
    for component in rgb:
        if not isinstance(component, (int, float)) or not math.isfinite(component):
            raise ColorError(f"non-finite rgb component {component!r}")

    xyz = _mat3_mul_vec(_RGB_TO_XYZ[src], [float(c) for c in rgb])
    out = _mat3_mul_vec(_mat3_inverse(_RGB_TO_XYZ[dst]), xyz)
    return [min(max(component, 0.0), 1.0) for component in out]


def _validate_code_value(name: str, value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ColorError(f"{name}: invalid 8-bit code value {value!r}")
    if not 0.0 <= value <= 255.0:
        raise ColorError(f"{name}: code value {value!r} outside [0, 255]")
    return float(value)


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
    values raise :class:`ColorError`.

    If the normalized source and target metadata are identical the pixel is
    copied through unchanged (quantized to integers).  Otherwise the
    conversion order is fixed:

    1. decode the source range (Y and chroma separately),
    2. apply the source YUV-to-RGB matrix,
    3. map the RGB gamut from source primaries to target primaries,
    4. encode the RGB result into the target range,
    5. round to the nearest integer and clamp to ``0..255``.

    No implicit tone mapping is performed.  The result is a ``tuple`` of three
    integers, so no 16-bit or unquantized float leaks into an 8-bit pipeline.
    Intermediate values use the usual absolute tolerance of ``1e-6``.
    """
    src = normalize(from_meta)
    dst = normalize(to_meta)

    y_code = _validate_code_value("y", y)
    u_code = _validate_code_value("u", u)
    v_code = _validate_code_value("v", v)

    # Identical sample properties: direct copy, no conversion.
    if all(src[field] == dst[field] for field in _REQUIRED_KEYS):
        return (
            min(max(int(round(y_code)), 0), 255),
            min(max(int(round(u_code)), 0), 255),
            min(max(int(round(v_code)), 0), 255),
        )

    # 1. decode the source range (luma and chroma have different bounds).
    y_lo, y_hi = range_scale(src)
    c_lo, c_hi = _chroma_scale(src["range"])
    y_norm = (y_code - y_lo) / (y_hi - y_lo)
    u_norm = (u_code - c_lo) / (c_hi - c_lo) - 0.5
    v_norm = (v_code - c_lo) / (c_hi - c_lo) - 0.5

    # 2. source YUV -> RGB.
    rgb = _mat3_mul_vec(_YUV_TO_RGB_STANDARD[src["matrix"]], [y_norm, u_norm, v_norm])

    # 3. gamut-map source primaries -> target primaries (clamps to [0, 1]).
    rgb = gamut_map(rgb, src["primaries"], dst["primaries"])

    # 4. encode the target range.
    t_lo, t_hi = range_scale(dst)
    encoded = [component * (t_hi - t_lo) + t_lo for component in rgb]

    # 5. quantize to 8-bit integers.
    return tuple(min(max(int(round(component)), 0), 255) for component in encoded)


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
    # Validate metadata up front so nothing is converted on bad metadata.
    normalize(from_meta)
    normalize(to_meta)

    if not isinstance(frames, (list, tuple)):
        raise ColorError(f"frames must be a list of (y, u, v) tuples, got {type(frames).__name__}")

    # Validate every frame before converting anything (all-or-nothing).
    for index, frame in enumerate(frames):
        if not isinstance(frame, (list, tuple)) or len(frame) != 3:
            raise ColorError(f"frame {index}: expected a (y, u, v) triple, got {frame!r}")
        for name, component in zip(("y", "u", "v"), frame):
            _validate_code_value(f"frame {index} {name}", component)

    converted = [convert_pixel(frame[0], frame[1], frame[2], from_meta, to_meta) for frame in frames]
    return {"frames": converted, "count": len(converted)}
