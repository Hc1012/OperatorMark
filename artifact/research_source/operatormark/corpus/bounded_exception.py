"""Adopted bounded exception retention A: bounded exception record.

Replaces unbounded str(exception)/traceback text in the v2 failure record and
run/unexpected_failure.json by one record whose canonical-JSON length has an
exact source bound. The full rendering (chains, notes, frames, source lines)
is kept whole when it fits; otherwise its head and tail are kept and the
middle is DECLARED omitted, with the exact total length and SHA-256 of the
full rendering so a reproduction can be checked. Nothing else is dropped
silently. This is a retention-policy change, not a lossless proof.
"""
import hashlib
import sys
import traceback

SCHEMA = 'operatormark.bounded_exception.v1'
# Explicitly adopted retention ceiling; formatting time/memory are not bounded.
RENDERED_CAP_BYTES = 16 * 2**20
NAME_CAP_BYTES = 1024
_MAXSIZE = sys.maxsize        # CPython length ceiling for str and bytes


def _escaped(ch):
    """Bytes one code point occupies in canonical JSON (ensure_ascii=False)."""
    if ch in '"\\\b\f\n\r\t':
        return 2
    if ch < ' ':
        return 6
    return len(ch.encode('utf-8'))


def clip(text, budget):
    """Head/tail of `text` whose canonical-JSON string payload is <= budget bytes."""
    try:
        text.encode('utf-8');form = 'utf8';units = text;cost = _escaped
    except UnicodeEncodeError:   # lone surrogates: exact bytes as hex instead
        form = 'utf8_surrogatepass_hex';units = text.encode('utf-8', 'surrogatepass').hex();cost = lambda _: 1
    if sum(map(cost, units)) <= budget:
        head, tail = units, ''
    else:
        half = budget // 2;used = 0;h = 0
        while h < len(units) and used + cost(units[h]) <= half:
            used += cost(units[h]);h += 1
        used = 0;t = len(units)
        while t > h and used + cost(units[t - 1]) <= half:
            used += cost(units[t - 1]);t -= 1
        if form != 'utf8':        # keep whole bytes (two hex digits)
            h -= h % 2;t += t % 2
        head, tail = units[:h], units[t:]
    return dict(form=form, head=head, tail=tail, total_units=len(units),
                omitted_units=len(units) - len(head) - len(tail))


def _class_text(cls, name):
    # No metaclass descriptors or user hooks (same rule as evidence._failure).
    if type(cls) is not type:
        return '?unsafe_exception_metaclass'
    value = type.__getattribute__(cls, '__dict__').get('__module__', 'builtins') if name == '__module__' \
        else type.__getattribute__(cls, '__qualname__')
    return value if type(value) is str else '?unavailable'


def bounded_exception_record(exc, *, cap=RENDERED_CAP_BYTES):
    if type(cap) is not int or not 0 <= cap <= RENDERED_CAP_BYTES:
        raise ValueError('adopted retained rendering ceiling')
    cls = type(exc)
    try:
        rendered = ''.join(traceback.format_exception(exc));unavailable = None
    except BaseException as failure:          # keep type/module even then
        rendered = '';unavailable = clip(_class_text(type(failure), '__qualname__'), NAME_CAP_BYTES)
    data = rendered.encode('utf-8', 'surrogatepass')
    return dict(schema=SCHEMA, cap_bytes=cap,
                type=clip(_class_text(cls, '__qualname__'), NAME_CAP_BYTES),
                module=clip(_class_text(cls, '__module__'), NAME_CAP_BYTES),
                rendered=clip(rendered, cap), rendering_complete=unavailable is None,
                rendered_utf8_bytes=len(data) if unavailable is None else None,
                rendered_sha256=hashlib.sha256(data).hexdigest() if unavailable is None else None,
                rendering_unavailable=unavailable)


def record_bound(cap=RENDERED_CAP_BYTES):
    """Exact upper bound of len(canonical_json_bytes(bounded_exception_record(...)))."""
    from .canonical import canonical_json_bytes
    worst = lambda: dict(form='utf8_surrogatepass_hex', head='', tail='', total_units=_MAXSIZE * 2,
                         omitted_units=_MAXSIZE * 2)
    skeleton = dict(schema=SCHEMA, cap_bytes=cap, type=worst(), module=worst(), rendered=worst(),
                    rendering_complete=False, rendered_utf8_bytes=_MAXSIZE, rendered_sha256='f' * 64, rendering_unavailable=worst())
    return len(canonical_json_bytes(skeleton)) + cap + 3 * NAME_CAP_BYTES
