"""Represent Torch's version string as a JSON string without changing its text.

torch.__version__ can be a str subclass. Frozen runtime records are JSON and
therefore contain an ordinary str. Only this representation boundary changes;
the strict comparator, expected version, Python build and numerical values do
not. Never use this function for scientific result comparison.
"""


def plain_runtime(observed):
    if type(observed) is not dict:
        raise ValueError('Runtime observation must be a dictionary')
    version = observed.get('torch')
    if not isinstance(version, str):
        raise ValueError('Observed Torch version must already be a string; got '
                         + type(version).__module__ + '.' + type(version).__qualname__)
    result = observed.copy()
    # Call the builtin implementation: do not accept custom __str__ conversions,
    # trim whitespace, parse versions, drop build suffixes or coerce numbers.
    result['torch'] = str.__str__(version)
    if type(result['torch']) is not str:
        raise ValueError('Torch version text did not become a plain string')
    return result
