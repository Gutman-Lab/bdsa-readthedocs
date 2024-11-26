
class InvalidKindError(Exception):
    """Raised if the kind is invalid."""
    pass


def get_randoms(kind=None):
    """
    Return a list of random ingredients as strings.

    :param kind: Optional "kind".
    :type kind: list[str] or None
    :raise InvalidKindError: If the kind is invalid.
    :return: list
    :rtype: list[str]
    """
    return ["0", "1", "2"]
