"""Custom exceptions for byutils."""


class BYUtilsError(Exception):
    """Base exception for byutils."""
    pass


class CacheMissError(BYUtilsError):
    """Raised when attempting to load from cache but resource not found."""

    def __init__(self, resource_id: str, cache_dir: str):
        self.resource_id = resource_id
        self.cache_dir = cache_dir
        super().__init__(
            f"Cache miss: '{resource_id}' not found in {cache_dir}\n\n"
            f"You are offline or on a compute node without internet access.\n"
            f"Please download on a login node first:\n\n"
            f"  from byutils import prefetch_model, prefetch_dataset\n"
            f"  prefetch_model('{resource_id}')\n\n"
            f"Then retry your computation on the compute node."
        )


class InternetRequiredError(BYUtilsError):
    """Raised when internet is required but not available."""

    def __init__(self, operation: str):
        super().__init__(
            f"Internet connection required for: {operation}\n\n"
            f"Please retry this operation on a login node with internet access.\n"
            f"Login nodes: login01, login02, login03, login04"
        )
