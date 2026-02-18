"""Internet connectivity detection utilities."""

import socket
import urllib.request
import urllib.error
from typing import Optional
from functools import lru_cache


@lru_cache(maxsize=1)
def is_login_node() -> bool:
    """
    Check if running on a login node (has internet access).

    Login nodes on BYU ORC have 'login' in their hostname.

    Returns:
        True if on a login node, False otherwise
    """
    hostname = socket.gethostname()
    return "login" in hostname.lower()


def check_hf_connectivity(timeout: float = 5.0) -> bool:
    """
    Check if HuggingFace Hub is accessible.

    Args:
        timeout: Connection timeout in seconds

    Returns:
        True if HuggingFace is accessible, False otherwise
    """
    try:
        # Try to reach HuggingFace status page
        urllib.request.urlopen("https://status.huggingface.co", timeout=timeout)
        return True
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def should_allow_download(
    force_download: Optional[bool] = None,
    check_internet: bool = True
) -> bool:
    """
    Determine whether to allow downloads.

    Args:
        force_download: Explicit override (True/False/None)
        check_internet: Whether to check internet connectivity

    Returns:
        True if downloads should be allowed
    """
    # Explicit override takes precedence
    if force_download is not None:
        return force_download

    # Default behavior: allow downloads on login nodes
    if is_login_node():
        return True

    # On compute nodes, check internet if requested
    if check_internet:
        return check_hf_connectivity(timeout=3.0)

    return False
