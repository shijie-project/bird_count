"""Windows-specific helpers for OpenCV windows.

All functions are safe no-ops on non-Windows platforms — callers don't
need to guard with platform checks.
"""

import ctypes
import logging
import platform
from ctypes import wintypes
from typing import Optional


logger = logging.getLogger(__name__)

_IS_WINDOWS = platform.system() == "Windows"

if _IS_WINDOWS:
    _user32 = ctypes.windll.user32

    # Type signatures — critical for 64-bit correctness. Without restype,
    # HWND values get truncated to 32 bits and silently break.
    _user32.FindWindowW.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR]
    _user32.FindWindowW.restype = wintypes.HWND

    _user32.SetWindowPos.argtypes = [
        wintypes.HWND,
        wintypes.HWND,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        wintypes.UINT,
    ]
    _user32.SetWindowPos.restype = wintypes.BOOL

    _user32.IsIconic.argtypes = [wintypes.HWND]
    _user32.IsIconic.restype = wintypes.BOOL
    _user32.IsWindow.argtypes = [wintypes.HWND]
    _user32.IsWindow.restype = wintypes.BOOL
    _user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
    _user32.ShowWindow.restype = wintypes.BOOL
    _user32.SetForegroundWindow.argtypes = [wintypes.HWND]
    _user32.SetForegroundWindow.restype = wintypes.BOOL
    _user32.GetSystemMenu.argtypes = [wintypes.HWND, wintypes.BOOL]
    _user32.GetSystemMenu.restype = wintypes.HMENU
    _user32.RemoveMenu.argtypes = [wintypes.HMENU, wintypes.UINT, wintypes.UINT]
    _user32.RemoveMenu.restype = wintypes.BOOL

    _HWND_TOPMOST = wintypes.HWND(-1)
    _HWND_NOTOPMOST = wintypes.HWND(-2)

# Win32 constants
_SW_RESTORE = 9
_SWP_NOSIZE = 0x0001
_SWP_NOMOVE = 0x0002
_SWP_FLAGS = _SWP_NOSIZE | _SWP_NOMOVE
_SC_CLOSE = 0xF060
_MF_BYCOMMAND = 0x00000000


def find_hwnd(window_name: str) -> Optional[int]:
    """Look up a window handle by exact title. Returns None if not found."""
    if not _IS_WINDOWS:
        return None
    hwnd = _user32.FindWindowW(None, window_name)
    return hwnd or None


def is_valid_hwnd(hwnd: Optional[int]) -> bool:
    """Check whether a previously-captured hwnd still refers to a live window."""
    if not _IS_WINDOWS or not hwnd:
        return False
    return bool(_user32.IsWindow(hwnd))


def force_foreground(hwnd: Optional[int], tag: str = "") -> None:
    """Briefly pop a window to the front, then drop the topmost flag.

    Uses the TOPMOST → NOTOPMOST trick to bypass Windows' SetForegroundWindow
    restrictions. After this returns the window can be covered normally.
    """
    if not _IS_WINDOWS or not hwnd:
        return
    try:
        if _user32.IsIconic(hwnd):
            _user32.ShowWindow(hwnd, _SW_RESTORE)
        _user32.SetWindowPos(hwnd, _HWND_TOPMOST, 0, 0, 0, 0, _SWP_FLAGS)
        _user32.SetWindowPos(hwnd, _HWND_NOTOPMOST, 0, 0, 0, 0, _SWP_FLAGS)
        _user32.SetForegroundWindow(hwnd)
    except OSError as e:
        logger.debug("[%s] force_foreground failed: %s", tag, e, exc_info=True)


def disable_close_button(hwnd: Optional[int], tag: str = "") -> None:
    """Remove the X (close) button from a window's system menu."""
    if not _IS_WINDOWS or not hwnd:
        return
    try:
        hmenu = _user32.GetSystemMenu(hwnd, False)
        if hmenu:
            _user32.RemoveMenu(hmenu, _SC_CLOSE, _MF_BYCOMMAND)
            logger.info("[%s] close button disabled", tag)
    except OSError as e:
        logger.debug("[%s] disable_close_button failed: %s", tag, e, exc_info=True)
