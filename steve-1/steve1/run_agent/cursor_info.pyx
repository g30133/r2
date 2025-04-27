from libc.stdlib cimport malloc, free
from libc.string cimport memcpy

cdef extern from "X11/Xlib.h":
    ctypedef void Display
    Display* XOpenDisplay(char* display_name)
    void XCloseDisplay(Display* display)

cdef extern from "X11/extensions/Xfixes.h":
    ctypedef struct XFixesCursorImage:
        int x, y
        unsigned int width, height
        int xhot, yhot
        unsigned int cursor_serial
        unsigned int* pixels

    XFixesCursorImage* XFixesGetCursorImage(Display* display)

def open():
    cdef Display* dpy = XOpenDisplay(NULL)
    if not dpy:
        return None
    return <long>dpy

def close(long dpy_ptr):
    cdef Display* dpy = <Display*>dpy_ptr
    if not dpy:
        return None
    XCloseDisplay(dpy)

def get(long dpy_ptr):
    cdef Display* dpy = <Display*>dpy_ptr
    if not dpy:
        return None
    cdef XFixesCursorImage* ci = XFixesGetCursorImage(dpy)
    if not ci:
        return None
    return [ci.x, ci.y, ci.width, ci.height, ci.xhot, ci.yhot]

