from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize

ext_modules = [
    Extension(
        "cursor_info",
        ["cursor_info.pyx"],
        libraries=["X11", "Xfixes"],  # Ensures linking
        include_dirs=["/usr/include", "/usr/include/X11"],  # X11 headers
        library_dirs=["/usr/lib", "/usr/lib/x86_64-linux-gnu"],  # X11 libraries
        extra_link_args=["-lX11", "-lXfixes"],
    )
]

setup(
    name="cursor_info",
    ext_modules=cythonize(ext_modules),
)

