# from distutils.core import setup, Extension
# import numpy.distutils.misc_util

# # Adding OpenCV to project
# # ************************

# # Adding sources of the project
# # *****************************

# m_name = "grid_subsampling"

# SOURCES = ["../cpp_utils/cloud/cloud.cpp",
#            "grid_subsampling/grid_subsampling.cpp",
#            "wrapper.cpp"]

# module = Extension(m_name,
#                    sources=SOURCES,
#                    extra_compile_args=['-std=c++11',
#                                        '-D_GLIBCXX_USE_CXX11_ABI=0'])

# setup(ext_modules=[module], include_dirs=numpy.distutils.misc_util.get_numpy_include_dirs())

from setuptools import setup, Extension, find_packages
import numpy as np
import os

# Module name
m_name = "grid_subsampling"

# C++ source files (adjust paths if needed)
SOURCES = [
    os.path.join("..", "cpp_utils", "cloud", "cloud.cpp"),
    os.path.join("grid_subsampling", "grid_subsampling.cpp"),
    "wrapper.cpp",
]

# If you need OpenCV headers/libs, uncomment and adjust these:
# import cv2
# opencv_inc = os.path.dirname(cv2.__file__) + "/../include"
# opencv_lib = os.path.dirname(cv2.__file__) + "/../lib"
# extra_include_dirs = [np.get_include(), opencv_inc]
# extra_library_dirs = [opencv_lib]
# extra_libraries = ["opencv_core", "opencv_imgproc"]

module = Extension(
    m_name,
    sources=SOURCES,
    include_dirs=[
        np.get_include(),
        # *extra_include_dirs  # uncomment if using OpenCV
    ],
    # library_dirs=extra_library_dirs,  # uncomment if linking OpenCV
    # libraries=extra_libraries,        # uncomment if linking OpenCV
    extra_compile_args=[
        "-std=c++11",
        "-D_GLIBCXX_USE_CXX11_ABI=0",
    ],
    language="c++",
)

setup(
    name=m_name,
    version="0.1.0",
    description="A PyBind11/C++ extension for fast grid subsampling",
    author="Your Name",
    packages=find_packages(),
    python_requires=">=3.6",
    install_requires=[
        "numpy>=1.16.0",
        # "opencv-python>=4.0.0",  # if you need cv2 at runtime
    ],
    ext_modules=[module],
)







