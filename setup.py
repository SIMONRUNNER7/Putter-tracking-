"""
PutterTrack Pro – setup.py
Allows installation as a package: pip install -e .
"""
from setuptools import setup, find_packages

setup(
    name="puttertrack-pro",
    version="1.0.0",
    description="Golf putter motion analysis application",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "PyQt6>=6.6.0",
        "opencv-python>=4.9.0",
        "opencv-contrib-python>=4.9.0",
        "numpy>=1.26.0",
        "scipy>=1.12.0",
        "Pillow>=10.2.0",
    ],
    extras_require={
        "yolo": ["ultralytics>=8.1.0"],
        "export": ["matplotlib>=3.8.0", "imageio>=2.33.0"],
    },
    entry_points={
        "console_scripts": [
            "puttertrack=main:main",
        ],
    },
)
