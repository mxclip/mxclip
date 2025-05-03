from setuptools import setup, find_packages

setup(
    name="mxclip",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "RealtimeSTT==0.3.100",
        "ffmpeg-python==0.2.0",
        "numpy>=1.24",
    ],
    entry_points={
        'console_scripts': [
            'mxclip=mxclip.main:cli',
        ],
    },
) 