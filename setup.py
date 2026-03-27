from setuptools import find_packages, setup

setup(
    name='supervised-segmentation-da',
    version='1.0.0',
    description='Supervised Segmentation with Dynamic Anchor Module',
    packages=find_packages(),
    install_requires=[
        'torch>=1.8',
        'mmcv-full>=1.3.7',
        'timm',
    ],
)
