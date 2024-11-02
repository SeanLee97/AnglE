# -*- coding: utf-8 -*-

try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup, find_packages


readme = open('README.md').read()

with open('requirements.txt', encoding='utf-8') as f:
    requirements = [l for l in f.read().splitlines() if l]

with open('dev-requirements.txt', encoding='utf-8') as f:
    test_requirements = [l for l in f.read().splitlines() if l][1:]

setup(
    name='angle_emb',
    version='0.5.3',
    description='AnglE-optimize Text Embeddings',
    long_description=readme,
    long_description_content_type="text/markdown",
    author='sean lee',
    author_email='xmlee97@gmail.com',
    packages=find_packages(exclude=("tests", "tests.*", "examples", "examples.*")),
    install_requires=requirements,
    zip_safe=False,
    keywords='angle_emb',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    test_suite='tests',
    tests_require=test_requirements,
    entry_points={
        'console_scripts': [
            'angle-trainer = angle_emb.angle_trainer:main',
        ],
    },
)
