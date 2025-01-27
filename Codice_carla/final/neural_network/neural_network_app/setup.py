from setuptools import setup, find_packages

setup(
    name='neural_network_app',
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='A neural network application for various tasks.',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'tensorflow>=2.0.0',
        'numpy>=1.18.0',
        'pandas>=1.0.0',
        'matplotlib>=3.0.0',
        'scikit-learn>=0.22.0'
    ],
    entry_points={
        'console_scripts': [
            'neural_network_app=neural_network:main',  # Adjust if main function is in a different file
        ],
    },
)