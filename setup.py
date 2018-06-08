from setuptools import setup, find_packages

install_requires = [
    'numpy',
    # 'tensorflow', # user install it
    'scipy',
    'scikit-image',
    'matplotlib',
]

setup(
    name = "tensorlayer",
    version = "1.8.5rc2",
    include_package_data=True,
    author='TensorLayer Contributors',
    author_email='hao.dong11@imperial.ac.uk',
    url = "https://github.com/tensorlayer/tensorlayer" ,
    license = "Apache 2.0" ,
    packages = find_packages(),
    install_requires=install_requires,
    description = "A TensorFlow-based Deep Learning Library for Researchers and Engineers.",
    keywords = "Deep learning, Reinforcement learning, TensorFlow",
    platform=['any'],
)