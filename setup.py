from setuptools import setup, find_packages

setup(
    name='ModelFlow',
    version='0.0.0',
    license='MIT',
    author="Giorgos Myrianthous",
    author_email='zhu.felix@outlook.com',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    url='https://github.com/gmyrianthous/example-publish-pypi',
    keywords='example project',
    install_requires=[
          'scikit-learn',
      ],

)