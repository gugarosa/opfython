from setuptools import setup
from setuptools import find_packages


setup(name='opfython',
      version='1.0.6',
      description='Optimum-Path Forest for Python',
      author='Recogna Laboratory',
      author_email='gth.rosa@uol.com.br',
      url='https://github.com/gugarosa/opfython',
      #download_url='https://github.com/gugarosa/opfython/tarball/1.0.0',
      license='Apache-2.0',
      install_requires=['numpy>=1.13.3'
                        'pylint>=1.7.4',
                        'pytest>=3.2.3',
                       ],
      extras_require={
          'tests': ['pytest',
                    'pytest-pep8',
                   ],
      },
      classifiers=[
          'Development Status :: 5 - Production/Stable',
          'Intended Audience :: Developers',
          'Intended Audience :: Education',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: Apache-2.0 License',
          'Programming Language :: Python :: 3.6',
          'Topic :: Software Development :: Libraries',
          'Topic :: Software Development :: Libraries :: Python Modules'
      ],
      packages=find_packages())
