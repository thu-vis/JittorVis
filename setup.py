  
from distutils.core import setup
setup(
  name = 'jittorvis',
  packages = ['jittorvis'],
  version = '1.1.2',
  license='MIT',
  description = 'Dynamic graph visualization for Jittor',
  author = 'Zhen Li',
  author_email = 'thu.lz@outlook.com',
  url = 'https://github.com/swordsbird/JittorVis',
  download_url = 'https://github.com/swordsbird/JittorVis/archive/jittorvis-1.1.0.tar.gz',
  keywords = ['Jittor', 'visualization'],
  install_requires=[
          'Flask',
          'numpy',
          'jittor',
      ],
  entry_points={
    'console_scripts':[
        'jittorvis = jittorvis.server:main'
    ]
  },
  include_package_data=True,
  package_data={"jittorvis": ["jittorvis/static/*",  "jittorvis/data/*",  "jittorvis/static/css/*",  "jittorvis/static/fonts/*",  "jittorvis/static/js/*",  "jittorvis/static/lib/*"]},
  classifiers=[  # Optional
    # How mature is this project? Common values are
    #   3 - Alpha
    #   4 - Beta
    #   5 - Production/Stable
    'Development Status :: 3 - Alpha',

    # Indicate who your project is intended for
    'Intended Audience :: Developers',
    'Topic :: Software Development :: Libraries',

    # Pick your license as you wish
    'License :: OSI Approved :: Apache Software License',

    # Specify the Python versions you support here. In particular, ensure
    # that you indicate whether you support Python 2, Python 3 or both.
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
  ],
)