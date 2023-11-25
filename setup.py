from setuptools import setup, find_packages

setup(
    # Basic package information
    name='HSE_LLM_Report_System',
    version='0.1',
    author='3L MNAJA Bilal & Ezziyani Ilyass ',
    author_email='elmanjabilal@gmail.com',
    description='A system for HSE reporting using LLM in industrial environments',
    
    # Long description, often from a README file
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',

    # URL to your project's main homepage
    url='https://github.com/BilalElmanja/HSE-Reporting-with-LLMs',
    
    # License details
    license='MIT',  # Or any other license you prefer

    # Classifiers help users find your project by categorizing it
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',

        # Specify the Python versions you support here
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
    ],

    # Keywords that describe your project
    keywords='HSE reporting LLM industrial safety',

    # You can find your project's packages automatically or specify them
    packages=find_packages(),

    # List run-time dependencies here. These will be installed by pip when
    # your project is installed
    install_requires=[
        # Add your project dependencies here
        # e.g., 'requests', 'numpy'
    ],

    # If there are any scripts in your project that should be accessible from the command line,
    # you can specify them here
    entry_points={
        'console_scripts': [
            # Example: 'script_name = module_name:function_name'
        ],
    },

    # If there are data files included in your packages that need to be installed, specify them here
    include_package_data=True,
    package_data={
        # Example: 'package_name': ['data/*.dat']
    },

    # Although 'package_data' is the preferred approach, you can also use 'data_files' for non-package data files
    data_files=[
        # Example: ('my_data', ['data/data_file'])
    ],
)
