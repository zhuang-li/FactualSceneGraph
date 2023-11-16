from setuptools import setup, find_packages

setup(
    name='FactualSceneGraph',
    version='0.4.4',
    author='Zhuang Li',
    author_email='lizhuang144@gmail.com',
    description='A package for scene graph parsing and evaluation',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/zhuang-li/FACTUAL',
    package_dir={'': "src"},
    packages=find_packages(where='src'),
    include_package_data=True,
    package_data={
        'factual_scene_graph.evaluation': ['resources/*'],
                  },
    install_requires=[
        'torch',
        'transformers',
        'tqdm',
        'nltk',
        'spacy',
        'sentence-transformers',
        'pandas',
        'numpy',
        'tabulate'
        # Add other dependencies needed for your package
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        # Add additional classifiers as appropriate for your project
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Operating System :: OS Independent',
    ],
)
