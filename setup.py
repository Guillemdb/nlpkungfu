from setuptools import setup


setup(
    name="nlpkf",
    description="Tools to solve various nlp tasks",
    version="0.0.1",
    license="MIT",
    author="Guillem Duran Ballester",
    author_email="guillem.db@gmail.com",
    url="https://github.com/Guillemdb/nlpkungfu",
    download_url="https://github.com/Guillemdb/nlpkungfu",
    keywords=[
        "nlp",
        "topic modeling",
        "ngrams",
        "vector embeddings",
        "tokenizer"
    ],
    install_requires=["spacy", "numpy", "gensim", "nltk",
                      "torch", "bokeh", "matplotlib", "pandas"],
    package_data={"": ["LICENSE", "README.md"]},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT license",
        "Operating System :: POSIX",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Topic :: Software Development :: Libraries",
],
)