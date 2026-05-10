"""
debug.py -- Global debug flag for the RAG pipeline.

Set DEBUG = True to enable verbose stdout tracing of every pipeline step:

  [DEBUG] User query      : "..."
  [DEBUG] Rewritten query : "..."
  [DEBUG] Expanded queries:
    [1] ...
    [2] ...
  [DEBUG] Chunks found    : 3
  [DEBUG] Chunk scores    :
    k1  score=0.8721  Law No. 05/L-085 -- Article 5
    k2  score=0.7340  ZRRE Network Code -- Article 8

Set DEBUG = False for production / when running under Streamlit.
"""

DEBUG: bool = True


def dprint(label: str, value: object = None) -> None:
    """
    Print a single debug line when DEBUG is True.

    Parameters
    ----------
    label : str
        Short description of what is being printed.
    value : object, optional
        The value to display. If None, label is printed as-is.
        Lists are printed one item per line, indented.
    """
    if not DEBUG:
        return

    prefix = "[DEBUG]"

    if value is None:
        print(f"{prefix} {label}")
    elif isinstance(value, list):
        print(f"{prefix} {label}")
        for i, item in enumerate(value, 1):
            print(f"  [{i}] {item}")
    else:
        print(f"{prefix} {label}: {value}")
