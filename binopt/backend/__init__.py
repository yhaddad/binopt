# Keras way of doing backends
from __future__ import absolute_import
from __future__ import print_function

# Default backend: Numpy
_BACKEND_ = 'numpy'

# Default backend: TensorFlow.
# _BACKEND_ = 'tensorflow'

elif _BACKEND_ == 'numpy':
    sys.stderr.write('Using Theano backend.\n')
    from .numpy_backend import *
elif _BACKEND_ == 'tensorflow':
    sys.stderr.write('Using TensorFlow backend.\n')
    from .tensorflow_backend import *
else:
    raise ValueError('Unknown backend: ' + str(_BACKEND_))


def backend():
    """Publicly accessible method
    for determining the current backend.
    # Returns
        String, the name of the backend Keras is currently using.
    # Example
    ```python
        >>> binopt.backend.backend()
        'tensorflow'
    ```
    """
    return _BACKEND_
