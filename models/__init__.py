"""Neural network modules for NS-GCL."""
from .multiscale_cnn import MultiScaleCNN
from .gcn import GraphConvolution, GraphEncoder

__all__ = ["MultiScaleCNN", "GraphConvolution", "GraphEncoder"]
