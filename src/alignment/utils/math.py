"""
Mathematical utility functions for the alignment package.

This module provides mathematical utilities for operations like
matrix calculations, linear algebra operations, and other math-related
functions used in alignment research.
"""

from typing import List, Optional, Tuple, Union, Any

import numpy as np
import torch
from torch import Tensor

from alignment.utils.core import to_numpy


def orthogonalize(A: Union[np.ndarray, Tensor], B: Union[np.ndarray, Tensor]) -> Union[np.ndarray, Tensor]:
    """
    Orthogonalize A with respect to B.
    
    Args:
        A: Matrix to orthogonalize
        B: Reference matrix
        
    Returns:
        Orthogonalized A
    """
    if isinstance(A, Tensor):
        A_np = to_numpy(A)
        B_np = to_numpy(B)
        result_np = orthogonalize(A_np, B_np)
        return torch.from_numpy(result_np).to(A.device)
    
    # Compute projection
    B_norm = B / np.linalg.norm(B)
    proj = np.outer(B_norm, B_norm) @ A
    
    # Subtract projection to orthogonalize
    return A - proj


def compute_correlation_matrix(X: Union[np.ndarray, Tensor], Y: Union[np.ndarray, Tensor]) -> Union[np.ndarray, Tensor]:
    """
    Compute the correlation matrix between X and Y.
    
    Args:
        X: First matrix
        Y: Second matrix
        
    Returns:
        Correlation matrix
    """
    is_tensor = isinstance(X, Tensor)
    
    if is_tensor:
        X_np, Y_np = to_numpy(X), to_numpy(Y)
    else:
        X_np, Y_np = X, Y
    
    # Standardize X
    X_centered = X_np - X_np.mean(axis=0, keepdims=True)
    X_std = X_np.std(axis=0, keepdims=True)
    X_std[X_std == 0] = 1.0  # Avoid division by zero
    X_standardized = X_centered / X_std
    
    # Standardize Y
    Y_centered = Y_np - Y_np.mean(axis=0, keepdims=True)
    Y_std = Y_np.std(axis=0, keepdims=True)
    Y_std[Y_std == 0] = 1.0  # Avoid division by zero
    Y_standardized = Y_centered / Y_std
    
    # Compute correlation
    n = X_np.shape[0]
    corr = (X_standardized.T @ Y_standardized) / n
    
    if is_tensor:
        return torch.from_numpy(corr).to(X.device)
    return corr


def matrix_angles(A: Union[np.ndarray, Tensor], B: Union[np.ndarray, Tensor]) -> Union[np.ndarray, Tensor]:
    """
    Compute the principal angles between subspaces spanned by A and B.
    
    Args:
        A: First matrix
        B: Second matrix
        
    Returns:
        Principal angles in radians
    """
    is_tensor = isinstance(A, Tensor)
    
    if is_tensor:
        A_np, B_np = to_numpy(A), to_numpy(B)
    else:
        A_np, B_np = A, B
        
    # Orthonormalize the matrices
    q_a, _ = np.linalg.qr(A_np)
    q_b, _ = np.linalg.qr(B_np)
    
    # Compute SVD of the product
    u, s, vh = np.linalg.svd(q_a.T @ q_b)
    
    # Clip singular values to [-1, 1] to handle numerical issues
    s = np.clip(s, -1.0, 1.0)
    
    # Compute angles in radians
    angles = np.arccos(s)
    
    if is_tensor:
        return torch.from_numpy(angles).to(A.device)
    return angles


def project_to_subspace(v: Union[np.ndarray, Tensor], subspace: Union[np.ndarray, Tensor]) -> Union[np.ndarray, Tensor]:
    """
    Project vector v onto subspace.
    
    Args:
        v: Vector to project
        subspace: Matrix whose columns form a basis for the subspace
        
    Returns:
        Projected vector
    """
    is_tensor = isinstance(v, Tensor)
    
    if is_tensor:
        v_np = to_numpy(v)
        subspace_np = to_numpy(subspace)
        result_np = project_to_subspace(v_np, subspace_np)
        return torch.from_numpy(result_np).to(v.device)
    
    # Orthonormalize the subspace basis
    Q, _ = np.linalg.qr(subspace)
    
    # Project onto the subspace
    return Q @ (Q.T @ v) 