"""
Weighted loss function for the weather prediction model.

This module provides GPU-aware loss functions that automatically detect and use
the same device as the input tensors, making them work seamlessly on both CPU and GPU.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


def compute_latitude_weights(latitude_values: np.ndarray) -> torch.Tensor:
    """Compute latitude-based area weights for grid cells."""
    # converting it to radians and computing the cos(lat) weights
    lat_rad = np.deg2rad(latitude_values)
    weights = np.cos(lat_rad)
    
    weights = weights / np.mean(weights)
    
    return torch.tensor(weights, dtype=torch.float32)


def compute_pressure_level_weights(pressure_levels: np.ndarray) -> torch.Tensor:
    """Compute pressure-level weights using proportio to pressure level."""
    
    weights = pressure_levels / np.mean(pressure_levels)
    
    return torch.tensor(weights, dtype=torch.float32)


def spatially_weighted_mse(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    latitude_weights: torch.Tensor,
    variable_weights: Optional[Dict[str, float]] = None,
    variable_slices: Optional[List[Tuple[int, int]]] = None,
    variable_names: Optional[List[str]] = None,
    pressure_level_weights: Optional[torch.Tensor] = None,
    eval_mode: bool = False
) -> torch.Tensor:
    """Compute spatially weighted MSE loss with pressure level weighting."""
    

    device = predictions.device
    
    predictions = predictions.to(device)
    targets = targets.to(device)
    latitude_weights = latitude_weights.to(device)
    if pressure_level_weights is not None:
        pressure_level_weights = pressure_level_weights.to(device)
    #print(pressure_level_weights)
    # handle batch dimension
    if predictions.dim() == 3: 
        batch_size = predictions.shape[0]
        num_nodes = predictions.shape[1]
        num_features = predictions.shape[2]
        
        predictions_flat = predictions.view(batch_size, -1)
        targets_flat = targets.view(batch_size, -1)
        
        lat_weights_expanded = latitude_weights.unsqueeze(1).expand(-1, num_features).contiguous().view(-1)
        
        # creating pressure level weights 
        if pressure_level_weights is not None and variable_slices is not None and variable_names is not None:
            pressure_weights_expanded = torch.ones(num_nodes * num_features, device=device, dtype=torch.float32)
            
            for (start, end), var_name in zip(variable_slices, variable_names):
                if var_name in variable_weights:  # multi-level variable
                    var_channels = end - start
                    if var_channels > 1:  # multi-level variable
                        # applying pressure level weights to this variable's channels
                        for level_idx in range(var_channels):
                            feature_idx = start + level_idx
                            # expanding pressure weight to all spatial locations for this feature
                            pressure_weights_expanded[feature_idx::num_features] = pressure_level_weights[level_idx]
        else:
            pressure_weights_expanded = torch.ones_like(lat_weights_expanded)
        
        sq_error = (predictions_flat - targets_flat) ** 2
        
        # applying spatial and pressure level weighting
        combined_weights = lat_weights_expanded.unsqueeze(0) * pressure_weights_expanded.unsqueeze(0)
        weighted_sq_error = sq_error * combined_weights
        
        # applying variable-level weighting 
        if variable_weights is not None and variable_slices is not None and variable_names is not None:
            var_weighted_error = torch.zeros_like(weighted_sq_error)
            
            for (start, end), var_name in zip(variable_slices, variable_names):
                if var_name in variable_weights:
                    weight = variable_weights[var_name]
                    # applying the weight to all grid points for this variable
                    var_weighted_error[:, start:end] = weighted_sq_error[:, start:end] * weight
                else:
                    var_weighted_error[:, start:end] = weighted_sq_error[:, start:end]
            
            weighted_sq_error = var_weighted_error
        
        loss = weighted_sq_error.mean()
        
    else: 
        num_nodes = predictions.shape[0]
        num_features = predictions.shape[1]
        #print(latitude_weights.shape)
        # creating latitude weights for all the features

        lat_weights_expanded = latitude_weights.unsqueeze(1).repeat(1,num_features).repeat(64,1)
        #lat_weights_expanded = latitude_weights.unsqueeze(1).expand(-1, num_features).contiguous().view(-1)
        #rint(lat_weights_expanded.shape)
        #print(pressure_level_weights)

        var_weights = torch.ones((11,1), device=device, dtype=torch.float32)
        for i, var_val in enumerate(variable_weights.values()):
            var_weights[i,:] = var_val            
        #print(var_weights)
        pressure_weights = pressure_level_weights.repeat(6) 

        pressure_weights_expanded = torch.ones((num_features,1), device=device, dtype=torch.float32)

        pressure_weights_expanded[:5,:] = var_weights[:5]

        pressure_weights_expanded[5:,:] = pressure_weights_expanded[5:,:] * pressure_weights.unsqueeze(1)
        #print(pressure_weights_expanded.shape)
        pressure_weights_expanded = pressure_weights_expanded.transpose(0,1).repeat(2048,1)
        


        
        sq_error = (predictions - targets) ** 2
        #print("sq_error")
        #print(sq_error.shape)
        # applying spatial and pressure level weighting
        if batch_size is None:
            combined_weights = (lat_weights_expanded * pressure_weights_expanded).repeat(4,1)
        else:
            combined_weights = (lat_weights_expanded * pressure_weights_expanded).repeat(batch_size,1)
        #print(combined_weights)
        #print(combined_weights.shape)
        if eval_mode:
            weighted_sq_error = sq_error
        else:
            weighted_sq_error = sq_error * combined_weights
        

        
        loss = weighted_sq_error.mean(0).sum()
    
    return loss


def get_default_pressure_levels() -> np.ndarray:
    """Get default pressure levels matching the GraphCast configuration."""
    return np.array([50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000], dtype=np.float32)


def get_variable_weights(grid_variables: List[str]) -> Dict[str, float]:
    """Get variable weights."""
    # we're using GraphCast's variable weights here
    graphcast_weights = {
        # single-level variables have lower weights
        '10m_u_component_of_wind': 0.1,
        '10m_v_component_of_wind': 0.1,
        'mean_sea_level_pressure': 0.1,
        'total_precipitation_6hr': 0.1,
        
        # 2m temperature is important though, thus gets a higher weighting
        '2m_temperature': 1.0,
        
        # multi-level atmospheric variables have a default to 1.0
        'temperature': 1.0,
        'geopotential': 1.0,
        'u_component_of_wind': 1.0,
        'v_component_of_wind': 1.0,
        'vertical_velocity': 1.0,
        'specific_humidity': 1.0,
    }
    
    return {var: graphcast_weights.get(var, 1.0) for var in grid_variables}


def masked_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    latitude_weights: torch.Tensor,
    variable_weights: Optional[Dict[str, float]] = None,
    variable_slices: Optional[List[Tuple[int, int]]] = None,
    variable_names: Optional[List[str]] = None,
    pressure_level_weights: Optional[torch.Tensor] = None,
    eval_mode: bool = False
) -> torch.Tensor:
    """Computes spatially weighted MSE loss with NaN/Inf masking."""

    if device is None:
        device = predictions.device
    
    # handling nan/inf values by masking
    mask = torch.isfinite(targets)
    if not mask.any():
        return torch.zeros((), device=device, dtype=predictions.dtype)
    
    # replacing nan/inf with zeros for computation
    pred_clean = torch.where(mask, predictions, torch.zeros_like(predictions))
    target_clean = torch.where(mask, targets, torch.zeros_like(targets))
    
    loss = spatially_weighted_mse(
        predictions=pred_clean,
        targets=target_clean,
        latitude_weights=latitude_weights,
        variable_weights=variable_weights,
        variable_slices=variable_slices,
        variable_names=variable_names,
        pressure_level_weights=pressure_level_weights,
        eval_mode=eval_mode
    )
    
    return loss
