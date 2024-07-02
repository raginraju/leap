#constants
MIN_MAX_CONSTANTS = {"ANNUALCO2": { "min":0, "max":37082560000},
                 "ANIMALPROTIEN": { "min":10.267731, "max":422.91956},
                 "FAT": { "min":116.299614, "max":1671.6952},
                 "CARB": { "min":895.17065, "max":2525.5974},
                 "FERTILITY": { "min":1.0902, "max":8.2497},
                 "RURAL": { "min":0, "max":0.960019133},
                 "FIXEDLINE": { "min":0, "max":91.398224},
                 "MOBILELINE": { "min":0, "max":221.30878}}

def reverse_min_max_scaling(scaled_value, predictor):
    """
    Reverse the min-max scaling of a value.
    
    Parameters:
    scaled_value (float): The scaled value to reverse.
    original_min (float): The original minimum value of the feature.
    original_max (float): The original maximum value of the feature.
    predictor: predictor type.
    
    Returns:
    float: The original value before scaling.
    """
    original_min = MIN_MAX_CONSTANTS[predictor]["min"]
    original_max = MIN_MAX_CONSTANTS[predictor]["max"]
    original_value = (scaled_value) * (original_max - original_min) + original_min
    return original_value


def min_max_scaling(value, predictor):
    """
    Perform min-max scaling on a value.

    Parameters:
    value (float): The value to be scaled.
    original_min (float): The original minimum value of the feature.
    original_max (float): The original maximum value of the feature.
    feature_range (tuple): The range to which the values are scaled (default is (0, 1)).

    Returns:
    float: The scaled value.
    """
    original_min = MIN_MAX_CONSTANTS[predictor]["min"]
    original_max = MIN_MAX_CONSTANTS[predictor]["max"]
    scaled_value = ((value - original_min) / (original_max - original_min))
    return scaled_value
