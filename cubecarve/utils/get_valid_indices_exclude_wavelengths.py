import numpy as np

def get_valid_indices_exclude_wavelengths(
          index,
          total_layers,
          wl_index,
          forbidden_wavelengths=None, 
          tol=1e-4):
        """
        Selects up to `total_layers` indices centered near `center_index`, 
        excluding layers where the wavelength is in `forbidden_wavelengths`.
        
        Parameters:
            wavelengths: 1D array of shape (N,) corresponding to cube layers.
            index: int, the target index to center around.
            total_layers: int, total number of layers to return.
            forbidden_wavelengths: 1D array or list of forbidden wavelength values.
            tol: float, tolerance for wavelength matching (default is small for float comparison).
        
        Returns:
            selected_indices: array of valid layer indices (within range and not forbidden).
        """

        if forbidden_wavelengths is None:
            forbidden_wavelengths = []

        # Create a boolean mask of allowed layers
        allowed_mask = np.ones_like(wl_index, dtype=bool)
        for bad_wl in forbidden_wavelengths:
            allowed_mask &= np.abs(wl_index - bad_wl) > tol

        allowed_indices = np.where(allowed_mask)[0]

        # If center_index isn't allowed, find the closest allowed one
        if index not in allowed_indices:
            index = allowed_indices[np.argmin(np.abs(allowed_indices - index))]

        # Find index of center in allowed list
        center_pos = np.where(allowed_indices == index)[0][0]

        # Now select around that index
        half = total_layers // 2
        start = max(0, center_pos - half)
        end = min(len(allowed_indices), center_pos + half + (total_layers % 2))

        selected = allowed_indices[start:end]

        # If not enough layers, try padding
        while len(selected) < total_layers:
            if start > 0:
                start -= 1
            if end < len(allowed_indices):
                end += 1
            selected = allowed_indices[start:end]
            if start == 0 and end == len(allowed_indices):
                break

        return selected