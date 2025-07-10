import numpy as np
class Spin:
    def __init__(self, L: int) -> None:
        """Initialize the Spin model with a lattice of size LxL."""
        ...

    def run(self, step: int, spacing: int) -> None:
        """Run the Spin model simulation.

        step: number of new recordings.
        
        spacing: interval at which to record the state.
        """
        ...

    def get_saving(self) -> np.ndarray:
        """Get the recorded states of the Spin model. Return as a numpy array in the shape (step, L, L, 3). """
        ...

    def set_parameters(self, t: float, J11: float, J12: float, J21: float, J22: float, K: float) -> None:
        """Set the parameters for the Spin model.

        t: temperature of the system.

        J11: nearest neighbour Heisenberg interaction.

        J12: nearest neighbour biquadratic interaction.

        J21: second nearest neighbour Heisenberg interaction.

        J22: second nearest neighbour biquadratic interaction.

        K: Kitaev interaction.
        """
    