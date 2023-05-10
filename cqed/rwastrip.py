# This file is part of cqed: quantum hardware modelling.
#
# Author: Konstantin Nesterov, 2017 and later.
########################################################################
"""The class for representing RWA strips.
"""

__all__ = ['RWAStrip']

import numpy as np

import qutip as qt


class RWAStrip(object):
    """A class for representing separate RWA strips in the system
    of coupled transmon and resonator in the rotating frame.

    Parameters
    ----------
    qubit : :class:`cqed.TransmonFull` or similar
        Qubit object supporting `qubit.a()` methods, etc..
    omega_c : float
        The resonator frequency.
    g : float
        Qubit-resonator coupling strength.
    nphotons : int or float
        The number of excitations in the RWA stip. In general, `nexcit`
        does not need to be integer, so the class is applicable to the
        model in Khezri, et.al., arXiv:2212.05097, describing RWA strips
        for a generic coherent state.
    """

    def __init__(self, qubit, omega_c, g, nphotons, nlev=None):
        self.qubit = qubit  # Qubit object.
        self.omega_c = omega_c  # Resonator frequency.
        self.g = g  # Coupling strength.
        self.nphotons = nphotons  # The total number of excitations.
        if nlev is None:
            self.nlev = qubit.nlev
        else:
            if nlev <= qubit.nlev:
                self.nlev = nlev
            else:
                raise Exception('nlev > qubit.nlev')
        self.reset()

    def reset(self):
        """Manual "reset" of the cached data in a class instance.

        It is necessary to call this method after a change in any
        parameter that is directly or indirectly used in the class
        instance. For example, after any of the attributes of
        an underlying qubit object has been changed, certain cached
        data such as eigenvalues have to be re-calculated.
        """
        self._eigvals = None
        self._eigvecs = None

    def levels_nonint(self):
        """Bare energy levels (qubit eigenenergies) in the rotating frame.

        Returns
        -------
        numpy.ndarray
            Array of bare eigenvalues in the order of qubit levels.
        """
        nlev = self.nlev
        return (self.qubit.levels(nlev=nlev)
                - self.omega_c * np.linspace(0, nlev-1, nlev))

    def level_nonint(self, level_ind):
        """Bare energy level in the rotating frame.

        Parameters
        ----------
        level_ind : int
            The level index starting from zero. The order of levels
            is the same as for the qubit.

        Returns
        -------
        float
            Energy of the level.
        """
        if level_ind < 0 or level_ind >= self.nlev-1:
            raise Exception('The level is out of bounds')
        return self.levels_nonint()[level_ind]

    def H_0(self):
        """Qubit Hamiltonian in the rotating frame."""
        return qt.Qobj(np.diag(self.levels_nonint()))

    def V(self):
        """JC interaction term in RWA."""
        off_diag = [self.g * np.sqrt(max(self.nphotons-ilev, 0) * (ilev+1))
                    for ilev in range(self.nlev-1)]
        return qt.Qobj(np.diag(off_diag, 1) + np.diag(off_diag, -1))

    def H(self):
        """The Hamiltonian of the interacting system in the qubit
        eigenbasis in the rotating frame."""
        return self.H_0() + self.V()

    def _spectrum_coupled(self):
        """Dressed eigenenergies and eigenstates in the order of
        qubit levels."""
        if self._eigvals is None or self._eigvecs is None:
            evals, evecs = self.H().eigenstates()
            labels = [np.argmax(np.abs(evecs[k].data))
                      for k in range(self.nlev)]
            self._eigvals = evals[np.argsort(labels)]
            self._eigvecs = evecs[np.argsort(labels)]
        return self._eigvals, self._eigvecs

    def levels(self, interaction='on'):
        """Dressed or bare energies in the order of qubit levels.

        Parameters
        ----------
        interaction : 'on' or 'off', optional
            Return energy levels with or without coupling.

        Returns
        -------
        numpy.ndarray
            Array of eigenvalues.
        """
        if interaction == 'on':
            eigvals, _ = self._spectrum_coupled()
            return eigvals
        elif interaction == 'off':
            return self.levels_nonint()
        else:
            raise Exception('Unrecognized parameter value')

    def level(self, level_ind, interaction='on'):
        """Energy of a single level.

        Parameters
        ----------
        level_ind : int
            The level index starting from zero in the order of
            qubit levels.
        interaction : 'on' or 'off', optional
            Return energy level with or without coupling.

        Returns
        -------
        float
            Energy of the level.
        """
        if level_ind < 0 or level_ind >= self.nlev-1:
            raise Exception('The level is out of bounds')
        return self.levels(interaction=interaction)[level_ind]

    def eigvecs(self, interaction='on'):
        """Dressed or bare eigenvectors in the order of qubit levels.

        Parameters
        ----------
        interaction : 'on' or 'off', optional
            Return energy levels with or without coupling.

        Returns
        -------
        1d array of :class:`qutip.Qobj`
            Eigenvectors.

        See Also
        --------
        levels
        """
        if interaction == 'on':
            _, eigvecs = self._spectrum_coupled()
            return eigvecs
        elif interaction == 'off':
            return np.array([qt.basis(self.nlev, k) for k in range(self.nlev)])
        else:
            raise Exception('Unrecognized parameter value')

    def eigvec(self, level_ind, interaction='on'):
        """Dressed or bare eigenvector for a single level.

        Parameters
        ----------
        level_ind : int
            The level index starting from zero in the order of
            qubit levels.
        interaction : 'on' or 'off', optional
            Return energy level with or without coupling.

        Returns
        -------
        :class:`qutip.Qobj`
            Eigenvector.

        See Also
        --------
        levels
        """
        if level_ind < 0 or level_ind >= self.nlev-1:
            raise Exception('The level is out of bounds')
        return self.eigvecs(interaction=interaction)[level_ind]
