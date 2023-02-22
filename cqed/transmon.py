# This file is part of cqed: quantum hardware modelling.
#
# Author: Konstantin Nesterov, 2017 and later.
########################################################################
"""The classes for representing superconducting transmon qubits.
"""

__all__ = ['TransmonSimple', 'TransmonFull']

import numpy as np

import qutip as qt


class TransmonSimple(object):
    """A class for representing transmons based on the Duffing
    oscillator model.

    Parameters
    ----------
    omega_q : float
        The frequency of the qubit 0-1 transition.
    alpha : float
        Anharmonicity. Defined as being negative for transmons.
    nlev : int, default=5
        The number of qubit eigenstates.
    omega_d : float, optional
        The drive frequency for the Hamiltonian in the rotating frame.
    """

    def __init__(self, omega_q, alpha, nlev=5, omega_d=None):
        # Most of these attributes are defined later as properties.
        self.omega_q = omega_q  # The qubit main transition frequency.
        self.omega_d = omega_d  # Drive frequency for rotating frame stuff.
        self.alpha = alpha  # The qubit anharmonicity (omega_12 - omega_01).
        self.nlev = nlev  # The number of eigenstates in the qubit.
        self.type = 'qubit'

    def __str__(self):
        s = ('A transmon qubit with omega_q = {} '.format(self.omega_q)
             + 'and alpha = {} '.format(self.alpha))
        return s

    @property
    def omega_q(self):
        return self._omega_q

    @omega_q.setter
    def omega_q(self, value):
        self._omega_q = value
        self._reset_cache()

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, value):
        if value >= 0:
            raise Exception('Anharmonicity must be negative.')
        self._alpha = value
        self._reset_cache()

    @property
    def nlev(self):
        return self._nlev

    @nlev.setter
    def nlev(self, value):
        if value <= 0:
            raise Exception('The number of levels must be positive.')
        self._nlev = value
        self._reset_cache()

    def _reset_cache(self):
        """Reset cached data that have already been calculated."""
        self._eigvals = None
        self._eigvecs = None

    def a(self):
        """Annihilation operator."""
        return qt.destroy(self.nlev)

    def H(self):
        """Qubit Hamiltonian."""
        omega_q = self.omega_q
        alpha = self.alpha
        nlev = self.nlev
        H_qubit = np.zeros((nlev, nlev))
        for k in range(1, nlev):
            H_qubit[k, k] = k * omega_q + 0.5 * k * (k-1) * alpha
        return qt.Qobj(H_qubit)

    def H_rotating(self):
        """Qubit Hamiltonian in the rotating frame."""
        a = self.a()
        return self.H() - self.omega_d * a.dag() * a

    def _eigenspectrum(self, eigvecs_flag=False):
        """Eigenenergies and eigenstates."""
        if not eigvecs_flag:
            if self._eigvals is None:
                H = self.H()
                self._eigvals = H.eigenenergies()
            return self._eigvals
        else:
            if self._eigvals is None or self._eigvecs is None:
                H = self.H()
                self._eigvals, self._eigvecs = H.eigenstates()
            return self._eigvals, self._eigvecs

    def levels(self, nlev=None):
        """Eigenenergies of the qubit.

        Parameters
        ----------
        nlev : int, optional
            The number of eigenstates if different from `self.nlev`.

        Returns
        -------
        numpy.ndarray
            Array of eigenvalues.
        """
        if nlev is None:
            nlev = self.nlev
        if nlev < 1 or nlev > self.nlev:
            raise Exception('`nlev` is out of bounds.')
        return self._eigenspectrum()[0:nlev]

    def level(self, level_ind):
        """Energy of a single qubit level.

        Parameters
        ----------
        level_ind : int
            The qubit level index starting from zero.

        Returns
        -------
        float
            Energy of the level.
        """
        if level_ind < 0 or level_ind >= self.nlev:
            raise Exception('The level is out of bounds')
        return self._eigenspectrum()[level_ind]

    def freq(self, level1, level2):
        """Transition energy/frequency between two levels of the qubit.

        Parameters
        ----------
        level1, level2 : int
            The qubit levels.

        Returns
        -------
        float
            Transition frequency between `level1` and `level2` defined
            as the difference of energies.
            Positive if `level1` < `level2`.
        """
        return self.level(level2) - self.level(level1)

    def eye(self):
        """Identity operator in the qubit eigenbasis.

        Returns
        -------
        :class:`qutip.Qobj`
            The identity operator.
        """
        return qt.qeye(self.nlev)


class TransmonFull(object):
    """A class for representing transmons using charge basis.

    Parameters
    ----------
    E_C : float
        The capacitive energy.
    E_J : float
        The Josephson energy.
    nlev : int, default=5
        The number of qubit eigenstates.
    nlim : int, default=20
        Truncation in the charge basis [-nlim, .., nlim].
    n_g : float, default=0
        Charge offset.
    """

    def __init__(self, E_C, E_J, nlev=5, nlim=20, n_g=0):
        # Most of these attributes are defined later as properties.
        self.E_C = E_C  # The charging energy.
        self.E_J = E_J  # The Josephson energy.
        self.n_g = n_g  # Charge offset.
        self.nlev = nlev  # The number of eigenstates in the qubit.
        self.nlim = nlim  # Cutoff in charge basis [-nlim, ..nlim].
        self.type = 'qubit'

    def __str__(self):
        s = ('A transmon qubit with E_J = {} '.format(self.E_J)
             + 'and E_C = {} '.format(self.E_C))
        return s

    @property
    def E_C(self):
        return self._E_C

    @E_C.setter
    def E_C(self, value):
        if value <= 0:
            raise Exception('Charging energy must be positive.')
        self._E_C = value
        self._reset_cache()

    @property
    def E_J(self):
        return self._E_J

    @E_J.setter
    def E_J(self, value):
        if value <= 0:
            print('*** Warning: Josephson energy is not positive. ***')
        self._E_J = value
        self._reset_cache()

    @property
    def n_g(self):
        return self._n_g

    @n_g.setter
    def n_g(self, value):
        self._n_g = value
        self._reset_cache()

    @property
    def nlim(self):
        return self._nlim

    @nlim.setter
    def nlim(self, value):
        if value <= 0:
            raise Exception('The number of levels must be positive.')
        self._nlim = value
        self._reset_cache()

    def _reset_cache(self):
        """Reset cached data that have already been calculated."""
        self._eigvals = None
        self._eigvecs = None

    def _cos_phi_charge(self):
        """cos phi operator in the charge basis."""
        ones = np.ones((2*self.nlim))
        return 0.5 * qt.Qobj(np.diag(ones, 1) + np.diag(ones, -1))

    def _n_charge(self):
        """Charge operator in the charge basis."""
        return qt.Qobj(np.diag(np.arange(-self.nlim, self.nlim+1)))

    def _hamiltonian_charge(self):
        """Qubit Hamiltonian in the charge basis."""
        E_C = self.E_C
        E_J = self.E_J
        n_g = self.n_g
        n = self._n_charge()
        cos_phi = self._cos_phi_charge()
        return 4 * E_C * (n-n_g)**2 - E_J * cos_phi

    def _eigenspectrum_charge(self, eigvecs_flag=False):
        """Eigenenergies and eigenstates in the charge basis."""
        if not eigvecs_flag:
            if self._eigvals is None:
                H_charge = self._hamiltonian_charge()
                self._eigvals = H_charge.eigenenergies()
            return self._eigvals
        else:
            if self._eigvals is None or self._eigvecs is None:
                H_charge = self._hamiltonian_charge()
                self._eigvals, self._eigvecs = H_charge.eigenstates()
            return self._eigvals, self._eigvecs

    def levels(self, nlev=None):
        """Qubit eigenenergies.

        Parameters
        ----------
        nlev : int, optional
            The number of eigenstates if different from `self.nlev`.

        Returns
        -------
        numpy.ndarray
            Array of eigenvalues.
        """
        if nlev is None:
            nlev = self.nlev
        if nlev < 1 or nlev > 2*self.nlim + 1:
            raise Exception('`nlev` is out of bounds.')
        return self._eigenspectrum_charge()[0:nlev]

    def level(self, level_ind):
        """Energy of a single qubit level.

        Parameters
        ----------
        level_ind : int
            The qubit level starting from zero.

        Returns
        -------
        float
            Energy of the level.
        """
        if level_ind < 0 or level_ind >= 2*self.nlim + 1:
            raise Exception('The level is out of bounds')
        return self._eigenspectrum_charge()[level_ind]

    def freq(self, level1, level2):
        """Transition frequency between two qubit levels.

        Parameters
        ----------
        level1, level2 : int
            The qubit levels.

        Returns
        -------
        float
            Transition frequency between `level1` and `level2` defined
            as the difference of energies.
            Positive if `level1` < `level2`.
        """
        return self.level(level2) - self.level(level1)

    def H(self, nlev=None):
        """Qubit Hamiltonian in its eigenbasis.

        Parameters
        ----------
        nlev : int, optional
            The number of eigenstates if different from `self.nlev`.

        Returns
        -------
        :class:`qutip.Qobj`
            The Hamiltonian operator.
        """
        return qt.Qobj(np.diag(self.levels(nlev=nlev)))

    def eye(self, nlev=None):
        """Identity operator in the qubit eigenbasis.

        Parameters
        ----------
        nlev : int, optional
            The number of qubit eigenstates if different from `self.nlev`.

        Returns
        -------
        :class:`qutip.Qobj`
            The identity operator.
        """
        if nlev is None:
            nlev = self.nlev
        if nlev < 1 or nlev > 2*self.nlim + 1:
            raise Exception('`nlev` is out of bounds.')
        return qt.qeye(nlev)

    def a(self, nlev=None):
        """Annihilation operator in the qubit eigenbasis."""
        if nlev is None:
            nlev = self.nlev
        return qt.destroy(self.nlev)

    def adag(self, nlev=None):
        """Creation operator in the qubit eigenbasis."""
        if nlev is None:
            nlev = self.nlev
        return qt.create(self.nlev)

    def cos_phi(self, nlev=None):
        """cos phi operator in the qubit eigenbasis.

        Parameters
        ----------
            The number of eigenstates if different from `self.nlev`.

        Returns
        -------
        :class:`qutip.Qobj`
            The cos phi operator.
        """
        if nlev is None:
            nlev = self.nlev
        if nlev < 1 or nlev > 2*self.nlim + 1:
            raise Exception('`nlev` is out of bounds.')
        _, evecs = self._eigenspectrum_charge(eigvecs_flag=True)
        cos_phi_op = np.zeros((nlev, nlev), dtype=complex)
        for ind1 in range(nlev):
            for ind2 in range(nlev):
                cos_phi_op[ind1, ind2] = self._cos_phi_charge().matrix_element(
                        evecs[ind1].dag(), evecs[ind2])
        return qt.Qobj(cos_phi_op)

    def n(self, nlev=None):
        """Charge operator in the qubit eigenbasis.

        Parameters
        ----------
            The number of eigenstates if different from `self.nlev`.

        Returns
        -------
        :class:`qutip.Qobj`
            The charge operator.
        """
        if nlev is None:
            nlev = self.nlev
        if nlev < 1 or nlev > 2*self.nlim + 1:
            raise Exception('`nlev` is out of bounds.')
        _, evecs = self._eigenspectrum_charge(eigvecs_flag=True)
        n_op = np.zeros((nlev, nlev), dtype=complex)
        for ind1 in range(nlev):
            for ind2 in range(nlev):
                n_op[ind1, ind2] = self._n_charge().matrix_element(
                        evecs[ind1].dag(), evecs[ind2])
        return qt.Qobj(n_op)

    def cos_phi_ij(self, level1, level2):
        """The cos phi matrix element between two eigenstates.

        Parameters
        ----------
        level1, level2 : int
            The qubit levels.

        Returns
        -------
        complex
            The matrix element of the flux operator.
        """
        if (level1 < 0 or level1 > 2*self.nlim + 1
                or level2 < 0 or level2 > 2*self.nlim + 1):
            raise Exception('Level index is out of bounds.')
        _, evecs = self._eigenspectrum_charge(eigvecs_flag=True)
        return self._cos_phi_charge().matrix_element(
                evecs[level1].dag(), evecs[level2])

    def n_ij(self, level1, level2):
        """The charge matrix element between two eigenstates.

        Parameters
        ----------
        level1, level2 : int
            The qubit levels.

        Returns
        -------
        complex
            The matrix element of the charge operator.
        """
        if (level1 < 0 or level1 > 2*self.nlim + 1
                or level2 < 0 or level2 > 2*self.nlim + 1):
            raise Exception('Level index is out of bounds.')
        _, evecs = self._eigenspectrum_charge(eigvecs_flag=True)
        return self._n_charge().matrix_element(
            evecs[level1].dag(), evecs[level2])
