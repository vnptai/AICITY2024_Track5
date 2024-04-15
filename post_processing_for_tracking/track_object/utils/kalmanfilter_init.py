from copy import deepcopy
import numpy as np
from numpy import dot, zeros, eye, isscalar
# import numpy.linalg as linalg
from post_processing_for_tracking.track_object.utils.common import reshape_z
from scipy.optimize import linear_sum_assignment
import scipy as sp


class KalmanFilter(object):
    r"""
    Parameters
    ----------
    dim_x : int
        Number of state variables for the Kalman filter. For example, if
        you are tracking the position and velocity of an object in two
        dimensions, dim_x would be 4.
        This is used to set the default size of P, Q, and u

    dim_z : int
        Number of of measurement inputs. For example, if the sensor
        provides you with position in (x,y), dim_z would be 2.

    """

    def __init__(self, dim_x, dim_z):
        if dim_x < 1:
            raise ValueError('dim_x must be 1 or greater')
        if dim_z < 1:
            raise ValueError('dim_z must be 1 or greater')

        self.dim_x = dim_x
        self.dim_z = dim_z

        self.x = zeros((dim_x, 1))  # state
        self.P = eye(dim_x)  # uncertainty covariance
        self.Q = eye(dim_x)  # process uncertainty
        self.B = None  # control transition matrix
        self.F = eye(dim_x)  # state transition matrix
        self.H = zeros((dim_z, dim_x))  # Measurement function
        self.R = eye(dim_z)  # state uncertainty
        self._alpha_sq = 1.  # fading memory control
        self.M = np.zeros((dim_z, dim_z))  # process-measurement cross correlation
        self.z = np.array([[None] * self.dim_z]).T

        # gain and residual are computed during the innovation step. We
        # save them so that in case you want to inspect them for various
        # purposes
        self.K = np.zeros((dim_x, dim_z))  # kalman gain
        self.y = zeros((dim_z, 1))
        self.S = np.zeros((dim_z, dim_z))  # system uncertainty
        self.SI = np.zeros((dim_z, dim_z))  # inverse system uncertainty

        # identity matrix. Do not alter this.
        self._I = np.eye(dim_x)

        # these will always be a copy of x,P after predict() is called
        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()

        # these will always be a copy of x,P after update() is called
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()

        # Only computed only if requested via property
        # self.inv = np.linalg.inv
        self.inv = sp.linalg.inv

    def predict(self, u=None, B=None, F=None, Q=None):
        """
        Predict next state (prior) using the Kalman filter state propagation
        equations.

        Parameters
        ----------

        u : np.array
            Optional control vector. If not `None`, it is multiplied by B
            to create the control input into the system.

        B : np.array(dim_x, dim_z), or None
            Optional control transition matrix; a value of None
            will cause the filter to use `self.B`.

        F : np.array(dim_x, dim_x), or None
            Optional state transition matrix; a value of None
            will cause the filter to use `self.F`.

        Q : np.array(dim_x, dim_x), scalar, or None
            Optional process noise matrix; a value of None will cause the
            filter to use `self.Q`.
        """
        if B is None:
            B = self.B
        if F is None:
            F = self.F
        if Q is None:
            Q = self.Q
        elif isscalar(Q):
            Q = eye(self.dim_x) * Q

        # x = Fx + Bu
        if B is not None and u is not None:
            self.x = dot(F, self.x) + dot(B, u)
        else:
            self.x = dot(F, self.x)

        # P = FPF' + Q
        self.P = self._alpha_sq * dot(dot(F, self.P), F.T) + Q

        # save prior
        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()

    def update(self, z, R=None, H=None):
        """
        Add a new measurement (z) to the Kalman filter.

        If z is None, nothing is computed. However, x_post and P_post are
        updated with the prior (x_prior, P_prior), and self.z is set to None.

        Parameters
        ----------
        z : (dim_z, 1): array_like
            measurement for this update. z can be a scalar if dim_z is 1,
            otherwise it must be convertible to a column vector.

        R : np.array, scalar, or None
            Optionally provide R to override the measurement noise for this
            one call, otherwise  self.R will be used.

        H : np.array, or None
            Optionally provide H to override the measurement function for this
            one call, otherwise self.H will be used.
        """

        # set to None to force recompute
        if z is None:
            self.z = np.array([[None] * self.dim_z]).T
            self.x_post = self.x.copy()
            self.P_post = self.P.copy()
            self.y = zeros((self.dim_z, 1))
            return
        z = reshape_z(z, self.dim_z, self.x.ndim)
        if R is None:
            R = self.R
        elif isscalar(R):
            R = eye(self.dim_z) * R
        if H is None:
            H = self.H
        # y = z - Hx
        # error (residual) between measurement and prediction
        self.y = z - dot(H, self.x)
        # common subexpression for speed
        PHT = dot(self.P, H.T)
        # S = HPH' + R
        # project system uncertainty into measurement space
        self.S = dot(H, PHT) + R
        ##################################### leak cpu

        self.SI = self.inv(self.S)

        ##########################
        # map system uncertainty into kalman gain
        self.K = dot(PHT, self.SI)

        # x = x + Ky
        # predict new x with residual scaled by the kalman gain
        self.x = self.x + dot(self.K, self.y)

        # P = (I-KH)P(I-KH)' + KRK'
        # This is more numerically stable
        # and works for non-optimal K vs the equation
        # P = (I-KH)P usually seen in the literature.

        I_KH = self._I - dot(self.K, H)
        self.P = dot(dot(I_KH, self.P), I_KH.T) + dot(dot(self.K, R), self.K.T)

        # save measurement and posterior state
        self.z = deepcopy(z)
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()
