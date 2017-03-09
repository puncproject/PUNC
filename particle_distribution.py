import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import sys

def integration(u, mu, sigma):
    """
    This function numerically integrates

        \int_{0}{2\pi} e^{v\mu\cos(\theta)/\sigma^2} d\theta

    This is needed in derivation of 2D drifting Maxwell-Boltzmann distribution.

    There is no analytical expression for the 2D drifting Maxwell-Boltzmann
    distribution
    """
    f = np.zeros(len(u))
    for i in range(len(u)):
        v = u[i]
        f[i] = integrate.quad(lambda x: np.exp(v*mu*np.cos(x)/sigma**2),
                              0., 2*np.pi)[0]
    return f

def gaussian_distribution(v, mu, sigma):
    """
    This function returns the Gaussian distribution
    """
    return 1/(sigma * np.sqrt(2 * np.pi))* np.exp( - (v-mu)**2 / (2*sigma**2) )

def boltzmann_distribution(v, mu, sigma, d):
    """
    This function returns the Maxwell-Boltzmann distribution for both
    drifting and non-drifting particles in 2D and 3D

    Note:
    1) There is no analytical expression for the 2D drifting
    Maxwell-Boltzmann distribution, so it is given by nummerical integration

    2) For drifting plasmas these expressions are valid only in the case where
    the standard deviation is the same in all directions
    """
    if d == 2:
        if mu == 0:
            return (v/sigma**2)*np.exp( - v**2 / (2.*sigma**2) )
        else:
            return (v/(2.*np.pi*sigma**2))*(np.exp( - (v**2 + mu**2) /\
                                         (2.*sigma**2)))*integration(v,mu,sigma)
    if d == 3:
        if mu == 0:
            return 4.*np.pi*(v)**2/(sigma*np.sqrt(2 * np.pi))**3*\
                                         np.exp( - (v - mu)**2 / (2.*sigma**2) )
        else:
            return (v/mu)*(1./(np.sqrt(2*np.pi)*sigma))*(np.exp(- (v - mu)**2 /\
                       (2.*sigma**2)) - np.exp( - (v + mu)**2 / (2.*sigma**2) ))

def hist_plot(v, mu, sigma, d, distribution, n_bins, file_name):
    """
    This function takes an array of particle velocities and plots both the
    histogram of the velocity distribution, and the plot of the corresponding
    theoretical distribution
    """
    fig = plt.figure()
    count, bins, ignored = plt.hist(v, n_bins, normed=True)
    if distribution == 'Gaussian':
        f = gaussian_distribution(bins, mu, sigma)
        title = 'Gaussian distribution'
        x_label = 'velocity, v_i'
        y_label = 'f(v_i)'
    if distribution == 'Boltzmann':
        f = boltzmann_distribution(bins, mu, sigma, d)
        title = 'Maxwell-Boltzmann distribution'
        x_label = 'speed, v'
        y_label = 'f(v)'
    plt.plot(bins,f,linewidth=2, color='r')
    fig.suptitle(title, fontsize=20)
    plt.xlabel(x_label, fontsize=18)
    plt.ylabel(y_label, fontsize=16)
    fig.savefig(file_name,bbox_inches='tight',dpi=100)

def speed_distribution(v, mu, sigma, ions=0):
    """
    This function takes a velocity vector (2D or 3D) and plots both the
    distribution of each velocity component, $v_i$, and the distribution of the
    speed,

                        v = \sqrt{\sum_{0}{d}v_i^2},

    where d = 2 or 3.

    Note: Currently implemented distributions are
            1) Gaussian distribution for velocity components
            2) Non-drifting Maxwell-Boltzmann speed istribution
            3) Drifting Maxwell-Boltzmann speed istribution
    """
    n_bins = 200
    d = len(v[0,:])
    v_2 = np.zeros(len(v))
    if ions == 1:
        file_names = ["i_velocity_xcomp.png", "i_velocity_ycomp.png",
                      "i_velocity_zcomp.png", "ion_speeds.png"]
    elif ions == 0:
        file_names = ["e_velocity_xcomp.png", "e_velocity_ycomp.png",
                      "e_velocity_zcomp.png", "electron_speeds.png"]
    for i in range(d):
        v_i = v[:,i]  # velocity components
        s = np.std(v_i, ddof=0)
        print "std:: ", s, ",   sigma:  ", sigma[i]
        print "mean: ", np.mean(v_i), "   mu:", mu[i]
        hist_plot(v_i, mu[i], sigma[i], d, 'Gaussian', n_bins, file_names[i])
        v_2 += v_i**2 # Speed squared

    v_sqrt = np.sqrt(v_2) # Speed
    mu_speed = np.sqrt(np.dot(mu,mu)) # Drift velocity
    assert all(i == sigma[0] for i in sigma)

    hist_plot(v_sqrt, mu_speed, sigma[0], d,
              'Boltzmann', n_bins, file_names[-1])


if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt

    n_particles = 10000000
    T = 300        # Temperature - electrons
    m = 9.10938356e-31 # particle mass - electron
    kB = 1.38064852e-23 # Boltzmann's constant

    alpha_e = np.sqrt(kB*T/m) # Boltzmann factor

    test_2d = True
    test_3d = False

    # 2D test
    if test_2d:
        d = 2
        mu = [[1.,1], [5.,9.]]
        sigma = [[1.,1.],[5.,5.]]
        for j in range(len(mu)):
            mu_ = mu[j]
            for k in range(len(sigma)):
                sigma_ = sigma[k]
                velocities = np.empty((n_particles,d))
                for i in range(d):
                    velocities[:,i] = np.random.normal(mu_[i],
                                                       sigma_[i],
                                                       n_particles)
                speed_distribution(velocities, mu_, sigma_)

    # 3D test
    if test_3d:
        d = 3
        mu = [[1.,1.,1.], [10.,10.,10.]]
        sigma = [[1.,1.,1.],[10.,10.,10.]]
        for j in range(len(mu)):
            mu_ = mu[j]
            for k in range(len(sigma)):
                sigma_ = sigma[k]
                velocities = np.empty((n_particles,d))
                for i in range(d):
                    velocities[:,i] = np.random.normal(mu_[i],
                                                       sigma_[i],
                                                       n_particles)
                speed_distribution(velocities, mu_, sigma_)
    plt.show()
