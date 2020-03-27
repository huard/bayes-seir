# pip install numpy scipy xarray

from scipy.integrate import odeint
import numpy as np
import xarray as xr


"""
Notes
-----

S : Susceptibles are not infected but could become infected.
E : Exposed
I : Infectious have the disease and can transmit it to susceptibles
R : Removed individuals may have had or not the disease but can't become infected or transmit it to the others. 
    They could be vaccinated, be immune after recovery, have died or be isolated. 

beta : the average number of contacts per person per time, multiplied by the probability of disease transmission in a 
       contact between a susceptible and an infectious subject.
gamma : 1 / T_infectious

T_infectious: number of days patient is infectious
T_incubation: number of days between exposition and infection
R0 = beta / gamma 


---

Median incubation period: 5 days
Symptoms within 11.5 days in 97.5% of cases

Stephen A. Lauer, Kyra H. Grantz, Qifang Bi, Forrest K. Jones, Qulu Zheng, Hannah R. Meredith, Andrew S. Azman, Nicholas G. Reich, Justin Lessler. The Incubation Period of Coronavirus Disease 2019 (COVID-19) From Publicly Reported Confirmed Cases: Estimation and Application. Annals of Internal Medicine, 2020; DOI: 10.7326/M20-0504

"""

# Variable names
vn = {"s": "Susceptible",
      "e": "Exposed",
      "i": "Infected",
      "r": "Recovered",
      "hm": "Recovering at home (mild)",
      "hs": "Recovering at home (severe)",
      "ho": "Recovering in hospital",
      "f": "Dying patient",
      "rm": "Recovered from mild symptoms",
      "rs": "Recovered from severe symptoms",
      "hc": "Hospitalized (critical care unit)",
      "rd": "Dead"
      }


class SIR:
    def __init__(self, N, I0, beta, gamma):
        """
        SIR model

        S -> I -> R

        Parameters
        ----------
        N : int
          Total population
        I0 : int
          Initial number of infected individuals.
        beta : array
          Contact rate
        gamma : float
          Mean recovery rate [1/days]


        Notes
        -----
        R_0 = beta / gamma
        """
        self.N = N
        self.I0 = I0
        self.beta = beta
        self.gamma = gamma
        self.S0 = N - I0
        self.days = len(beta)
        self.args = (self.N, self.beta, self.gamma)
        self.variables = ("s", "i", "r")

    @staticmethod
    def _deriv(y, t, N, beta, gamma):
        """
        The SIR model differential equations
        """
        S, I, R = y
        # print(t)
        dSdt = -beta[int(t)] * S * I / N
        dIdt = beta[int(t)] * S * I / N - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt

    def run(self):
        # Initial conditions vector
        y0 = self.S0, self.I0, 0
        # Integrating the SIR equations over the time grid, t
        t = list(range(0, self.days))
        # Getting results
        result = odeint(self._deriv, y0, t, args=self.args, hmax=1)
        return self.to_xr(result.T)

    def to_xr(self, results):
        # Convert to xarray
        days = np.arange(self.days)
        data_vars = {k: xr.Variable(("time",), x, {"name": vn[k]}) for (k, x) in zip(self.variables, results.T)}
        coords = {"time": days}
        return xr.Dataset(data_vars, coords)

    __call__ = run


class SEIR(SIR):
    def __init__(self, N, I0, E0, beta, gamma, sigma):
        """
        SEIR model

        S -> E -> I -> R

        Parameters
        ----------
        N : int
          Total population
        I0 : int
          Initial number of infected individuals.
        E0 : int
          Initial number of exposed individuals
        beta : array
          Contact rate
        gamma : float
          Mean recovery rate [1/days]
        sigma: float
          Incubation rate. Average duration of incubation is 1/sigma
          """

        self.sigma = sigma
        self.E0 = E0
        super().__init__(N, I0, beta, gamma)
        self.args = (self.N, self.beta, self.gamma, self.sigma)
        self.S0 = N - I0 - E0
        self.variables = ("s", "e", "i", "r")

    @staticmethod
    def _deriv(y, t, N, beta, gamma, sigma):
        """
        The SIR model differential equations
        """
        S, E, I, R = y
        # print(t)
        dSdt = -beta[int(t)] * S * I / N
        dEdt = beta[int(t)] * S * I / N - sigma * E
        dIdt = sigma * E - gamma * I
        dRdt = gamma * I
        return dSdt, dEdt, dIdt, dRdt

    def run(self):
        # Initial conditions vector
        y0 = self.S0, self.E0, self.I0, 0
        # Integrating the SIR equations over the time grid, t
        t = list(range(0, self.days))
        # Getting results
        result = odeint(self._deriv, y0, t, args=self.args, hmax=1)
        return self.to_xr(result)


class SEIRC(SEIR):
    def __init__(self, **kwargs):
        """
        SEIR model with clinical dynamics emulating the epidemic calculator.

        Parameters
        ----------
        N : int
          Total population
        I0 : int
          Initial number of infected individuals.
        E0 : int
          Initial number of exposed individuals
        beta : array
          Contact rate times probability of infection.
        D_incubation : float
          Average duration of incubation.
        D_infectious
          Duration patient is infectious.
        p_severe : float
          Fraction of recoveries with severe symptoms
        p_fatal : float
          Fraction of recoveries that end up being fatal
        D_hospital_lag : float
          Hospitalization rate [0,1]
        D_recovery_mild : float
          Recovery time for mild cases (days)
        D_recovery_severe : float
          Recovery time for severe cases (days)
        D_death : float
          Days to death after infection

        Notes
        -----
        p_mild : 1 - p_severe - p_fatal
          Fraction of recoveries with mild symptoms

        References
        ----------
        http://gabgoh.github.io/COVID/index.html

        """
        d = dict(
            N=7e6,
            I0=1,
            E0=0,
            R=2.2,
            D_incubation=5.2,
            D_infectious=2.9,
            D_recovery_mild=14 - 2.9,
            D_recovery_severe=31.5 - 2.9,
            D_hospital_lag=5,
            D_death=32 - 2.9,
            p_fatal=0.02,
            p_severe=0.2
        )
        p = self.params = d
        d.update(kwargs)

        p["S0"] = p['N'] - p['I0'] + p["E0"]
        p['beta'] = p['R'] / p['D_infectious']
        p['gamma'] = 1 / p['D_infectious']
        p['a'] = 1 / p['D_incubation']
        p['p_mild'] = 1 - p['p_severe'] - p['p_fatal']

        self.days = len(p["beta"])
        self.variables = ("s", "e", "i", "hm", "hs", "ho", "f", "rm", "rs", "rd")

    @staticmethod
    def _deriv(y, t, p):
        """
        dS        = -beta*I*S
        dE        =  beta*I*S - a*E
        dI        =  a*E - gamma*I
        dMild     =  p_mild*gamma*I   - (1/D_recovery_mild)*Mild
        dSevere   =  p_severe*gamma*I - (1/D_hospital_lag)*Severe
        dSevere_H =  (1/D_hospital_lag)*Severe - (1/D_recovery_severe)*Severe_H
        dFatal    =  p_fatal*gamma*I  - (1/D_death)*Fatal
        dR_Mild   =  (1/D_recovery_mild)*Mild
        dR_Severe =  (1/D_recovery_severe)*Severe_H
        dR_Fatal  =  (1/D_death)*Fatal

        Mild, severe, severe hospital and fatal are all sick but not infectious. They do not contribute to further
        infections.
        """
        s, e, i, mild, severe, severe_h, fatal, r_mild, r_severe, r_fatal = y
        it = int(t)

        ds = -p['beta'][it] * s * i
        de = p['beta'][it] * s * i - p['a'] * e
        di = p['a'] * e - p['gamma'] * i
        dmild = p['p_mild'] * p['gamma'] * i - 1 / p["D_recovery_mild"] * mild
        dsevere = p['p_severe'] * p['gamma'] * i - 1 / p["D_hospital_lag"] * severe
        dsevere_h = 1 / p["D_hospital_lag"] * severe - 1 / p['D_recovery_severe'] * severe_h
        dfatal = p['p_fatal'] * p['gamma'] * i - 1 / p['D_death'] * fatal
        dr_mild = 1 / p['D_recovery_mild'] * mild
        dr_severe = 1 / p['D_recovery_severe'] * severe_h
        dr_fatal = 1 / p['D_death'] * fatal

        return ds, de, di, dmild, dsevere, dsevere_h, dfatal, dr_mild, dr_severe, dr_fatal

    def run(self):
        # Initial conditions vector
        p = self.params

        y0 = p['S0'] / p["N"], p['E0'] / p["N"], p['I0'] / p["N"], 0, 0, 0, 0, 0, 0, 0
        # Integrating the equations over the time grid, t
        t = list(range(0, self.days))
        # Getting results
        result = odeint(self._deriv, y0, t, args=(self.params,), hmax=1)

        out = self.to_xr(result)
        return self.post_process(out)

    def post_process(self, ds):
        p = self.params
        ds['hospital'] = p['N'] * (ds['ho'] + ds['f'])
        ds['recovered'] = p['N'] * (ds['rm'] + ds['rs'])
        ds['dead'] = p['N'] * ds['rd']
        ds['infectious'] = p['N'] * ds['i']
        ds['exposed'] = p['N'] * ds['e']
        return ds


# WIP
class SEIRC_ICU(SEIR):
    def __init__(self, **kwargs):
        """
        SEIR model with clinical dynamics, including ICU bed limitations.

        Also, we differentiate between infections instead of recovery.

        Parameters
        ----------
        N : int
          Total population
        I0 : int
          Initial number of infected individuals.
        E0 : int
          Initial number of exposed individuals
        beta : array
          Contact rate times probability of infection.
        D_incubation : float
          Average duration of incubation.
        D_infectious
          Duration patient is infectious.
        p_severe : float
          Fraction of recoveries with severe symptoms
        p_fatal : float
          Fraction of recoveries that end up being fatal
        D_hospital_lag : float
          Hospitalization rate [0,1]
        D_recovery_mild : float
          Recovery time for mild cases (days)
        D_recovery_severe : float
          Recovery time for severe cases (days)
        D_death : float
          Days to death after infection

        Notes
        -----
        p_mild : 1 - p_severe - p_fatal
          Fraction of recoveries with mild symptoms

        """
        d = dict(
            N=7e6,
            I0=1,
            E0=0,
            R=2.2,
            D_incubation=5.2,
            D_infectious=2.9,
            D_recovery_mild=14 - 2.9,
            D_recovery_severe=31.5 - 2.9,
            D_hospital_lag=5,
            D_death=32 - 2.9,
            p_fatal=0.02,
            p_severe=0.2
        )
        p = self.params = d
        d.update(kwargs)

        p["S0"] = p['N'] - p['I0'] + p["E0"]
        p['beta'] = p['R'] / p['D_infectious']
        p['gamma'] = 1 / p['D_infectious']
        p['sigma'] = 1 / p['D_incubation']
        p['p_mild'] = 1 - p['p_severe'] - p['p_fatal']

        self.days = len(p["beta"])
        self.variables = ("s", "e", "i", "hm", "hs", "ho", "f", "rm", "rs", "rd")

    @staticmethod
    def _deriv(y, t, p):
        """
        dS        = -(beta_a * Ia + beta_m * Im + beta_s * Is + beta_h * Ih) * S
        dE        =  (beta_a * Ia + beta_m * Im + beta_s * Is + beta_h * Ih) * S - sigma*E
        dIa       =  p_a * sigma*E - gamma_a * Ia                      # asymptomatic (50%)
        dIm       =  p_m * sigma*E - gamma_m * Im                      # mild         (30%)
        dIs       =  p_s * sigma*E - gamma_s * Is - 1/hosp_lag * Is    # severe       (15%)
        dIc       =  p_s * sigma*E - 1/hosp_lag * Ic - 1/T_death * Ic  # critical at home (they die) (5%)
        dIh       =  1 / hosp_lag * Ic                                 # critical at hospital
        dRa       =  gamma_a * Ia
        dRm       =  gamma_m * Im
        dRs       =  gamma_s * Is -

        dSevere_H =  (1/D_hospital_lag)*Severe - (1/D_recovery_severe)*Severe_H
        dFatal    =  p_fatal*gamma*I  - (1/D_death)*Fatal
        dR_Mild   =  (1/D_recovery_mild)*Mild
        dR_Severe =  (1/D_recovery_severe)*Severe_H
        dR_Fatal  =  (1/D_death)*Fatal
        """
        s, e, i, mild, severe, severe_h, fatal, r_mild, r_severe, r_fatal = y
        it = int(t)

        ds = -p['beta'][it] * s * i
        de = p['beta'][it] * s * i - p['sigma'] * e
        di = p['sigma'] * e - p['gamma'] * i
        dmild = p['p_mild'] * p['gamma'] * i - 1 / p["D_recovery_mild"] * mild
        dsevere = p['p_severe'] * p['gamma'] * i - 1 / p["D_hospital_lag"] * severe
        dsevere_h = 1 / p["D_hospital_lag"] * severe - 1 / p['D_recovery_severe'] * severe_h
        dfatal = p['p_fatal'] * p['gamma'] * i - 1 / p['D_death'] * fatal
        dr_mild = 1 / p['D_recovery_mild'] * mild
        dr_severe = 1 / p['D_recovery_severe'] * severe_h
        dr_fatal = 1 / p['D_death'] * fatal
        # dd = p['p_fatal_home'] * rs / p["T_death_home"] + p['p_fatal_hospital'] * hs / p["T_death_hospital"]
        # dr = 1 / p['T_recovery_mild'] * rm + 1 / p["T_recovery_severe"] * rs

        return ds, de, di, dmild, dsevere, dsevere_h, dfatal, dr_mild, dr_severe, dr_fatal

    def run(self):
        # Initial conditions vector
        p = self.params

        y0 = p['S0'] / p["N"], p['E0'] / p["N"], p['I0'] / p["N"], 0, 0, 0, 0, 0, 0, 0
        # Integrating the equations over the time grid, t
        t = list(range(0, self.days))
        # Getting results
        result = odeint(self._deriv, y0, t, args=(self.params,), hmax=1)

        out = self.to_xr(result)
        return self.post_process(out)

    def post_process(self, ds):
        p = self.params
        ds['hospital'] = p['N'] * (ds['ho'] + ds['f'])
        ds['recovered'] = p['N'] * (ds['rm'] + ds['rs'])
        ds['dead'] = p['N'] * ds['rd']
        ds['infectious'] = p['N'] * ds['i']
        ds['exposed'] = p['N'] * ds['e']
        return ds


def test_SEIR_gabgoh():
    """Results roughly compare with http://gabgoh.github.io/COVID/

    I suspect differences are due to time stepping schemes."""
    Rt = np.array(100 * [2.2] + 100 * [0.73])
    Tinc = 5.2
    Tinf = 2.9
    s = SEIR(7E7, 1, 0, beta=Rt / Tinf, gamma=1 / Tinf, sigma=1 / Tinc)
    return s.run()


def test_SEIRC_gabgoh():
    """Results roughly compare with http://gabgoh.github.io/COVID/

    I suspect differences are due to time stepping schemes. gabgoh uses RK4 integration"""

    Rt = np.array(100 * [2.2] + 100 * [0.73])
    s = SEIRC(R=Rt)
    return s.run()


def qc():
    import COVID19Py
    covid19 = COVID19Py.COVID19()

    return covid19.getLocationById(44)
    # N = 8485000  # Population 2019
