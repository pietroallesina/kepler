import kepler as kp
import numpy as np

sun = kp.Body("sun", 1.99e30, np.array([0, 0]), np.array([0, 0]))
mercury = kp.Body("mercury", 3.3e23, np.array([4.6e10, 0]), np.array([0, 6.22e4]))
venus = kp.Body("venus", 4.87e24, np.array([1.08e11, 0]), np.array([0, 3.5e4]))
earth = kp.Body("earth", 5.97e24, np.array([1.5e11, 0]), np.array([1e1, 3e4]))
moon = kp.Body("moon", 7.35e22, np.array([1.5e11, 4.5e8]), np.array([-1e3, 3e4]))
mars = kp.Body("mars", 6.39e23, np.array([2.28e11, 0]), np.array([0, 2.4e4]))
jupiter = kp.Body("jupiter", 1.9e27, np.array([7.78e11, 0]), np.array([0, 1.3e4]))
saturn = kp.Body("saturn", 5.68e26, np.array([1.43e12, 0]), np.array([0, 9.7e3]))

sys = kp.System([earth, moon], softening=0e4)

t0 = 0
tf = kp.days2secs(200)
steps = 1e4
sim = kp.Simulation(sys, t0, tf, steps)

sim.log()
sim.plot_trajectory()
sim.plot_energy()
sim.plot_momentum()
# sim.plot_animation(duration=20)