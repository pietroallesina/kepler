import kepler as kp
import numpy as np

particles = []
for i in range(5):
    # particles.append( kp.Body(i, 10, np.random.randint(low=-100, high=100, size=2), np.random.random_sample(size=2)-0.5) )
    particles.append( kp.Body(i, 1, np.random.randint(low=-500, high=500, size=2), np.array([0, 0])) )
    print(particles[i].position)

sys = kp.System(particles, bigG=1e0, softening=1e1)

sim = kp.Simulation(sys, t0=0, tf=1e4, steps=1e3)

# sim.log()
sim.plot_trajectory()
# sim.plot_animation(duration=20)
sim.plot_energy()