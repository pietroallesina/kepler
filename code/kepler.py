import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import matplotlib.animation as anm8

def secs2days(secs):
    return secs / 24 / 60 / 60

def days2secs(days):
    return days * 24 * 60 * 60

def days2ms(days):
    return days * 24 * 60 * 60 * 1000

# e: scientific, f: decimal, g: general
def sci(x):
    return "{:.2g}".format(x)

class Body:
    def __init__(b, name, mass, position, velocity):
        b.name = name
        b.mass = mass
        b.position = position
        b.velocity = velocity

class System:

    # initialize system and initial values
    def __init__(sys, bodies, bigG=6.6743e-11, softening=0):
        x0 = np.array([b.position for b in bodies])
        v0 = np.array([b.velocity for b in bodies])
        sys.y0 = np.concatenate((x0, v0), axis=None)
        sys.bodies = bodies
        sys.n = len(bodies)
        sys.G = bigG  # Nm^2/kg^2
        sys.softening = softening

    # Define the function to compute the derivatives
    def differential(sys, t, y: np.array) -> np.array :
        n = sys.n
        x = y[:n*2].reshape(n, 2)
        v = y[n*2:].reshape(n, 2)
        a = np.zeros((n, 2))
        for i in range(n):
            for j in range(n):
                if i != j:
                    r = x[j] - x[i]
                    a[i] += sys.bodies[j].mass * r / (np.linalg.norm(r) + sys.softening)**3
            a[i] *= sys.G
        return np.concatenate((v, a), axis=None)

class Simulation:

    # initialize simulation
    def __init__(sim, system, t0, tf, steps):

        print(f"start: {t0}s, end: {tf}s, steps: {steps}, timestep: {(tf-t0)/steps}s")
        sim.system = system
        sim.bodies = system.bodies
        sim.n = system.n
        sim.t0 = t0
        sim.tf = tf
        sim.steps = int(steps)
        sim.t = np.linspace(sim.t0, sim.tf, sim.steps)
        sim.sol = integrate.solve_ivp( fun=system.differential, t_span=(sim.t0, sim.tf), y0=system.y0, t_eval=sim.t)

        sim.pos = sim.sol.y[:sim.n*2, :]
        sim.vel = sim.sol.y[sim.n*2:, :]

        # update bodies, overwrites initial values
        for j in range(system.n):
            sim.bodies[j].position = (sim.pos[j*2, :], sim.pos[j*2+1, :])
            sim.bodies[j].velocity = (sim.vel[j*2, :], sim.vel[j*2+1, :])

    # compute energy
    def energy(sim):
        n = sim.n
        E = np.zeros(sim.steps)
        for i in range(sim.steps):
            for j in range(n):
                E[i] += 0.5 * sim.bodies[j].mass * np.linalg.norm(sim.sol.y[n*2+j*2:n*2+j*2+2, i])**2
                for k in range(n):
                    if j != k:
                        r = sim.sol.y[k*2:k*2+2, i] - sim.sol.y[j*2:j*2+2, i]
                        E[i] -= sim.system.G * sim.bodies[j].mass * sim.bodies[k].mass / np.linalg.norm(r)
        return E

    # write solution to file
    def log(sim):
        n = sim.n
        with open("solution.txt", "w") as f:
            for body in sim.bodies:
                f.write(f"{body.name}\n")
                for i in range(sim.steps):
                    f.write(f"t: {sci(sim.t[i])}\n")
                    f.write(f" x=({sci(body.position[0][i])}, {sci(body.position[1][i])}), v=({sci(body.velocity[0][i])}, {sci(body.velocity[1][i])})\n")
                f.write("\n")

    # plot trajectories
    def plot_trajectory(sim):

        fig, ax = plt.subplots()

        for body in sim.bodies:
            ax.plot(body.position[0], body.position[1], label=body.name)

        ax.set_title("Trajectory")
        ax.legend()
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        # ax.set_aspect('equal') # comment for earth-moon
        plt.show()

    # plot energy
    def plot_energy(sim):
        fig, ax = plt.subplots()
        ax.plot(sim.t, sim.energy(), label=f"Energy")
        ax.set_title("Energy")
        ax.legend()
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Energy (J)")
        plt.show()

    # plot animation
    def plot_animation(sim, duration=10):

        print(f"speed: {sci((sim.tf-sim.t0)/duration)}x")

        fig, ax = plt.subplots()

        ax.set_title("Animation")
        # ax.set_aspect('equal') # comment for earth-moon
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        """     
        x_max = max([max(body.position[0]) for body in sim.bodies])
        x_min = min([min(body.position[0]) for body in sim.bodies])
        y_max = max([max(body.position[1]) for body in sim.bodies])
        y_min = min([min(body.position[1]) for body in sim.bodies])
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        """

        lines = []
        for body in sim.bodies:
            lines.append(ax.plot(body.position[0], body.position[1], label=body.name)[0])

        def update(t):
            for i in range(sim.n):
                lines[i].set_data(sim.bodies[i].position[0][:t], sim.bodies[i].position[1][:t])
            return lines

        ani = anm8.FuncAnimation(fig, func=update, frames=sim.steps, interval=1000*duration/sim.steps, repeat=False, blit=False)
        plt.show()

        return ani
    
    # compute momentum
    def momentum(sim):
        n = sim.n
        P = np.zeros(sim.steps)
        for i in range(sim.steps):
            for j in range(n):
                P[i] += sim.bodies[j].mass * np.linalg.norm(sim.sol.y[n*2+j*2:n*2+j*2+2, i])
        return P

    # plot momentum
    def plot_momentum(sim):
        fig, ax = plt.subplots()
        ax.plot(sim.t, sim.momentum(), label=f"Momentum")
        ax.set_title("Momentum")
        ax.legend()
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Momentum (kg m/s)")
        plt.show()