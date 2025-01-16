import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp


# Description: This was a small project to familiarize myself with
#              matplotlib in advance of a class starting which will
#              use the library.


# CONSTANTS
G = 1.0 # gravity


# Create the figure and axis
fig, ax = plt.subplots()
ax.set_xlim(-5, 5)  # Set x-axis limits
ax.set_ylim(-5, 5)  # Set y-axis limits

ax.set_title("Three Body Problem")
ax.set_xlabel("X Position")
ax.set_ylabel("Y Position")


# Create 3 placeholder 'planet' dots
dot1, = ax.plot([], [], 'o', color='green', label = 'body1')
dot2, = ax.plot([], [], 'o', color='red', label = 'body2') 
dot3, = ax.plot([], [], 'o', color='blue', label = 'body3')  

# call legend to place labels
ax.legend()

# set their initial masses
m1, m2, m3 = 1.0, 1.0, 1.0

# set initial positons 
x1, y1 = 1 , 1
x2, y2 = -1 , -1
x3, y3, = 0 , 0

# set initial velocities
vx1, vy1 = 0, .5
vx2, vy2 = -.5 , 0
vx3, vy3 = 0, -.2

# define time for simulation to run as tuple (start, finish)
time = (0,20)
num_evals = np.linspace(*time, 1000)

# define initial state array for solve_ivp function
state = [x1, y1, x2, y2, x3, y3, vx1, vy1, vx2, vy2, vx3, vy3]

# function to determine position and velocities of planets given
# any time, state and mass
def equation_system(t, state, m1, m2, m3):
    # t will remain unused as this system does not depend on time. 
    x1, y1, x2, y2, x3, y3, vx1, vy1, vx2, vy2, vx3, vy3 = state
    # compute distances
    r12 = np.sqrt((x2-x1)**2 + (y2-y1)**2)
    r13 = np.sqrt((x3-x1)**2 + (y3-y1)**2)
    r23 = np.sqrt((x3-x2)**2 + (y3-y2)**2)

    # Newton universal acceleration
    # ex: ax1 = gravity * mass object 2 * (difference between obj 1 & 2 x position) / (distance between them ^3)
    #           + gravity * mass object 3 * (difference between obj 1 & 3 x position) / (distance between them ^3)
    # -- this represents the acceleration from the two objects applied to object 1 in the x position. --
    ax1 = G * m2 * (x2 - x1) / r12**3 + G * m3 * (x3 - x1) / r13**3
    ay1 = G * m2 * (y2 - y1) / r12**3 + G * m3 * (y3 - y1) / r13**3
    ax2 = G * m1 * (x1 - x2) / r12**3 + G * m3 * (x3 - x2) / r23**3
    ay2 = G * m1 * (y1 - y2) / r12**3 + G * m3 * (y3 - y2) / r23**3
    ax3 = G * m1 * (x1 - x3) / r13**3 + G * m2 * (x2 - x3) / r23**3
    ay3 = G * m1 * (y1 - y3) / r13**3 + G * m2 * (y2 - y3) / r23**3

    return [vx1, vy1, vx2, vy2, vx3, vy3, ax1, ay1, ax2, ay2, ax3, ay3]

# call solver for the system of differential equations
solution = solve_ivp(
    equation_system,
    time,
    state,
    t_eval=num_evals,
    args=(m1, m2, m3),
    rtol=1e-8,
    atol=1e-8
)

# solutons -->
# x1_sol and y1_sol are arrays of the x1 and y1 object positions over time
x1_sol, y1_sol = solution.y[0], solution.y[1]

# x2_sol and y2_sol are arrays of the x2 and y2 object positions over time
x2_sol, y2_sol = solution.y[2], solution.y[3]

# x3_sol and y3_sol are arrays of the x3and y3 object positions over time
x3_sol, y3_sol = solution.y[4], solution.y[5]

# Initialize function for the animation (set the starting position)
def init():
    dot1.set_data([], [])  # Start with an empty position (no dot visible initially)
    dot2.set_data([], [])
    dot3.set_data([], [])
    return dot1, dot2, dot3

# Update function for the animation (move the dot)
def update(frame):
    dot1.set_data([x1_sol[frame]], [y1_sol[frame]])
    dot2.set_data([x2_sol[frame]], [y2_sol[frame]])
    dot3.set_data([x3_sol[frame]], [y3_sol[frame]])
    return dot1, dot2, dot3

# Create the animation
ani = FuncAnimation(fig, update, frames=len(num_evals), init_func=init, blit=True, interval=30)

# Show the animation
plt.show()
