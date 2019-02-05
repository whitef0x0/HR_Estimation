"""
Matplotlib Animation Example

author: Jake Vanderplas
email: vanderplas@astro.washington.edu
website: http://jakevdp.github.com
license: BSD
Please feel free to use and modify this, but keep the above information. Thanks!
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation

# First set up the figure, the axis, and the plot element we want to animate


class AnimateGraph():
	# initialization function: plot the background of each frame
	def __init__(self, x_data, y_data, interval):
		self.interval = interval
		self.x_data = x_data
		self.y_data = y_data

		self.fig = plt.figure()
		self.ax = plt.axes(xlim=(0, 200), ylim=(-1, 1))
		self.line, = ax.plot([], [], lw=2)

	def init_animation(self):
		line.set_data(self.x_data, self.y_data)
	    return line,

	# animation function.  This is called sequentially
	def animate(self, i):
	    x = np.linspace(0, 2, 1000)
	    y = np.sin(2 * np.pi * (x - 0.01 * i))
	    line.set_data(x, y)
	    return line,

	def run(self)
		# call the animator.  blit=True means only re-draw the parts that have changed.
		anim = animation.FuncAnimation(self.fig, self.animate, init_func=self.init_animation,
		                               frames=len(self.x_data), interval=self.interval, blit=True)

		# save the animation as an mp4.  This requires ffmpeg or mencoder to be
		# installed.  The extra_args ensure that the x264 codec is used, so that
		# the video can be embedded in html5.  You may need to adjust this for
		# your system: for more information, see
		# http://matplotlib.sourceforge.net/api/animation_api.html
		anim.save('basic_animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

		plt.show()