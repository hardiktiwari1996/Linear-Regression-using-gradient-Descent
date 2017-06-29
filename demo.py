from numpy import *

def compute_error_for_line_given_points(b, m, points):
	total_error = 0 
	for i in range(len(points)):
		x = points[i, 0]
		y = points[i, 1]

		total_error += (y - (m * x + b))**2

	return total_error / float(len(points))


def step_gradient(b_current, m_current, points, learning_rate):
	b_gradient = 0
	m_gradient = 0

	N = float(len(points))

	for i in range(len(points)):
		x = points[i, 0]
		y = points[i, 1]

		b_gradient += -(2/N) * (y - ((m_current * x) + b_current))
		m_gradient += -(2/N) * x * (y - ((m_current * x) + b_current))


	new_b = b_current - (learning_rate * b_gradient)
	new_m = m_current - (learning_rate * m_gradient)

	return [new_b, new_m]


def gradient_descent_runner(points, b, m, learning_rate, num_iternations):
	for i in range(num_iternations):
		b, m = step_gradient(b, m, array(points), learning_rate)

	return [b, m]


def run():
	# Step 1 - Collect our data
	points = genfromtxt('data.csv', delimiter=',')

	# Step 2 - define our hyperparameter
	# How fast should our model converge?
	learning_rate = 0.0001
	# Equation of line y = mx + b 
	initial_b = 0
	initial_m = 0
	num_iternations = 1000

	# Step 3 - train our model
	print 'Starting gradient descent at b={0}, m={1}, error={2}'.format(initial_b, initial_m, 
	compute_error_for_line_given_points(initial_b, initial_m, points))

	print 'Running...'

	[b, m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iternations)

	print 'Final b={0}, m={1}, error={2}'.format(b, m, compute_error_for_line_given_points(b, m, points))

if __name__ == '__main__':
	run()
