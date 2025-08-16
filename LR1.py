from numpy import *
import matplotlib.pyplot as plt

def cost(b, m, points):
    total = 0 # Set total error to 0

    # Loop through all points
    for i in range(len(points)):
        x = points[i, 0] # Get x value - the first column in the array
        y = points[i, 1] # Get y value - the second column in the array
        total += (y - (m * x + b)) ** 2 # Calculate the squared error, with m*x+b being the predicted value
        
    # Return the average of the squared errors
    return total / float(len(points))

# Gradient descent algorithm
def step_gradient(b_curr, m_curr, points, learning_rate):  # Takes current values of b and m, the points, and the learning rate as inputs
    b_grad = 0 # Initialize gradient for b to 0
    m_grad = 0 # Initialize gradient for m to 0

    N = float(len(points)) # Number of points

    # Loop through all points to calculate the gradients
    for i in range(len(points)):
        x = points[i, 0]
        y = points[i, 1]
        b_grad += -(2 / N) * (y - (m_curr * x + b_curr)) # Calculate gradient for b 
        m_grad += -(2 / N) * x * (y - (m_curr * x + b_curr)) # Calculate gradient for m
    
    new_b = b_curr - learning_rate * b_grad # Update b using the learning rate and gradient 
    new_m = m_curr - learning_rate * m_grad # Update m using the learning rate and gradient
    
    return new_b, new_m

def gradient_descent(points, start_b, start_m, learning_rate, num_iterations):
    b = start_b # Initialize b with the starting value
    m = start_m # Initialize m with the starting value
    history = [cost(start_b, start_m, points)] # Initialize a history list to store the cost values
    
    # Perform gradient descent for a specified number of iterations
    for i in range(num_iterations):
        b, m = step_gradient(b, m, array(points), learning_rate) # Update b and m using the step_gradient function
        if i % 100 == 0:
            print(f"Iteration {i}: b = {b}, m = {m}, error = {cost(b, m, points)}")
        history.append(cost(b, m, points)) # Store the cost for each iteration
        
    # for i in range(len(history)):
    #     print(f"{history[i]} ")

    return b, m, history # Return the final values of b and m, along with the history of costs

def run():
    points = genfromtxt('E:\Python\Linear Regression\data.csv', delimiter=',') # Load data from a CSV file
    learning_rate = 0.0000005 # Set the learning rate
    initial_b = 0 # Initial value for b
    initial_m = 0 # Initial value for m
    num_iterations = 2000 # Number of iterations for gradient descent

    print(f"Start gradient descent at b = {initial_b}, m = {initial_m}, error = {cost(initial_b, initial_m, points)}")
    
    # Perform gradient descent to find the best b and m
    b, m, history = gradient_descent(points, initial_b, initial_m, learning_rate, num_iterations)
    
    # print(f"After {num_iterations} iterations b = {b}, m = {m}, error = {cost(b, m, points)}")
    # Plot the Cost over iterations
    plt.plot(range(len(history)), history)
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.title('Cost over Iterations')
    plt.show()

if __name__ == "__main__":
    run()
