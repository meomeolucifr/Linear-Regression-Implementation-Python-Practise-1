# Linear Regression Implementation - Study Time vs. Score

This is my Day 1 practice implementation of Linear Regression as part of my journey learning AI and Data Science. The project implements a simple linear regression model to predict student scores based on their study time, inspired by Siraj Raval's tutorial.

## Project Overview

This project demonstrates the implementation of Linear Regression from scratch using Python. The model predicts the relationship between study time and scores, using gradient descent to find the optimal parameters of the linear equation.

### Features

- Implementation of Linear Regression using gradient descent
- Visualization of the data points and regression line using matplotlib
- Cost function implementation to measure model performance
- Step-by-step gradient descent optimization

## Dependencies

- NumPy - For numerical computations
- Matplotlib - For data visualization

## Dataset

The dataset (`data.csv`) contains two columns:
- First column: Study time (hours)
- Second column: Score achieved

## Implementation Details

The code implements the following key components:

1. **Cost Function**: Calculates the Mean Squared Error (MSE) between predicted and actual values
2. **Gradient Descent**: Optimizes the parameters (slope and intercept) of the linear regression line
3. **Data Visualization**: Plots the original data points and the regression line

## How to Run

1. Make sure you have Python installed along with the required dependencies
2. Clone this repository
3. Run the script:
```bash
python LR1.py
```

## Learning Resources

This implementation is based on the tutorial by Siraj Raval, as part of my learning journey in AI and Data Science.

## Future Improvements

- Add more sophisticated error handling
- Implement cross-validation
- Add support for multiple features
- Include model evaluation metrics

## License

This project is open source and available under the MIT License.

---
*Note: This is a learning project created as part of my AI and Data Science practice.*
