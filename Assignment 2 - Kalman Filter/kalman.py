import numpy as np
import matplotlib.pyplot as plt

class KalmanFilter:
    """
    Like Low-Pass Filter with dynamic alpha.
    """
    def __init__(self, A, B, C, P, Q, R, x0):
        self.A = A
        self.B = B
        self.C = C
        self.P = P
        self.Q = Q
        self.R = R
        self.x = x0
    
    def predict(self, u = None):
        self.x = self.A @ self.x
        if u:
            self.x += self.B @ u
        self.P = self.A @ self.P @ self.A.T + self.Q
    
    def update(self, y):
        self.gain = self.P @ self.C.T / (self.C @ self.P @ self.C.T + self.R)
        
        self.x = self.x + self.gain @ (y - self.C @ self.x)
        self.P = (np.eye(self.gain.shape[0]) - self.gain @ self.C) @ self.P
    
    def filter(self, measurement, control_input = None):
        self.predict(u = control_input)
        self.update(y = measurement)
        return self.x

class KalmanExperiment:
    def __init__(self, A, B, C, P0, Q, R, x0, num_points, sigma=1, verbose=True, labels=None):
        self.A = A
        self.check_eigenvalues()
        self.B = B
        self.C = C
        self.P0 = P0
        self.Q = Q
        self.R = R
        self.x0 = x0
        
        self.num_points = num_points
        self.n = self.A.shape[0]
        self.sigma = sigma

        self.x = np.zeros((num_points, self.n))
        self.y = np.zeros((num_points, C.shape[0]))

        self.kalman_filter = KalmanFilter(A, B, C, P0, Q, R, x0) 
        self.verbose = verbose
        self.labels = list(map(str, range(self.n))) if labels is None else labels
    
    def check_eigenvalues(self):
        for eigenval in np.abs(np.linalg.eigvals(self.A)):
            assert eigenval <= 1, f'Unstable A: {eigenval}'

    def generate_data(self):
        self.w = np.random.normal(0, self.sigma, (num_points, 1))
        
        self.x[0] = self.x0

        for k in range(1, self.num_points):
            self.x[k] = self.A @ self.x[k-1] + self.B @ self.w[k]
            measurement_noise = np.random.multivariate_normal(np.zeros(self.y.shape[1]), self.R)
            self.y[k] = self.C @ self.x[k] + measurement_noise
        
        if self.verbose:
            plt.figure()
            plt.plot(self.w, label='Noise')
            plt.title('Noise')
            plt.legend()
            plt.show()

            plt.figure()
            for i in range(self.n):
                plt.plot(self.x[..., i], label=self.labels[i])
            plt.title('Generated Data')
            plt.legend()
            plt.show()

    def run(self):
        self.generate_data()

        estimated_states = np.zeros((self.num_points, self.n))
        for k in range(self.num_points):
            measurement = self.y[k]
            estimated_state = self.kalman_filter.filter(measurement)
            estimated_states[k] = estimated_state
        
        if self.verbose:
            plt.figure()
            for i in range(self.n):
                plt.plot(self.x[..., i], label='True '+self.labels[i])
                plt.plot(estimated_states[..., i], label='Estimated '+self.labels[i], linestyle='dashed')
            plt.title('Estimated Data')
            plt.legend()
            plt.show()

        return estimated_states

if __name__ == "__main__":
    # A = np.array([[1, 1], 
    #               [0, 1]])
    
    # B = np.array([[0.05], 
    #               [0.1]])
    
    # C = np.array([[1.0, 
    #                0.1]])  
    
    # P0 = np.eye(2) 
    # Q = np.eye(2) * 10
    # R = np.array([[1]]) 
    # x0 = np.array([10000, -2])

    dt = 1.0/60

    A = np.array([[1, dt, 0], 
                  [0, 1, dt],
                  [0, 0, 1]])
    
    B = np.array([[0.05], 
                  [0.05], 
                  [0.1]])
    
    C = np.array([[1.0, 
                   0,
                   0]])  
    
    P0 = np.eye(3) 
    Q = np.array([[0.05, 0.05, 0], 
                  [0.05, 0.05, 0],
                  [0, 0, 1]])
    R = np.array([[0.5]]) 
    x0 = np.array([10, -1, 10])

    num_points = 10000

    experiment = KalmanExperiment(A, B, C, P0, Q, R, x0, num_points, labels=['Displacement', 'Velocity', 'Acceleration'])
    estimated_states = experiment.run()