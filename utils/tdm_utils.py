from matplotlib import pyplot as plt
import numpy as np
import torch
import pandas as pd

def create_sinusodial_data_samples(n_samples, len_ts, f, amp, ylims = 10 ):
    t = np.linspace(0, len_ts, n_samples)
    s = np.sin(2*np.pi*f*t)*amp*t
    x = s 
    x = x.reshape(-1, 1)

    plt.figure(figsize=(16, 6))
    plt.subplot(212)
    plt.plot(t, s, 'r', label='values',alpha =0.7)
    plt.ylim(-ylims*2, ylims*2)  
    plt.legend()
    plt.show()

    return x

def generate_points_in_2D(n):
    """
    Generates n random points in 2D and scales them by a random factor to make them either very far away or extremely close to each other.

    Parameters:
    n (int): Number of points to generate.

    Returns:
    numpy.ndarray: Scaled points.
    """
    points = np.random.rand(n, 2)  # Generate n random points in 2D
    scale_factors = np.random.choice([0.01, 0.1, 1, 10, 20,50,60, 100,120,110], size=n)  # Randomly choose scale factors from a wider range
    scaled_points = points * scale_factors[:, np.newaxis]  # Scale the points
    return scaled_points

def ppoly_data():
	"""
	if x < -10, then y = -3 · x + 4

	if -10 < x < 15, then y = x + 44

	if x > 15, then y = -4 · x + 119
	"""
	np.random.seed(9999)
	x = np.random.normal(0, 1, 1000) * 10
	y = np.where(x < -10, -3 * x + 4 , np.where(x < 15, x + 44, -4 * x + 119)) + np.random.normal(0, 3, 1000)

	return x, y

def TS_1(x1=20, samples=300, with_trend=True):
    """
    Returns:
    ========
    x,y - timestamps and values of time series for SAX exercise.
    """
    x = np.linspace(0, x1, samples)

    trend = lambda x : 0.05*x+1
    np.random.seed(5000)
    y = 0.5*np.sin(x) + np.random.uniform(size=len(x)) + (trend(x) if with_trend else 0)
    return x,y

def TS_2(x1=20, samples=300, with_trend=True):
    """
    Returns:
    ========
    x,y - timestamps and values of time series for SAX exercise.
    """
    x = np.linspace(0, x1, samples)

    trend = lambda x : 0.02*x+1
    np.random.seed(5000)
    y = 0.5*np.cos(x) + np.random.uniform(size=len(x)) + (trend(x) if with_trend else 0)
    return x,y

def draw_timeseries_plot(x, y, plot_title):
    plt.figure(figsize=(16,9))
    plt.plot(x, y, alpha=0.8, color='teal')
    plt.title(plot_title)
    plt.xlabel("Samples")
    plt.ylabel("Signal")
    plt.grid()
    
def set_default(figsize=(10, 10)):
    plt.style.use(['dark_background', 'bmh'])
    plt.rc('axes', facecolor='k')
    plt.rc('figure', facecolor='k')
    plt.rc('figure', figsize=figsize)

def plot_data(X, y, d=0, auto=False, zoom=1):
    plt.scatter(X.numpy()[:, 0], X.numpy()[:, 1], c=y, s=20, cmap=plt.cm.Spectral)
    plt.axis('square')
    plt.axis(np.array((-1.1, 1.1, -1.1, 1.1)) * zoom)
    if auto is True: plt.axis('equal')
    plt.axis('off')

    _m, _c = 0, '.15'
    plt.axvline(0, ymin=_m, color=_c, lw=1, zorder=0)
    plt.axhline(0, xmin=_m, color=_c, lw=1, zorder=0)

def plot_model(X, y, model):
    mesh = np.arange(-1.1, 1.1, 0.01)
    xx, yy = np.meshgrid(mesh, mesh)
    with torch.no_grad():
        data = torch.from_numpy(np.vstack((xx.reshape(-1), yy.reshape(-1))).T).float()
        Z = model(data).detach()
    Z = np.argmax(Z, axis=1).reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.3)
    plot_data(X, y)

def show_scatterplot(X, colors, title=''):
    colors = colors.numpy()
    X = X.numpy()
    plt.figure()
    plt.axis('equal')
    plt.scatter(X[:, 0], X[:, 1], c=colors, s=30)
    # plt.grid(True)
    plt.title(title)
    plt.axis('off')

def plot_bases(bases, width=0.04):
    bases[2:] -= bases[:2]
    plt.arrow(*bases[0], *bases[2], width=width, color=(1,0,0), zorder=10, alpha=1., length_includes_head=True)
    plt.arrow(*bases[1], *bases[3], width=width, color=(0,1,0), zorder=10, alpha=1., length_includes_head=True)

import numpy as np

def get_ts_steps(n, m, noise):
    """Sample a simple random two-class time series"""
    s = np.random.random((n)) * noise - noise / 2
    k = int(len(s) / m)
    for i in range(m):
        if i % 2 == 0:
            continue
        s[k * i:k * (i + 1)] += +1
    return s

def get_ts_wave(n, m, noise):
    """Sample a simple random two-class time series"""
    x = np.sin(np.linspace(-np.pi, m*np.pi, n))
    s = np.random.random((n)) * noise - noise / 2
    return x+s

def get_Sawtooth_wave(n, m, noise):
    """Sample a simple random two-class time series"""
    s = np.random.random((n)) * noise - noise / 2 - 1
    k = int(len(s) / m)
    lin_func = lambda x: 2/k * x
    lin = lin_func(np.arange(k))
    for i in range(m):
        if i % 2 == 0:
            continue
        s[k * i:k * (i + 1)] += 1*lin
    return s

def get_random(n, noise):
    return (np.random.rand(n)-.5)*noise

def data_shift(train, test, lags=7):
    train_shifted =  pd.concat([train.shift(lags-i) for i in range(lags+1)], axis=1)
    train_shifted.columns = [f't-{lags-i}' for i in range(lags)] + ['t']
    train_shifted = train_shifted.iloc[lags:]

    test_shifted =  pd.concat([test.shift(lags-i) for i in range(lags+1)], axis=1)
    test_shifted.columns = [f't-{lags-i}' for i in range(lags)] + ['t']
    test_shifted = test_shifted.iloc[lags:]
    
    return train_shifted, test_shifted

if __name__ == '__main__':
    pass

