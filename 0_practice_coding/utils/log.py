import seaborn as sns
import matplotlib.pyplot as plt


def log_loss(x, y, title):
    sns.lineplot(x=x, y=y)
    plt.title(title)
    plt.show()
