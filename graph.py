import matplotlib.pyplot as plt

class Graph(object):
    def __init__(self, title, numPlots):
        self.windowTitle = title
        self.rows = numPlots
        self.cols = 1
        self.reset()

    def reset(self):
        self.graph = plt.figure(self.windowTitle)
        self.index = 1

    def addSubPlot(self, title, x_axis_title, y_axis_title, x_data, y_data):
        current_subplot = self.graph.add_subplot(self.rows, self.cols, self.index)    
        current_subplot.set_title(title)
        current_subplot.set_xlabel(x_axis_title)
        current_subplot.set_ylabel(y_axis_title)
        current_subplot.plot(x_data, y_data, color = "blue")
        self.index = self.index + 1

    def show(self):
        self.graph.show()
        plt.show()
        
