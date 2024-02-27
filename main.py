import math
import matplotlib.pyplot as plt

class Network:

    """Инициализация нейронной сети с заданными параметрами."""
    def __init__(self, synapses: list, sample: list, N, p, end, a, b):
        self.synapses = synapses
        self.sample = sample
        self.errors = []
        self.N = N
        self.p = p
        self.end = end
        self.a = a
        self.b = b
        print("Создана нейронная сеть со следующими параметрами: ")
        print("Вектор весов: ", self.synapses)

    """ Рассчитывает выходной сигнал нейронной сети для заданных входных данных. """
    def iteration(self, inputs: list):
        net = 0
        for i in range(len(self.synapses)):
            net += self.synapses[i] * inputs[i]
        return net

    """ Прогнозирует следующие значения на заданное количество шагов. """
    def predict(self, count_steps: int, previous: list):
        out = []
        for i in range(count_steps):
            previous = previous[1:] + [self.iteration([1] + previous)]  # сдвигаем окно и добавляем предсказание
            out.append(previous[-1])
        return out

    """ Обучает нейронную сеть на основе предоставленных обучающих данных. """
    def learning(self, p, norm_learning):
        error = 0
        for i in range(len(self.sample)-1):
            err = self.sample[i+1][-1] - self.iteration(self.sample[i])
            for j in range(p+1):
                self.synapses[j] += norm_learning * err * self.sample[i][j]
            error += (err**2)
        self.errors.append(self.get_err())
        # self.errors.append(math.sqrt(error))

    def get_err(self):
        predicted = self.predict(self.N-self.p+self.end, [func(self.a + (self.b-self.a)/(self.N-1) * i) for i in range(self.p)])
        ideal = [func(self.a + (self.b-self.a)/(self.N-1) * i) for i in range(self.p, self.N+self.end)]
        # ideal = ideal[-len(predicted):]
        # print(ideal)
        # print(predicted)
        return math.sqrt(sum([(ideal[i] - predicted[i])**2 for i in range(len(predicted))])/self.p)

def func(x):
    return (float)(0.5 * math.exp(0.5 * math.cos(0.5 * x)) + math.sin(0.5 * x))

""" обучающая выборка.создаются списки Y(x) со сдвигом на 1 каждый """
def create_sample(m, p, X, out=False):
    sample = []
    # создаем выборку
    for i in range(m-p):
        sample.append([1])  # Добавляем единицу в качестве первого элемента каждого образца
        for j in range(p):
            sample[i].append(X[j+i])  # Добавляем p элементов из входного массива X в образец
    # вывод выборки при необходимости в читаемом виде
    if out:
        for i in range(m-p):
            print(sample[i])  # Выводим образцы, если параметр out равен True
    return sample  # Возвращаем сформированную выборку

def main():
    a = -5
    b = 3
    p = 10  # Размер окна
    NORM_LEARNING = 0.5  # Норма обучения
    ITERATIONS = 1000  # Количество итераций обучения

    N = 20  # Количество шагов
    end = 2 * b - a  # Конец прогнозируемой функции
    T = end
    step = (b-a)/(N-1)  # Расстояние между t
    X = [func(a + step * i) for i in range(N)]  # Список прогнозируемых точек

    Net = Network([0] * (p + 1), create_sample(N, p, X), N, p, T, a, b)  # Создаем НС
    for i in range(ITERATIONS):
        Net.learning(p, NORM_LEARNING)  # Обучаем НС
    print("Размер конечной ошибки: {}".format(round(Net.errors[-1], 3)))  # Выводим размер конечной ошибки

    x = [(a + step * i) for i in range(N+T)]
    y = Net.predict(N+T-p, [func(a + step * i) for i in range(p)])  # Предсказываем значения

    # Добавляем значения из окна для красивого вывода
    y = [func(a + step * i) for i in range(p)] + y

    plt.plot(x, y, 'bo')  # Выводим предсказанные точки
    y = [func(a + step * i) for i in range(N+T)]
    plt.plot(x, y)  # Выводим исходную функцию
    plt.show()  # Отображаем график

    plt.semilogy([i for i in range(len(Net.errors))], Net.errors)  # Строим график логарифмической ошибки
    plt.show()  # Отображаем график


if __name__ == "__main__":
    main()

