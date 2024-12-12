from scipy.stats import norm
from scipy.integrate import quad

class SlippageSolver:

    def __init__(self, historical_OHLC) -> None:
        '''
        Min-Max Scaled 데이터에 적합
        작동 양상: 매수호가 - 매도호가 차이가 0에 근접(Min-Max Scaled)하거나 1틱(Raw)에 근접할 경우, 50% 미만.
        쓰임새: 스프레드가 벌어졌을 때, 제공되는 10호가를 넘어서 주문 가능
        '''
        self.historical = historical_OHLC

    def pdf(self, x, current_price):
        mu = current_price
        sigma = 0.025 * mu
        return norm.pdf(x, loc=mu, scale=sigma)
    def probability_between_previous_and_next(self, current_price, previous_price, next_price):
        lower_limit = previous_price
        upper_limit = next_price
        def integrand(x):
            return self.pdf(x, current_price)
        probability, _ = quad(integrand, lower_limit, upper_limit)
        return probability
    
ss = SlippageSolver([])
print(ss.probability_between_previous_and_next(0.03, 0.029, 0.0301))