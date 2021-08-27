import numpy as np
from scipy import stats


class Evaluation:
    def __init__(self, data, action_label, initial_investment, trading_cost_ratio=0.001):
        """

        :param data:
        :param action_label: The label of the column of action in data in order to choose between human and robot
        :param initial_investment:
        """
        # TODO: calculate the amount of share you own
        if not ('action' in data.columns):
            raise Exception('action is not in data columns')
        self.data = data
        self.initial_investment = initial_investment
        self.action_label = action_label
        self.trading_cost_ratio = trading_cost_ratio

    def evaluate(self):
        arithmetic_return = self.arithmetic_daily_return()
        # logarithmic_return = self.logarithmic_daily_return()
        average_daily_return = self.average_daily_return()
        daily_return_variance = self.daily_return_variance()
        daily_return_variance_log = self.daily_return_variance("logarithmic")
        time_weighted_return = self.time_weighted_return()
        total_return = self.total_return()
        sharp_ratio = self.sharp_ratio()
        value_at_risk = self.value_at_risk()
        volatility = self.volatility()
        portfolio = self.get_daily_portfolio_value()

        print('#' * 50)
        print(f'Arithmetic Return: {arithmetic_return}')
        # print('#' * 50)
        # print(f'Logarithmic Return: {logarithmic_return}')
        print('#' * 50)
        print(f'Average daily return: {average_daily_return}')
        print('#' * 50)
        print(f'Daily return variance (return type: Arithmetic): {daily_return_variance}')
        print('#' * 50)
        print(f'Daily return variance (return type: Logarithmic): {daily_return_variance_log}')
        print('#' * 50)
        print(f'Time weighted return: {time_weighted_return}')
        print('#' * 50)
        print(f'Total Return: {total_return} %')
        print('#' * 50)
        print(f'Sharp Ratio: {sharp_ratio}')
        print('#' * 50)
        print(f'Value at Risk (Monte Carlo method): {value_at_risk}')
        print('#' * 50)
        print(f'Volatility: {volatility}')
        print('#' * 50)
        print(f'Initial Investment: {self.initial_investment}')
        print('#' * 50)
        print(f'Final Portfolio Value: {portfolio[-1]}')
        print('#' * 50)

        print(
            f"{format(arithmetic_return, '.0f')} & {format(average_daily_return, '.2f')} & {format(daily_return_variance, '.2f')} & "
            f"{format(time_weighted_return, '.3f')} & {format(total_return, '.0f')} \% & {format(sharp_ratio, '.3f')} & {format(value_at_risk, '.2f')} & {format(volatility, '.1f')} & "
            f"{format(self.initial_investment, '.0f')} & {format(portfolio[-1], '.1f')} \\\\")

    def arithmetic_daily_return(self):
        # TODO: should we consider the days we have bought the share or we should act in general?
        # TODO: 1 + arithemtic_return
        self.arithmetic_return()
        return self.data[f'arithmetic_daily_return_{self.action_label}'].sum()

    def logarithmic_daily_return(self):
        self.logarithmic_return()
        return self.data[f'logarithmic_daily_return_{self.action_label}'].sum()

    def average_daily_return(self):
        # TODO 1 + arithemtic return
        self.arithmetic_return()
        return self.data[f'arithmetic_daily_return_{self.action_label}'].mean()

    def daily_return_variance(self, daily_return_type="arithmetic"):
        if daily_return_type == 'arithmetic':
            self.arithmetic_return()
            return self.data[f'arithmetic_daily_return_{self.action_label}'].var()
        elif daily_return_type == "logarithmic":
            self.logarithmic_return()
            return self.data[f'logarithmic_daily_return_{self.action_label}'].var()

    def time_weighted_return(self):
        rate_of_return = self.get_rate_of_return()
        mult = 1
        for i in rate_of_return:
            mult = mult * (i + 1)
        return np.power(mult, 1 / len(rate_of_return)) - 1

    def total_return(self):
        """
        TODO: Portfolio chart
        TODO: How to calculate the number of shares
        TODO: calculate the total return for annual data
        By using the initial investment, we calculate the number of shares someone can buy. Then in the end,
        we can calculate the portfolio value
        :return:
        """
        portfolio_value = self.get_daily_portfolio_value()
        return (portfolio_value[-1] - self.initial_investment) / self.initial_investment * 100

    def sharp_ratio(self):
        """
        TODO: We may need to add Risk Free Value in the future
        TODO: How to calculate Risk Free amount?
        https://www.investopedia.com/terms/s/sharperatio.asp

        Since we always have risk, we emit Risk Free Value from the formula

        Subtracting the risk-free rate from the mean return allows an investor to better isolate the profits associated
        with risk-taking activities. Generally, the greater the value of the Sharpe ratio,
        the more attractive the risk-adjusted return.

        The Sharpe ratio can also help explain whether a portfolio's excess returns are due to smart investment decisions
        or a result of too much risk. Although one portfolio or fund can enjoy higher returns than its peers, it is only
        a good investment if those higher returns do not come with an excess of additional risk.

        The greater a portfolio's Sharpe ratio, the better its risk-adjusted-performance. If the analysis results in a
        negative Sharpe ratio, it either means the risk-free rate is greater than the portfolio’s return, or the
        portfolio's return is expected to be negative. In either case, a negative Sharpe ratio does not convey any
        useful meaning.

        :return:
        """
        # self.arithmetic_return()
        # return self.data[f'arithmetic_daily_return_{self.action_label}'].mean() / self.data[
        #     f'arithmetic_daily_return_{self.action_label}'].std()
        rate_of_return = self.get_rate_of_return()
        return np.mean(rate_of_return) / np.std(rate_of_return)

    def value_at_risk(self, significance_level=5):
        """
        https://www.investopedia.com/articles/04/092904.asp

        For investors, the risk is about the odds of losing money, and VAR is based on that common-sense fact.
        By assuming investors care about the odds of a really big loss, VAR answers the question,
        "What is my worst-case scenario?" or "How much could I lose in a really bad month?"

        You can see how the "VAR question" has three elements: a relatively high level of confidence
        (typically either 95% or 99%), a time period (a day, a month or a year) and an estimate of investment loss
        (expressed either in dollar or percentage terms).

        The Variance-Covariance Method:
        This method assumes that stock returns are normally distributed. In other words, it requires that we estimate
        only two factors - an expected (or average) return and a standard deviation - which allow us to plot a normal
        distribution curve.

        :param significance_level: the level of significance for the amount of loss in historical data
        we chose significance level of 5% which means that we are sure 95% that our loss will be lower than
        k. k is in the 5% part of data
        :return:
        """
        self.arithmetic_return()
        returns = self.data[f'arithmetic_daily_return_{self.action_label}']
        avg = returns.mean()
        std = returns.std()

        historical_sorted = np.array(np.floor(sorted(self.data[f'arithmetic_daily_return_{self.action_label}'].values)),
                                     dtype=int)

        HistVAR = np.percentile(historical_sorted, significance_level)

        var_cov_VAR_95 = -1.65 * std  # For 95% confidence
        var_cov_VAR_99 = -2.33 * std  # For 99% confidence

        print(f'Historical VAR is {HistVAR}')
        print(f'Variance-Covariance VAR with 95% confidence is {var_cov_VAR_95}')
        print(f'Variance-Covariance VAR with 99% confidence is {var_cov_VAR_99}')

        np.random.seed(42)
        n_sims = 1000000
        sim_returns = np.random.normal(avg, std, n_sims)
        SimVAR = np.percentile(sim_returns, significance_level)
        return SimVAR

    def volatility(self):
        """
        Volatility is a statistical measure of the dispersion of returns for a given security or market index. In most
        cases, the higher the volatility, the riskier the security. Volatility is often measured as either the standard
        deviation or variance between returns from that same security or market index.

        The most popular and traditional measure of risk is volatility. The main problem with volatility, however,
        is that it does not care about the direction of an investment's movement: stock can be volatile because it
        suddenly jumps higher. Of course, investors aren't distressed by gains.

        :return:
        """
        self.arithmetic_return()
        return np.sqrt(len(self.data) * self.data[f'arithmetic_daily_return_{self.action_label}'].var())

    def logarithmic_return(self):
        """
        R = ln(V_close / V_open)
        :return:
        """
        self.data[f'logarithmic_daily_return_{self.action_label}'] = 0.0

        own_share = False
        for i in range(len(self.data)):
            if self.data[self.action_label][i] == 'buy' or (own_share and self.data[self.action_label][i] == 'None'):
                own_share = True
                if i < len(self.data) - 1:
                    self.data[f'logarithmic_daily_return_{self.action_label}'][i] = np.log(
                        self.data['close'][i] / self.data['close'][i + 1])
            elif self.data[self.action_label][i] == 'sell':
                own_share = False

        self.data[f'logarithmic_daily_return_{self.action_label}'] = self.data[
                                                                         f'logarithmic_daily_return_{self.action_label}'] * 100

    def arithmetic_return(self):
        """
        TODO: SELL
        R = (V_close - V_open) / V_open
        :return:
        """
        self.data[f'arithmetic_daily_return_{self.action_label}'] = 0.0

        own_share = False
        for i in range(len(self.data)):
            if (self.data[self.action_label][i] == 'buy') or (own_share and self.data[self.action_label][i] == 'None'):
                own_share = True
                if i < len(self.data) - 1:
                    self.data[f'arithmetic_daily_return_{self.action_label}'][i] = (self.data['close'][i + 1] -
                                                                                    self.data['close'][i]) / \
                                                                                   self.data['close'][i]

            elif self.data[self.action_label][i] == 'sell':
                own_share = False

        self.data[f'arithmetic_daily_return_{self.action_label}'] = self.data[
                                                                        f'arithmetic_daily_return_{self.action_label}'] * 100
        # else you have sold your share, so you would not lose or earn any more money

    def get_daily_portfolio_value(self):
        portfolio_value = [self.initial_investment]
        self.arithmetic_return()

        arithmetic_return = self.data[f'arithmetic_daily_return_{self.action_label}'] / 100
        num_shares = 0

        for i in range(len(self.data)):
            action = self.data[self.action_label][i]
            if action == 'buy' and num_shares == 0:  # then buy and pay the transaction cost
                num_shares = portfolio_value[-1] * (1 - self.trading_cost_ratio) / \
                             self.data.iloc[i]['close']
                if i + 1 < len(self.data):
                    portfolio_value.append(num_shares * self.data.iloc[i + 1]['close'])

            elif action == 'sell' and num_shares > 0:  # then sell and pay the transaction cost
                portfolio_value.append(num_shares * self.data.iloc[i]['close'] * (1 - self.trading_cost_ratio))
                num_shares = 0

            elif (action == 'None' or action == 'buy') and num_shares > 0:  # hold shares and get profit
                profit = arithmetic_return[i] * portfolio_value[len(portfolio_value) - 1]
                portfolio_value.append(portfolio_value[-1] + profit)

            elif (action == 'sell' or action == 'None') and num_shares == 0:
                portfolio_value.append(portfolio_value[-1])

        return portfolio_value

    def get_rate_of_return(self):
        portfolio = self.get_daily_portfolio_value()
        rate_of_return = [(portfolio[p + 1] - portfolio[p]) / portfolio[p] for p in range(len(portfolio) - 1)]
        return rate_of_return

    def calculate_match_actions(self, human_actions, agent_actions):
        match = 0
        total = 0
        for i in range(len(self.data)):
            if self.data.iloc[i][human_actions] == self.data.iloc[i][agent_actions] == 'buy':
                match += 1
                total += 1
            elif self.data.iloc[i][human_actions] == self.data.iloc[i][agent_actions] == 'sell':
                match += 1
                total += 1
            elif self.data.iloc[i][human_actions] != 'None' and self.data.iloc[i][agent_actions] != 'None':
                total += 1
        return match / total
