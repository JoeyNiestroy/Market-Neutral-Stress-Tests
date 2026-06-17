# Market-Neutral-Stress-Tests
IC-controlled stress testing framework for market-neutral mean-variance portfolios under forecast error and model misspecification.


This project studies how different forms of forecast error affect the performance of market-neutral mean-variance portfolios.

The workflow is:

1. Build a top-N equity universe from database and compute daily returns.
2. Estimate rolling alpha and beta coefficients and obtain out-of-sample idiosyncratic returns. Where market returns are determined by the universe
3. Treat future idiosyncratic returns as an oracle signal and degrade them into 'synthetic' alpha forecasts with controlled ICs via:
$\hat{\alpha} = \rho \alpha_{\text{true}} + \sqrt{1-\rho^2}\ * \varepsilon$

where $\(\alpha_{\text{true}}\)$ is the true future idio return calculated in 2, $\(\varepsilon\)$ is a noise vector, and $\(\rho\)$ is the desired IC. 

4. Introduce various forecast-error types including Gaussian noise, heavy tails, sparse outliers, sector based, volatility bias, beta bias, and adversarial.

5. Solve a market-neutral mean-variance optimization problem with turnover and max holding penalties for each scenario.

6. Evaluate the effects of forecast quality and model misspecification on performance, beta neutrality, turnover, drawdowns, and transaction costs.

The goal is not to produce a tradable strategy, but to provide a research framework for understanding the sensitivity of portfolio optimization to noisy and structurally biased alpha forecasts.

