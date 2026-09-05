# Reflection: Global Portfolio Optimization and Risk Model

The hardest part of building an out of sample portfolio model isn't writing the optimization code; it's keeping your own backtest honest.

When I first compared my mean variance optimized portfolio against an "in sample" baseline, the numbers looked suspiciously clean. That's because the baseline weights were fit on the entire dataset, including the exact periods the walk forward test was supposed to evaluate. That's a subtle form of look ahead bias: the baseline wasn't a fair ceiling; it was a portfolio that had already peeked at the answer.

To fix it, I had to build a proper apples to apples comparison using just the first 36-month training window, matching the information set actually available at the time. It's a small code adjustment, but it stops the model from overstating what optimization actually adds.

That instinct to distrust results that look too neat, should have gone further. While I applied Ledoit Wolf shrinkage to the covariance matrix, a necessary safeguard given 39 assets across a decade of monthly data, I left expected returns entirely unregularized, relying on raw historical averages. In mean variance optimization, estimation error in expected returns wrecks the math far faster than covariance noise. Hand an optimizer noisy means, and it will happily double down on whatever asset just got lucky. Regularizing those return estimates via a factor model or Black Litterman priors is the glaring gap that needs closing next.

A few other structural blind spots remain unaddressed. The model treats the risk free rate as a static line through an eleven year window where global rates swung from zero to over 5%. It also converts local prices to USD via spot FX before computing returns, which blurs the line between actual stock performance and currency movement. Factor in a survivorship biased ticker universe drawn from today's known constituents, and the reality is clear: this framework is a rigorous learning exercise, not a production ready mandate.
