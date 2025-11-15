# Chapter 292: SAC (Soft Actor-Critic) for Trading

## Introduction: Maximum Entropy Reinforcement Learning for Trading

Soft Actor-Critic (SAC) is an off-policy, actor-critic deep reinforcement learning algorithm based on the maximum entropy framework. Originally introduced by Haarnoja et al. (2018), SAC simultaneously maximizes expected return and policy entropy, producing agents that are both performant and robust. For trading, this entropy-augmented objective translates into strategies that maintain a healthy degree of exploration, avoid premature convergence to brittle deterministic policies, and adapt gracefully to shifting market regimes.

Traditional reinforcement learning algorithms for trading -- such as DQN or vanilla policy gradient methods -- often converge to deterministic policies that overfit to historical market conditions. When regime changes occur (e.g., a shift from trending to mean-reverting markets), these policies can fail catastrophically. SAC addresses this fundamental challenge by explicitly encouraging the agent to remain stochastic, exploring multiple viable trading strategies simultaneously rather than committing prematurely to a single approach.

In this chapter we build a complete SAC implementation in Rust targeting continuous position sizing for cryptocurrency trading on the Bybit exchange. Our agent learns not just whether to buy or sell, but precisely how much capital to allocate to a position -- a continuous action space that SAC handles naturally through its Gaussian policy with reparameterization.

## Mathematical Foundations

### The Maximum Entropy Objective

Standard RL maximizes the expected sum of discounted rewards:

```
J_standard(pi) = E_{tau ~ pi} [ sum_{t=0}^{T} gamma^t * r(s_t, a_t) ]
```

SAC augments this with an entropy bonus, yielding the maximum entropy objective:

```
J_SAC(pi) = E_{tau ~ pi} [ sum_{t=0}^{T} gamma^t * ( r(s_t, a_t) + alpha * H(pi(.|s_t)) ) ]
```

where `H(pi(.|s_t)) = -E_{a ~ pi} [log pi(a|s_t)]` is the entropy of the policy at state `s_t`, and `alpha > 0` is the temperature parameter controlling the trade-off between reward maximization and entropy maximization.

The temperature `alpha` is crucial: when `alpha` is large, the agent favors exploration and randomness; when `alpha` approaches zero, SAC recovers the standard RL objective. For trading, the right `alpha` balances exploitation of known profitable patterns against exploration of new strategies -- critical when markets evolve.

### Soft Q-Functions and Twin Critics

SAC uses soft Q-functions that incorporate the entropy term. The soft Bellman equation becomes:

```
Q_soft(s_t, a_t) = r(s_t, a_t) + gamma * E_{s_{t+1}} [ V_soft(s_{t+1}) ]
```

where the soft value function is:

```
V_soft(s_t) = E_{a_t ~ pi} [ Q_soft(s_t, a_t) - alpha * log pi(a_t | s_t) ]
```

To combat overestimation bias (a known issue in Q-learning), SAC employs **twin critics** (Q1 and Q2), taking the minimum:

```
Q_target(s_t, a_t) = r + gamma * ( min(Q1_target(s', a'), Q2_target(s', a')) - alpha * log pi(a' | s') )
```

where `a' ~ pi(.|s')`. This clipped double-Q trick, borrowed from TD3, prevents the policy from exploiting overestimated Q-values -- particularly important in noisy financial environments where spurious correlations abound.

### Reparameterization Trick

SAC parameterizes the policy as a Gaussian distribution. To enable gradient-based optimization through the sampling process, it uses the reparameterization trick:

```
a_t = tanh( mu_theta(s_t) + sigma_theta(s_t) * epsilon ),   epsilon ~ N(0, I)
```

The `tanh` squashing ensures actions lie in `[-1, 1]`, which maps naturally to position sizing (from full short to full long). The log-probability must account for this transformation:

```
log pi(a|s) = log mu_gaussian(u|s) - sum_i log(1 - tanh^2(u_i))
```

where `u` is the pre-squash action. This correction is essential for correct entropy computation.

### Automatic Temperature Tuning

Rather than manually setting `alpha`, SAC can learn it by solving a constrained optimization:

```
alpha* = argmin_{alpha} E_{a ~ pi*} [ -alpha * log pi*(a|s) - alpha * H_target ]
```

where `H_target` is a target entropy (typically set to `-dim(A)` for continuous actions). The loss for `alpha` is:

```
L(alpha) = E_{a_t ~ pi} [ -alpha * ( log pi(a_t | s_t) + H_target ) ]
```

This automatic tuning is crucial for trading because the appropriate level of stochasticity changes as the agent learns and as market conditions evolve.

## Why Entropy Maximization Helps Trading

### 1. Exploration in Non-Stationary Environments

Financial markets are fundamentally non-stationary. A strategy that works in a bull market may fail in a bear market. By maintaining entropy in the policy, SAC agents continue exploring alternative strategies even after finding profitable ones. This ongoing exploration means the agent can adapt more quickly when market regimes shift.

### 2. Robustness to Market Changes

A deterministic policy (like those from DDPG) puts all its weight on a single action for each state. If the Q-function has even small errors -- inevitable given market noise -- the policy exploits these errors. SAC's stochastic policy averages over a range of actions, providing natural regularization against Q-function estimation errors.

### 3. Multi-Modal Strategy Discovery

Markets often present multiple viable strategies simultaneously (e.g., momentum and mean reversion at different time scales). SAC's entropy objective naturally discovers and maintains multiple modes in the policy, effectively hedging across strategies.

### 4. Improved Sample Efficiency

SAC is off-policy, meaning it can learn from historical data collected by any policy. Combined with the replay buffer, this makes it significantly more sample-efficient than on-policy methods like PPO -- critical when market data is limited or expensive.

### 5. Continuous Position Sizing

Unlike discrete-action RL methods that can only output {buy, hold, sell}, SAC naturally handles continuous actions. The agent can output a precise position size in [-1, 1], representing the fraction of capital to allocate. This fine-grained control allows more nuanced risk management.

## Rust Implementation

Our Rust implementation includes the following core components:

### Actor Network (Gaussian Policy)

The actor outputs mean and log-standard-deviation for a Gaussian distribution over actions. The reparameterization trick enables backpropagation through the sampling operation. A `tanh` squashing function constrains actions to valid position sizes.

### Twin Critics (Q1, Q2)

Two independent Q-networks estimate action values. During target computation, we take the minimum of the two to prevent overestimation. Each critic receives state-action pairs and outputs a scalar Q-value.

### Replay Buffer

An experience replay buffer stores transitions (state, action, reward, next_state, done) and supports uniform random sampling for mini-batch training. This enables off-policy learning from historical experiences.

### Automatic Temperature Tuning

The log-alpha parameter is optimized to maintain a target entropy level. This adapts the exploration-exploitation trade-off dynamically throughout training.

### Training Loop

The SAC update performs these steps per iteration:
1. Sample mini-batch from replay buffer
2. Update critics using soft Bellman backup with twin clipping
3. Update actor to maximize Q-value minus alpha * log_prob
4. Update alpha to match target entropy
5. Soft-update target networks using Polyak averaging

## Bybit Data Integration

Our implementation fetches real OHLCV data from the Bybit public API for backtesting. The agent observes a feature vector constructed from:

- Price returns over multiple lookback periods
- Normalized volume
- Simple volatility estimates
- Current position

This provides the agent with sufficient market microstructure information to learn meaningful trading strategies.

## Comparison with DDPG

| Feature | SAC | DDPG |
|---------|-----|------|
| Policy type | Stochastic (Gaussian) | Deterministic |
| Exploration | Entropy-driven (intrinsic) | External noise (OU process) |
| Overestimation | Twin critics + clipping | Single critic |
| Temperature | Auto-tuned alpha | Manual noise schedule |
| Robustness | High (entropy regularization) | Lower (brittle to noise) |
| Sample efficiency | High (off-policy + entropy) | Moderate (off-policy) |

SAC consistently outperforms DDPG in trading environments due to its inherent robustness and better exploration properties. The automatic temperature tuning removes a major hyperparameter burden compared to DDPG's noise schedule.

## Key Takeaways

1. **Maximum Entropy RL** adds an entropy bonus to the reward, encouraging the agent to remain stochastic and exploratory -- ideal for non-stationary financial markets.

2. **Twin Critics** (clipped double-Q) prevent overestimation bias, which is especially harmful in noisy financial environments where the Q-function can latch onto spurious patterns.

3. **Reparameterization Trick** enables gradient-based optimization of a stochastic Gaussian policy, naturally supporting continuous position sizing in [-1, 1].

4. **Automatic Temperature Tuning** adapts the exploration-exploitation balance dynamically, removing a critical hyperparameter and allowing the agent to self-regulate its stochasticity.

5. **Off-Policy Learning** with a replay buffer makes SAC sample-efficient, able to learn from historical market data without requiring online interaction.

6. **Robustness** through entropy regularization makes SAC policies less likely to overfit to specific market regimes, providing more consistent out-of-sample performance compared to deterministic alternatives like DDPG.

7. **Rust Implementation** provides the performance characteristics needed for real-time trading applications, with memory safety guarantees that are critical for financial systems.

## References

- Haarnoja, T., et al. (2018). "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor." ICML 2018.
- Haarnoja, T., et al. (2018). "Soft Actor-Critic Algorithms and Applications." arXiv:1812.05905.
- Fujimoto, S., et al. (2018). "Addressing Function Approximation Error in Actor-Critic Methods." ICML 2018. (TD3 / Twin critics)
- Lillicrap, T., et al. (2015). "Continuous control with deep reinforcement learning." (DDPG)
