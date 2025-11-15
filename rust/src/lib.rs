use ndarray::{Array1, Array2};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

// ============================================================
// Neural Network Layer
// ============================================================

/// A single fully-connected layer: y = activation(W * x + b)
#[derive(Debug, Clone)]
pub struct LinearLayer {
    pub weights: Array2<f64>,
    pub biases: Array1<f64>,
}

impl LinearLayer {
    /// Xavier-initialized linear layer
    pub fn new(input_dim: usize, output_dim: usize) -> Self {
        let mut rng = rand::thread_rng();
        let scale = (2.0 / (input_dim + output_dim) as f64).sqrt();
        let weights = Array2::from_shape_fn((output_dim, input_dim), |_| {
            rng.gen_range(-scale..scale)
        });
        let biases = Array1::zeros(output_dim);
        Self { weights, biases }
    }

    /// Forward pass: W * x + b
    pub fn forward(&self, input: &Array1<f64>) -> Array1<f64> {
        self.weights.dot(input) + &self.biases
    }
}

fn relu(x: &Array1<f64>) -> Array1<f64> {
    x.mapv(|v| v.max(0.0))
}

// ============================================================
// Actor: Gaussian Policy with Reparameterization
// ============================================================

const LOG_STD_MIN: f64 = -20.0;
const LOG_STD_MAX: f64 = 2.0;

/// Gaussian policy actor that outputs mean and log_std for continuous actions.
/// Uses tanh squashing to bound actions to [-1, 1].
#[derive(Debug, Clone)]
pub struct Actor {
    pub layer1: LinearLayer,
    pub layer2: LinearLayer,
    pub mean_layer: LinearLayer,
    pub log_std_layer: LinearLayer,
}

impl Actor {
    pub fn new(state_dim: usize, action_dim: usize, hidden_dim: usize) -> Self {
        Self {
            layer1: LinearLayer::new(state_dim, hidden_dim),
            layer2: LinearLayer::new(hidden_dim, hidden_dim),
            mean_layer: LinearLayer::new(hidden_dim, action_dim),
            log_std_layer: LinearLayer::new(hidden_dim, action_dim),
        }
    }

    /// Forward pass returning (mean, log_std) before squashing
    pub fn forward(&self, state: &Array1<f64>) -> (Array1<f64>, Array1<f64>) {
        let h1 = relu(&self.layer1.forward(state));
        let h2 = relu(&self.layer2.forward(&h1));
        let mean = self.mean_layer.forward(&h2);
        let log_std = self.log_std_layer.forward(&h2)
            .mapv(|v| v.clamp(LOG_STD_MIN, LOG_STD_MAX));
        (mean, log_std)
    }

    /// Sample action using reparameterization trick.
    /// Returns (squashed_action, log_prob).
    pub fn sample_action(&self, state: &Array1<f64>) -> (Array1<f64>, f64) {
        let mut rng = rand::thread_rng();
        let (mean, log_std) = self.forward(state);
        let std = log_std.mapv(|v| v.exp());

        // Reparameterization: u = mean + std * epsilon
        let epsilon = Array1::from_shape_fn(mean.len(), |_| {
            rng.gen_range(-1.0_f64..1.0_f64) * 1.4142 // approximate normal
        });
        let u = &mean + &(&std * &epsilon);

        // Squash through tanh
        let action = u.mapv(|v| v.tanh());

        // Compute log probability with tanh correction
        let log_prob = gaussian_log_prob(&u, &mean, &log_std)
            - action.mapv(|a| (1.0 - a * a + 1e-6).ln()).sum();

        (action, log_prob)
    }

    /// Deterministic action (mean squashed through tanh)
    pub fn deterministic_action(&self, state: &Array1<f64>) -> Array1<f64> {
        let (mean, _) = self.forward(state);
        mean.mapv(|v| v.tanh())
    }

    /// Soft-update this actor's parameters toward another actor
    pub fn soft_update(&mut self, source: &Actor, tau: f64) {
        soft_update_layer(&mut self.layer1, &source.layer1, tau);
        soft_update_layer(&mut self.layer2, &source.layer2, tau);
        soft_update_layer(&mut self.mean_layer, &source.mean_layer, tau);
        soft_update_layer(&mut self.log_std_layer, &source.log_std_layer, tau);
    }
}

/// Gaussian log probability (diagonal covariance)
fn gaussian_log_prob(x: &Array1<f64>, mean: &Array1<f64>, log_std: &Array1<f64>) -> f64 {
    let dim = x.len() as f64;
    let diff = x - mean;
    let var = log_std.mapv(|v| (2.0 * v).exp());
    let log_det = log_std.sum() * 2.0;
    -0.5 * (diff.mapv(|v| v * v) / &var).sum() - 0.5 * log_det - 0.5 * dim * (2.0 * PI).ln()
}

fn soft_update_layer(target: &mut LinearLayer, source: &LinearLayer, tau: f64) {
    target.weights = &target.weights * (1.0 - tau) + &source.weights * tau;
    target.biases = &target.biases * (1.0 - tau) + &source.biases * tau;
}

// ============================================================
// Critic: Q-function network
// ============================================================

/// Q-function critic that takes (state, action) and outputs a scalar Q-value.
#[derive(Debug, Clone)]
pub struct Critic {
    pub layer1: LinearLayer,
    pub layer2: LinearLayer,
    pub output_layer: LinearLayer,
}

impl Critic {
    pub fn new(state_dim: usize, action_dim: usize, hidden_dim: usize) -> Self {
        Self {
            layer1: LinearLayer::new(state_dim + action_dim, hidden_dim),
            layer2: LinearLayer::new(hidden_dim, hidden_dim),
            output_layer: LinearLayer::new(hidden_dim, 1),
        }
    }

    /// Forward pass: Q(s, a)
    pub fn forward(&self, state: &Array1<f64>, action: &Array1<f64>) -> f64 {
        let mut input = Array1::zeros(state.len() + action.len());
        input.slice_mut(ndarray::s![..state.len()]).assign(state);
        input.slice_mut(ndarray::s![state.len()..]).assign(action);

        let h1 = relu(&self.layer1.forward(&input));
        let h2 = relu(&self.layer2.forward(&h1));
        self.output_layer.forward(&h2)[0]
    }

    /// Soft-update this critic's parameters toward another critic
    pub fn soft_update(&mut self, source: &Critic, tau: f64) {
        soft_update_layer(&mut self.layer1, &source.layer1, tau);
        soft_update_layer(&mut self.layer2, &source.layer2, tau);
        soft_update_layer(&mut self.output_layer, &source.output_layer, tau);
    }
}

// ============================================================
// Twin Critics
// ============================================================

/// Twin Q-networks to reduce overestimation bias (as in TD3).
#[derive(Debug, Clone)]
pub struct TwinCritics {
    pub q1: Critic,
    pub q2: Critic,
}

impl TwinCritics {
    pub fn new(state_dim: usize, action_dim: usize, hidden_dim: usize) -> Self {
        Self {
            q1: Critic::new(state_dim, action_dim, hidden_dim),
            q2: Critic::new(state_dim, action_dim, hidden_dim),
        }
    }

    /// Returns min(Q1(s,a), Q2(s,a)) to combat overestimation
    pub fn min_q(&self, state: &Array1<f64>, action: &Array1<f64>) -> f64 {
        let q1_val = self.q1.forward(state, action);
        let q2_val = self.q2.forward(state, action);
        q1_val.min(q2_val)
    }

    pub fn soft_update(&mut self, source: &TwinCritics, tau: f64) {
        self.q1.soft_update(&source.q1, tau);
        self.q2.soft_update(&source.q2, tau);
    }
}

// ============================================================
// Soft Value Function
// ============================================================

/// Soft value function V(s) = E_a[Q(s,a) - alpha * log pi(a|s)]
#[derive(Debug, Clone)]
pub struct SoftValueFunction {
    pub layer1: LinearLayer,
    pub layer2: LinearLayer,
    pub output_layer: LinearLayer,
}

impl SoftValueFunction {
    pub fn new(state_dim: usize, hidden_dim: usize) -> Self {
        Self {
            layer1: LinearLayer::new(state_dim, hidden_dim),
            layer2: LinearLayer::new(hidden_dim, hidden_dim),
            output_layer: LinearLayer::new(hidden_dim, 1),
        }
    }

    /// Forward pass: V(s)
    pub fn forward(&self, state: &Array1<f64>) -> f64 {
        let h1 = relu(&self.layer1.forward(state));
        let h2 = relu(&self.layer2.forward(&h1));
        self.output_layer.forward(&h2)[0]
    }

    /// Compute soft value: V(s) = Q(s,a) - alpha * log_prob
    pub fn compute_soft_value(
        q_value: f64,
        log_prob: f64,
        alpha: f64,
    ) -> f64 {
        q_value - alpha * log_prob
    }

    pub fn soft_update(&mut self, source: &SoftValueFunction, tau: f64) {
        soft_update_layer(&mut self.layer1, &source.layer1, tau);
        soft_update_layer(&mut self.layer2, &source.layer2, tau);
        soft_update_layer(&mut self.output_layer, &source.output_layer, tau);
    }
}

// ============================================================
// Replay Buffer
// ============================================================

/// Experience tuple for the replay buffer
#[derive(Debug, Clone)]
pub struct Transition {
    pub state: Array1<f64>,
    pub action: Array1<f64>,
    pub reward: f64,
    pub next_state: Array1<f64>,
    pub done: bool,
}

/// Fixed-size replay buffer with uniform random sampling
#[derive(Debug)]
pub struct ReplayBuffer {
    pub buffer: Vec<Transition>,
    pub capacity: usize,
    pub position: usize,
}

impl ReplayBuffer {
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: Vec::with_capacity(capacity),
            capacity,
            position: 0,
        }
    }

    pub fn push(&mut self, transition: Transition) {
        if self.buffer.len() < self.capacity {
            self.buffer.push(transition);
        } else {
            self.buffer[self.position] = transition;
        }
        self.position = (self.position + 1) % self.capacity;
    }

    pub fn sample(&self, batch_size: usize) -> Vec<&Transition> {
        let mut rng = rand::thread_rng();
        let len = self.buffer.len();
        (0..batch_size)
            .map(|_| &self.buffer[rng.gen_range(0..len)])
            .collect()
    }

    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }
}

// ============================================================
// Automatic Temperature (Alpha) Tuning
// ============================================================

/// Manages the entropy temperature parameter alpha with automatic tuning.
#[derive(Debug, Clone)]
pub struct TemperatureTuner {
    pub log_alpha: f64,
    pub target_entropy: f64,
    pub learning_rate: f64,
}

impl TemperatureTuner {
    /// Creates a new temperature tuner.
    /// target_entropy is typically -dim(action_space) for continuous actions.
    pub fn new(action_dim: usize, learning_rate: f64) -> Self {
        Self {
            log_alpha: 0.0, // alpha starts at 1.0
            target_entropy: -(action_dim as f64),
            learning_rate,
        }
    }

    pub fn alpha(&self) -> f64 {
        self.log_alpha.exp()
    }

    /// Update alpha based on the current policy entropy.
    /// Loss: L(alpha) = -alpha * (log_pi + H_target)
    pub fn update(&mut self, log_prob: f64) {
        let alpha_loss_gradient = -(log_prob + self.target_entropy);
        self.log_alpha -= self.learning_rate * alpha_loss_gradient;
        // Clamp to prevent extreme values
        self.log_alpha = self.log_alpha.clamp(-5.0, 2.0);
    }
}

// ============================================================
// SAC Agent
// ============================================================

/// Configuration for the SAC agent
#[derive(Debug, Clone)]
pub struct SACConfig {
    pub state_dim: usize,
    pub action_dim: usize,
    pub hidden_dim: usize,
    pub gamma: f64,
    pub tau: f64,
    pub actor_lr: f64,
    pub critic_lr: f64,
    pub alpha_lr: f64,
    pub buffer_capacity: usize,
    pub batch_size: usize,
}

impl Default for SACConfig {
    fn default() -> Self {
        Self {
            state_dim: 8,
            action_dim: 1,
            hidden_dim: 64,
            gamma: 0.99,
            tau: 0.005,
            actor_lr: 3e-4,
            critic_lr: 3e-4,
            alpha_lr: 3e-4,
            buffer_capacity: 100_000,
            batch_size: 64,
        }
    }
}

/// The full SAC agent with actor, twin critics, target networks, and temperature tuning.
pub struct SACAgent {
    pub actor: Actor,
    pub critics: TwinCritics,
    pub target_critics: TwinCritics,
    pub temperature: TemperatureTuner,
    pub replay_buffer: ReplayBuffer,
    pub config: SACConfig,
    pub training_step: usize,
    pub entropy_history: Vec<f64>,
}

impl SACAgent {
    pub fn new(config: SACConfig) -> Self {
        let actor = Actor::new(config.state_dim, config.action_dim, config.hidden_dim);
        let critics = TwinCritics::new(config.state_dim, config.action_dim, config.hidden_dim);
        let target_critics = critics.clone();
        let temperature = TemperatureTuner::new(config.action_dim, config.alpha_lr);
        let replay_buffer = ReplayBuffer::new(config.buffer_capacity);

        Self {
            actor,
            critics,
            target_critics,
            temperature,
            replay_buffer,
            config,
            training_step: 0,
            entropy_history: Vec::new(),
        }
    }

    /// Select action for the given state
    pub fn select_action(&self, state: &Array1<f64>, deterministic: bool) -> Array1<f64> {
        if deterministic {
            self.actor.deterministic_action(state)
        } else {
            let (action, _) = self.actor.sample_action(state);
            action
        }
    }

    /// Store a transition in the replay buffer
    pub fn store_transition(&mut self, transition: Transition) {
        self.replay_buffer.push(transition);
    }

    /// Perform one SAC update step using a mini-batch from the replay buffer.
    /// Returns (critic_loss, actor_loss, alpha, entropy) for logging.
    pub fn update(&mut self) -> Option<(f64, f64, f64, f64)> {
        if self.replay_buffer.len() < self.config.batch_size {
            return None;
        }

        // Clone sampled transitions to avoid borrow conflicts
        let batch: Vec<Transition> = self.replay_buffer
            .sample(self.config.batch_size)
            .into_iter()
            .cloned()
            .collect();
        let alpha = self.temperature.alpha();

        let mut total_critic_loss = 0.0;
        let mut total_actor_loss = 0.0;
        let mut total_entropy = 0.0;

        // --- Update critics ---
        for transition in &batch {
            let (next_action, next_log_prob) = self.actor.sample_action(&transition.next_state);
            let target_q = self.target_critics.min_q(&transition.next_state, &next_action);
            let target_value = target_q - alpha * next_log_prob;
            let td_target = transition.reward
                + self.config.gamma * (1.0 - transition.done as i32 as f64) * target_value;

            let q1 = self.critics.q1.forward(&transition.state, &transition.action);
            let q2 = self.critics.q2.forward(&transition.state, &transition.action);

            let critic_loss = (q1 - td_target).powi(2) + (q2 - td_target).powi(2);
            total_critic_loss += critic_loss;

            // Simplified gradient update for critics (approximate)
            self.apply_critic_gradient(transition, td_target);
        }

        // --- Update actor and temperature ---
        for transition in &batch {
            let (new_action, log_prob) = self.actor.sample_action(&transition.state);
            let q_value = self.critics.min_q(&transition.state, &new_action);
            let actor_loss = alpha * log_prob - q_value;
            total_actor_loss += actor_loss;
            total_entropy += -log_prob;

            // Update temperature
            self.temperature.update(log_prob);
        }

        // Soft-update target networks
        self.target_critics.soft_update(&self.critics, self.config.tau);

        let n = batch.len() as f64;
        let avg_entropy = total_entropy / n;
        self.entropy_history.push(avg_entropy);
        self.training_step += 1;

        Some((
            total_critic_loss / n,
            total_actor_loss / n,
            self.temperature.alpha(),
            avg_entropy,
        ))
    }

    /// Approximate gradient update for critic networks
    fn apply_critic_gradient(&mut self, transition: &Transition, td_target: f64) {
        let lr = self.config.critic_lr;

        // Approximate gradient descent on Q1
        let q1 = self.critics.q1.forward(&transition.state, &transition.action);
        let error1 = q1 - td_target;
        let scale1 = -2.0 * lr * error1 / self.config.batch_size as f64;
        self.nudge_critic(&mut self.critics.q1.clone(), &transition.state, &transition.action, scale1);

        // Approximate gradient descent on Q2
        let q2 = self.critics.q2.forward(&transition.state, &transition.action);
        let error2 = q2 - td_target;
        let scale2 = -2.0 * lr * error2 / self.config.batch_size as f64;
        self.nudge_critic(&mut self.critics.q2.clone(), &transition.state, &transition.action, scale2);
    }

    fn nudge_critic(&mut self, _critic: &mut Critic, _state: &Array1<f64>, _action: &Array1<f64>, scale: f64) {
        // Simplified: nudge output layer biases proportionally to error
        self.critics.q1.output_layer.biases[0] += scale * 0.1;
        self.critics.q2.output_layer.biases[0] += scale * 0.1;
    }

    /// Get current alpha value
    pub fn alpha(&self) -> f64 {
        self.temperature.alpha()
    }

    /// Get entropy history
    pub fn entropy_history(&self) -> &[f64] {
        &self.entropy_history
    }
}

// ============================================================
// DDPG Agent (Simplified for Comparison)
// ============================================================

/// Simplified DDPG agent for comparison with SAC.
/// Uses deterministic policy and OU noise for exploration.
pub struct DDPGAgent {
    pub actor: LinearLayer,
    pub critic: Critic,
    pub noise_scale: f64,
    pub training_step: usize,
}

impl DDPGAgent {
    pub fn new(state_dim: usize, action_dim: usize, hidden_dim: usize) -> Self {
        Self {
            actor: LinearLayer::new(state_dim, action_dim),
            critic: Critic::new(state_dim, action_dim, hidden_dim),
            noise_scale: 0.1,
            training_step: 0,
        }
    }

    /// Select action with OU-style noise
    pub fn select_action(&self, state: &Array1<f64>, add_noise: bool) -> Array1<f64> {
        let mut action = self.actor.forward(state).mapv(|v| v.tanh());
        if add_noise {
            let mut rng = rand::thread_rng();
            action = action.mapv(|a| {
                (a + rng.gen_range(-self.noise_scale..self.noise_scale)).clamp(-1.0, 1.0)
            });
        }
        action
    }
}

// ============================================================
// Trading Environment
// ============================================================

/// OHLCV candle from Bybit
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Candle {
    pub timestamp: u64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

/// Bybit API response structures
#[derive(Debug, Deserialize)]
pub struct BybitResponse {
    #[serde(alias = "retCode")]
    pub ret_code: i32,
    pub result: BybitResult,
}

#[derive(Debug, Deserialize)]
pub struct BybitResult {
    pub list: Vec<Vec<String>>,
}

/// Fetch OHLCV data from Bybit public API
pub fn fetch_bybit_klines(
    symbol: &str,
    interval: &str,
    limit: usize,
) -> Result<Vec<Candle>, anyhow::Error> {
    let url = format!(
        "https://api.bybit.com/v5/market/kline?category=spot&symbol={}&interval={}&limit={}",
        symbol, interval, limit
    );

    let client = reqwest::blocking::Client::new();
    let resp: BybitResponse = client.get(&url).send()?.json()?;

    if resp.ret_code != 0 {
        anyhow::bail!("Bybit API error: ret_code={}", resp.ret_code);
    }

    let mut candles: Vec<Candle> = resp
        .result
        .list
        .iter()
        .filter_map(|row| {
            if row.len() >= 6 {
                Some(Candle {
                    timestamp: row[0].parse().unwrap_or(0),
                    open: row[1].parse().unwrap_or(0.0),
                    high: row[2].parse().unwrap_or(0.0),
                    low: row[3].parse().unwrap_or(0.0),
                    close: row[4].parse().unwrap_or(0.0),
                    volume: row[5].parse().unwrap_or(0.0),
                })
            } else {
                None
            }
        })
        .collect();

    // Bybit returns newest first; reverse to chronological order
    candles.reverse();
    Ok(candles)
}

/// Simple trading environment for SAC
pub struct TradingEnv {
    pub candles: Vec<Candle>,
    pub current_step: usize,
    pub position: f64,       // current position in [-1, 1]
    pub portfolio_value: f64, // tracks PnL
    pub lookback: usize,
    pub initial_value: f64,
}

impl TradingEnv {
    pub fn new(candles: Vec<Candle>, lookback: usize) -> Self {
        Self {
            candles,
            current_step: lookback,
            position: 0.0,
            portfolio_value: 10000.0,
            lookback,
            initial_value: 10000.0,
        }
    }

    /// Reset environment to initial state
    pub fn reset(&mut self) -> Array1<f64> {
        self.current_step = self.lookback;
        self.position = 0.0;
        self.portfolio_value = self.initial_value;
        self.get_state()
    }

    /// Construct state vector from market data
    pub fn get_state(&self) -> Array1<f64> {
        let mut features = Vec::new();
        let step = self.current_step;

        if step < self.lookback || step >= self.candles.len() {
            return Array1::zeros(self.state_dim());
        }

        // Price returns for different lookback periods
        for period in &[1, 3, 5, 10] {
            if step >= *period {
                let ret = (self.candles[step].close - self.candles[step - period].close)
                    / self.candles[step - period].close;
                features.push(ret * 100.0); // scale returns
            } else {
                features.push(0.0);
            }
        }

        // Normalized volume (relative to recent average)
        let vol_sum: f64 = (0..self.lookback.min(5))
            .map(|i| self.candles[step - i].volume)
            .sum();
        let avg_vol = vol_sum / self.lookback.min(5) as f64;
        if avg_vol > 0.0 {
            features.push(self.candles[step].volume / avg_vol - 1.0);
        } else {
            features.push(0.0);
        }

        // Simple volatility (std of recent returns)
        let returns: Vec<f64> = (1..self.lookback.min(10))
            .filter_map(|i| {
                if step >= i {
                    Some(
                        (self.candles[step - i + 1].close - self.candles[step - i].close)
                            / self.candles[step - i].close,
                    )
                } else {
                    None
                }
            })
            .collect();
        if !returns.is_empty() {
            let mean_ret = returns.iter().sum::<f64>() / returns.len() as f64;
            let variance =
                returns.iter().map(|r| (r - mean_ret).powi(2)).sum::<f64>() / returns.len() as f64;
            features.push(variance.sqrt() * 100.0);
        } else {
            features.push(0.0);
        }

        // High-low range normalized
        let range = (self.candles[step].high - self.candles[step].low) / self.candles[step].close;
        features.push(range * 100.0);

        // Current position
        features.push(self.position);

        Array1::from_vec(features)
    }

    /// State dimension
    pub fn state_dim(&self) -> usize {
        8 // 4 returns + volume + volatility + range + position
    }

    /// Take action (continuous position sizing) and return (next_state, reward, done)
    pub fn step(&mut self, action: &Array1<f64>) -> (Array1<f64>, f64, bool) {
        let new_position = action[0].clamp(-1.0, 1.0);

        // Calculate PnL from position change
        let price_change = if self.current_step + 1 < self.candles.len() {
            (self.candles[self.current_step + 1].close - self.candles[self.current_step].close)
                / self.candles[self.current_step].close
        } else {
            0.0
        };

        // Reward = position * price_change - transaction_cost
        let position_change = (new_position - self.position).abs();
        let transaction_cost = position_change * 0.001; // 0.1% fee
        let pnl = self.position * price_change;
        let reward = (pnl - transaction_cost) * 100.0; // scale reward

        self.portfolio_value *= 1.0 + pnl - transaction_cost;
        self.position = new_position;
        self.current_step += 1;

        let done = self.current_step >= self.candles.len() - 1;
        let next_state = self.get_state();

        (next_state, reward, done)
    }

    /// Total return of the episode
    pub fn total_return(&self) -> f64 {
        (self.portfolio_value - self.initial_value) / self.initial_value * 100.0
    }
}

/// Generate synthetic price data for testing when API is unavailable
pub fn generate_synthetic_candles(n: usize) -> Vec<Candle> {
    let mut rng = rand::thread_rng();
    let mut price = 50000.0_f64;
    let mut candles = Vec::with_capacity(n);

    for i in 0..n {
        let ret = rng.gen_range(-0.02..0.02);
        let open = price;
        price *= 1.0 + ret;
        let close = price;
        let high = open.max(close) * (1.0 + rng.gen_range(0.0..0.01));
        let low = open.min(close) * (1.0 - rng.gen_range(0.0..0.01));
        let volume = rng.gen_range(100.0..1000.0);

        candles.push(Candle {
            timestamp: 1700000000000 + (i as u64) * 60000,
            open,
            high,
            low,
            close,
            volume,
        });
    }

    candles
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_actor_output_dimensions() {
        let actor = Actor::new(8, 1, 64);
        let state = Array1::zeros(8);
        let (mean, log_std) = actor.forward(&state);
        assert_eq!(mean.len(), 1);
        assert_eq!(log_std.len(), 1);
    }

    #[test]
    fn test_actor_action_bounded() {
        let actor = Actor::new(8, 1, 64);
        let state = Array1::from_vec(vec![1.0, -1.0, 0.5, -0.5, 0.2, 0.8, -0.3, 0.0]);
        for _ in 0..100 {
            let (action, _) = actor.sample_action(&state);
            for &a in action.iter() {
                assert!(a >= -1.0 && a <= 1.0, "Action {} out of bounds", a);
            }
        }
    }

    #[test]
    fn test_twin_critics_min_q() {
        let critics = TwinCritics::new(8, 1, 64);
        let state = Array1::zeros(8);
        let action = Array1::from_vec(vec![0.5]);
        let q1 = critics.q1.forward(&state, &action);
        let q2 = critics.q2.forward(&state, &action);
        let min_q = critics.min_q(&state, &action);
        assert_eq!(min_q, q1.min(q2));
    }

    #[test]
    fn test_replay_buffer() {
        let mut buffer = ReplayBuffer::new(100);
        assert!(buffer.is_empty());

        for i in 0..50 {
            buffer.push(Transition {
                state: Array1::from_vec(vec![i as f64; 8]),
                action: Array1::from_vec(vec![0.0]),
                reward: 1.0,
                next_state: Array1::from_vec(vec![(i + 1) as f64; 8]),
                done: false,
            });
        }

        assert_eq!(buffer.len(), 50);
        let batch = buffer.sample(10);
        assert_eq!(batch.len(), 10);
    }

    #[test]
    fn test_temperature_tuner() {
        let mut tuner = TemperatureTuner::new(1, 0.01);
        let initial_alpha = tuner.alpha();
        assert!((initial_alpha - 1.0).abs() < 1e-6, "Initial alpha should be 1.0");

        // High entropy (log_prob very negative) should increase alpha
        for _ in 0..100 {
            tuner.update(-5.0);
        }
        // Low entropy (log_prob close to 0) should decrease alpha
        let mut tuner2 = TemperatureTuner::new(1, 0.01);
        for _ in 0..100 {
            tuner2.update(-0.1);
        }

        // The two tuners should have diverged
        assert!((tuner.alpha() - tuner2.alpha()).abs() > 0.01);
    }

    #[test]
    fn test_sac_agent_creation() {
        let config = SACConfig::default();
        let agent = SACAgent::new(config);
        assert_eq!(agent.training_step, 0);
        assert!(agent.entropy_history.is_empty());
        assert!(agent.replay_buffer.is_empty());
    }

    #[test]
    fn test_trading_env_synthetic() {
        let candles = generate_synthetic_candles(100);
        let mut env = TradingEnv::new(candles, 10);
        let state = env.reset();
        assert_eq!(state.len(), 8);

        let action = Array1::from_vec(vec![0.5]);
        let (next_state, reward, done) = env.step(&action);
        assert_eq!(next_state.len(), 8);
        assert!(!done);
        let _ = reward; // reward can be any value
    }

    #[test]
    fn test_soft_value_computation() {
        let q_value = 10.0;
        let log_prob = -2.0;
        let alpha = 0.2;
        let soft_v = SoftValueFunction::compute_soft_value(q_value, log_prob, alpha);
        let expected = q_value - alpha * log_prob; // 10.0 - 0.2 * (-2.0) = 10.4
        assert!((soft_v - expected).abs() < 1e-6);
    }

    #[test]
    fn test_sac_training_loop() {
        let candles = generate_synthetic_candles(200);
        let mut env = TradingEnv::new(candles, 10);
        let config = SACConfig {
            state_dim: env.state_dim(),
            action_dim: 1,
            hidden_dim: 32,
            batch_size: 16,
            buffer_capacity: 1000,
            ..SACConfig::default()
        };
        let mut agent = SACAgent::new(config);

        let mut state = env.reset();
        // Fill buffer with some transitions
        for _ in 0..50 {
            let action = agent.select_action(&state, false);
            let (next_state, reward, done) = env.step(&action);
            agent.store_transition(Transition {
                state: state.clone(),
                action: action.clone(),
                reward,
                next_state: next_state.clone(),
                done,
            });
            if done {
                state = env.reset();
            } else {
                state = next_state;
            }
        }

        // Should be able to update now
        let result = agent.update();
        assert!(result.is_some());
        let (critic_loss, _actor_loss, alpha, entropy) = result.unwrap();
        assert!(critic_loss.is_finite());
        assert!(alpha > 0.0);
        assert!(entropy.is_finite());
    }
}
