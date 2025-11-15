use sac_trading::*;

fn main() {
    println!("=== Chapter 292: SAC (Soft Actor-Critic) for Trading ===\n");

    // --- Step 1: Fetch or generate market data ---
    println!("[1] Fetching BTCUSDT data from Bybit...");
    let candles = match fetch_bybit_klines("BTCUSDT", "15", 500) {
        Ok(c) if c.len() >= 100 => {
            println!("    Fetched {} candles from Bybit", c.len());
            c
        }
        Ok(_) | Err(_) => {
            println!("    API unavailable, using synthetic data (500 candles)");
            generate_synthetic_candles(500)
        }
    };

    let lookback = 10;
    let state_dim = 8;
    let action_dim = 1;

    // --- Step 2: Train SAC agent ---
    println!("\n[2] Training SAC agent for continuous position sizing...");
    let sac_config = SACConfig {
        state_dim,
        action_dim,
        hidden_dim: 64,
        gamma: 0.99,
        tau: 0.005,
        actor_lr: 3e-4,
        critic_lr: 3e-4,
        alpha_lr: 3e-4,
        buffer_capacity: 10_000,
        batch_size: 32,
    };
    let mut sac_agent = SACAgent::new(sac_config);

    let num_episodes = 5;
    let mut sac_returns = Vec::new();

    for episode in 0..num_episodes {
        let mut env = TradingEnv::new(candles.clone(), lookback);
        let mut state = env.reset();
        let mut episode_reward = 0.0;
        let mut steps = 0;

        loop {
            let action = sac_agent.select_action(&state, false);
            let (next_state, reward, done) = env.step(&action);
            episode_reward += reward;

            sac_agent.store_transition(Transition {
                state: state.clone(),
                action: action.clone(),
                reward,
                next_state: next_state.clone(),
                done,
            });

            // Update every step if buffer has enough samples
            if let Some((critic_loss, actor_loss, alpha, entropy)) = sac_agent.update() {
                if steps % 100 == 0 {
                    println!(
                        "    Episode {}, Step {}: critic_loss={:.4}, actor_loss={:.4}, alpha={:.4}, entropy={:.4}",
                        episode + 1, steps, critic_loss, actor_loss, alpha, entropy
                    );
                }
            }

            if done {
                break;
            }
            state = next_state;
            steps += 1;
        }

        let total_ret = env.total_return();
        sac_returns.push(total_ret);
        println!(
            "  Episode {}: total_return={:.2}%, episode_reward={:.2}, steps={}, alpha={:.4}",
            episode + 1,
            total_ret,
            episode_reward,
            steps,
            sac_agent.alpha()
        );
    }

    // --- Step 3: Show entropy evolution ---
    println!("\n[3] Entropy evolution during SAC training:");
    let history = sac_agent.entropy_history();
    if history.len() >= 5 {
        let chunk_size = history.len() / 5;
        for (i, chunk) in history.chunks(chunk_size).enumerate().take(5) {
            let avg: f64 = chunk.iter().sum::<f64>() / chunk.len() as f64;
            let bar_len = ((avg.abs() * 2.0).min(40.0)) as usize;
            let bar: String = std::iter::repeat('#').take(bar_len).collect();
            println!("    Phase {}: avg_entropy={:.4} |{}", i + 1, avg, bar);
        }
    } else {
        println!("    Not enough data for entropy visualization");
    }

    // --- Step 4: Compare with DDPG (deterministic policy) ---
    println!("\n[4] Comparing SAC with DDPG (deterministic baseline)...");
    let ddpg_agent = DDPGAgent::new(state_dim, action_dim, 64);
    let mut ddpg_returns = Vec::new();

    for episode in 0..num_episodes {
        let mut env = TradingEnv::new(candles.clone(), lookback);
        let mut state = env.reset();

        loop {
            let action = ddpg_agent.select_action(&state, true);
            let (next_state, _reward, done) = env.step(&action);
            if done {
                break;
            }
            state = next_state;
        }

        let total_ret = env.total_return();
        ddpg_returns.push(total_ret);
        println!(
            "  DDPG Episode {}: total_return={:.2}%",
            episode + 1,
            total_ret
        );
    }

    // --- Step 5: Summary ---
    println!("\n[5] Summary:");
    let sac_avg: f64 = sac_returns.iter().sum::<f64>() / sac_returns.len() as f64;
    let ddpg_avg: f64 = ddpg_returns.iter().sum::<f64>() / ddpg_returns.len() as f64;

    println!("    SAC  avg return: {:.2}%", sac_avg);
    println!("    DDPG avg return: {:.2}%", ddpg_avg);
    println!("    Final alpha (temperature): {:.4}", sac_agent.alpha());
    println!("    SAC training steps: {}", sac_agent.training_step);
    println!();

    println!("=== Key SAC Properties ===");
    println!("  - Stochastic policy (Gaussian + tanh squashing)");
    println!("  - Twin critics to prevent Q-value overestimation");
    println!("  - Automatic temperature (alpha) tuning");
    println!("  - Entropy maximization for robust exploration");
    println!("  - Continuous position sizing in [-1, 1]");
    println!("  - Off-policy learning with replay buffer");
    println!();
    println!("=== Done ===");
}
