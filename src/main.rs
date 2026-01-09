mod engine;
mod loss;
mod make_moons;
mod nn;
mod viz;
use engine::Arena;
use loss::loss;
use make_moons::make_moons;
use nn::MLP;

fn main() {
    // 1. Setup Data
    let (x_train, y_train) = make_moons(500, 0.1);
    viz::visualize_data(&x_train, &y_train, "Original Moons Data");
    // 2. Setup Model & Arena
    let mut arena = Arena::new();

    // MLP with 2 inputs (x,y coords), two hidden layers of 16, and 1 output
    let model = MLP::new(&mut arena, 2, vec![16, 16, 1]);

    // IMPORTANT: Capture where the weights end in the Arena nodes.
    // Everything after this index is "temporary math" that we reset every loop.
    let weights_end_idx = arena.len();

    // 3. Training Loop
    for step in 0..100 {
        // --- CLEANUP ---
        arena.reset_to_size(weights_end_idx);

        // --- FORWARD & LOSS ---
        // Pass references &x_train and &y_train
        let (loss_id, accuracy) = loss(&mut arena, &model, &x_train, &y_train);

        // --- BACKWARD ---
        arena.zero_grad();
        arena.backward(loss_id);

        // --- OPTIMIZATION (SGD) ---
        let learning_rate = 1.0 - 0.9 * (step as f64 / 100.0);

        for &p_id in &model.parameters() {
            let grad = arena.get_value(p_id).grad;
            let node = arena.get_value_mut(p_id);
            node.data -= learning_rate * grad;
        }

        if step % 1 == 0 {
            let loss_val = arena.get_value(loss_id).data;
            println!(
                "step {} loss {:.4}, accuracy {:.4}%",
                step,
                loss_val,
                accuracy * 100.0
            );
        }
    }
}
