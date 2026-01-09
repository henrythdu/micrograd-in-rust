use crate::engine::Arena;
use crate::nn::MLP;

pub fn loss(arena: &mut Arena, model: &MLP, X: &Vec<Vec<f64>>, y: &Vec<f64>) -> (usize, f64) {
    let scores: Vec<usize> = X
        .iter()
        .map(|xrow| {
            // 1. Convert the current row to Arena IDs
            let input_ids: Vec<usize> = xrow.iter().map(|&v| arena.scalar(v)).collect();

            // 2. Pass those IDs into the model
            model.forward(arena, input_ids)[0]
        })
        .collect();
    // svm "max-margin" loss
    let mut losses = Vec::with_capacity(scores.len());
    let mut correct_count = 0.0;

    for (&yi, &scorei) in y.iter().zip(scores.iter()) {
        let neg_yi = arena.scalar(-yi);
        let one = arena.scalar(1.0);
        let weighted_score = arena.mul(neg_yi, scorei);
        let margin = arena.add(weighted_score, one);
        let loss = arena.relu(margin);
        losses.push(loss);

        let score_val = arena.get_value(scorei).data;
        if (yi > 0.0) == (score_val > 0.0) {
            correct_count += 1.0;
        }
    }

    let zero = arena.scalar(0.0);
    let data_loss_sum = losses.iter().fold(zero, |acc, &loss| arena.add(acc, loss));
    let len_losses = arena.scalar(losses.len() as f64);
    let data_loss = arena.div(data_loss_sum, len_losses);
    // L2 Regularization Loss
    let alpha = arena.scalar(1e-4);
    let parameters = model.parameters();
    let reg_sum = parameters.iter().fold(arena.scalar(0.0), |acc, &param| {
        let p_sq = arena.pow(param, 2.0);
        arena.add(acc, p_sq)
    });
    let reg_loss = arena.mul(alpha, reg_sum);
    let total_loss = arena.add(data_loss, reg_loss);
    let accuracy = correct_count / y.len() as f64;
    return (total_loss, accuracy);
}
