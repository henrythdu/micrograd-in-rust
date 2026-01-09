mod engine;
mod loss;
mod nn;
use engine::Arena;
use loss::loss;
use nn::{Layer, MLP};

fn main() {
    let mut arena = Arena::new();
    let mlp = MLP::new(&mut arena, 4, vec![8, 16, 32]);
    let input = vec![arena.scalar(2.0), arena.scalar(3.0), arena.scalar(4.0)];
    let output = mlp.forward(&mut arena, input);
    let (loss, acc) = loss(
        &mut arena,
        &mlp,
        &vec![vec![2.0, 3.0, 4.0, 5.0]],
        &vec![1.0, 0.0],
    );
    println!("Output: {:?}", output);
    println!("Parameters: {:?}", mlp.parameters());
    println!("Loss: {}, Accuracy: {}", arena.get_value(loss).data, acc);
}
