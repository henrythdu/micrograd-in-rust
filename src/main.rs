mod engine;
mod nn;
use engine::Arena;
use nn::{Layer, MLP};

fn main() {
    let mut arena = Arena::new();
    let mlp = MLP::new(&mut arena, 4, vec![8, 16, 32]);
    let input = vec![arena.scalar(2.0), arena.scalar(3.0), arena.scalar(4.0)];
    let output = mlp.forward(&mut arena, input);
    println!("Output: {:?}", output);
}
