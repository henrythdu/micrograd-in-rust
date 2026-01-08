mod engine;
mod nn;
use engine::Area;
use nn::{Layer, MLP};

fn main() {
    let mut area = Area::new();
    let mlp = MLP::new(&mut area, 4, vec![8, 16, 32]);
    // let input = vec![2.0, 3.0, 4.0];
    // let output = neuron.forward(&mut area, input);
    println!("Output: {:?}", mlp);
}
