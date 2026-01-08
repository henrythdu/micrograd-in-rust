use crate::engine::Area;
use rand::Rng;
#[derive(Debug)]
pub struct Neuron {
    pub weights: Vec<usize>,
    pub bias: usize,
}

impl Neuron {
    pub fn new(area: &mut Area, nin: usize) -> Self {
        let mut rng = rand::thread_rng();
        let mut weights = Vec::with_capacity(nin);
        for _ in 0..nin {
            let val = rng.gen_range(-1.0..=1.0);
            weights.push(area.scalar(val));
        }
        let bias = area.scalar(0.0);
        Neuron { weights, bias }
    }
    pub fn forward(&self, area: &mut Area, x: Vec<usize>) -> usize {
        // 1. Start with the bias
        let mut sum = self.bias;

        // 2. zip weights and inputs, multiply them, and add to the sum
        for (w_id, x_id) in self.weights.iter().zip(x.iter()) {
            let product = area.mul(*w_id, *x_id);
            sum = area.add(sum, product);
        }

        // 3. Apply ReLU
        area.relu(sum)
    }
}

#[derive(Debug)]
pub struct Layer {
    pub neurons: Vec<Neuron>,
}

impl Layer {
    pub fn new(area: &mut Area, nin: usize, nout: usize) -> Self {
        let mut neurons = Vec::with_capacity(nout);
        for _ in 0..nout {
            neurons.push(Neuron::new(area, nin));
        }
        Layer { neurons }
    }
    pub fn forward(&self, area: &mut Area, x: Vec<usize>) -> Vec<usize> {
        let mut outputs = Vec::with_capacity(self.neurons.len());
        for neuron in &self.neurons {
            outputs.push(neuron.forward(area, x.clone()));
        }
        outputs
    }
}

#[derive(Debug)]
pub struct MLP {
    pub layers: Vec<Layer>,
}

impl MLP {
    pub fn new(area: &mut Area, nin: usize, nouts: Vec<usize>) -> Self {
        let mut current_in = nin;
        let mut layers = Vec::with_capacity(nouts.len());
        for nout in nouts {
            layers.push(Layer::new(area, current_in, nout));
            current_in = nout;
        }
        MLP { layers }
    }
    pub fn forward(&self, area: &mut Area, x: Vec<usize>) -> Vec<usize> {
        let mut outputs = x;
        for layer in &self.layers {
            outputs = layer.forward(area, outputs);
        }
        outputs
    }
}
