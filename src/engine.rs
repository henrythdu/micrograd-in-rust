#[derive(Debug, Clone, Copy)]
pub enum Op {
    Scalar,
    Add,
    Mul,
    Pow(f64), // To keep track of the exponent for backward pass
    ReLU,
}

#[derive(Debug)]
pub struct Value {
    pub data: f64,
    pub grad: f64,
    pub children: Vec<usize>,
    pub op: Op,
}

pub struct Arena {
    nodes: Vec<Value>,
}

impl Arena {
    pub fn new() -> Self {
        Arena { nodes: Vec::new() }
    }
    pub fn get_value(&self, id: usize) -> &Value {
        &self.nodes[id]
    }

    pub fn get_value_mut(&mut self, id: usize) -> &mut Value {
        &mut self.nodes[id]
    }

    pub fn scalar(&mut self, val: f64) -> usize {
        let new_id = self.nodes.len();
        let value = Value {
            data: val,
            grad: 0.0,
            children: Vec::new(),
            op: Op::Scalar,
        };
        self.nodes.push(value);
        new_id
    }
}

// Math Operations
impl Arena {
    pub fn add(&mut self, a: usize, b: usize) -> usize {
        let new_id = self.nodes.len();
        let value = Value {
            data: self.nodes[a].data + self.nodes[b].data,
            grad: 0.0,
            children: vec![a, b],
            op: Op::Add,
        };
        self.nodes.push(value);
        new_id
    }

    pub fn mul(&mut self, a: usize, b: usize) -> usize {
        let new_id = self.nodes.len();
        let value = Value {
            data: self.nodes[a].data * self.nodes[b].data,
            grad: 0.0,
            children: vec![a, b],
            op: Op::Mul,
        };
        self.nodes.push(value);
        new_id
    }

    pub fn pow(&mut self, a: usize, n: f64) -> usize {
        let new_id = self.nodes.len();
        let val = self.nodes[a].data.powf(n);

        let node = Value {
            data: val,
            grad: 0.0,
            children: vec![a],
            op: Op::Pow(n),
        };

        self.nodes.push(node);
        new_id
    }

    #[allow(dead_code)]
    pub fn sub(&mut self, a: usize, b: usize) -> usize {
        let neg_one = self.scalar(-1.0);
        let neg_b = self.mul(b, neg_one);
        self.add(a, neg_b)
    }

    pub fn div(&mut self, a: usize, b: usize) -> usize {
        let b_inv = self.pow(b, -1.0);
        self.mul(a, b_inv)
    }

    pub fn relu(&mut self, a: usize) -> usize {
        let new_id = self.nodes.len();
        let value = Value {
            data: self.nodes[a].data.max(0.0),
            grad: 0.0,
            children: vec![a],
            op: Op::ReLU,
        };
        self.nodes.push(value);
        new_id
    }
}

// Backward Pass
impl Arena {
    pub fn zero_grad(&mut self) {
        // Reset grad to 0
        for node in &mut self.nodes {
            node.grad = 0.0;
        }
    }
    pub fn backward(&mut self, root_id: usize) {
        if let Some(root) = self.nodes.get_mut(root_id) {
            root.grad = 1.0;
        }

        for i in (0..=root_id).rev() {
            let node_grad = self.nodes[i].grad;
            let children = self.nodes[i].children.clone();
            let op = self.nodes[i].op;

            match op {
                Op::Add => {
                    for child in children {
                        self.nodes[child].grad += node_grad;
                    }
                }
                Op::Mul => {
                    let a = self.nodes[children[0]].data;
                    let b = self.nodes[children[1]].data;
                    self.nodes[children[0]].grad += node_grad * b;
                    self.nodes[children[1]].grad += node_grad * a;
                }
                Op::Pow(n) => {
                    let a = self.nodes[children[0]].data;
                    self.nodes[children[0]].grad += node_grad * n * a.powf(n - 1.0);
                }
                Op::ReLU => {
                    let child_data = self.nodes[children[0]].data;
                    let local_derivative = if child_data > 0.0 { 1.0 } else { 0.0 };
                    self.nodes[children[0]].grad += node_grad * local_derivative;
                }
                Op::Scalar => {}
            }
        }
    }
}

// Reset arena to prevevnt memory explosion
impl Arena {
    pub fn reset_to_size(&mut self, size: usize) {
        self.nodes.truncate(size);
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }
}
