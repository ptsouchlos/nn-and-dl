use crate::{
    training_data::Sample,
    types::{DataVector, Scalar},
};
use std::iter::zip;

pub struct Network {
    sizes: Vec<usize>,
    biases: Vec<DataVector>,
    weights: Vec<DataVector>,
}

fn sigmoid_vec(z: DataVector) -> DataVector {
    z.map(|val| sigmoid(val))
}

fn sigmoid(z: Scalar) -> Scalar {
    1 as Scalar / (1 as Scalar + (-z).exp())
}

impl Network {
    pub fn new(sizes: Vec<usize>) -> Self {
        let mut biases = Vec::new();
        let mut weights = Vec::new();
        for sz in &sizes {
            biases.push(DataVector::new_random(*sz));
            weights.push(DataVector::new_random(*sz));
        }
        Self {
            sizes,
            biases,
            weights,
        }
    }

    pub fn number_of_layers(&self) -> usize {
        self.sizes.len()
    }

    pub fn biases(&self) -> &Vec<DataVector> {
        &self.biases
    }

    pub fn weights(&self) -> &Vec<DataVector> {
        &self.weights
    }

    pub fn feed_forward(&mut self, input: DataVector) -> DataVector {
        let mut result = input;
        for (b, w) in zip(&self.biases, &self.weights) {
            let z = b.add_scalar(w.dot(&result));
            result = sigmoid_vec(z)
        }

        result
    }

    pub fn sgd(
        &self,
        _training_data: Vec<Sample>,
        _epochs: usize,
        _mini_batch_size: usize,
        _learning_rate: f32,
    ) {
        // For each epoch, run a mini back and update the biases and weights
        // After each epoch, print out the error
    }
}

#[cfg(test)]
mod tests {
    use crate::network::Network;

    #[test]
    fn create_network() {
        let sizes = vec![784, 15, 10];
        let nn = Network::new(sizes.clone());
        assert_eq!(nn.number_of_layers(), 3);
        assert_eq!(nn.biases().len(), 3);
        assert_eq!(nn.weights().len(), 3);

        for (idx, row) in nn.biases().iter().enumerate() {
            assert_eq!(row.len(), sizes[idx]);
        }

        for (idx, row) in nn.weights().iter().enumerate() {
            assert_eq!(row.len(), sizes[idx]);
        }
    }
}
