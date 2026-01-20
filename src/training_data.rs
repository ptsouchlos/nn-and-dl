//! Training input data for neural network training

use crate::types::{DataVector, Scalar};

pub struct Sample {
    image: DataVector,
    label: Scalar,
}
